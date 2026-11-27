"""Viewer slots, route failures, and file results at local and child boundaries."""

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import autorig_api
import finish_api
import generate_api as api
import pytest
import rig_api

from image_to_3dlab.glb_turn import build_glb, parse_glb
from tests.test_coverage_viewer_jobs_http import studio as http_studio
from tests.test_coverage_viewer_workers import Child
from tests.test_viewer_model_info_api import _sized_glb
from tests.test_viewer_rig_api import valid_sidecar

studio = http_studio


def test_mac_trellis_job_passes_attention_selection_to_child(tmp_path, monkeypatch):
    manager = api.JobManager()
    monkeypatch.setattr(api, "JOBS", manager)
    spec = dataclasses.replace(api.BACKENDS["trellis"], wrapper=api.WRAPPER, build_args=api._trellis_build_args)
    monkeypatch.setitem(api.BACKENDS, "trellis", spec)
    monkeypatch.setattr(api, "LEARNED_BASELINE_PATH", tmp_path / "baseline.json")
    job = manager.create(tmp_path / "source.png", spec.validate_settings({"sparse_attn_backend": "mlx-fp16"}),
                         "trellis", "prop", None, tmp_path)
    threads = []

    class Thread:
        def __init__(self, *, target, args, daemon, name=None):
            assert daemon
            self.target, self.args = target, args
            threads.append(target)

        def start(self):
            if self.target is api._read_process:
                self.target(*self.args)

        def join(self, timeout):
            assert timeout == 5

    def launch(argv, *, env, stdout, **kw):
        assert argv[argv.index("--sparse-attn-backend") + 1] == "mlx"
        assert env["I2L_MLX_ATTN_DTYPE"] == "fp16"
        job.output_path.write_bytes(_sized_glb(1))
        return NS(pid=3210, stdout=io.BytesIO(b"Loading TRELLIS\n"), wait=lambda: 0)

    monkeypatch.setattr(api.threading, "Thread", Thread)
    monkeypatch.setattr(api.subprocess, "Popen", launch)
    api._run_job(job)
    assert threads == [api._read_process, api._rss_monitor], job.events[-1]
    assert job.status == "done" and manager.active is None
    assert job.output_path.is_file()
    assert not (job.directory / "pid").exists()


def test_sf3d_without_optional_legacy_record_still_moves_output(tmp_path):
    job = api.Job("a" * 32, tmp_path, tmp_path / "source.png", tmp_path / "result.glb", {}, "sf3d")
    folder = tmp_path / "commercial"
    folder.mkdir()
    generated = folder / "source__sf3d__model.glb"
    generated.write_bytes(_sized_glb(1))
    api._sf3d_finalize(job)
    assert job.output_path.is_file() and not generated.exists()
    assert not job.manifest_path.exists()
    api._sf3d_parse_line(job, "     ")
    assert job.events == []


def test_fully_local_sf3d_setup_has_no_mac_weights_warning(tmp_path, monkeypatch):
    import backend_catalog

    path = tmp_path / "sf3d" / "system.py"
    path.parent.mkdir()
    path.touch()
    monkeypatch.setattr(api, "SF3D_REPO_DEFAULT", tmp_path)
    monkeypatch.setitem(backend_catalog.BY_ID, "sf3d", NS(describe=lambda host: {
        "weights": [{"label": "encoder", "bytes_expected": 100, "bytes_present": 100}],
    }))
    result = api._sf3d_readiness(api.APPLE)
    assert result["ready"] and result["warning"] is None
    assert result["missing_weights"] == []


def test_prop_turn_waits_for_active_generation(studio):
    generation = api.JOBS.create(api.OUTPUT_ROOT / "picture.png", {}, "sf3d", "prop", None, api.OUTPUT_ROOT)
    status, _, body = studio("/api/props/runs/missing/turn", {"prop": "barrel"})
    assert status == 409
    assert "wait" in body["error"]
    assert api.JOBS.active == generation.id
    assert not api.PROPS_JOBS.jobs


def test_skin_front_approval_is_recorded_even_when_back_needs_setup(studio, monkeypatch):
    store = api.skins_api.STORE
    skin = api.skins_api.Skin("blue", "paint blue", {"front": "source.png", "back": "back.png"})
    attempt = api.OUTPUT_ROOT / "try.png"
    attempt.touch()
    skin.tries["front"].append({"number": 1, "picture": "try.png"})
    store.save(skin)
    monkeypatch.setattr(api.image_api, "BINARY", api.OUTPUT_ROOT / "no-picture-engine")
    status, _, body = studio(f"/api/skins/{skin.id}/approve", {"number": 1})
    assert status == 503 and body["needs_setup"]
    stored = store.get(skin.id)
    assert stored.approved["front"] == "try.png" and stored.tries["back"] == []
    assert not api.image_api.MANAGER.jobs


@pytest.mark.parametrize(("case", "message"), [("unchanged", "no corrections"), ("unbound", "no Blender binding")])
def test_rebind_rejects_semantically_incomplete_sidecars(tmp_path, case, message):
    book = rig_api.RigJobManager(tmp_path)
    asset, scene = b"model", b"blend"
    document = json.loads(valid_sidecar(asset, scene))
    if case == "unchanged":
        document["corrections"].clear()
    else:
        document["binding"]["joints"].clear()
    with pytest.raises(ValueError, match=message):
        book.create("sample.glb", asset, scene, json.dumps(document).encode())
    assert not book.jobs and book.active is None
    assert not list(tmp_path.iterdir())


def test_finish_hero_carries_height_and_provenance_into_the_result(tmp_path, monkeypatch):
    book = finish_api.FinishJobManager(tmp_path, tmp_path)
    job = finish_api.FinishJob("a" * 32, tmp_path / "hero")
    job.directory.mkdir()
    job.settings.update(hero=True, height_m=1.75)
    book.active = job.id
    book.jobs[job.id] = job
    sidecar = {"license": {"name": "fixture"}}

    def launch(argv, **kwargs):
        assert finish_api.HERO_WORKER.name in Path(argv[2]).name
        assert argv[argv.index("--result") + 1] == str(job.result_glb)
        document, binary = parse_glb(_sized_glb(1))
        document["scenes"] = [{"nodes": [0]}]
        job.result_glb.write_bytes(build_glb(document, binary))
        job.result_glb.with_suffix(".provenance.json").write_text(json.dumps(sidecar))
        return Child("I2L_STAGE::compress::Packing\n")

    monkeypatch.setattr(finish_api.subprocess, "Popen", launch)
    finish_api.run_job(job, book)
    assert job.status == "done", job.events[-1]
    assert book.active is None
    result = job.result_glb.read_bytes()
    record = json.loads(job.record_path.read_text())
    assert record["license"] == sidecar["license"]
    assert record["output"]["sha256"] == hashlib.sha256(result).hexdigest()
    document, _ = parse_glb(result)
    position = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    assert document["accessors"][position]["max"][1] == pytest.approx(1.75)


def test_finish_source_lookup_skips_licensed_record_for_other_mesh(tmp_path, monkeypatch):
    unwanted = tmp_path / "old.json"
    wanted = tmp_path / "new.json"
    unwanted.write_text('{"license":{"name":"sample"},"output":{"sha256":"different"}}')
    asset = b"finished mesh"
    wanted.write_text(json.dumps({"license": {"name": "sample"},
                                  "output": {"sha256": hashlib.sha256(asset).hexdigest()}}))
    original = Path.rglob
    monkeypatch.setattr(Path, "rglob", lambda path, pat: iter([unwanted, wanted])
                        if path == tmp_path and pat == "*.json" else original(path, pat))
    assert finish_api.find_source_record(asset, tmp_path) == wanted


def test_cancel_second_rig_removes_first_try_without_replacing_cancelled_result(tmp_path, monkeypatch):
    model = tmp_path / "rig.glb"
    model.write_bytes(b"first rig")
    job = autorig_api.AnimateJob("a" * 32, "rig", tmp_path, model, [(["fixture-rig"], tmp_path)], 1, {})
    job.expect_humanoid = True
    monkeypatch.setattr(autorig_api, "rig_fit", lambda path: {"humanoid": False, "detail": "leg missing"})
    monkeypatch.setattr(autorig_api, "free_ports", lambda count: (8111, 8112))

    def rerig(argv, **kwargs):
        assert argv == ["fixture-rig"]
        assert model.with_name("rig_try1.glb").read_bytes() == b"first rig"
        job.cancel_requested = True
        return Child("", code=7)

    monkeypatch.setattr(autorig_api.subprocess, "Popen", rerig)
    assert autorig_api.rig_again_if_not_humanoid(job) == 7
    assert job.cancel_requested
    assert not model.with_name("rig_try1.glb").exists()
    assert not model.exists()


def test_late_picture_completion_does_not_release_a_new_job(tmp_path):
    book = api.image_api.ImageJobManager(tmp_path)
    first = book.create("first", {})
    first.status = "done"
    second = book.create("second", {})
    book.finish(first)
    assert book.active == second.id
    assert book.jobs == {first.id: first, second.id: second}


def test_replacing_first_picture_creates_no_archive(tmp_path):
    image = tmp_path / "result.new.png"
    final = tmp_path / "result.png"
    image.write_bytes(b"new picture")
    assert api.image_api.take_place(image, final) is None
    assert final.read_bytes() == b"new picture"
    assert not image.exists()


def test_turn_record_ignores_stage_that_did_not_return_selected_prop(tmp_path):
    import props_api

    job = props_api.PropsJob("a" * 32, tmp_path)
    job.only = "vase"
    live = job.finished_dir
    staged = job.finished_dir = tmp_path / "turn" / "finished"
    for area, name in ((live, "vase"), (staged, "barrel")):
        area.mkdir(parents=True)
        (area / "finish_props.json").write_text(json.dumps({"props": [{"name": name}]}), encoding="utf-8")
    assert props_api.turned_record(job) is None
    assert json.loads((live / "finish_props.json").read_text())["props"][0]["name"] == "vase"
    assert props_api.generated_models(tmp_path / "removed") == []


def test_proxy_requests_use_forwarded_host_when_listening_publicly(monkeypatch):
    import request_utils

    monkeypatch.delenv("I3D_ALLOWED_HOSTS", raising=False)
    forwarded = "studio.example"
    handler = NS(headers={
        "Host": "localhost",
        "X-Forwarded-Host": forwarded,
        "X-AssetFurnace-Request": "studio",
        "Origin": f"https://{forwarded}",
    }, server=NS(server_address=("0.0.0.0", 5000)))
    assert request_utils.check_request(handler, mutation=True) is None
    handler.headers["Host"] = "localhost/path"
    with pytest.raises(PermissionError, match="Invalid request host"):
        request_utils.check_request(handler, mutation=True)


def test_report_for_failure_without_note_or_log_leaves_sections_out(tmp_path):
    from activity_api import append_history
    from bug_report_api import _chip, details

    append_history(tmp_path, {"result": "error", "step": "model", "chain": "fixture"})
    report = details(tmp_path, machine="fixture machine", home="")
    assert "Step:" in report
    assert "What it said:" not in report and "Last lines of the log:" not in report
    assert _chip("unsupported-system") == ""
