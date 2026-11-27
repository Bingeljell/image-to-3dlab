"""Generation settings, progress, and setup with local fixtures."""

import dataclasses
import json
import subprocess
import sys
from types import SimpleNamespace

import backend_catalog
import generate_api as api
import patch_hunyuan_rocm
import pytest


@pytest.mark.parametrize(("route", "numeric", "number"), [
    ("trellis", "seed", 42),
    ("sf3d", "target_vertices", 1000),
    ("hunyuan-mlx", "paint_steps", 12),
    ("hunyuan-mlx-xiong", "paint_steps", 12),
    ("pixal3d", "seed", 42),
    ("hunyuan-cuda", "steps", 12),
    ("hunyuan-rocm", "steps", 12),
])
def test_backend_settings_accept_numeric_strings_and_reject_invalid_shapes(route, numeric, number):
    validator = api.BACKENDS[route].validate_settings
    assert validator(None) == validator({})
    assert validator({numeric: str(number)})[numeric] == number
    with pytest.raises(ValueError, match="JSON object"):
        validator([])
    with pytest.raises(ValueError, match="integer"):
        validator({numeric: "twelve"})


@pytest.mark.parametrize(("field", "value", "message"), [
    ("steps", 0, "steps must be at least 1"),
    ("octree_resolution", 128, "octree_resolution must be 256, 384 or 512"),
    ("max_num_view", 0, "max_num_view must be 1 to 9"),
    ("paint_resolution", 512, "paint_resolution must be 128 or 256"),
])
def test_amd_settings_reject_values_outside_supported_limits(field, value, message):
    with pytest.raises(ValueError, match=message):
        api.BACKENDS["hunyuan-rocm"].validate_settings({field: value})


def test_amd_does_not_pass_unset_paint_overrides(tmp_path):
    spec = api.BACKENDS["hunyuan-rocm"]
    job = api.Job("a" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb",
                  spec.validate_settings({"max_num_view": "1", "paint_resolution": "128"}), spec.id)
    assert spec.build_args(job)[-4:] == ["--max-num-view", "1", "--paint-resolution", "128"]
    job.settings = spec.validate_settings(None)
    assert "--max-num-view" not in spec.build_args(job)
    assert "--paint-resolution" not in spec.build_args(job)


def test_invalid_xiong_model_is_rejected():
    with pytest.raises(ValueError, match="model must be one of"):
        api.BACKENDS["hunyuan-mlx-xiong"].validate_settings({"model": "unknown"})


@pytest.mark.parametrize(("value", "expected"), [
    (None, None), ("", None), ("02:37", 157), ("01:02:03", 3723), ("01", None), ("bad:time", None),
])
def test_progress_time_formats(value, expected):
    assert api._seconds(value) == expected


def test_trellis_progress_tracks_two_shape_passes_and_eta(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_baseline", lambda: {"bake": 42.0})
    job = api.Job("a" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {}, "trellis")
    spec = api.BACKENDS["trellis"]
    for line in (
        "Loading TRELLIS.2", "Pipeline loaded", "decode_latent done in 1s", "Pre-cap done",
        "Sparse:  10%|# | 1/10 [00:01<00:09, 2.00it/s]",
        "Shape:   0%|  | 0/12 [00:00, 1.00s/it]",
        "Shape: 100%|##| 12/12 [00:12<00:00, 1.00s/it]",
        "Shape:   0%|  | 0/12 [00:00, 1.00s/it]",
        "Texture:  50%|# | 6/12 [00:05<00:07, 1.00s/it]",
        "Export: 100%|##| 1/1 [00:03, 3.00s/it]",
        "to_glb + export done in 3s", "plain text",
    ):
        spec.parse_line(job, line)

    assert [event["phase"] for event in job.events] == [
        "load", "load", "decode", "bake", "sparse_structure", "shape_slat_coarse",
        "shape_slat_coarse", "shape_slat_fine", "texture_slat", "bake", "bake",
    ]
    assert job.events[4]["s_per_it"] == 0.5
    assert job.events[5]["stage_eta_seconds"] is None
    assert job.events[7]["overall_pct"] > job.events[5]["overall_pct"]
    assert job.events[8]["total_eta_seconds"] == 49.0
    assert job.stage_durations == {"shape_slat_coarse": 12.0, "bake": 3.0}
    assert job.events[-1]["overall_pct"] == 100


@pytest.mark.parametrize("phase_percent", [api._overall_pct, api._hunyuan_overall_pct])
def test_unknown_progress_stage_has_no_completion(phase_percent):
    assert phase_percent("unexpected", 60) == 0


def test_optional_pixal_banners_do_not_spoof_progress(tmp_path):
    job = api.Job("a" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {}, "pixal3d")
    for line in ("[1/6] Load", "[bad/6] Message", "[7/6] Message", "[2/5] Message", "Plain"):
        api.BACKENDS["pixal3d"].parse_line(job, line)
    assert len(job.events) == 1
    assert job.events[0]["phase"] == api.PIXAL3D_BANNERS[1]


def test_amd_readiness_requires_build_patches_and_every_weight(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(api, "HUNYUAN_ROCM_VENDOR", tmp_path)
    present = False
    patched = False

    def describe():
        return {"weights": [{"label": "Shape", "present": present, "human_present": "8 GB" if present else "0 B"}]}

    def check_patches(root):
        assert root == tmp_path / "Hunyuan3D-2.1"
        return patched

    monkeypatch.setitem(backend_catalog.BY_ID, "hunyuan-rocm", SimpleNamespace(describe=describe))
    monkeypatch.setattr(patch_hunyuan_rocm, "is_patched", check_patches)
    paths = []
    for attribute in ("HUNYUAN_ROCM_MARKER", "HUNYUAN_ROCM_PYTHON", "HUNYUAN_ROCM_WRAPPER"):
        path = tmp_path / attribute
        paths.append(path)
        monkeypatch.setattr(api, attribute, path)
    check = api.BACKENDS["hunyuan-rocm"].readiness

    assert not check()["ready"]
    assert "not installed" in check()["build"]["hint"]
    for path in paths:
        path.touch()
    assert not check()["build"]["present"]
    assert "missing the AMD patches" in check()["build"]["hint"]
    patched = True
    assert check()["build"]["present"]
    assert check()["missing_weights"] == ["Shape"]
    assert "weights are missing" in check()["build"]["hint"]
    present = True
    assert check()["ready"]
    assert check()["build"]["hint"] is None


def test_pixal_readiness_requires_cli_wrapper_and_nine_weight_files(tmp_path, monkeypatch):
    cli, wrapper, models = [tmp_path / name for name in ("cli", "wrapper", "models")]
    monkeypatch.setattr(api, "PIXAL3D_CLI", cli)
    monkeypatch.setattr(api, "PIXAL3D_WRAPPER", wrapper)
    monkeypatch.setattr(api, "PIXAL3D_MODELS", models)
    check = api.BACKENDS["pixal3d"].readiness
    assert len(check()["missing_weights"]) == 2
    cli.touch()
    wrapper.touch()
    models.mkdir()
    for index in range(8):
        (models / f"{index}.gguf").touch()
    assert check()["weights"]["pixal3d-sv-q8_0"]["human"] == "8/9 GGUF files"
    assert not check()["ready"]
    (models / "eighth.gguf").touch()
    assert check()["ready"]
    wrapper.unlink()
    assert not check()["ready"]


@pytest.fixture
def advisor(tmp_path, monkeypatch):
    python = tmp_path / "python"
    wrapper = tmp_path / "advisor"
    source = tmp_path / "image.png"
    source.write_bytes(b"png")
    original = api.trellis_spec()
    monkeypatch.setattr(api, "trellis_spec", lambda: dataclasses.replace(original, interpreter=python))
    monkeypatch.setattr(api, "TINYCLIP_ADVISOR", wrapper)
    return SimpleNamespace(python=python, wrapper=wrapper, image=source)


def test_advisor_requires_local_programs_before_starting(advisor, monkeypatch):
    monkeypatch.setattr(api.subprocess, "run", lambda *args, **kw: pytest.fail("started missing advisor"))
    with pytest.raises(RuntimeError, match="environment is not installed"):
        api.run_trellis_input_advisor(advisor.image)
    advisor.python.touch()
    with pytest.raises(RuntimeError, match="advisor is missing"):
        api.run_trellis_input_advisor(advisor.image)


@pytest.mark.parametrize(("stdout", "error", "message"), [
    ("not JSON", None, "invalid JSON"),
    ('{"verdict": "invalid", "flat_risk": 0.8}', None, "invalid verdict"),
    ('{"verdict": "uncertain", "flat_risk": "0.8"}', None, "invalid score"),
    ("", subprocess.TimeoutExpired([], 3), "input check timed out"),
    ("", subprocess.CalledProcessError(1, [], stderr="details\nmissing weights\n"), "failed: missing weights"),
    ("", subprocess.CalledProcessError(1, []), "failed$"),
    ('{"verdict": "likely_flat", "flat_risk": 0.8}', None, None),
])
def test_advisor_translates_child_process_result(advisor, monkeypatch, stdout, error, message):
    advisor.python.touch()
    advisor.wrapper.touch()

    def run(args, **kwargs):
        assert args == [str(advisor.python), str(advisor.wrapper), str(advisor.image)]
        assert kwargs["timeout"] == api.TINYCLIP_TIMEOUT_SECONDS
        if error:
            raise error
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(api.subprocess, "run", run)
    if message:
        with pytest.raises(RuntimeError, match=message):
            api.run_trellis_input_advisor(advisor.image)
    else:
        assert api.run_trellis_input_advisor(advisor.image) == json.loads(stdout)


class ControlledThread:
    """Run pipe readers immediately, and leave the periodic memory probe scheduled."""

    def __init__(self, target, args=(), *, daemon, name=""):
        assert daemon
        self.target, self.args, self.name = target, args, name

    def start(self):
        if not self.name.startswith("rss-"):
            self.target(*self.args)

    def join(self, timeout):
        assert timeout == 5


@pytest.mark.parametrize(("result", "status", "message"), [
    ("success", "done", "Generation complete"),
    ("missing", "error", "produced no output file"),
    ("exit", "error", "working\nlast message"),
    ("diagnostic", "error", "CUDA out of memory"),
    ("cancel", "cancelled", "Generation cancelled"),
])
def test_generator_stops_and_releases_its_slot(tmp_path, monkeypatch, result, status, message):
    import io

    manager = api.JobManager()
    monkeypatch.setattr(api, "JOBS", manager)
    monkeypatch.setattr(api, "LEARNED_BASELINE_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(api.threading, "Thread", ControlledThread)
    spec = api.BACKENDS["sf3d"]
    job = manager.create(tmp_path / "in.png", spec.validate_settings(None), "sf3d", "prop", None, tmp_path)
    if result == "success":
        folder = job.directory / "commercial_conditional"
        folder.mkdir()
        (folder / "in__sf3d__run.glb").write_bytes(b"model")
        (folder / "in__sf3d__run.provenance.json").write_text("{}", encoding="utf-8")
    log = b"\rworking\n\nlast message"
    if result == "diagnostic":
        log += b"\nCUDA out of memory."
    job.cancel_requested = result == "cancel"
    code = 3 if result in {"exit", "diagnostic"} else 0

    def popen(args, **kwargs):
        assert args[:2] == [str(spec.interpreter), str(spec.wrapper)]
        assert args[-1] == str(job.image_path)
        assert kwargs["stdout"] == subprocess.PIPE
        return SimpleNamespace(pid=3210, stdout=io.BytesIO(log), wait=lambda: code)

    monkeypatch.setattr(api.subprocess, "Popen", popen)
    api._run_job(job)
    assert job.status == status
    assert message.lower() in job.events[-1]["message"].lower()
    assert any(event.get("message") == "last message" for event in job.events)
    if result == "success":
        assert not job.directory.joinpath("run.log").exists()
    else:
        assert "last message" in job.directory.joinpath("run.log").read_text(encoding="utf-8")
    assert manager.active is None
    assert not (job.directory / "pid").exists()
    if result == "success":
        assert job.output_path.read_bytes() == b"model"
        assert job.manifest_path.is_file()
        assert job.events[-1]["manifest_url"].endswith("/manifest.json")
    else:
        assert not job.output_path.exists()


@pytest.mark.parametrize("code", [0, 5])
def test_setup_reader_emits_completion_and_frees_the_slot(monkeypatch, code):
    import io

    monkeypatch.setattr(api, "SETUP_ACTIVE", "fixture")
    monkeypatch.setattr(api.threading, "Thread", ControlledThread)

    def popen(argv, **kwargs):
        assert argv == ["fixture", "--setup"]
        assert kwargs["stdout"] == subprocess.PIPE
        return SimpleNamespace(stdout=io.BytesIO(b"setting up\n\nfinished\n"), wait=lambda: code)

    monkeypatch.setattr(api.subprocess, "Popen", popen)
    run = api._start_setup_run("fixture", ["fixture", "--setup"])
    assert run.status == ("done" if code == 0 else "error")
    assert [e["message"] for e in run.events[:2]] == ["setting up", "finished"]
    assert f"code {code}" in run.events[-1]["message"]
    assert api.SETUP_ACTIVE is None


def test_setup_kills_worker_if_it_cannot_start_the_reader(monkeypatch):
    from unittest.mock import Mock, call

    proc = Mock(pid=3210)
    proc.wait.side_effect = [subprocess.TimeoutExpired([], 5), -9]
    group = Mock()

    class RefusingThread(ControlledThread):
        def start(self):
            raise RuntimeError("no worker threads")

    monkeypatch.setattr(api.threading, "Thread", RefusingThread)
    monkeypatch.setattr(api.processes, "terminate_group", group)
    monkeypatch.setattr(api.subprocess, "Popen", lambda *args, **kwargs: proc)
    with pytest.raises(api.WorkerStartError, match="no worker threads"):
        api._start_setup_run("fixture")
    group.assert_called_once_with(3210)
    assert proc.method_calls == [call.terminate(), call.wait(timeout=5), call.kill(), call.wait(timeout=5)]


def test_hunyuan_apple_routes_report_which_local_setup_pieces_are_missing(tmp_path, monkeypatch):
    for var in ("HUNYUAN_PYTHON", "HUNYUAN_WRAPPER", "HUNYUAN_PAINT_VENV", "HUNYUAN_PAINT_WEIGHTS",
                "HUNYUAN_SHAPE_REPO", "HUNYUAN_XIONG_SHAPE_VENV", "HUNYUAN_XIONG_WRAPPER"):
        monkeypatch.setattr(api, var, tmp_path / var.lower())
    choices = {name: tmp_path / name for name in ("2.0", "2.0-turbo")}
    monkeypatch.setattr(api, "HUNYUAN_XIONG_SHAPE_MODELS", choices)
    paired = api.BACKENDS["hunyuan-mlx"].readiness
    xiong = api.BACKENDS["hunyuan-mlx-xiong"].readiness

    absent = paired()["missing_weights"]
    assert len(absent) == 4
    assert any("paint weights" in item for item in absent)
    assert "dgrauet shape" in paired()["warning"]
    assert not xiong()["ready"]
    assert xiong()["model_availability"] == {"2.0": False, "2.0-turbo": False}

    for path in (api.HUNYUAN_PYTHON, api.HUNYUAN_WRAPPER, api.HUNYUAN_PAINT_VENV,
                 api.HUNYUAN_XIONG_SHAPE_VENV, api.HUNYUAN_XIONG_WRAPPER):
        path.touch()
    api.HUNYUAN_PAINT_WEIGHTS.mkdir()
    api.HUNYUAN_SHAPE_REPO.mkdir()
    (api.HUNYUAN_SHAPE_REPO / "cache.bin").write_bytes(b"placeholder")
    assert paired()["ready"]
    assert paired()["warning"] is None

    choices["2.0-turbo"].mkdir()
    # A non-default model can be picked explicitly.
    alternative = xiong()
    assert alternative["ready"]
    assert alternative["model_availability"] == {"2.0": False, "2.0-turbo": True}
    assert any("default model" in item for item in alternative["missing_weights"])
    choices["2.0"].mkdir()
    assert not xiong()["missing_weights"]


def test_amd_stable_fast_3d_requires_declared_weights(tmp_path, monkeypatch):
    for var in ("SF3D_ROCM_MARKER", "SF3D_ROCM_PYTHON"):
        path = tmp_path / var
        path.touch()
        monkeypatch.setattr(api, var, path)
    row = {"label": "Encoder", "bytes_expected": 1000, "bytes_present": 5}
    monkeypatch.setattr(api, "_sf3d_weights", lambda: [row])
    def ready():
        return api._sf3d_readiness(api.AMD)

    assert ready()["build"]["present"]
    assert ready()["missing_weights"] == ["Encoder"]
    assert not ready()["ready"]
    row["bytes_present"] = 1000
    assert ready()["ready"]
    assert ready()["build"]["hint"] is None
    row.update(bytes_expected=0, bytes_present=0)
    assert not api.weight_blocks_generate(row)


def test_setup_is_available_when_uv_and_mac_bootstrap_are_present(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPO", tmp_path)
    monkeypatch.setattr(api, "find_uv", lambda: "uv")
    bootstrap = tmp_path / "scripts" / "bootstrap_trellis_space_macos.py"
    bootstrap.parent.mkdir()
    assert not api.setup_available(api.APPLE)[0]
    bootstrap.touch()
    assert api.setup_available(api.APPLE) == (True, None)


def test_corrupt_picture_fails_alpha_check_and_missing_numpy_disables_border_check(tmp_path, monkeypatch):
    picture = tmp_path / "bad.png"
    picture.write_bytes(b"bad picture")
    assert not api.image_has_transparent_alpha(picture)
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert api.image_border_opaque_fraction(picture) is None


def test_generation_stale_slot_does_not_release_newly_reserved_job(tmp_path):
    manager = api.JobManager()
    first = manager.create(tmp_path / "front", {}, "sf3d", "front", "known", tmp_path)
    with pytest.raises(RuntimeError, match="already running"):
        manager.create(tmp_path / "front", {}, "sf3d", "front", "known", tmp_path)
    first.status = "done"
    second = manager.create(tmp_path / "front", {}, "sf3d", "front", "known", tmp_path)
    assert second.directory.name == "known-2"
    manager.finish(first)
    assert manager.active == second.id


def test_failure_summary_keeps_only_the_requested_tail_lines():
    assert api.failure_reason(["progress", "second message", "third message"], limit=2) == (
        "second message\nthird message"
    )


def test_parser_ignores_multipart_parts_with_no_form_name():
    body = (b"--example\r\nContent-Disposition: attachment\r\n\r\nskip\r\n"
            b"--example\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nA Sample\r\n"
            b"--example--\r\n")
    assert api.parse_multipart("multipart/form-data; boundary=example", body) == {
        "title": {"filename": None, "data": b"A Sample", "value": "A Sample"},
    }
