"""Worker lifecycle tests. Pipes stand in for external programs."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import backend_catalog
import download_api
import finish_api
import preview_api
import pytest
import rig_api


class Child:
    def __init__(self, output, *, code=0, wait_error=None):
        self.pid = 3210
        self.stdout = io.BytesIO(output) if isinstance(output, bytes) else io.StringIO(output)
        self.code = code
        self.wait_error = wait_error
        self.exited = False

    def wait(self):
        if self.wait_error:
            raise self.wait_error
        self.exited = True
        return self.code

    def poll(self):
        return self.code if self.exited else None


@pytest.fixture(params=["rig", "finish"])
def model_worker(request, tmp_path, monkeypatch):
    if request.param == "rig":
        module = rig_api
        job = module.RigJob("a" * 32, tmp_path)
        manager = module.RigJobManager(tmp_path)
        monkeypatch.setattr(module, "blender_executable", lambda: Path("fixture-blender"))
        outputs = [job.result_glb, job.result_blend, job.result_sidecar, job.report_path]
        banner = "I2L_STAGE::weights::Painting progress\n"
    else:
        module = finish_api
        job = module.FinishJob("a" * 32, tmp_path)
        job.settings.update(skip_paint=True, skip_compress=True, skip_bake=True)
        job.resume = True
        manager = module.FinishJobManager(tmp_path, tmp_path)
        monkeypatch.setattr(module.blender_lookup, "find_blender", lambda: Path("fixture-blender"))
        outputs = [job.result_glb]
        banner = "I2L_STAGE::retopologise::Painting progress\n"
    manager.jobs[job.id] = job
    manager.active = job.id
    return SimpleNamespace(module=module, job=job, manager=manager, outputs=outputs, banner=banner)


@pytest.mark.parametrize(("result", "state", "message"), [
    ("complete", "done", None), ("empty", "error", "without"),
    ("exit", "error", "exited with code 4"), ("cancel", "cancelled", "cancelled"),
    ("broken_pipe", "error", "pipe failed"),
])
def test_model_workers_emit_progress_before_terminal_result(model_worker, monkeypatch, result, state, message):
    work = model_worker
    if result == "complete":
        for path in work.outputs:
            path.write_bytes(b"artifact")
    work.job.cancel_requested = result == "cancel"
    child = Child(work.banner + "\nnormal log\n", code=4 if result == "exit" else 0,
                  wait_error=OSError("pipe failed") if result == "broken_pipe" else None)
    launched = Mock(return_value=child)

    def reap(process, *, grace):
        assert process is child
        assert grace == 30
        process.exited = True
        assert work.manager.active == work.job.id
        return "reaped after signal"

    monkeypatch.setattr(work.module.subprocess, "Popen", launched)
    monkeypatch.setattr(work.module.processes, "stop_and_reap", reap)
    work.module.run_job(work.job, work.manager)
    launched.assert_called_once()
    assert work.job.status == state
    assert work.manager.active is None
    assert not (work.job.directory / "pid").exists()
    assert any(e.get("message") == "Painting progress" for e in work.job.events)
    if message:
        assert message in work.job.events[-1]["message"]
    if result == "broken_pipe":
        assert "cleanup warning: reaped after signal" in work.job.log_lines
    if result == "complete":
        assert work.job.events[-1]["result_url"].endswith("/result.glb")


def test_model_workers_refuse_missing_program(model_worker, tmp_path, monkeypatch):
    work = model_worker
    if work.module is rig_api:
        monkeypatch.setattr(rig_api, "blender_executable", lambda: None)
    else:
        monkeypatch.setattr(finish_api, "WORKER", tmp_path / "missing.py")
    popen = Mock()
    monkeypatch.setattr(work.module.subprocess, "Popen", popen)
    work.module.run_job(work.job, work.manager)
    popen.assert_not_called()
    assert work.job.status == "error"
    assert work.job.events[-1]["phase"] == "error"
    assert work.manager.active is None


def test_model_workers_cancel_only_the_active_child(model_worker, monkeypatch):
    work = model_worker
    group = Mock()
    monkeypatch.setattr(work.module.processes, "terminate_group", group)
    work.job.process = Child("")
    work.module.cancel_job(work.job)
    group.assert_called_once_with(work.job.process.pid)
    assert work.job.status == "cancelling"
    assert work.job.cancel_requested


def test_finish_adoption_distinguishes_missing_resume_data_and_busy_slot(tmp_path):
    manager = finish_api.FinishJobManager(tmp_path, tmp_path)
    missing = "sample__finish__20260101-080000"
    with pytest.raises(RuntimeError, match="no such run"):
        manager.adopt(missing)
    directory = tmp_path / missing
    directory.mkdir()
    with pytest.raises(RuntimeError, match="predates resume support"):
        manager.adopt(missing)
    job = finish_api.FinishJob("a" * 32, directory)
    manager.jobs[job.id] = job
    manager.active = job.id
    with pytest.raises(RuntimeError, match="already running"):
        manager.adopt(missing)


@pytest.fixture
def download(tmp_path, monkeypatch):
    backend = backend_catalog.resolve("sf3d")
    setup = download_api.DownloadRun(backend)
    setup.directory = tmp_path / "run"
    setup.command = ["fake-python", "--weights-only"]
    monkeypatch.setattr(download_api, "DOWNLOADS", {setup.backend.id: setup})
    return setup


@pytest.mark.parametrize(("result", "state"), [
    ("success", "done"), ("cancel", "cancelled"), ("exit", "error"),
    ("broken", "error"), ("missing", "error"), ("build", "done"),
])
def test_download_results_stop_progress_and_clean_pid(download, monkeypatch, result, state):
    watchers = []
    monkeypatch.setattr(download_api.shutil, "which",
                        lambda command: None if result == "missing" else command)

    class Watcher:
        def __init__(self, target, args, daemon):
            assert daemon
            self.target, self.args = target, args

        def start(self):
            watchers.append((self.target, *self.args))

    child = Child(b"\x1b[31mone line\x1b[0m\n\n", code=2 if result == "exit" else 0,
                  wait_error=OSError("pipe failed") if result == "broken" else None)
    download.cancelled = result == "cancel"
    if result == "build":
        download.rebuild = True

    def reap(process, *, grace):
        assert process is child
        assert grace == 5
        process.exited = True
        return "forced exit"

    popen = Mock(return_value=child)
    monkeypatch.setattr(download_api.threading, "Thread", Watcher)
    monkeypatch.setattr(download_api.subprocess, "Popen", popen)
    monkeypatch.setattr(download_api.processes, "stop_and_reap", reap)
    download_api._run(download)
    assert download.status == state
    assert not (download.directory / "pid").exists()
    if result == "missing":
        popen.assert_not_called()
        assert "not installed" in download.events[-1]["detail"]
        assert watchers == []
    else:
        popen.assert_called_once()
        (target, run, stop), = watchers
        assert run is download
        assert stop.is_set()
        expected = download_api._watch_elapsed if result == "build" else download_api._watch_size
        assert target is expected
        assert download.log[0] == "one line"
    if result == "broken":
        assert download.log[-1] == "cleanup warning: forced exit"
        assert download.events[-1]["detail"] == "pipe failed"
    if result == "cancel":
        assert "resuming" in download.events[-1]["detail"]


def test_download_cancel_checks_job_before_signalling(download, monkeypatch):
    group = Mock()
    monkeypatch.setattr(download_api.processes, "terminate_group", group)
    with pytest.raises(RuntimeError, match="no download"):
        download_api.cancel("missing")
    download_api.cancel("sf3d")
    assert download.cancelled
    group.assert_not_called()
    download.process = Child(b"")
    download_api.cancel("sf3d")
    group.assert_called_once_with(download.process.pid)
    download.status = "done"
    with pytest.raises(RuntimeError, match="no download"):
        download_api.cancel("sf3d")


def test_download_reports_bytes_arriving_and_build_reports_elapsed(download, monkeypatch):
    samples = iter([10, 34, 34])
    mock_clock = iter([10.0, 13.0, 13.0, 16.0, 16.0])
    monkeypatch.setattr(download, "bytes_present", lambda: next(samples))
    monkeypatch.setattr(download_api, "time", SimpleNamespace(monotonic=lambda: next(mock_clock)))
    wait = Mock(side_effect=[False, False, True])
    download.log.append("fetching encoder")
    download_api._watch_size(download, SimpleNamespace(wait=wait))
    assert len(download.events) == 2
    assert download.events[1]["bytes_present"] == 34
    assert download.events[0]["phase"] == "downloading"

    download.events.clear()
    download.rebuild = True
    estimate = download_api.REBUILD_MINUTES * 60
    monkeypatch.setattr(download_api, "time",
                        SimpleNamespace(monotonic=lambda: download.started + estimate / 2))
    download_api._watch_elapsed(download, SimpleNamespace(wait=Mock(side_effect=[False, True])))
    assert download.events[0]["overall_pct"] == 50
    assert "rebuilding" in download.events[0]["detail"]


@pytest.mark.parametrize("move", [None, "turn"])
def test_preview_composes_and_caches_output_from_child_render(tmp_path, monkeypatch, move):
    from PIL import Image

    model = tmp_path / "model.glb"
    model.write_bytes(b"fixture")
    fixture = {"name": "Stone prop", "model": model.name, "clips": [{"name": "turn", "file": model.name}]}
    scripts = []

    def render(args, **kwargs):
        scripts.append(Path(args[4]))
        assert scripts[-1].is_file() and scripts[-1].name == "blender_preview_worker.py"
        assert args[5:7] == ["--", str(model)]
        assert args[8:] == [
            "0.0", "512", "640", "1", preview_api.shots_arg(preview_api.shots_for(bool(move))),
        ]
        folder = Path(args[7])
        times = {}
        for name, _, fraction in preview_api.shots_for(bool(move)):
            Image.new("RGB", (500, 400), (40, 50, 60)).save(folder / f"{name}.png")
            times[name] = fraction
        if move:
            (folder / "times.json").write_text(json.dumps(times), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preview_api.subprocess, "run", render)
    blender = tmp_path / "fixture-blender"
    path, shown = preview_api.preview(tmp_path, fixture, blender=blender, look="clay", move=move)
    assert shown == ("move: turn" if move else "model")
    assert path.parent.name == preview_api.PREVIEWS_DIR
    assert scripts[0].is_file()
    with Image.open(path) as result:
        assert result.format == "PNG"
        assert result.width <= preview_api.SHEET_WIDTH
    assert preview_api.preview(tmp_path, fixture, blender=blender, look="clay", move=move) == (path, shown)
    assert len(scripts) == 1


def test_preview_reports_missing_frames_from_the_child(tmp_path, monkeypatch):
    def render(args, **kwargs):
        assert args[8:] == ["0", "512", "640", "0", "front=0="]
        return SimpleNamespace(returncode=0, stderr="", stdout="no frames")

    monkeypatch.setattr(preview_api.subprocess, "run", render)
    with pytest.raises(RuntimeError, match="no frames"):
        preview_api.render_shots(tmp_path / "stub", tmp_path / "model.glb", tmp_path, 0, "paint", [
            ("front", 0, None),
        ])


@pytest.mark.parametrize("scenario", ["no-props", "lost-child", "cancel", "turn-missing-prop", "turn-works", "turn-empty"])
def test_props_worker_keeps_turn_and_live_files_consistent(tmp_path, monkeypatch, scenario):
    import props_api

    from tests.test_viewer_model_info_api import _sized_glb

    baked = _sized_glb(1.0)
    manager = props_api.PropsJobManager(tmp_path)
    monkeypatch.setattr(props_api, "blender_executable", lambda: Path("fixture-blender"))
    monkeypatch.setattr(props_api, "find_gltfpack", lambda: None)
    original = manager.create("sheet.glb", b"model", {"compress": True})
    previous = original.directory / "split"
    previous.mkdir()
    text = '{"props": [{"name": "vase"}]}'
    (previous / "props.json").write_text(text, encoding="utf-8")
    manager.finish(original)

    turned = scenario.startswith("turn")
    if turned:
        job = manager.turn(original.directory.name, "vase", 90)
    else:
        job = original
        manager.active = job.id
        (previous / "props.json").unlink()
    processes = []
    live_finished = job.directory / "finished"
    live_finished.mkdir()
    (live_finished / "finish_props.json").write_text('{"props":[]}', encoding="utf-8")

    def popen(command, **kwargs):
        if not processes:
            assert str(props_api.SPLITTER) in command
            if turned:
                job.split_dir.mkdir(parents=True)
                new = text if scenario != "turn-missing-prop" else '{"props": [{"name": "barrel"}]}'
                (job.split_dir / "props.json").write_text(new, encoding="utf-8")
                (job.split_dir / "vase.glb").write_bytes(b"split")
            child = Child("SPLIT::Split sheet\n\n", wait_error=OSError("lost pipe") if scenario == "lost-child" else None)
            if scenario == "cancel":
                job.cancel_requested = True
        else:
            assert str(props_api.FINISHER) in command
            assert "--no-compress" in command
            if scenario == "turn-works":
                target = job.finished_dir / "vase"
                target.mkdir(parents=True)
                (target / "vase_LOD0.glb").write_bytes(baked)
                (job.finished_dir / "finish_props.json").write_text(text, encoding="utf-8")
            child = Child("FINISH::{\"prop\": \"vase\", \"lod\": 0}\n")
        processes.append(child)
        return child

    def reap(process, *, grace):
        assert process is processes[-1]
        assert grace == 30
        process.exited = True
        return "worker reaped"

    monkeypatch.setattr(props_api.subprocess, "Popen", popen)
    monkeypatch.setattr(props_api.processes, "stop_and_reap", reap)
    props_api.run_job(job, manager)
    assert manager.active is None
    assert not (job.directory / "pid").exists()
    assert not job.staging.exists()
    assert job.events[1]["message"] == "Split sheet"
    if scenario == "turn-works":
        final = job.directory / "finished" / "vase" / "vase_LOD0.glb"
        from image_to_3dlab.glb_turn import parse_glb

        document, payload = parse_glb(final.read_bytes())
        assert payload == parse_glb(baked)[1]
        assert "AssetFurnace" in document["asset"]["generator"]
        assert json.loads((live_finished / "finish_props.json").read_text(encoding="utf-8")) == json.loads(text)
        assert json.loads(job.settings_path.read_text(encoding="utf-8"))["turns"] == {"vase": 90}
        assert job.status == "done", job.events[-1]
    else:
        assert json.loads(job.settings_path.read_text(encoding="utf-8"))["turns"] == {}
        message = {
            "turn-empty": "did not produce its files",
            "turn-missing-prop": "no longer has a prop",
            "no-props": "found no props",
            "lost-child": "lost pipe",
            "cancel": "Cancelled",
        }
        assert message[scenario] in job.events[-1]["message"]
        assert job.status == ("cancelled" if scenario == "cancel" else "error")
    if scenario == "lost-child":
        assert "cleanup warning: worker reaped" in job.log_lines


def test_prop_worker_refuses_missing_script_before_launch(tmp_path, monkeypatch):
    import props_api

    book = props_api.PropsJobManager(tmp_path)
    job = book.create("sheet.glb", b"model", {})
    monkeypatch.setattr(props_api, "blender_executable", lambda: Path("fixture-blender"))
    monkeypatch.setattr(props_api, "FINISHER", tmp_path / "missing.py")
    child = Mock()
    monkeypatch.setattr(props_api.subprocess, "Popen", child)
    props_api.run_job(job, book)
    child.assert_not_called()
    assert job.status == "error"
    assert "script missing" in job.events[-1]["message"]
    assert book.active is None


@pytest.mark.parametrize("turns", ["barrel", {"../../other": 90}])
def test_prop_settings_refuse_invalid_turns(turns):
    import props_api

    with pytest.raises(ValueError, match="turns"):
        props_api.normalise_settings({"turns": turns})
    assert props_api.PropsProgress().begin_props([], 0)["overall_pct"] == 0


def test_incomplete_prop_run_cannot_start_turn(tmp_path):
    import props_api

    root = props_api.PropsJobManager(tmp_path)
    run = tmp_path / "sheet__props__20260101-080000"
    with pytest.raises(RuntimeError, match="no such run"):
        root.turn(run.name, "prop", 90)
    run.mkdir()
    with pytest.raises(RuntimeError, match="missing its source.glb or settings.json"):
        root.turn(run.name, "prop", 90)


def test_cancel_prop_worker_signals_running_child(tmp_path, monkeypatch):
    import props_api

    job = props_api.PropsJob("a" * 32, tmp_path)
    job.process = Child("")
    signal = Mock()
    monkeypatch.setattr(props_api.processes, "terminate_group", signal)
    props_api.cancel_job(job)
    signal.assert_called_once_with(job.process.pid)
    assert job.cancel_requested


@pytest.mark.parametrize("repair", ["needed", "already-current", "unavailable"])
def test_rig_prepares_existing_install_then_honours_process_cancellation(tmp_path, monkeypatch, repair):
    import autorig_api

    monkeypatch.setattr(autorig_api, "SKINTOKENS", tmp_path)
    monkeypatch.setattr(autorig_api, "free_ports", lambda count: (8101, 8102))
    monkeypatch.setattr(autorig_api, "card_problem", lambda: None)
    command = ["fixture-python", "demo.py"]
    job = autorig_api.AnimateJob("a" * 32, "rig", tmp_path, tmp_path / "rig.glb",
                                 [(command, tmp_path), (["should-not-run"], tmp_path)], 10, {})
    manager = autorig_api.AnimateJobManager(tmp_path)
    manager.jobs[job.id] = job
    manager.active = job.id
    monkeypatch.setattr(autorig_api, "ensure_port_patch", lambda root: "updated ports" if repair == "needed" else None)

    def diffusers(root):
        if repair == "unavailable":
            raise OSError("no temporary space")
        return "repaired diffusers" if repair == "needed" else None

    monkeypatch.setattr(autorig_api, "ensure_diffusers", diffusers)
    monitor = Mock()
    monkeypatch.setattr(autorig_api.threading, "Thread", monitor)

    def start(args, **kwargs):
        assert args == command
        env = kwargs["env"]
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["SKINTOKENS_PORT"] == "8101"
        assert env["SKINTOKENS_BPY_PORT"] == "8102"
        job.cancel_requested = True
        return Child("first rig attempt\n\n")

    child = Mock(side_effect=start)
    monkeypatch.setattr(autorig_api.subprocess, "Popen", child)
    autorig_api.run_job(job, manager)
    assert job.status == "cancelled"
    assert child.call_count == 1
    assert manager.active is None
    assert not (tmp_path / "pid").exists()
    assert "first rig attempt" in job.log_lines
    if repair == "needed":
        assert "updated ports" in job.log_lines
        assert "repaired diffusers" in job.log_lines
    elif repair == "unavailable":
        assert any("no temporary space" in line for line in job.log_lines)


@pytest.mark.parametrize("kind, expected", [("rig", "Fitting a skeleton"), ("animate", "Playing the clip")])
def test_rig_and_animation_estimated_progress(kind, expected, tmp_path, monkeypatch):
    import autorig_api

    job = autorig_api.AnimateJob("a" * 32, kind, tmp_path, tmp_path / "rig.glb", [], 10, {})
    monkeypatch.setattr(autorig_api, "time", SimpleNamespace(monotonic=lambda: job.started + 4))
    stop = SimpleNamespace(wait=Mock(side_effect=[False, True]))
    autorig_api._tick(job, stop)
    assert job.events[0]["overall_pct"] == 40
    assert job.events[0]["message"] == expected


def test_cancel_animation_signals_worker_and_invalid_speed_is_reported(tmp_path, monkeypatch):
    import autorig_api

    job = autorig_api.AnimateJob("a" * 32, "animate", tmp_path, tmp_path / "clip.glb", [], 10, {})
    job.process = Child("")
    group = Mock()
    monkeypatch.setattr(autorig_api.processes, "terminate_group", group)
    autorig_api.cancel_job(job)
    assert job.cancel_requested
    group.assert_called_once_with(job.process.pid)
    with pytest.raises(ValueError, match="speed must be a number"):
        autorig_api.clamp_speed("quick", {})


def test_download_cannot_start_while_another_run_holds_slot(download, monkeypatch):
    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.APPLE)
    launch = Mock()
    monkeypatch.setattr(download_api.threading, "Thread", launch)
    with pytest.raises(RuntimeError, match="already running"):
        download_api.start("sf3d")
    launch.assert_not_called()
    assert download_api.active() is download


@pytest.mark.symlinks
def test_removing_absent_weights_removes_only_stale_snapshot_link(tmp_path, monkeypatch):
    from dataclasses import replace

    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "vision.gguf"
    stale.symlink_to(cache / "gone")
    (cache / "model.gguf").write_bytes(b"other weights")
    backend = backend_catalog.resolve("sf3d")
    weights = (
        backend_catalog.WeightSet("Vision", "example/vision", 100, cache, pattern="vision*.gguf"),
        backend_catalog.WeightSet("Missing", "example/missing", 50, tmp_path / "missing"),
    )
    monkeypatch.setattr(download_api, "REPO", tmp_path)
    monkeypatch.setattr(download_api, "DOWNLOADS", {})
    monkeypatch.setitem(download_api.BY_ID, "sf3d", replace(backend, weights=weights))
    assert weights[0].state()[:2] == (False, 0)
    removed = download_api.remove("sf3d")
    assert removed["freed_bytes"] == 0
    assert removed["removed"] == [str(stale)]
    assert not stale.is_symlink()
    assert (cache / "model.gguf").read_bytes() == b"other weights"


@pytest.mark.parametrize("result", ["cancel", "error"])
def test_addon_worker_does_not_publish_result_when_cancelled_or_failed(tmp_path, monkeypatch, result):
    import addons_api

    job = addons_api.AddonJob("a" * 32, "shape-tools", "check", {}, "sample")
    manager = addons_api.AddonJobManager()
    manager.jobs[job.id] = job
    manager.active = job.id
    monkeypatch.setattr(addons_api, "OUTPUT_DIR", tmp_path)
    run = tmp_path / addons_api.output_dir(job.addon_id, job.run_stem)
    run.mkdir(parents=True)

    def worker(addon, tool, inputs, *, on_start, cancel_event, **kwargs):
        assert (addon, tool, inputs) == ("shape-tools", "check", {})
        on_start(Child(""))
        assert (run / "pid").is_file()
        if result == "cancel":
            manager.cancel(job.id)
            assert cancel_event.is_set()
            return run / "result.glb"
        raise RuntimeError("mesh checker failed")

    monkeypatch.setattr(addons_api.ai, "run_addon", worker)
    addons_api._run_addon_job(job, manager)
    assert not (run / "pid").exists()
    assert manager.active is None
    assert job.result_path is None
    assert job.status == ("cancelled" if result == "cancel" else "error")
    if result == "error":
        assert job.error_message == "mesh checker failed"
