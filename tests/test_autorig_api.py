"""The Animate tab's backend: what it offers, what it refuses, and the records it writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import autorig_api as aa


def _glb(path: Path, payload: bytes = b"glTF") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    _glb(root / "commercial" / "knight" / "knight.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "knight_40k.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "steps" / "1_retopo.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_rigged.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_walk.glb")
    _glb(root / "props" / "sheet" / "sheet.glb")
    return root


@pytest.fixture
def motions(tmp_path):
    root = tmp_path / "motions"
    root.mkdir()
    (root / "walk.npz").write_bytes(b"x")
    (root / "kick.npz").write_bytes(b"x")
    (root / "presets.json").write_text(json.dumps({"presets": [
        {"id": "walk", "label": "Walk", "seconds": 4},
        {"id": "kick", "label": "Kick", "seconds": 2, "speed": 1.25},
        {"id": "missing", "label": "No file"},
        {"id": "../evil", "label": "Bad id"},
    ]}))
    return root


def test_offers_generated_finished_and_rigged_models_only(output):
    models = {m["path"]: m["kind"] for m in aa.pickable_models(output)}
    assert models == {
        "commercial/knight/knight.glb": "generated",
        "finish/knight__finish__20261003-135502/knight_40k.glb": "finished",
        "animate/knight__rig__20261005-120000/knight_rigged.glb": "rigged",
    }


def test_resolve_refuses_anything_not_offered(output):
    assert aa.resolve_model("commercial/knight/knight.glb", output).is_file()
    for path in ("props/sheet/sheet.glb", "../secret.glb",
                 "animate/knight__rig__20261005-120000/knight_walk.glb"):
        with pytest.raises(ValueError):
            aa.resolve_model(path, output)


def test_presets_need_a_file_and_a_safe_id(motions):
    assert [p["id"] for p in aa.presets(motions)] == ["walk", "kick"]


def test_rig_job_copies_the_source_and_records_both_licences(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    source = {"license": {"name": "MIT"}, "output": {"classification": "commercial"}}
    job = manager.create_rig("My Knight.glb", b"glTF-data", source, "generated")
    assert job.directory.name.startswith("My-Knight__rig__")
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glTF-data"
    assert job.result.name == "My-Knight_rigged.glb"
    (rig, rig_cwd), (clean, _) = job.commands
    assert rig[1:3] == ["demo.py", "--input"] and "--use_transfer" in rig and rig_cwd == aa.SKINTOKENS
    # The clean-up rewrites SkinTokens' result in place, after it.
    assert clean[1].endswith("clean_skin_weights.py") and clean[-1] == str(job.result)
    assert job.record["license"] == {"name": "MIT"}
    assert job.record["classification"] == "commercial"
    assert job.record["rig"]["license"]["name"] == "MIT"
    with pytest.raises(RuntimeError, match="already running"):
        manager.create_rig("other.glb", b"x", None, "uploaded")


def test_animation_needs_a_rigged_model_a_preset_and_a_sane_spread(tmp_path, motions):
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    with pytest.raises(ValueError, match="rigged"):
        manager.create_animation(_glb(tmp_path / "plain.glb"), "walk", 0)
    with pytest.raises(ValueError, match="preset"):
        manager.create_animation(rigged, "missing", 0)
    with pytest.raises(ValueError, match="between"):
        manager.create_animation(rigged, "walk", 90)
    with pytest.raises(ValueError, match="number"):
        manager.create_animation(rigged, "walk", "wide")
    job = manager.create_animation(rigged, "walk", 12)
    assert job.result == run / "knight_walk.glb"
    (command, _), = job.commands
    assert command[command.index("--arm-spread") + 1] == "12"
    assert job.record["motion"]["made_with"] == "Kimodo"
    assert "NVIDIA" in job.record["motion"]["license"]["name"]


def _fake_job(tmp_path, script: str) -> aa.AnimateJob:
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", script.format(out=result)]
    return aa.AnimateJob("0" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                         {"license": {"name": "MIT"}})


def test_a_finished_job_writes_its_provenance(tmp_path):
    job = _fake_job(tmp_path, "open(r'{out}', 'wb').write(b'glb'); print('hello')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"
    assert json.loads(job.result.with_suffix(".provenance.json").read_text())["license"]
    assert job.events[-1]["phase"] == "done"
    assert "hello" in job.log_path.read_text()


def test_done_is_said_only_after_the_done_event_names_the_file(tmp_path, monkeypatch):
    """A plan polls the status and reads the last event: if "done" came first (the fit check takes
    seconds), the plan saw no file and skipped the move. Lost a robot's rig on 2026-10-25."""
    job = _fake_job(tmp_path, "open(r'{out}', 'wb').write(b'glb')")
    seen = []

    def slow_fit(path):
        seen.append(job.status)          # what a poll would see while the fit check runs
        return {}

    monkeypatch.setattr(aa, "rig_fit", slow_fit)
    real_emit = job.emit

    def emit(event):
        if event.get("phase") == "done":
            seen.append(job.status)      # what a poll would see as the path arrives
        real_emit(event)

    job.emit = emit
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert seen and "done" not in seen
    assert job.status == "done" and job.events[-1]["phase"] == "done"


def test_a_job_that_writes_nothing_is_an_error_with_its_log(tmp_path):
    job = _fake_job(tmp_path, "import sys; print('boom'); sys.exit(3)")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error"
    assert "code 3" in job.events[-1]["message"]
    assert "boom" in job.events[-1]["log_tail"]


def test_progress_estimate_never_claims_done():
    assert aa.estimated_pct(0, 70) == 0
    assert aa.estimated_pct(35, 70) == 50
    assert aa.estimated_pct(500, 70) == 95


def test_a_post_launch_error_terminates_the_worker_before_releasing_the_slot(tmp_path, monkeypatch):
    """If the worker process is alive when an exception fires, it must be stopped and reaped
    before the active slot is freed."""
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", "print('working')"]
    job = aa.AnimateJob("2" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                        {"license": {"name": "MIT"}})
    manager = aa.AnimateJobManager(tmp_path, tmp_path)
    manager.active = job.id

    calls = []

    class RaisingStdout:
        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("broken pipe")

    class Worker:
        pid = 4242
        returncode = 0
        stdout = RaisingStdout()

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            calls.append(("wait", timeout, manager.active))
            return 0

        def poll(self):
            return None

    def fake_terminate_group(pid):
        calls.append(("terminate_group", pid, manager.active))

    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):
        if self.name == "pid":
            raise OSError("No space left on device")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(aa.subprocess, "Popen", Worker)
    monkeypatch.setattr(aa.processes, "terminate_group", fake_terminate_group)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    aa.run_job(job, manager)

    assert job.status == "error"
    assert calls == [("terminate_group", 4242, job.id), ("wait", pytest.approx(30, rel=0.1), job.id)]
    assert manager.active is None


def test_an_animate_run_records_its_pid_then_clears_it(tmp_path):
    from image_to_3dlab import processes

    directory = tmp_path / "clip"
    directory.mkdir()
    result = directory / "out.glb"
    result.write_bytes(b"glb")
    script = (
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(0.2)\n"
        "print(Path('pid').read_text().strip())\n"
    )
    job = aa.AnimateJob(
        "ab" * 16, "animate", directory, result,
        [([sys.executable, "-c", script], directory)], 1.0, {"license": "test"},
    )
    manager = aa.AnimateJobManager(tmp_path, tmp_path / "motions")
    manager.jobs[job.id] = job
    manager.active = job.id

    aa.run_job(job, manager)

    assert job.status == "done", job.log_lines
    assert job.log_lines[0] == processes.pid_record(job.process.pid)
    assert not (directory / "pid").exists()


def test_source_record_prefers_a_sidecar_that_names_a_licence(tmp_path):
    glb = _glb(tmp_path / "m.glb")
    (tmp_path / "m.json").write_text(json.dumps({"seed": 1}))
    assert aa.source_record_for(glb) is None
    (tmp_path / "m.retopo-repaint.json").write_text(json.dumps({"license": {"name": "X"}}))
    assert aa.source_record_for(glb)["license"]["name"] == "X"


def test_friendly_names_drop_run_stamps_and_keep_the_face_count(output):
    names = {m["kind"]: m["name"] for m in aa.pickable_models(output)}
    assert names["finished"] == "knight · 40k"
    assert names["rigged"] == "knight"
    long = aa.friendly_name("a-cute-anime-girl-standing-in-a-perfect-t-pose__20261003__pixal3d")
    assert long.startswith("a cute anime girl") and long.endswith("…") and len(long) == 40


def test_a_failing_first_step_stops_the_chain(tmp_path):
    result = tmp_path / "out.glb"
    marker = tmp_path / "second-ran"
    commands = [([sys.executable, "-c", "import sys; sys.exit(2)"], tmp_path),
                ([sys.executable, "-c", f"open(r'{marker}', 'w')"], tmp_path)]
    job = aa.AnimateJob("1" * 32, "rig", tmp_path, result, commands, 1.0, {})
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error" and not marker.exists()


def test_speed_defaults_to_the_presets_own_and_can_be_overridden(tmp_path, motions):
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    for preset, asked, expected in (("walk", None, "1"), ("kick", None, "1.25"), ("kick", 1.5, "1.5")):
        manager = aa.AnimateJobManager(tmp_path / "animate", motions)
        job = manager.create_animation(rigged, preset, 0, asked)
        (command, _), = job.commands
        assert command[command.index("--speed") + 1] == expected
        assert job.record["motion"]["speed"] == float(expected)
    with pytest.raises(ValueError, match="between"):
        aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(rigged, "walk", 0, 5)


def test_primary_moves_are_the_first_take():
    """A second take ends in `_b`. Everything else, including Walk (woman), is the move itself."""
    path = Path(__file__).resolve().parents[1] / "image_to_3dlab" / "motions" / "presets.json"
    shipped = json.loads(path.read_text())["presets"]
    primary = [preset for preset in shipped if not str(preset["id"]).endswith("_b")]
    assert "walk_woman" in {preset["id"] for preset in primary}
    assert [preset["id"] for preset in primary if preset.get("take")] == []


def test_the_shipped_preset_moves_are_all_in_the_repo():
    """The moves live in git: a fresh clone with an empty motions folder offers no moves."""
    shipped = aa.presets()
    assert len(shipped) >= 42, len(shipped)  # 21 moves, takes A and B
    assert {"idle", "walk", "run", "dance"} <= {p["id"] for p in shipped}


def test_free_ports_are_two_different_ports_that_bind():
    import socket
    ports = aa.free_ports(2)
    assert len(set(ports)) == 2
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))  # where SkinTokens' Blender server binds


def test_a_rig_job_hands_skintokens_free_ports_not_fixed_ones(tmp_path):
    # Windows can reserve SkinTokens' fixed ports for Hyper-V/WSL (#193): every run picks
    # its own, through the variables the patched spec.py reads.
    job = _fake_job(tmp_path, "import os; open(r'{out}', 'w').write("
                              "os.environ['SKINTOKENS_PORT'] + ' ' + os.environ['SKINTOKENS_BPY_PORT'])")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    port, bpy_port = (int(p) for p in job.result.read_text().split())
    assert port != bpy_port


def test_an_older_install_gets_the_port_edit_before_it_rigs(tmp_path, monkeypatch):
    spec = tmp_path / "src" / "server" / "spec.py"
    spec.parent.mkdir(parents=True)
    spec.write_text("import os\nPORT = 59875\nBPY_PORT = 59876\n")
    monkeypatch.setattr(aa, "port_edits", lambda: [e for e in _portable().EDITS
                                                   if "PORT" in e.old])
    assert aa.ensure_port_patch(tmp_path) is None
    assert 'os.environ.get("SKINTOKENS_BPY_PORT"' in spec.read_text()
    assert aa.ensure_port_patch(tmp_path) is None  # already there: nothing to do


def test_a_tree_the_port_edit_no_longer_fits_still_rigs(tmp_path):
    # Upstream moved on: the rig keeps upstream's ports (today's behaviour) and says why.
    (tmp_path / "src" / "server").mkdir(parents=True)
    (tmp_path / "src" / "server" / "spec.py").write_text("PORT = 1234\n")
    assert "port" in aa.ensure_port_patch(tmp_path).lower()


def _portable():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import patch_skintokens_portable
    return patch_skintokens_portable
