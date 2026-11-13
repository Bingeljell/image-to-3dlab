"""The Animate tab's backend: what it offers, what it refuses, and the records it writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from glb_fixtures import valid_glb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import autorig_api as aa


@pytest.fixture(autouse=True)
def no_real_rig_install(tmp_path_factory, monkeypatch):
    """Jobs here never reach this machine's real auto-rig.

    Before a Rig job, run_job asks the installed SkinTokens Python about the graphics card and
    reads its diffusers version. On a machine with Rig installed that reached the real venv,
    and a test that fakes subprocess.Popen broke inside the probe instead of where it meant to.
    """
    monkeypatch.setattr(aa, "SKINTOKENS", tmp_path_factory.mktemp("no-skintokens"))
    monkeypatch.setattr(aa.cuda_arch, "probe", lambda *a, **k: None)


def _glb(path: Path, payload: bytes = valid_glb()) -> Path:
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
    (root / "idle.npz").write_bytes(b"x")
    (root / "presets.json").write_text(json.dumps({"aliases": {"walk_old": "walk"}, "presets": [
        {"id": "walk", "label": "Walk", "seconds": 4},
        {"id": "kick", "label": "Kick", "seconds": 2, "speed": 1.25},
        {"id": "idle", "label": "Idle (restless)", "seconds": 7, "loop": False},
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
    assert [p["id"] for p in aa.presets(motions)] == ["walk", "kick", "idle"]


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
    assert "--loop" in command and "--in-place" in command
    assert job.record["motion"]["made_with"] == "Kimodo"
    assert job.record["motion"]["loop"] is True and job.record["motion"]["grounded"] is True
    assert job.record["motion"]["in_place"] is True
    assert "NVIDIA" in job.record["motion"]["license"]["name"]


def test_animate_command_loops_repeating_clips_and_leaves_actions_alone(tmp_path):
    rigged = tmp_path / "knight_rigged.glb"
    motion = tmp_path / "walk.npz"
    walk = aa.animate_command(rigged, motion, tmp_path / "walk.glb", 0, 1.0,
                              aa.repeats("walk"), aa.repeats("walk"))
    kick = aa.animate_command(rigged, motion, tmp_path / "kick.glb", 0, 1.0,
                              aa.repeats("kick"), aa.repeats("kick"))
    assert "--loop" in walk and "--in-place" in walk
    assert "--loop" not in kick and "--in-place" not in kick
    # Grounding is on by default, so the studio passes no opt-out.
    assert "--no-ground" not in walk


def test_an_old_move_name_still_finds_the_move_that_replaced_it(tmp_path, motions):
    # Run, Wave, Sit down and Strafe left were renamed or folded into their second take; a saved
    # plan or an agent asking by the old name must still get the move, not "no such preset".
    assert aa.resolve_move("run") == "run_b" and aa.resolve_move("strafe_left") == "strafe_left_b"
    assert aa.resolve_move("walk") == "walk"           # a current name is left alone
    shipped = {p["id"] for p in aa.presets()}
    for old in ("idle_b", "fight_stance", "punch_combo", "run", "wave", "sit", "strafe_left"):
        assert aa.resolve_move(old) in shipped, old    # every alias lands on a move that exists
    # Every move id 0.5.8 shipped still works, except the moves dropped on purpose.
    dropped = {"kata", "kata_b", "sword_swing", "sword_swing_b", "roundhouse", "roundhouse_b",
               "flying_kick", "flying_kick_b"}
    v058 = {"cheer", "cheer_b", "dance", "dance_b", "death", "death_b", "fight_stance",
            "fight_stance_b", "hit_react", "hit_react_b", "idle", "idle_b", "jump", "jump_b",
            "kick", "kick_b", "punch", "punch_b", "punch_combo", "punch_combo_b", "roll", "roll_b",
            "run", "run_b", "sit", "sit_b", "sneak", "sneak_b", "strafe_left", "strafe_left_b",
            "walk", "walk_b", "walk_woman", "wave", "wave_b"} | dropped
    for old in sorted(v058 - dropped):
        assert aa.resolve_move(old) in shipped, old
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    job = manager.create_animation(rigged, "walk_old", 0)      # the fixture's alias
    assert job.result == run / "knight_walk.glb"
    manager.finish(job)
    with pytest.raises(ValueError, match="The moves are"):
        manager.create_animation(rigged, "nonsense", 0)


def test_each_move_says_whether_it_stays_in_place(tmp_path, motions):
    # Loops travel nowhere in a game; one-offs keep their travel. presets.json says which, per move.
    shipped = {p["id"]: p for p in aa.presets()}
    for move in ("idle", "walk", "run_b", "sneak", "strafe_left_b", "strafe_right", "walk_woman"):
        assert aa.default_in_place(shipped[move]) is True, move
    for move in ("jump", "roll", "punch", "death", "cheer", "robot_dance", "sit_b"):
        assert aa.default_in_place(shipped[move]) is False, move


def test_the_in_place_switch_reaches_the_retarget_command(tmp_path, motions):
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")

    walk = manager.create_animation(rigged, "walk", 0)              # the move's own default
    (command, _), = walk.commands
    assert "--in-place" in command and walk.record["motion"]["in_place"] is True
    manager.finish(walk)

    kick = manager.create_animation(rigged, "kick", 0)               # a one-off: travels by default
    (command, _), = kick.commands
    assert "--in-place" not in command and kick.record["motion"]["in_place"] is False
    manager.finish(kick)

    with pytest.raises(ValueError, match="true or false"):
        manager.create_animation(rigged, "walk", 0, in_place="yes")

    stuck = manager.create_animation(rigged, "kick", 0, in_place=True)  # flipped on
    (command, _), = stuck.commands
    assert "--in-place" in command and stuck.record["motion"]["in_place"] is True


def test_a_move_that_keeps_its_whole_clip_only_has_its_seam_blended(tmp_path, motions):
    """The restless idle is 7 s of distinct beats; trimming it cut 5 s out, so it keeps every
    frame and the retargeter only blends the seam. It still passes --loop for that blend."""
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    job = manager.create_animation(rigged, "idle", 0)
    (command, _), = job.commands
    assert "--loop" in command and "--keep-whole" in command
    assert job.record["motion"]["loop"] is False and job.record["motion"]["keep_whole"] is True
    manager.finish(job)
    # A normal loop is blended but trimmed, and a one-off gets neither flag.
    walk = manager.create_animation(rigged, "walk", 0)
    (command, _), = walk.commands
    assert "--loop" in command and "--keep-whole" not in command
    manager.finish(walk)
    kick = manager.create_animation(rigged, "kick", 0)
    (command, _), = kick.commands
    assert "--loop" not in command and "--keep-whole" not in command


def test_repeating_presets_are_listed_in_one_place():
    for preset in ("idle", "walk", "run", "sneak", "strafe",
                   "walk_b", "run_b", "walk_woman", "strafe_left"):
        assert aa.repeats(preset), preset
    for preset in ("kick", "jump", "punch", "wave", "death", "roll"):
        assert not aa.repeats(preset), preset


def test_a_move_can_keep_its_whole_clip(tmp_path):
    # Trimming to the shortest repeat cut a 7 s look-left, look-right idle to 1.7 s and dropped the
    # looks; `"loop": false` keeps every frame. Without the field, the name rule still decides.
    assert aa.loops({"id": "idle", "loop": False}) is False
    assert aa.loops({"id": "crouch_sneak", "loop": True}) is True
    assert aa.loops({"id": "walk"}) is True and aa.loops({"id": "kick"}) is False

    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "idle.npz").write_bytes(b"x")
    (motions / "presets.json").write_text(json.dumps({"presets": [
        {"id": "idle", "label": "Idle", "seconds": 7, "in_place": True, "loop": False}]}))
    rigged = _glb(tmp_path / "animate" / "knight__rig__20261005-120000" / "knight_rigged.glb")
    job = aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(rigged, "idle", 0)
    (command, _), = job.commands
    # It keeps every frame but still passes --loop, so the retargeter blends the seam.
    assert "--loop" in command and "--keep-whole" in command and "--in-place" in command
    assert job.record["motion"]["loop"] is False and job.record["motion"]["keep_whole"] is True


def test_the_shipped_moves_keep_whole_clips_and_strike_faster():
    shipped = {p["id"]: p for p in aa.presets()}
    for move in ("idle", "idle_restless", "sprint", "zombie_walk"):   # beats, or already a loop
        assert aa.loops(shipped[move]) is False, move
    assert aa.loops(shipped["crouch_sneak"]) is True
    assert shipped["punch"]["speed"] == shipped["punch_b"]["speed"] == 1.3
    assert shipped["one_two"]["speed"] == 1.5


def _write_glb_script(payload: bytes | None = None) -> str:
    """A worker stand-in that writes a real GLB, so the generator stamp can read it."""
    return f"open(r'{{out}}', 'wb').write(bytes.fromhex('{(payload or valid_glb()).hex()}'))"


def _fake_job(tmp_path, script: str) -> aa.AnimateJob:
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", script.format(out=result)]
    return aa.AnimateJob("0" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                         {"license": {"name": "MIT"}})


def test_a_finished_job_writes_its_provenance(tmp_path):
    job = _fake_job(tmp_path, _write_glb_script() + "; print('hello')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"
    assert json.loads(job.result.with_suffix(".provenance.json").read_text())["license"]
    assert job.events[-1]["phase"] == "done"
    assert "hello" in job.log_path.read_text()


def test_done_is_said_only_after_the_done_event_names_the_file(tmp_path, monkeypatch):
    """A plan polls the status and reads the last event: if "done" came first (the fit check takes
    seconds), the plan saw no file and skipped the move. Lost a robot's rig on 2026-10-25."""
    job = _fake_job(tmp_path, _write_glb_script())
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
    result.write_bytes(valid_glb())
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


def test_a_hidden_second_take_always_has_a_visible_first_one():
    """A take kept alongside another stays hidden (`take` 1) and keeps its `_b` id. When a move was
    kept as its second take only (Jog is Run B), that take becomes the move: it shows, id unchanged."""
    path = Path(__file__).resolve().parents[1] / "image_to_3dlab" / "motions" / "presets.json"
    shipped = json.loads(path.read_text())["presets"]
    primary = {p["id"] for p in shipped if not p.get("take")}
    assert "walk_woman" in primary and "run_b" in primary and "walk_b" not in primary
    for preset in shipped:
        if preset.get("take"):
            assert str(preset["id"]).endswith("_b"), preset["id"]
            assert str(preset["id"]).removesuffix("_b") in primary, preset["id"]


def test_the_shipped_preset_moves_are_all_in_the_repo():
    """The moves live in git: a fresh clone with an empty motions folder offers no moves."""
    shipped = aa.presets()
    assert len(shipped) >= 29, len(shipped)  # the reviewed move set, takes A and B where both are kept
    assert {"idle", "walk", "run_b", "dance", "strafe_right"} <= {p["id"] for p in shipped}
    assert not {"roundhouse", "flying_kick", "kata", "sword_swing", "kick_b"} & {p["id"] for p in shipped}
    assert {"sprint", "crouch_sneak", "zombie_walk", "robot_dance", "one_two", "death_2"} <= {
        p["id"] for p in shipped}


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


# --- a card Rig cannot run on says so, before the model loads (Quadro P5000, #? 2026-11-13) ---

def test_an_old_card_is_refused_with_one_sentence_and_no_subprocess(tmp_path, monkeypatch):
    """The card dies on its first kernel; loading 8 GB of model to learn that is the bug."""
    sentence = ("Your graphics card (NVIDIA Quadro P5000) is older than Rig supports: it needs "
                "an NVIDIA RTX 20-series or newer.")
    monkeypatch.setattr(aa, "card_problem", lambda python=None: sentence)
    job = _fake_job(tmp_path, "print('should never run')")
    manager = aa.AnimateJobManager(tmp_path, tmp_path)
    aa.run_job(job, manager)

    assert job.status == "error"
    assert job.events[-1]["message"] == sentence and job.events[-1]["phase"] == "error"
    assert job.process is None                      # nothing was started
    assert not job.result.exists()
    assert manager.active is None


def test_the_refusal_reaches_the_bug_report(tmp_path, monkeypatch):
    """Report a bug reads the newest error record's note and log: the sentence must be there."""
    from activity_api import read_history

    sentence = ("Your graphics card (NVIDIA Quadro P5000) is older than Rig supports: it needs "
                "an NVIDIA RTX 20-series or newer.")
    monkeypatch.setattr(aa, "card_problem", lambda python=None: sentence)
    job = _fake_job(tmp_path, "print('never')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))

    (record,) = read_history(tmp_path)
    assert record["result"] == "error" and record["step"] == "rigged"
    assert record["note"] == sentence
    assert sentence in record["log"]

    # ... and Report a bug reads it from there, which is the only reason it is written.
    import bug_report_api

    text = bug_report_api.details(tmp_path, home=str(tmp_path), machine="Windows · NVIDIA Quadro P5000")
    assert f"What it said: {sentence}" in text


def test_a_card_that_works_is_not_mentioned(tmp_path, monkeypatch):
    monkeypatch.setattr(aa, "card_problem", lambda python=None: None)
    job = _fake_job(tmp_path, _write_glb_script())
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"


def test_a_rig_that_could_not_even_start_is_in_the_report_too(tmp_path, monkeypatch):
    """A launch that raises is a failure like any other, and Report a bug only reads the file."""
    from activity_api import read_history

    def no_popen(*args, **kwargs):
        raise OSError("SkinTokens is not installed")

    monkeypatch.setattr(aa, "card_problem", lambda python=None: None)
    monkeypatch.setattr(aa.subprocess, "Popen", no_popen)
    job = _fake_job(tmp_path, "print('never')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))

    assert job.status == "error"
    (record,) = read_history(tmp_path)
    assert record["note"] == "SkinTokens is not installed"


def test_a_refused_rig_lands_where_report_a_bug_looks():
    """The studio's own manager writes history into output/, beside every other step's."""
    import generate_api

    assert aa.ACTIVITY_ROOT == aa.ANIMATE_JOBS.activity_root
    assert aa.ACTIVITY_ROOT == generate_api.OUTPUT_ROOT
    assert aa.ANIMATE_JOBS.output_root == aa.ACTIVITY_ROOT / "animate"


def test_the_check_asks_the_venvs_own_torch(monkeypatch):
    """SkinTokens' venv carries its own torch; the host's means nothing to it."""
    seen = []
    monkeypatch.setattr(aa.cuda_arch, "card_problem",
                        lambda python, run=None: seen.append(python) or None)
    aa.card_problem(Path("/venv/python"))
    assert seen == [Path("/venv/python")]


def _site(root, windows=False):
    site = root / ".venv" / ("Lib" if windows else "lib/python3.11") / "site-packages"
    site.mkdir(parents=True)
    return site


@pytest.mark.parametrize("windows", [False, True])
def test_the_installed_diffusers_is_read_from_its_record(tmp_path, windows):
    (_site(tmp_path, windows) / "diffusers-0.43.0.dist-info").mkdir()
    assert aa.installed_version(tmp_path, "diffusers") == "0.43.0"
    assert aa.installed_version(tmp_path / "nothing", "diffusers") is None


@pytest.mark.parametrize("version, broken", [
    ("0.40.0", False), ("0.42.1", False), ("0.43.0", True), ("0.43.0rc1", True), ("0.44.2", True),
    ("1.0.0", True), (None, False),
])
def test_diffusers_from_0_43_cannot_run_the_rig(version, broken):
    assert aa.diffusers_too_new(version) is broken


def test_a_too_new_diffusers_is_stepped_down_before_rig(tmp_path):
    (_site(tmp_path) / "diffusers-0.43.0.dist-info").mkdir()
    ran = []
    note = aa.ensure_diffusers(tmp_path, runner=lambda cmd, **kw: ran.append(cmd), uv="/bin/uv")
    assert ran and ran[0][:3] == ["/bin/uv", "pip", "install"] and ran[0][-1] == "diffusers<0.43"
    assert "0.43.0" in note


def test_a_good_diffusers_is_left_alone(tmp_path):
    (_site(tmp_path) / "diffusers-0.42.1.dist-info").mkdir()
    ran = []
    assert aa.ensure_diffusers(tmp_path, runner=lambda cmd, **kw: ran.append(cmd), uv="/bin/uv") is None
    assert ran == []


def test_no_uv_says_how_to_fix_it_instead_of_crashing(tmp_path, monkeypatch):
    (_site(tmp_path) / "diffusers-0.43.0.dist-info").mkdir()
    monkeypatch.setattr(aa.host, "find_uv", lambda *a, **k: None)
    assert "Setup & Status" in aa.ensure_diffusers(tmp_path, runner=lambda *a, **k: None)

def test_the_bone_names_option_is_kept_on_the_job_and_in_its_record(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_rig("knight.glb", b"glTF-data", None, "generated", bone_names="unity")
    assert job.bone_names == "unity"
    assert job.record["rig"]["bone_names"] == "unity"
    plain = aa.AnimateJobManager(tmp_path / "plain", tmp_path).create_rig("knight.glb", b"x", None, "uploaded")
    assert plain.bone_names is None and "bone_names" not in plain.record["rig"]
    numbered = aa.AnimateJobManager(tmp_path / "numbered", tmp_path).create_rig(
        "knight.glb", b"x", None, "uploaded", bone_names="numbered")
    assert numbered.bone_names is None   # SkinTokens' own bone_0, bone_1, ... are the default


def test_an_unknown_bone_names_choice_is_refused_before_anything_is_written(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    with pytest.raises(ValueError, match="unity"):
        manager.create_rig("knight.glb", b"x", None, "generated", bone_names="maya")
    assert not (tmp_path / "animate").exists()
    assert not manager.busy()


def test_a_rig_with_bone_names_names_them_before_the_file_is_reported(tmp_path, monkeypatch):
    job = _fake_job(tmp_path, _write_glb_script())
    job.bone_names = "unity"
    calls = []

    def name_bones(path, preset):
        calls.append((path, preset, job.status))
        return {"preset": preset, "named": 52, "bones": 60, "kept": ["bone_58"], "capped": 0, "missing": []}

    monkeypatch.setattr(aa.bone_names, "name_bones", name_bones)
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert calls == [(job.result, "unity", "running")]
    assert job.status == "done"
    assert "Unity Humanoid names on 52 of 60 bones" in job.events[-1]["bone_names"]
    assert "Unity Humanoid names on 52 of 60 bones" in job.log_path.read_text()
    record = json.loads(job.result.with_suffix(".provenance.json").read_text())
    assert record["rig"]["bone_names"] == {"preset": "unity", "named": 52, "bones": 60, "missing": []}


def test_a_skeleton_that_is_no_humanoid_keeps_its_numbered_names_and_still_rigs(tmp_path, monkeypatch):
    job = _fake_job(tmp_path, _write_glb_script())
    job.bone_names = "unity"

    def name_bones(path, preset):
        raise LookupError("no humanoid body found in the skeleton")

    monkeypatch.setattr(aa.bone_names, "name_bones", name_bones)
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"
    assert "kept the numbered names" in job.events[-1]["bone_names"]


def test_a_rig_without_the_option_never_renames(tmp_path, monkeypatch):
    job = _fake_job(tmp_path, _write_glb_script())
    monkeypatch.setattr(aa.bone_names, "name_bones", lambda *a: pytest.fail("renamed without being asked"))
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done" and "bone_names" not in job.events[-1]


def test_the_rig_form_s_bone_names_field_is_read_and_checked():
    assert aa.bone_names_choice({}) is None
    assert aa.bone_names_choice({"bone_names": {"value": " unity "}}) == "unity"
    assert aa.bone_names_choice({"bone_names": {"value": "numbered"}}) is None
    with pytest.raises(ValueError, match="unity"):
        aa.bone_names_choice({"bone_names": {"value": "maya"}})
