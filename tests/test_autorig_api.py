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
    (root / "presets.json").write_text(json.dumps({"presets": [
        {"id": "walk", "label": "Walk", "seconds": 4},
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
    assert [p["id"] for p in aa.presets(motions)] == ["walk"]


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
