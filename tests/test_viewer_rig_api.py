from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from viewer import rig_api


def valid_sidecar(asset: bytes, scene: bytes) -> bytes:
    source = {
        "schemaVersion": 1,
        "rigProfile": "rigify.quadruped.v1",
        "assetFingerprint": "sha256:" + hashlib.sha256(asset).hexdigest(),
        "coordinateSpace": "armature-local",
        "mirror": {"axis": "X", "origin": 0},
        "joints": {
            "elbow": {
                "label": "Elbow", "position": [0, 0, 0], "sourceBone": "DEF-arm",
                "parent": None, "mirrorOf": None,
            }
        },
        "corrections": {
            "elbow": {
                "sourcePosition": [0, 0, 0], "targetPosition": [0, 0, 1],
                "delta": [0, 0, 1], "mirrored": False,
            }
        },
        "binding": {
            "adapter": "test", "sceneFingerprint": "sha256:" + hashlib.sha256(scene).hexdigest(),
            "metarigObjectId": "object-id",
            "joints": {"elbow": {"targets": [
                {"boneId": "bone-id", "boneName": "arm", "endpoint": "tail"}
            ]}},
        },
    }
    return json.dumps(source).encode()


def test_manager_validates_and_persists_a_rebind_bundle(tmp_path):
    manager = rig_api.RigJobManager(tmp_path)
    asset, scene = b"glb", b"blend"

    job = manager.create("Creature.glb", asset, scene, valid_sidecar(asset, scene))

    assert job.asset_path.read_bytes() == asset
    assert job.scene_path.read_bytes() == scene
    assert manager.active == job.id
    assert job.directory.parent == tmp_path


def test_manager_removes_a_rejected_bundle(tmp_path):
    manager = rig_api.RigJobManager(tmp_path)

    with pytest.raises(Exception, match="fingerprint mismatch"):
        manager.create("Creature.glb", b"wrong", b"blend", valid_sidecar(b"glb", b"blend"))
    assert list(tmp_path.iterdir()) == []


def test_blender_command_is_noninteractive_and_typed(tmp_path):
    manager = rig_api.RigJobManager(tmp_path)
    job = manager.create("Creature.glb", b"glb", b"blend", valid_sidecar(b"glb", b"blend"))
    command = rig_api.build_command(job, Path("/Applications/Blender"))

    # The uploaded .blend's own Python must never run, whatever Auto Run says: both flags
    # come before the file, as Blender applies arguments in order.
    assert command[:5] == ["/Applications/Blender", "--background", "--factory-startup",
                           "--disable-autoexec", str(job.scene_path)]
    # Without this Blender exits 0 when the worker raises, so a crash reads as success.
    # It must come before --python: Blender applies arguments in order.
    assert command[5:7] == ["--python-exit-code", "1"]
    assert command[7:9] == ["--python", str(rig_api.WORKER)]
    assert command[-4:] == [
        str(job.result_glb), str(job.result_blend), str(job.result_sidecar), str(job.report_path)
    ]


def test_stage_lines_become_stable_progress_events():
    assert rig_api._stage_event("I2L_STAGE::weights::Transferring") == {
        "phase": "weights", "overall_pct": 55, "message": "Transferring",
    }
    assert rig_api._stage_event("ordinary Blender output") is None


def test_runner_records_a_pid_for_server_restart_reconciliation():
    source = Path(rig_api.__file__).read_text()

    assert 'joinpath("pid").write_text' in source
    assert 'joinpath("pid").unlink(missing_ok=True)' in source


def test_a_post_launch_error_terminates_the_worker_before_releasing_the_slot(tmp_path, monkeypatch):
    """If the pid-record write fails after Popen, the child must be stopped and reaped
    before the active slot is freed."""
    job = rig_api.RigJob("2" * 32, tmp_path / "out" / "creature__rebind__20261027-020202")
    manager = rig_api.RigJobManager(tmp_path / "out")
    manager.active = job.id

    calls = []

    class Worker:
        pid = 4242
        returncode = 0
        stdout = iter([])

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

    monkeypatch.setattr(rig_api, "blender_executable", lambda: Path("/bin/blender"))
    monkeypatch.setattr(rig_api.subprocess, "Popen", Worker)
    monkeypatch.setattr(rig_api.processes, "terminate_group", fake_terminate_group)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    rig_api.run_job(job, manager)

    assert job.status == "error"
    assert calls == [("terminate_group", 4242, job.id), ("wait", pytest.approx(30, rel=0.1), job.id)]
    assert manager.active is None
