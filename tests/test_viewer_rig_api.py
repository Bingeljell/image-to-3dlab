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

    assert command[:3] == ["/Applications/Blender", "--background", str(job.scene_path)]
    assert command[3:5] == ["--python", str(rig_api.WORKER)]
    assert command[-4:] == [
        str(job.result_glb), str(job.result_blend), str(job.result_sidecar), str(job.report_path)
    ]


def test_stage_lines_become_stable_progress_events():
    assert rig_api._stage_event("I2L_STAGE::weights::Transferring") == {
        "phase": "weights", "overall_pct": 55, "message": "Transferring",
    }
    assert rig_api._stage_event("ordinary Blender output") is None
