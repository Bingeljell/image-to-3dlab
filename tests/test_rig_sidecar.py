from __future__ import annotations

import hashlib
import json

import pytest

from image_to_3dlab.rig_sidecar import (
    RigSidecarError,
    asset_fingerprint,
    load_sidecar,
    plan_corrections,
    validate_sidecar,
    verify_asset,
    verify_scene,
)


def sidecar_for(payload: bytes) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "schemaVersion": 1,
        "rigProfile": "rigify.quadruped.v1",
        "assetFingerprint": f"sha256:{digest}",
        "coordinateSpace": "armature-local",
        "mirror": {"axis": "X", "origin": 0},
        "joints": {
            "chest": {
                "label": "Chest", "position": [0, 1, 0],
                "sourceBone": "DEF-spine", "parent": None, "mirrorOf": None,
            },
            "shoulder.L": {
                "label": "Left shoulder", "position": [0.2, 1, 0],
                "sourceBone": "DEF-upper_arm.L", "parent": "chest",
                "mirrorOf": None,
            },
        },
        "corrections": {
            "shoulder.L": {
                "sourcePosition": [0.2, 1, 0],
                "targetPosition": [0.25, 1.1, 0],
                "delta": [0.05, 0.1, 0],
                "mirrored": False,
            }
        },
    }


def with_binding(sidecar: dict, scene_payload: bytes) -> dict:
    sidecar["binding"] = {
        "adapter": "rigify.basic-quadruped.blender-5.2.v1",
        "sceneFingerprint": f"sha256:{hashlib.sha256(scene_payload).hexdigest()}",
        "metarigObjectId": "metarig-uuid",
        "joints": {
            "shoulder.L": {
                "targets": [
                    {
                        "boneId": "bone-uuid",
                        "boneName": "front_thigh.L",
                        "endpoint": "head",
                    }
                ]
            }
        },
    }
    return sidecar


def test_validate_and_plan_browser_sidecar():
    validated = validate_sidecar(sidecar_for(b"glb"))
    plan = plan_corrections(validated)

    assert len(plan) == 1
    assert plan[0].joint_id == "shoulder.L"
    assert plan[0].source_bone == "DEF-upper_arm.L"
    assert plan[0].target_position == (0.25, 1.1, 0.0)


def test_fingerprint_verification_is_exact(tmp_path):
    asset = tmp_path / "creature.glb"
    asset.write_bytes(b"glb")
    sidecar = validate_sidecar(sidecar_for(b"glb"))

    assert asset_fingerprint(asset) == sidecar["assetFingerprint"]
    assert verify_asset(sidecar, asset) == sidecar["assetFingerprint"]
    asset.write_bytes(b"different")
    with pytest.raises(RigSidecarError, match="fingerprint mismatch"):
        verify_asset(sidecar, asset)


def test_scene_binding_is_asset_specific_and_supplies_stable_bone_targets(tmp_path):
    scene = tmp_path / "source.blend"
    scene.write_bytes(b"blend")
    sidecar = validate_sidecar(with_binding(sidecar_for(b"glb"), b"blend"))

    assert verify_scene(sidecar, scene) == sidecar["binding"]["sceneFingerprint"]
    correction = plan_corrections(sidecar)[0]
    assert correction.targets[0].bone_id == "bone-uuid"
    assert correction.targets[0].bone_name == "front_thigh.L"
    assert correction.targets[0].endpoint == "head"

    scene.write_bytes(b"different scene")
    with pytest.raises(RigSidecarError, match="scene fingerprint mismatch"):
        verify_scene(sidecar, scene)


def test_rebind_requires_a_binding_manifest(tmp_path):
    scene = tmp_path / "source.blend"
    scene.write_bytes(b"blend")

    with pytest.raises(RigSidecarError, match="binding manifest is required"):
        verify_scene(validate_sidecar(sidecar_for(b"glb")), scene)


def test_load_sidecar_normalizes_optional_joint_references(tmp_path):
    source = sidecar_for(b"glb")
    del source["joints"]["chest"]["parent"]
    del source["joints"]["chest"]["mirrorOf"]
    path = tmp_path / "creature.rig.json"
    path.write_text(json.dumps(source))

    loaded = load_sidecar(path)

    assert loaded["joints"]["chest"]["parent"] is None
    assert loaded["joints"]["chest"]["mirrorOf"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schemaVersion=2), "schemaVersion"),
        (lambda value: value.update(coordinateSpace="world"), "coordinateSpace"),
        (lambda value: value["joints"]["chest"].update(parent="missing"), "unknown joint"),
        (
            lambda value: value["corrections"]["shoulder.L"].update(
                sourcePosition=[0.3, 1, 0]
            ),
            "sourcePosition does not match",
        ),
        (
            lambda value: value["corrections"]["shoulder.L"].update(delta=[0, 0, 0]),
            "delta does not match",
        ),
    ],
)
def test_invalid_or_stale_sidecars_are_rejected(mutate, message):
    source = sidecar_for(b"glb")
    mutate(source)

    with pytest.raises(RigSidecarError, match=message):
        validate_sidecar(source)
