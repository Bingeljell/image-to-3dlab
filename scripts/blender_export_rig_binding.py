#!/usr/bin/env python3
"""Prepare an open Blender metarig scene and export its browser rig sidecar.

Run through Blender, opening the source scene before this script:

    blender --background source.blend --python scripts/blender_export_rig_binding.py -- \
        source.glb prepared.blend prepared.rig.json \
        rigify.basic-quadruped.blender-5.2.v1

The source scene is never overwritten. Persistent object/bone IDs are stamped into the
prepared copy, and the sidecar fingerprints that exact saved copy plus the reviewed GLB.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.rig_profile import load_rig_profile
from image_to_3dlab.rig_sidecar import asset_fingerprint

OBJECT_ID = "i2l_metarig_id"
BONE_ID = "i2l_bone_id"


def parse_args(argv: list[str]) -> tuple[Path, Path, Path, str]:
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(args) != 4:
        raise SystemExit(
            "usage: blender ... --python blender_export_rig_binding.py -- "
            "SOURCE.glb PREPARED.blend OUTPUT.rig.json ADAPTER"
        )
    asset, scene, sidecar = (Path(value).expanduser().resolve() for value in args[:3])
    if asset.suffix.lower() != ".glb":
        raise SystemExit("SOURCE must be a .glb")
    if scene.suffix.lower() != ".blend":
        raise SystemExit("PREPARED must be a .blend")
    if not sidecar.name.lower().endswith(".rig.json"):
        raise SystemExit("OUTPUT must end with .rig.json")
    return asset, scene, sidecar, args[3]


def _find_metarig(bpy, profile):
    tagged = [
        obj for obj in bpy.data.objects
        if obj.type == "ARMATURE" and obj.get("i2l_adapter") == profile["id"]
    ]
    if len(tagged) == 1:
        return tagged[0]
    named = [
        bpy.data.objects.get(name) for name in profile["blenderMetarigNames"]
        if bpy.data.objects.get(name) is not None
        and bpy.data.objects.get(name).type == "ARMATURE"
    ]
    named = list(dict.fromkeys(named))
    if len(named) != 1:
        raise RuntimeError(
            f"adapter {profile['id']} expected one metarig named from "
            f"{profile['blenderMetarigNames']}, found {len(named)}"
        )
    return named[0]


def _ensure_ids(metarig, profile) -> None:
    metarig[OBJECT_ID] = metarig.get(OBJECT_ID) or uuid.uuid4().hex
    metarig["i2l_adapter"] = profile["id"]
    required_names = {
        target["bone"]
        for joint in profile["joints"].values()
        for target in joint["targets"]
    }
    missing = sorted(required_names - set(metarig.data.bones.keys()))
    if missing:
        raise RuntimeError("metarig is missing adapter bones: " + ", ".join(missing))
    for name in required_names:
        bone = metarig.data.bones[name]
        bone[BONE_ID] = bone.get(BONE_ID) or uuid.uuid4().hex


def _joint_position(metarig, targets) -> list[float]:
    target = targets[0]
    bone = metarig.data.bones[target["bone"]]
    value = bone.head_local if target["endpoint"] == "head" else bone.tail_local
    return [float(axis) for axis in value]


def build_sidecar(metarig, profile, asset_path: Path, scene_path: Path) -> dict:
    joints = {}
    bindings = {}
    for joint_id, joint in profile["joints"].items():
        joints[joint_id] = {
            "label": joint["label"],
            "position": _joint_position(metarig, joint["targets"]),
            "sourceBone": joint["sourceBone"],
            "parent": joint["parent"],
            "mirrorOf": joint["mirrorOf"],
        }
        bindings[joint_id] = {
            "targets": [
                {
                    "boneId": metarig.data.bones[target["bone"]][BONE_ID],
                    "boneName": target["bone"],
                    "endpoint": target["endpoint"],
                }
                for target in joint["targets"]
            ]
        }
    return {
        "schemaVersion": 1,
        "rigProfile": profile["semanticProfile"],
        "assetFingerprint": asset_fingerprint(asset_path),
        "coordinateSpace": "armature-local",
        "mirror": {"axis": "X", "origin": 0.0},
        "joints": joints,
        "corrections": {},
        "binding": {
            "adapter": profile["id"],
            "sceneFingerprint": asset_fingerprint(scene_path),
            "metarigObjectId": metarig[OBJECT_ID],
            "joints": bindings,
        },
    }


def main() -> int:
    asset_path, scene_path, sidecar_path, adapter = parse_args(sys.argv)
    if not asset_path.is_file():
        raise SystemExit(f"source GLB does not exist: {asset_path}")
    import bpy

    profile = load_rig_profile(adapter)
    installed = ".".join(str(value) for value in bpy.app.version[:2])
    if installed != profile["blenderVersion"]:
        raise RuntimeError(
            f"adapter {adapter} targets Blender {profile['blenderVersion']}, installed is {installed}"
        )
    metarig = _find_metarig(bpy, profile)
    _ensure_ids(metarig, profile)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path), check_existing=False)
    sidecar = build_sidecar(metarig, profile, asset_path, scene_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print("I2L_RIG_BINDING::" + json.dumps({
        "adapter": adapter,
        "joints": len(sidecar["joints"]),
        "scene": str(scene_path),
        "sidecar": str(sidecar_path),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
