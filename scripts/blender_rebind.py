#!/usr/bin/env python3
"""Apply browser fit-joint corrections to an opened prepared Blender scene.

    blender --background prepared.blend --python scripts/blender_rebind.py -- \
        source.glb corrected.rig.json result.glb result.blend result.rig.json report.json

This script never opens or overwrites the source scene itself; Blender opens it before the
script, and every output path must be distinct. Bone names are diagnostics only—the binding
manifest's persistent object and bone IDs are authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_to_3dlab.rig_sidecar import (  # noqa: E402
    JointCorrection,
    asset_fingerprint,
    load_sidecar,
    plan_corrections,
    verify_asset,
    verify_scene,
)
from blender_rebind_weights import transfer_weights  # noqa: E402

OBJECT_ID = "i2l_metarig_id"
BONE_ID = "i2l_bone_id"


def parse_args(argv: list[str]) -> tuple[Path, ...]:
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(args) != 6:
        raise SystemExit(
            "usage: blender ... PREPARED.blend --python blender_rebind.py -- "
            "SOURCE.glb CORRECTED.rig.json RESULT.glb RESULT.blend RESULT.rig.json REPORT.json"
        )
    paths = tuple(Path(value).expanduser().resolve() for value in args)
    suffixes = (".glb", ".rig.json", ".glb", ".blend", ".rig.json", ".json")
    for path, suffix in zip(paths, suffixes, strict=True):
        if not str(path).lower().endswith(suffix):
            raise SystemExit(f"{path.name} must end with {suffix}")
    if len(set(paths)) != len(paths):
        raise SystemExit("rebind input and output paths must be distinct")
    return paths


def _find_metarig(bpy, object_id: str):
    matches = [
        obj for obj in bpy.data.objects
        if obj.type == "ARMATURE" and obj.get(OBJECT_ID) == object_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metarig with persistent id {object_id}, found {len(matches)}")
    return matches[0]


def _bones_by_id(metarig) -> dict[str, Any]:
    result = {}
    for bone in metarig.data.bones:
        bone_id = bone.get(BONE_ID)
        if not bone_id:
            continue
        if bone_id in result:
            raise RuntimeError(f"duplicate persistent bone id {bone_id}")
        result[bone_id] = bone
    return result


def apply_corrections(bpy, metarig, corrections: list[JointCorrection]) -> list[dict[str, Any]]:
    bones_by_id = _bones_by_id(metarig)
    resolved = []
    for correction in corrections:
        if not correction.targets:
            raise RuntimeError(f"joint {correction.joint_id} has no Blender binding targets")
        for target in correction.targets:
            bone = bones_by_id.get(target.bone_id)
            if bone is None:
                raise RuntimeError(
                    f"joint {correction.joint_id} target {target.bone_id} is missing from metarig"
                )
            resolved.append((correction, target, bone.name))

    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    metarig.hide_set(False)
    metarig.hide_select = False
    metarig.select_set(True)
    bpy.context.view_layer.objects.active = metarig
    bpy.ops.object.mode_set(mode="EDIT")
    applied = []
    try:
        for correction, target, bone_name in resolved:
            edit_bone = metarig.data.edit_bones.get(bone_name)
            if edit_bone is None:
                raise RuntimeError(f"persistent target bone {bone_name} is unavailable in edit mode")
            setattr(edit_bone, target.endpoint, correction.target_position)
            applied.append({
                "joint": correction.joint_id,
                "boneId": target.bone_id,
                "boneName": bone_name,
                "expectedBoneName": target.bone_name,
                "endpoint": target.endpoint,
                "targetPosition": list(correction.target_position),
            })
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return applied


def regenerate_rigify(bpy, metarig):
    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.select_all(action="DESELECT")
    metarig.select_set(True)
    bpy.context.view_layer.objects.active = metarig
    bpy.ops.pose.rigify_generate()
    rig = getattr(metarig.data, "rigify_target_rig", None)
    if rig is None:
        rig = bpy.data.objects.get("rig")
    if rig is None or rig.type != "ARMATURE":
        raise RuntimeError("Rigify generation completed without a target armature")
    return rig


def _bound_meshes(bpy, rig) -> list[Any]:
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name.startswith("WGT-"):
            continue
        modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
        if obj.parent == rig or any(modifier.object == rig for modifier in modifiers):
            meshes.append(obj)
    if not meshes:
        raise RuntimeError("generated Rigify armature has no bound mesh to export")
    return meshes


def export_glb(bpy, rig, meshes, output_path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    rig.hide_set(False)
    rig.hide_select = False
    rig.select_set(True)
    for mesh in meshes:
        mesh.hide_set(False)
        mesh.hide_select = False
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
    )
    if not output_path.is_file():
        raise RuntimeError("Blender export completed without producing the result GLB")


def refreshed_sidecar(sidecar: dict, asset_path: Path, scene_path: Path) -> dict:
    refreshed = json.loads(json.dumps(sidecar))
    for joint_id, correction in refreshed["corrections"].items():
        refreshed["joints"][joint_id]["position"] = correction["targetPosition"]
    refreshed["corrections"] = {}
    refreshed["assetFingerprint"] = asset_fingerprint(asset_path)
    refreshed["binding"]["sceneFingerprint"] = asset_fingerprint(scene_path)
    return refreshed


def main() -> int:
    source_glb, sidecar_path, output_glb, output_blend, output_sidecar, report_path = parse_args(sys.argv)
    for source in (source_glb, sidecar_path):
        if not source.is_file():
            raise SystemExit(f"required input does not exist: {source}")
    for output in (output_glb, output_blend, output_sidecar, report_path):
        output.parent.mkdir(parents=True, exist_ok=True)

    import bpy

    opened_scene = Path(bpy.data.filepath).resolve()
    sidecar = load_sidecar(sidecar_path)
    verify_asset(sidecar, source_glb)
    verify_scene(sidecar, opened_scene)
    corrections = plan_corrections(sidecar)
    if not corrections:
        raise RuntimeError("sidecar contains no fit-joint corrections to rebind")

    metarig = _find_metarig(bpy, sidecar["binding"]["metarigObjectId"])
    print("I2L_STAGE::apply::Applying fit-joint corrections", flush=True)
    applied = apply_corrections(bpy, metarig, corrections)
    print("I2L_STAGE::rigify::Regenerating Rigify controls", flush=True)
    rig = regenerate_rigify(bpy, metarig)
    meshes = _bound_meshes(bpy, rig)
    print("I2L_STAGE::weights::Building voxel proxy and transferring weights", flush=True)
    weight_reports = [transfer_weights(bpy, mesh, rig) for mesh in meshes]
    print("I2L_STAGE::export::Saving scene and exporting GLB", flush=True)
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), check_existing=False)
    export_glb(bpy, rig, meshes, output_glb)

    refreshed = refreshed_sidecar(sidecar, output_glb, output_blend)
    output_sidecar.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n")
    report = {
        "schemaVersion": 1,
        "status": "done",
        "adapter": sidecar["binding"]["adapter"],
        "correctionsApplied": len(corrections),
        "targetsApplied": applied,
        "rig": rig.name,
        "meshes": [mesh.name for mesh in meshes],
        "weightStrategy": "voxel-proxy-transfer",
        "weights": weight_reports,
        "artifacts": {
            "glb": output_glb.name,
            "blend": output_blend.name,
            "sidecar": output_sidecar.name,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("I2L_REBIND::" + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
