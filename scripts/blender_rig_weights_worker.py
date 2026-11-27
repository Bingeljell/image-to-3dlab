"""Build a quadruped rig and transfer weights from a voxel proxy in Blender.

Run (internal): bundled by blender_build_rig.py and blender_voxel_weights.py.
"""

from __future__ import annotations

import json

if __package__:
    from .blender_scene import (
        private_scene_collections,
        remove_scene_objects,
        scene_marker_objects,
    )
elif __name__ != "_assetfurnace_worker":
    from blender_scene import (
        private_scene_collections,
        remove_scene_objects,
        scene_marker_objects,
    )

# (bone, head marker, tail marker, parent bone, connected)
#
# The foot is two segments, not one. A single wrist->paw bone pivots the whole foot as a
# peg, so lifting it raises the toe instead of dropping it. With the paw joint at the
# *heel* and a toe segment ahead of it, the foot rolls the way a foot does: heel plants,
# foot flattens, toes push off.
SKELETON: list[tuple[str, str, str, str | None, bool]] = [
    ("spine_01", "pelvis", "spine_mid", None, False),
    ("spine_02", "spine_mid", "chest", "spine_01", True),
    ("neck", "chest", "neck_base", "spine_02", True),
    ("head", "neck_base", "head", "neck", True),
    ("jaw", "head", "jaw", "head", False),
    ("tail_01", "tail_base", "tail_mid", "spine_01", False),
    ("tail_02", "tail_mid", "tail_tip", "tail_01", True),
]
for _side in ("L", "R"):
    SKELETON += [
        (f"front{_side}_upperarm", f"front{_side}_shoulder", f"front{_side}_elbow", "spine_02", False),
        (f"front{_side}_forearm", f"front{_side}_elbow", f"front{_side}_wrist", f"front{_side}_upperarm", True),
        (f"front{_side}_paw", f"front{_side}_wrist", f"front{_side}_paw", f"front{_side}_forearm", True),
        (f"front{_side}_toe", f"front{_side}_paw", f"front{_side}_toe", f"front{_side}_paw", True),
        (f"back{_side}_thigh", f"back{_side}_hip", f"back{_side}_stifle", "spine_01", False),
        (f"back{_side}_shin", f"back{_side}_stifle", f"back{_side}_ankle", f"back{_side}_thigh", True),
        (f"back{_side}_paw", f"back{_side}_ankle", f"back{_side}_paw", f"back{_side}_shin", True),
        (f"back{_side}_toe", f"back{_side}_paw", f"back{_side}_toe", f"back{_side}_paw", True),
    ]

# Ears have only a base marker, so their bone is extrapolated outward from the head
# along the head->ear direction; that makes the bone lie along the ear and pivot at its
# base, which is how an ear actually moves.
EAR_BONES = [("ear_L", "ear_L"), ("ear_R", "ear_R")]


def run(arguments: dict[str, object], *, bpy=None, vector=None) -> dict:
    """Build a rig or transfer its weights in the active Blender scene."""
    if bpy is None:
        import bpy

    args = dict(arguments)
    operation = args.pop("operation")
    if operation == "build":
        if vector is None:
            from mathutils import Vector as vector
        return build_rig(bpy, vector, **args)
    if operation == "voxel":
        return voxel_weights(bpy, **args)
    raise ValueError(f"unknown rig operation: {operation!r}")


def ear_tip(base, head, length: float, vector):
    """Project an ear from the head towards its base marker."""
    direction = base - head
    direction = direction.normalized() if direction.length > 1e-6 else vector((0, 0, 1))
    return base + direction * length


def weight_report(vertices, groups, names, head_z: float, weight_threshold: float) -> dict:
    """Summarise accepted bone weights and flag influences that reach the head."""
    zs = [vertex.co.z for vertex in vertices]
    span = max(zs) - min(zs)
    influence = {}
    for vertex in vertices:
        for group in vertex.groups:
            if group.weight >= weight_threshold:
                name = groups.get(group.group)
                if name:
                    record = influence.setdefault(name, [0, -1e9])
                    record[0] += 1
                    record[1] = max(record[1], vertex.co.z)
    report = {}
    for name in names:
        record = influence.get(name)
        report[name] = {
            "verts": record[0] if record else 0,
            "max_z": round(record[1], 4) if record else None,
            "reaches_head": bool(record and record[1] > head_z - span * 0.05),
        }
    return report


def flagged_groups(report: dict) -> tuple[list[str], list[str]]:
    legs = [name for name in report if "front" in name or "back" in name]
    contaminated = sorted(name for name in legs if report[name]["reaches_head"])
    empty = sorted(name for name in report if report[name]["verts"] == 0)
    return contaminated, empty


def _build_bones(bpy, Vector, arm, markers, ear_length):
    arm_data = arm.data
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        eb = arm_data.edit_bones
        for name, head_m, tail_m, parent, connect in SKELETON:
            bone = eb.new(name)
            bone.head = markers[head_m]
            bone.tail = markers[tail_m]
            if (bone.tail - bone.head).length < 1e-5:
                bone.tail = bone.head + Vector((0, 0, 1e-3))
        head_c = markers["head"]
        for name, marker in EAR_BONES:
            bone = eb.new(name)
            base = markers[marker]
            bone.head = base
            bone.tail = ear_tip(base, head_c, ear_length, Vector)
        for name, _h, _t, parent, connect in SKELETON:
            if parent:
                eb[name].parent = eb[parent]
                eb[name].use_connect = connect
        for name, _m in EAR_BONES:
            eb[name].parent = eb["head"]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def build_rig(bpy, Vector, *, ear_length: float, weight_threshold: float) -> dict:
    scene = bpy.context.scene
    markers = {name: obj.matrix_world.translation.copy()
               for name, obj in scene_marker_objects(bpy, scene, "JOINT_").items()}
    missing = sorted(({m for b in SKELETON for m in (b[1], b[2])}
                      | {m for _, m in EAR_BONES}) - set(markers))
    if missing:
        raise RuntimeError("missing markers: " + ", ".join(missing))

    mesh_objs = [o for o in scene.objects if o.type == "MESH"]
    if len(mesh_objs) != 1:
        raise RuntimeError(f"expected exactly one mesh, found {len(mesh_objs)}")
    mesh = mesh_objs[0]
    local_collections = private_scene_collections(bpy, scene)
    if set(mesh.users_collection) - local_collections:
        raise RuntimeError("the mesh is shared with another scene")

    old_modifiers = [mod for mod in mesh.modifiers if mod.type == "ARMATURE"]
    old_rigs = {mod.object for mod in old_modifiers if mod.object is not None}
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        old_rigs.add(mesh.parent)
    existing_binding = bool(old_rigs or old_modifiers or mesh.vertex_groups)

    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    arm_data = bpy.data.armatures.new("QuadRig")
    arm = bpy.data.objects.new("QuadRig", arm_data)
    candidate = mesh.copy()
    candidate.data = mesh.data.copy()
    candidate_data = candidate.data
    applied = False
    try:
        scene.collection.objects.link(arm)
        scene.collection.objects.link(candidate)
        world = mesh.matrix_world.copy()
        candidate.parent = None
        candidate.matrix_world = world
        candidate.hide_select = False
        for group in list(candidate.vertex_groups):
            candidate.vertex_groups.remove(group)
        for mod in list(candidate.modifiers):
            if mod.type == "ARMATURE":
                candidate.modifiers.remove(mod)

        _build_bones(bpy, Vector, arm, markers, ear_length)
        bpy.ops.object.select_all(action="DESELECT")
        candidate.select_set(True)
        arm.select_set(True)
        view_layer.objects.active = arm
        heat_error = None
        try:
            if "FINISHED" not in bpy.ops.object.parent_set(type="ARMATURE_AUTO"):
                heat_error = "automatic weighting was cancelled"
        except RuntimeError as exc:
            heat_error = str(exc)

        groups = {group.index: group.name for group in candidate.vertex_groups}
        names = [bone[0] for bone in SKELETON] + [ear[0] for ear in EAR_BONES]
        report = weight_report(candidate.data.vertices, groups, names, markers["head"].z, weight_threshold)
        contaminated, empty = flagged_groups(report)
        result = {
            "bones": len(arm_data.bones),
            "heat_error": heat_error,
            "vertex_groups": len(candidate.vertex_groups),
            "empty_groups": empty,
            "legs_reaching_head": contaminated,
            "verdict": "ENVELOPE FALLBACK" if contaminated else ("EMPTY GROUPS" if empty else "CLEAN"),
            "report": report,
        }

        if existing_binding and (heat_error is not None or empty or contaminated):
            print("!!! RIG REPLACEMENT FAILED. Previous binding kept.")
        else:
            mesh.data = candidate_data
            mesh.hide_select = False
            mesh.parent = arm
            mesh.matrix_parent_inverse = candidate.matrix_parent_inverse
            mesh.matrix_world = world
            for mod in old_modifiers:
                mesh.modifiers.remove(mod)
            binding = next(
                (mod for mod in candidate.modifiers if mod.type == "ARMATURE" and mod.object == arm),
                None,
            )
            mesh.modifiers.new(binding.name if binding else "Armature", "ARMATURE").object = arm
            mesh.select_set(True)
            applied = True

            # A prior rig can still drive another mesh or be linked into another scene.
            for old in old_rigs:
                if set(old.users_scene) != {scene}:
                    continue
                users = bpy.data.user_map(subset={old})[old]
                if users <= local_collections | {scene}:
                    remove_scene_objects(bpy, [old])
            arm.name = "QuadRig"
            if empty:
                print("!!! MESH IS NOT USABLY WEIGHTED — run scripts/blender_voxel_weights.py now")
    finally:
        bpy.data.objects.remove(candidate, do_unlink=True)
        if not applied:
            bpy.data.meshes.remove(candidate_data)
            bpy.data.objects.remove(arm, do_unlink=True)
            bpy.data.armatures.remove(arm_data)
            bpy.ops.object.select_all(action="DESELECT")
            for selected in previous_selected:
                selected.select_set(True)
            view_layer.objects.active = previous_active

    result["applied"] = applied
    print(json.dumps(result, indent=2))
    return result


def voxel_weights(bpy, *, voxel_size: float, weight_threshold: float, keep_proxy: bool) -> dict:
    scene = bpy.context.scene
    mesh_objs = [o for o in scene.objects if o.type == "MESH" and not o.name.startswith("PROXY_")]
    arms = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(mesh_objs) == 1:
        mesh = mesh_objs[0]
        if mesh.parent in arms:
            arms = [mesh.parent]
        else:
            bindings = {
                mod.object for mod in mesh.modifiers
                if mod.type == "ARMATURE" and mod.object is not None
            }
            if bindings:
                arms = [arm for arm in arms if arm in bindings]
    if len(mesh_objs) != 1 or len(arms) != 1:
        raise RuntimeError(f"need exactly one mesh and one armature, found {len(mesh_objs)} and {len(arms)}")
    mesh, arm = mesh_objs[0], arms[0]
    mesh.hide_select = False

    remove_scene_objects(
        bpy, [o for o in scene.objects if o.type == "MESH" and o.name.startswith("PROXY_")]
    )

    # --- 1. voxel proxy -------------------------------------------------------------
    proxy = mesh.copy()
    proxy.data = mesh.data.copy()
    proxy.name = "PROXY_weights"
    proxy.parent = None
    bpy.context.scene.collection.objects.link(proxy)
    for mod in list(proxy.modifiers):
        proxy.modifiers.remove(mod)
    for vg in list(proxy.vertex_groups):
        proxy.vertex_groups.remove(vg)

    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    rem = proxy.modifiers.new("Remesh", "REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=rem.name)
    proxy_faces = len(proxy.data.polygons)

    # --- 2. heat-weight the proxy ---------------------------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    proxy_error = None
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        proxy_error = str(exc)

    proxy_nonempty = 0
    for vg in proxy.vertex_groups:
        for v in proxy.data.vertices:
            if any(g.group == vg.index and g.weight >= weight_threshold for g in v.groups):
                proxy_nonempty += 1
                break

    # --- 3. transfer weights proxy -> real mesh -------------------------------------
    for vg in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(vg)
    for mod in list(mesh.modifiers):
        if mod.type in {"ARMATURE", "DATA_TRANSFER"}:
            mesh.modifiers.remove(mod)
    mesh.parent = None

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    dt = mesh.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = proxy
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    bpy.ops.object.datalayout_transfer(modifier=dt.name)
    bpy.ops.object.modifier_apply(modifier=dt.name)

    # --- 4. bind the real mesh with the transferred groups --------------------------
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_NAME")

    if not keep_proxy:
        bpy.data.objects.remove(proxy, do_unlink=True)

    # --- 5. verification ------------------------------------------------------------
    head = scene_marker_objects(bpy, scene, "JOINT_").get("head")
    groups = {vg.index: vg.name for vg in mesh.vertex_groups}
    head_z = head.matrix_world.translation.z if head else max(v.co.z for v in mesh.data.vertices)
    report = weight_report(mesh.data.vertices, groups, groups.values(), head_z, weight_threshold)
    contaminated, empty = flagged_groups(report)
    unweighted = sum(1 for v in mesh.data.vertices
                     if not any(g.weight >= weight_threshold for g in v.groups))

    result = {
        "proxy_faces": proxy_faces,
        "proxy_heat_error": proxy_error,
        "proxy_groups_with_weight": proxy_nonempty,
        "mesh_vertex_groups": len(mesh.vertex_groups),
        "empty_groups": empty,
        "legs_reaching_head": contaminated,
        "unweighted_vertices": unweighted,
        "total_vertices": len(mesh.data.vertices),
        "verdict": ("ENVELOPE-LIKE BLEED" if contaminated
                    else "EMPTY GROUPS" if empty
                    else "CLEAN"),
        "report": report,
    }
    print(json.dumps(result, indent=2))
    return result
