"""Append and place a rigged character in the live Blender scene.

Run (internal): bundled and sent to Blender by blender_import_character.py.
"""

from __future__ import annotations

import json
import math


def sk_to_blender(x, y, z):
    """Rotate SceneKit's Y-up coordinates into Blender's Z-up coordinates."""
    return x, -z, y


def import_character(
    bpy,
    *,
    path,
    prefix,
    armature_name,
    mesh_name,
    position,
    facing,
    action_name,
    frame,
    do_save,
):
    from mathutils import Vector

    before = set(bpy.data.objects.keys())
    requested_actions = [action_name] if action_name else []
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        data_to.objects = [
            name for name in data_from.objects if name in (armature_name, mesh_name)
        ]
        data_to.actions = [
            name for name in data_from.actions if name in requested_actions
        ]

    imported = [obj for obj in data_to.objects if obj is not None]
    collection = bpy.context.collection
    for obj in imported:
        collection.objects.link(obj)

    armature = next((obj for obj in imported if obj.type == "ARMATURE"), None)
    mesh = next((obj for obj in imported if obj.type == "MESH"), None)
    if armature is None or mesh is None:
        raise RuntimeError(
            f"expected an armature named {armature_name!r} and a mesh named {mesh_name!r} "
            f"in {path!r}, found {[obj.name for obj in imported]}"
        )

    armature.name = f"{prefix}_rig"
    mesh.name = f"{prefix}_mesh"

    # Retain Rigify widgets, hide them from render, and remove other imported extras.
    new_names = set(bpy.data.objects.keys()) - before - {armature.name, mesh.name}
    removed = []
    widgets_hidden = 0
    for name in new_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if name.startswith("WGT-"):
            obj.hide_render = True
            widgets_hidden += 1
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed.append(name)

    root = bpy.data.objects.new(f"{prefix}_root", None)
    root.empty_display_size = 0.5
    collection.objects.link(root)
    for obj in (armature, mesh):
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()

    root.location = sk_to_blender(*position)
    if facing is not None:
        target = Vector(sk_to_blender(*facing))
        to_target = target - Vector(root.location)
        to_target.z = 0
        if to_target.length > 0.0001:
            root.rotation_euler = (0, 0, math.atan2(to_target.x, to_target.y))

    if action_name:
        if armature.animation_data is None:
            armature.animation_data_create()
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"action {action_name!r} was not found in {path!r}")
        armature.animation_data.action = action

    if frame is not None:
        bpy.context.scene.frame_set(frame)
    if do_save:
        bpy.ops.wm.save_mainfile()

    return {
        "root": root.name,
        "armature": armature.name,
        "mesh": mesh.name,
        "location": list(root.location),
        "removed_strays": removed,
        "widgets_hidden": widgets_hidden,
        "saved": do_save,
    }


def run(arguments):
    import bpy

    result = import_character(bpy, **arguments)
    print("IMPORT_RESULT " + json.dumps(result))
    return result
