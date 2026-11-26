"""Verify short-clip retargeting through real GLB export and re-import."""

import sys

import bpy
import numpy as np


def check_short_retarget(workspace):
    import kimodo_retarget as retarget

    bpy.ops.wm.read_factory_settings(use_empty=True)
    data = bpy.data.armatures.new("Synthetic humanoid")
    rig = bpy.data.objects.new("Synthetic humanoid", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    def bone(name, parent, position):
        item = data.edit_bones.new(name)
        item.head = position
        item.tail = np.asarray(position) + (0, 0.02, 0)
        if parent:
            item.parent = data.edit_bones[parent]

    bone("hips", None, (0, 0, 1))
    for name, parent, height in (("spine1", "hips", 1.1), ("spine2", "spine1", 1.2),
                                 ("chest", "spine2", 1.35), ("neck", "chest", 1.5), ("head", "neck", 1.6)):
        bone(name, parent, (0, 0, height))
    for side, sign in (("L", -1), ("R", 1)):
        parent = "chest"
        for name, x in (("shoulder", .05), ("arm", .15), ("forearm", .4), ("hand", .65)):
            bone(side + name, parent, (sign * x, 0, 1.45))
            parent = side + name
        parent = "hips"
        for name, y, z in (("leg", 0, .95), ("shin", 0, .5), ("foot", 0, .08), ("toe", .12, .02)):
            bone(side + name, parent, (sign * .1, y, z))
            parent = side + name
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.primitive_cube_add(size=.1)
    mesh = bpy.context.object
    mesh.parent = rig
    mesh.vertex_groups.new(name="hips").add(list(range(8)), 1, "REPLACE")
    mesh.modifiers.new("Skin", "ARMATURE").object = rig
    source = workspace / "short-rig.glb"
    bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLB")

    rotations = np.broadcast_to(np.eye(3), (2, 77, 3, 3)).copy()
    rotations[1, retarget.SOMA_INDEX["Head"]] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    joints = np.zeros((2, 77, 3))
    joints[:, 0, 1] = .9
    joints[1, 0, 2] = .3
    motion = workspace / "two-poses.npz"
    np.savez(motion, global_rot_mats=rotations, posed_joints=joints)
    output = workspace / "short-retargeted.glb"
    previous_argv = sys.argv
    try:
        sys.argv = ["kimodo_retarget", "--rig", str(source), "--motion", str(motion),
                    "--out", str(output), "--loop", "--no-ground", "--no-align", "--no-clearance"]
        retarget.main()
    finally:
        sys.argv = previous_argv
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(output))
    imported = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    action = imported.animation_data.action
    first, last = action.frame_range
    assert last > first, "Short fallback exported a zero-duration animation"
    bpy.context.scene.frame_set(int(first))
    start = imported.pose.bones["head"].matrix.copy()
    bpy.context.scene.frame_set(int(last), subframe=last - int(last))
    end = imported.pose.bones["head"].matrix.copy()
    assert not np.allclose(start, end), "Short fallback lost its distinct final pose"
    assert abs((last - first) / bpy.context.scene.render.fps - 1 / 30) < 1e-5
    print("PASS retargeted two distinct poses survived GLB export and re-import")
    return {"distinct_poses": 2, "duration_seconds": 1 / 30}
