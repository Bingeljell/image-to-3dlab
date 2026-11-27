"""Write a sampled gait to a Rigify armature.

Run (internal): bundled and sent to Blender by blender_rigify_walk_cycle.py.
"""

import json
import math


def run(arguments: dict) -> dict:
    import bpy

    poses = arguments["poses"]
    action_name = arguments["action_name"]

    rig_name = arguments.get("rig_name", "rig")
    bone_names = arguments.get("bone_names") or {}
    arm = bpy.data.objects.get(rig_name)
    if arm is None:
        raise RuntimeError(f"no object named {rig_name!r} in the scene")

    ik_fk_bones = [
        "front_thigh_parent.L",
        "front_thigh_parent.R",
        "thigh_parent.L",
        "thigh_parent.R",
    ]
    not_fk = [
        b
        for b in (bone_names.get(role, role) for role in ik_fk_bones)
        if arm.pose.bones.get(b) and arm.pose.bones[b].get("IK_FK", 1.0) < 0.99
    ]
    if not_fk:
        print(
            json.dumps(
                {
                    "warning": "these legs are not in FK mode, rotations will be ignored by the deform bones",
                    "bones": not_fk,
                }
            )
        )

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")

    # Only retire this rig's previous take. Other rigs, NLA strips and fake
    # users keep their actions; Blender assigns a unique name on collision.
    previous = arm.animation_data.action if arm.animation_data else None
    if previous is not None:
        arm.animation_data.action = None
        base, separator, suffix = previous.name.rpartition(".")
        same_take = previous.name == action_name or (
            separator and base == action_name and suffix.isdecimal()
        )
        if same_take and previous.users == 0:
            bpy.data.actions.remove(previous)

    for pb in arm.pose.bones:
        if pb.rotation_mode != "XYZ":
            pb.rotation_mode = "XYZ"

    driven_names = set()
    for pose in poses:
        driven_names.update(pose.keys())

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(poses)

    missing = []
    for frame, pose in enumerate(poses, start=1):
        scene.frame_set(frame)
        for role in driven_names:
            name = bone_names.get(role, role)
            pb = arm.pose.bones.get(name)
            if pb is None:
                missing.append(name)
                continue
            deg = pose.get(role, 0.0)
            pb.rotation_euler = (math.radians(deg), 0.0, 0.0)
            pb.keyframe_insert("rotation_euler", frame=frame)

    action = arm.animation_data.action
    action.name = action_name
    scene.frame_set(1)
    bpy.ops.object.mode_set(mode="OBJECT")

    summary = {
        "frames": len(poses),
        "action": action.name,
        "driven_bones": sorted(bone_names.get(role, role) for role in driven_names),
        "missing_bones": sorted(set(missing)),
    }
    print(json.dumps(summary, indent=2))
    return summary
