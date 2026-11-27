"""Write a sampled slam attack to an armature.

Run (internal): bundled and sent to Blender by blender_attack_cycle.py.
"""

import json
import math


def run(arguments: dict) -> dict:
    import bpy

    poses = arguments["poses"]
    action_name = arguments["action_name"]
    toe_follow = arguments["toe"]

    bone_names = arguments.get("bone_names") or {}
    rig_name = arguments.get("rig_name")
    armatures = (
        o
        for o in bpy.data.objects
        if o.type == "ARMATURE" and (rig_name is None or o.name == rig_name)
    )
    arm = next(armatures)
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
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(poses)

    missing = []
    driven = set()
    for frame, pose in enumerate(poses, start=1):
        scene.frame_set(frame)
        for pb in arm.pose.bones:
            pb.rotation_euler = (0.0, 0.0, 0.0)
        for role, degrees in pose.items():
            bone = bone_names.get(role, role)
            pb = arm.pose.bones.get(bone)
            if pb is None:
                missing.append(bone)
                continue
            pb.rotation_euler = (math.radians(degrees), 0.0, 0.0)
            driven.add(bone)
        # The toes follow the paw at a fraction of its angle.
        for side in ("L", "R"):
            paw = pose.get("front" + side + "_paw", 0.0)
            toe_role = f"front{side}_toe"
            toe_name = bone_names.get(toe_role, toe_role)
            toe = arm.pose.bones.get(toe_name)
            if toe is not None:
                toe.rotation_euler = (math.radians(paw * toe_follow), 0.0, 0.0)
                driven.add(toe_name)
        for pb in arm.pose.bones:
            pb.keyframe_insert("rotation_euler", frame=frame)

    action = arm.animation_data.action
    action.name = action_name
    scene.frame_set(1)
    bpy.ops.object.mode_set(mode="OBJECT")

    summary = {
        "frames": len(poses),
        "action": action.name,
        "driven_bones": sorted(driven),
        "missing_bones": sorted(set(missing)),
    }
    print(json.dumps(summary, indent=2))
    return summary
