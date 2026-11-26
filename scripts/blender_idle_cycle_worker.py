"""Write idle motion and interpolation to an armature.

Run (internal): bundled and sent to Blender by blender_idle_cycle.py.
"""

import json
import math


def _fcurves(act):
    """Blender 4.4 moved F-curves into layered action slots."""
    direct = getattr(act, "fcurves", None)
    if direct is not None:
        return list(direct)
    found = []
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                found.extend(bag.fcurves)
    return found


def run(arguments: dict) -> dict:
    import bpy
    from mathutils import Matrix

    frames = arguments["frames"]
    breath_angle = math.radians(arguments["breath"])
    head_angle = math.radians(arguments["head"])
    ear_angle = math.radians(arguments["ear"])
    tail_angle = math.radians(arguments["tail"])
    flick_angle = math.radians(arguments["flick"])
    flick_at = arguments["flick_at"]
    head_yaw = math.radians(arguments["head_yaw"])
    shift = arguments["shift"]
    ear_turn_angle = math.radians(arguments["ear_turn"])
    turn_at = arguments["ear_turn_at"]
    turn_hold = arguments["ear_turn_hold"]

    armatures = (
        o
        for o in bpy.data.objects
        if o.type == "ARMATURE"
    )
    arm = next(armatures, None)
    if arm is None:
        raise RuntimeError("no armature in the scene — build the rig first")
    arm.hide_viewport = False

    if arm.animation_data and arm.animation_data.action:
        arm.animation_data.action = None
    arm.animation_data_clear()

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
    arm.location = (0.0, 0.0, 0.0)

    missing = []

    def put(role, axis, value, frame):
        name = role
        pb = arm.pose.bones.get(name)
        if pb is None:
            missing.append(name)
            return
        setattr(pb.rotation_euler, axis, value)
        pb.keyframe_insert("rotation_euler", frame=frame)

    for frame in range(1, frames + 2):
        t = (frame - 1) / frames
        bpy.context.scene.frame_set(frame)
        breath = math.sin(2.0 * math.pi * t)

        # Ribcage leads, belly follows a beat later — the lag is what makes it breathing
        # rather than a uniform swell.
        put("spine_02", "x", -breath_angle * breath, frame)
        put(
            "spine_01",
            "x",
            breath_angle * 0.45 * math.sin(2.0 * math.pi * t - 0.55),
            frame,
        )

        # The head rides the breath, lagging further still, plus a slow independent drift so
        # it never returns to exactly the same place.
        head_rot = head_angle * math.sin(
            2.0 * math.pi * t - 0.9
        ) + head_angle * 0.35 * math.sin(4.0 * math.pi * t + 1.7)
        put("head", "x", head_rot, frame)
        put("jaw", "x", head_angle * 0.15 * math.sin(2.0 * math.pi * t - 1.2), frame)

        # The neck carries the static yaw correction for a head generated turned; applied as
        # a world rotation expressed in bone space, since a bone's axes run along the bone.
        neck = arm.pose.bones.get("neck")
        if neck is not None:
            rest = neck.bone.matrix_local.to_3x3()
            world = Matrix.Rotation(head_yaw, 3, "Z")
            e = (rest.inverted() @ world @ rest).to_euler("XYZ")
            neck.rotation_euler = (
                e.x + head_angle * 0.4 * math.sin(2.0 * math.pi * t - 0.7),
                e.y,
                e.z,
            )
            neck.keyframe_insert("rotation_euler", frame=frame)

        # One sharp ear flick per loop, decaying fast, deliberately on one ear only. A purely
        # sinusoidal idle reads as dead after two cycles; the eye catches the irregularity.
        phase = (t - flick_at) % 1.0
        flick = (
            math.exp(-phase * 26.0) * math.sin(phase * 58.0) if phase < 0.35 else 0.0
        )
        put(
            "ear_L",
            "x",
            ear_angle * math.sin(2.0 * math.pi * t - 0.4) + flick_angle * flick,
            frame,
        )

        # One ear swivels back to track a sound behind the character, then returns. Independent
        # ear direction is the single most characteristically alert thing a character can make, and
        # because only one ear moves it reads as attention rather than as a pose change.
        # The turn is a world-up rotation expressed in bone space, so the ear pivots at its
        # base instead of shearing; amplitude is kept well inside anatomical range.
        turn_phase = (t - turn_at) % 1.0
        if turn_phase < turn_hold:
            progress = turn_phase / turn_hold
            if progress < 0.3:
                env = min(1.0, progress / 0.3)
            elif progress < 0.7:
                env = 1.0
            else:
                env = max(0.0, (1.0 - progress) / 0.3)
        else:
            env = 0.0
        ear_r_name = "ear_R"
        ear_r = arm.pose.bones.get(ear_r_name)
        if ear_r is not None:
            base = (
                ear_angle * math.sin(2.0 * math.pi * t - 0.15)
                + flick_angle * 0.25 * flick
            )
            rest_r = ear_r.bone.matrix_local.to_3x3()
            swivel = Matrix.Rotation(ear_turn_angle * env, 3, "Z")
            er = (rest_r.inverted() @ swivel @ rest_r).to_euler("XYZ")
            ear_r.rotation_euler = (er.x + base, er.y, er.z)
            ear_r.keyframe_insert("rotation_euler", frame=frame)
        else:
            missing.append(ear_r_name)

        # A lazy tail, slower than the breath and drifting sideways as well, so it never
        # traces the same arc twice within the loop.
        put("tail_01", "z", tail_angle * math.sin(2.0 * math.pi * t - 1.1), frame)
        put("tail_02", "z", tail_angle * 0.8 * math.sin(2.0 * math.pi * t - 1.9), frame)
        put("tail_01", "x", tail_angle * 0.3 * math.sin(4.0 * math.pi * t), frame)

        # A weight shift far smaller than a breath: enough to avoid looking frozen, small
        # enough that the planted feet do not visibly slide.
        arm.location.z = shift * breath
        arm.keyframe_insert("location", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")

    action = arm.animation_data.action

    curves = _fcurves(action)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"

    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = frames
    scn.frame_set(1)

    # hide_render, never hide_viewport: the latter drops the armature from the viewport
    # dependency graph and silently freezes the mesh while renders keep animating.
    for o in bpy.data.objects:
        if o.name.startswith("JOINT_"):
            o.hide_render = True
            o.hide_set(True)
    arm.hide_render = True
    arm.hide_set(True)

    summary = {
        "frames": frames,
        "fcurves": len(curves),
        "keyframes": sum(len(fc.keyframe_points) for fc in curves),
        "missing_bones": sorted(set(missing)),
    }
    print(json.dumps(summary, indent=2))
    return summary
