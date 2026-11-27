"""Write a looping quadruped gait and keep the paws on the rest floor.

Run (internal): bundled and sent to Blender by blender_walk_cycle.py.
"""

import json
import math

# WALK is a four-beat lateral gait: each foot lands a quarter-cycle after the last,
# so at least two feet are always down. TROT is a two-beat diagonal gait: opposite
# corners swing together and the body vaults over each pair in turn, which is where
# the trot's characteristic bounce comes from.
PHASES = {
    "walk": {"backL": 0.0, "frontL": 0.25, "backR": 0.5, "frontR": 0.75},
    "trot": {"frontL": 0.0, "backR": 0.0, "frontR": 0.5, "backL": 0.5},
}


def _fcurves(act):
    """Blender 4.4 moved F-curves into layered action slots; 4.3 and earlier expose
    them directly on the action."""
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
    """Key the gait, optionally remapping standard bone and paw marker roles."""
    import bpy
    from mathutils import Matrix

    frames = arguments["frames"]
    swing_front = math.radians(arguments["swing_front"])
    swing_back = math.radians(arguments["swing_back"])
    bend_front = math.radians(arguments["bend_front"])
    bend_back = math.radians(arguments["bend_back"])
    wrist = math.radians(arguments["wrist"])
    ankle = math.radians(arguments["ankle"])
    base_bend = math.radians(arguments["base_bend"])
    bob = arguments["bob"]
    sway = math.radians(arguments["sway"])
    head_yaw = math.radians(arguments["head_yaw"])
    head_pitch = math.radians(arguments["head_pitch"])
    head_level = math.radians(arguments["head_level"])
    chest_drop = math.radians(arguments["chest_drop"])
    shoulder_tuck = math.radians(arguments["shoulder_tuck"])
    hind_lead_l = math.radians(arguments["hind_lead_l"])
    hind_lead_r = math.radians(arguments["hind_lead_r"])
    body_drop = arguments["body_drop"]
    front_crouch = math.radians(arguments["front_crouch"])
    crouch_angle = math.radians(arguments["crouch"])
    front_fold_sign = arguments["front_fold_sign"]
    toe = math.radians(arguments["toe"])
    auto_plant = arguments["auto_plant"]
    bone_names = arguments.get("bone_names", {})
    marker_names = arguments.get("marker_names", {})

    rig_name = arguments.get("rig_name")
    armatures = (
        o
        for o in bpy.data.objects
        if o.type == "ARMATURE" and (rig_name is None or o.name == rig_name)
    )
    arm = next(armatures, None)
    if arm is None:
        raise RuntimeError("no armature in the scene — build the rig first")
    mesh_name = arguments.get("mesh_name")
    if mesh_name is None:
        mesh = None
        for o in bpy.data.objects:
            if o.type == "MESH" and not o.name.startswith("PROXY_"):
                mesh = o
    else:
        mesh = bpy.data.objects.get(mesh_name)
        if mesh is None or mesh.type != "MESH":
            raise RuntimeError(f"no mesh named {mesh_name!r} in the scene")

    arm.hide_viewport = False
    arm.hide_render = False

    phases = PHASES[arguments["gait"]]

    # (upper bone, middle bone, lower bone, swing, middle fold, lower fold).
    # Rotating a bone pivots it about its head, so the middle bone folds the elbow/knee
    # and the lower bone folds the wrist/ankle. Splitting the fold between the two is
    # what stops the whole lower limb hinging up as one rigid piece.
    chain = {
        "frontL": (
            "frontL_upperarm",
            "frontL_forearm",
            "frontL_paw",
            swing_front,
            bend_front,
            wrist,
        ),
        "frontR": (
            "frontR_upperarm",
            "frontR_forearm",
            "frontR_paw",
            swing_front,
            bend_front,
            wrist,
        ),
        "backL": (
            "backL_thigh",
            "backL_shin",
            "backL_paw",
            swing_back,
            bend_back,
            ankle,
        ),
        "backR": (
            "backR_thigh",
            "backR_shin",
            "backR_paw",
            swing_back,
            bend_back,
            ankle,
        ),
    }

    if arm.animation_data and arm.animation_data.action:
        arm.animation_data.action = None
    arm.animation_data_clear()

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)

    # The rig's rest position is the origin. Reading arm.location.z here instead would
    # pick up whatever the *previous* run's keyframes left behind, so each regeneration
    # would treat the already-lowered position as its new baseline and sink the character
    # further every time.
    arm.location = (0.0, 0.0, 0.0)
    base_z = 0.0
    missing = []

    # Probe vertices at each paw, and the floor height they rest at unposed. Crouching
    # folds the legs, which lifts the paws; re-planting them on this same floor is what
    # turns "legs fold" into "chest comes down while the feet stay where they are".
    paw_probe = []
    rest_floor = 0.0
    if mesh is not None:
        rest_co = [v.co.copy() for v in mesh.data.vertices]
        for role in ("frontL_paw", "frontR_paw", "backL_paw", "backR_paw"):
            marker = bpy.data.objects.get(marker_names.get(role, f"JOINT_{role}"))
            if marker is not None:
                target = marker.matrix_world.translation
                paw_probe.append(
                    min(
                        range(len(rest_co)),
                        key=lambda i: (rest_co[i] - target).length_squared,
                    )
                )
        if paw_probe:
            rest_floor = min(rest_co[i].z for i in paw_probe)

    for frame in range(1, frames + 2):
        t = (frame - 1) / frames
        bpy.context.scene.frame_set(frame)

        for leg, phase in phases.items():
            upper, mid, low, swing_amp, mid_amp, low_amp = chain[leg]
            theta = 2.0 * math.pi * (t + phase)
            swing = swing_amp * math.cos(theta)
            is_front = leg.startswith("front")
            if is_front:
                # The front legs hang off the chest, so pitching the chest down carries
                # them backward with it and the character reads as flying with its arms
                # trailing. Counter-rotate by the same amount to keep them vertical,
                # then apply the tuck on top of that corrected zero.
                swing += shoulder_tuck - chest_drop
            else:
                # swing_amp*cos(theta) is symmetric, so a bigger forward reach always buys
                # an equally bigger backward kick — there is no way to get one without the
                # other from amplitude alone. hind_lead_l/R is a static bias added after the
                # cosine: it shifts one side's swing range forward without changing its span,
                # so the forward peak can stay put while the rearward extreme shrinks. Split
                # by side, not shared, because hand-placed markers are rarely symmetric —
                # one leg needing this fix doesn't mean the other one does.
                swing += hind_lead_l if leg.endswith("L") else hind_lead_r
            # Rectified sine, shaped: the exponent widens the airborne plateau so the
            # fold eases in and out instead of snapping at the contact frames.
            lift = max(0.0, math.sin(theta)) ** 0.7
            # A real leg is never locked straight, even bearing weight; base_bend keeps a
            # standing bend so the limb reads as a limb rather than a stick.
            # Front legs hang at full extension once compensated for the chest drop,
            # while back legs carry a permanent fold — so without a matching front
            # crouch the front paws sink below the back ones and the animal stands on
            # two different floors.
            leg_crouch = (
                front_crouch + crouch_angle if is_front else crouch_angle * 0.55
            )
            # Which rotation sign *closes* a joint depends on the rest geometry, and the
            # front leg's differs from the back's once the elbow sits behind the shoulder.
            # Applying the wrong sign straightens the limb instead of folding it.
            sign = front_fold_sign if is_front else 1.0
            mid_fold = sign * (base_bend + leg_crouch + mid_amp * lift)
            # The wrist/ankle folds slightly later than the elbow/knee — the joint
            # closest to the ground is the last to leave it and the first to reach for it.
            low_lift = max(0.0, math.sin(theta - 0.5)) ** 0.7
            low_fold = sign * (base_bend * 0.5 + leg_crouch * 0.6 + low_amp * low_lift)
            # Toes flex late in stance, as the foot rolls forward and pushes off, then
            # relax while airborne. That roll is the whole reason the foot is two segments.
            push = max(0.0, math.sin(theta + 2.2)) ** 1.4
            toe_fold = sign * toe * push
            for role, value in (
                (upper, swing),
                (mid, mid_fold),
                (low, low_fold),
                (leg + "_toe", toe_fold),
            ):
                name = bone_names.get(role, role)
                pb = arm.pose.bones.get(name)
                if pb is None:
                    missing.append(name)
                    continue
                pb.rotation_euler.x = value
                pb.keyframe_insert("rotation_euler", frame=frame)

        # Hips rise twice per stride, as each diagonal support pair passes under the body.
        plant = 0.0
        if auto_plant and paw_probe and mesh is not None:
            arm.location.z = base_z
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            ev = mesh.evaluated_get(dg)
            plant = rest_floor - min(ev.data.vertices[i].co.z for i in paw_probe)
        arm.location.z = base_z + plant - body_drop + bob * math.cos(4.0 * math.pi * t)
        arm.keyframe_insert("location", frame=frame)

        # The subject was generated with its head turned, so a static counter-yaw is
        # baked into every frame. It is applied about the *world* up axis and then
        # expressed in the bone's own space — a bone's local axes run along the bone,
        # so setting rotation_euler.z directly would tilt the head rather than yaw it.
        # head_pitch additionally drops the head forward and down. A moving quadruped carries
        # its head low and reaching ahead; an upright head reads as a stationary animal
        # whose legs happen to be moving, which is most of the uncanniness.
        if abs(head_yaw) > 1e-6 or abs(head_pitch) > 1e-6:
            neck = arm.pose.bones.get(bone_names.get("neck", "neck"))
            if neck is not None:
                rest = neck.bone.matrix_local.to_3x3()
                world = Matrix.Rotation(head_pitch, 3, "X") @ Matrix.Rotation(
                    head_yaw, 3, "Z"
                )
                neck.rotation_euler = (rest.inverted() @ world @ rest).to_euler("XYZ")
                neck.keyframe_insert("rotation_euler", frame=frame)

        # Pitching the chest down is what actually lowers the shoulders, and the neck
        # and head ride down with it. Rotating the neck alone only pivots the muzzle.
        if abs(chest_drop) > 1e-6:
            chest = arm.pose.bones.get(bone_names.get("spine_02", "spine_02"))
            if chest is not None:
                crest = chest.bone.matrix_local.to_3x3()
                cworld = Matrix.Rotation(chest_drop, 3, "X")
                chest.rotation_euler = (crest.inverted() @ cworld @ crest).to_euler(
                    "XYZ"
                )
                chest.keyframe_insert("rotation_euler", frame=frame)

        body = {
            "spine_01": ("z", sway * math.sin(2.0 * math.pi * t)),
            "head": ("x", head_level - sway * 0.7 * math.cos(4.0 * math.pi * t)),
            "tail_01": ("z", -sway * 1.6 * math.sin(2.0 * math.pi * t)),
            "tail_02": ("z", -sway * 1.2 * math.sin(2.0 * math.pi * t - 0.6)),
            "ear_L": ("x", sway * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
            "ear_R": ("x", sway * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
        }
        for role, (axis, value) in body.items():
            name = bone_names.get(role, role)
            pb = arm.pose.bones.get(name)
            if pb is None:
                missing.append(name)
                continue
            setattr(pb.rotation_euler, axis, value)
            pb.keyframe_insert("rotation_euler", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")

    action = arm.animation_data.action

    curves = _fcurves(action)
    for fcurve in curves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"

    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = frames  # frame frames+1 duplicates frame 1, so this loops
    scn.frame_set(1)

    # Markers are hidden only from the render, never from the viewport. `hide_viewport`
    # removes an object from the viewport dependency graph entirely — on the armature that
    # silently stops the mesh deforming, so playback looks frozen while renders still
    # animate. Keep the rig live and visible; use hide_render to keep previews clean.
    custom_markers = set(marker_names.values())
    for o in bpy.data.objects:
        if o.name.startswith("JOINT_") or o.name in custom_markers:
            o.hide_render = True
    arm.hide_viewport = False
    arm.hide_render = True
    arm.show_in_front = True
    arm.data.pose_position = "POSE"

    summary = {
        "frames": frames,
        "fcurves": len(curves),
        "keyframes": sum(len(fc.keyframe_points) for fc in curves),
        "missing_bones": sorted(set(missing)),
        "frame_range": [scn.frame_start, scn.frame_end],
    }
    print(json.dumps(summary, indent=2))
    return summary
