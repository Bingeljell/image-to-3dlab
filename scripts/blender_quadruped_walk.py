#!/usr/bin/env python3
"""Author a natural walk, trot or scamper on any Rigify basic-quadruped rig.

    blender --background rigged.blend --python scripts/blender_quadruped_walk.py -- \
        --gait walk --frames 32 --out walked.blend

The gait maths lives in `quadruped_gait.py` as pure functions; this file is the thin
`bpy` layer that measures the rig, converts world-space offsets into each bone's local
space, and keys them.

Two decisions worth knowing. The legs are keyed as **IK foot targets**, so the script
sets `IK_FK` to 0.0 on the four leg parents rather than assuming it -- keying IK controls
while the rig is in FK mode authors curves that the constraints then ignore, which looks
exactly like the script silently failing. And the **body chain is keyed too**: the
rejected generic cycle left `hips`, `torso` and `chest` with zero motion, which is what
made it read as a marionette rather than an animal.

Stride and lift are multiples of *this* rig's own leg lengths, never absolute numbers
copied between creatures -- the ram and the fox are within 5% in height but have inverted
limb segments, so shared absolutes over-reach the IK targets and skate the feet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quadruped_gait import (
    BODY_CHAIN,
    FOOT_IK,
    FRONT_FK,
    FRONT_LIMBS,
    GAITS,
    HEAD_CHAIN,
    LIMBS,
    PAW,
    RAKE_CHAIN,
    SHOULDER,
    TAIL_CHAIN,
    body_drop_for_reach,
    check_gait,
    gait_plan,
    max_half_stride,
    rest_extent,
    shoulder_angle,
)

LEG_PARENTS = ("front_thigh_parent.L", "front_thigh_parent.R",
               "thigh_parent.L", "thigh_parent.R")
HIND_CHAIN = ("thigh.L", "shin.L", "foot.L")
FRONT_CHAIN = ("front_thigh.L", "front_shin.L", "front_foot.L")


def find_rig(objects):
    """The generated Rigify rig -- the one carrying DEF- bones, not the metarig."""
    armatures = [o for o in objects if o.type == "ARMATURE"]
    generated = [a for a in armatures
                 if any(b.name.startswith("DEF-") for b in a.data.bones)]
    if len(generated) == 1:
        return generated[0]
    if not generated:
        raise RuntimeError(
            "no generated Rigify rig found (no DEF- bones); generate the rig first, "
            "e.g. blender_bind_rig.py --generate-rigify")
    raise RuntimeError(f"several generated rigs: {[a.name for a in generated]}")


def chain_length(rig, metarig, names) -> float:
    """Summed rest length of one limb, from the metarig or the rig's ORG- mirrors."""
    for source, prefix in ((metarig, ""), (rig, "ORG-")):
        if source is None:
            continue
        bones = source.data.bones
        if all(prefix + n in bones for n in names):
            return sum((bones[prefix + n].tail_local
                        - bones[prefix + n].head_local).length for n in names)
    raise RuntimeError(f"cannot measure limb {names}: no metarig and no ORG- bones")


def forward_axis(rig) -> tuple[int, float]:
    """Which world axis the creature faces, as (index, sign).

    Derived from the rig rather than assumed: a head control ahead of the hips defines
    forward, so a rig built facing +Y works as well as one facing -Y.
    """
    bones = rig.pose.bones
    if "head" not in bones or "hips" not in bones:
        return 1, -1.0
    delta = (rig.matrix_world @ bones["head"].head) - (rig.matrix_world @ bones["hips"].head)
    index = max((0, 1), key=lambda i: abs(delta[i]))
    return index, (1.0 if delta[index] > 0 else -1.0)


def swing_axis(bpy, rig, bone_name, tip_name=None, probe=0.15):
    """Which local rotation axis and sign swings this limb forward, measured not assumed.

    Rigify's FK limb bones do not agree on axis or sign between rigs (or between the
    front and hind chains of one rig), and guessing wrong drives the leg backwards or
    sideways while every number in the report still looks correct.
    """
    from mathutils import Vector

    bone = rig.pose.bones[bone_name]
    tip = rig.pose.bones[tip_name or bone_name.replace("thigh", "foot")]
    original_mode, bone.rotation_mode = bone.rotation_mode, "XYZ"
    rest = Vector(bone.rotation_euler)
    axis_index, axis_sign, best = 0, 1.0, 0.0
    forward, sign = forward_axis(rig)
    for index in range(3):
        for direction in (1.0, -1.0):
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.rotation_euler[index] = probe * direction
            bpy.context.view_layer.update()
            moved = (rig.matrix_world @ tip.tail)[forward] * sign
            if moved > best:
                best, axis_index, axis_sign = moved, index, direction
    bone.rotation_euler = rest
    bone.rotation_mode = original_mode
    bpy.context.view_layer.update()
    return axis_index, axis_sign

def reach_geometry(rig, metarig, upper, lower):
    """Rest geometry of one two-bone IK chain: its length, and where its ankle sits.

    Taken from rest bone data rather than the posed rig, so the answer does not depend on
    whatever frame the scene happens to be sitting on.
    """
    bones = (metarig or rig).data.bones
    prefix = "" if metarig else "ORG-"
    chain = sum((bones[prefix + n].tail_local - bones[prefix + n].head_local).length
                for n in (upper, lower))
    shoulder = rig.matrix_world @ rig.data.bones["ORG-" + upper].head_local
    ankle = rig.matrix_world @ rig.data.bones["ORG-" + lower].tail_local
    offset = ankle - shoulder
    return {"chain_length": chain,
            "vertical_drop": abs(offset.z),
            "horizontal_at_rest": abs(offset.y)}


def axis_for(bpy, rig, bone_name, tip_name, component, probe=0.12):
    """Local rotation axis and sign that moves `tip_name` furthest along a world axis.

    Measured, never assumed: Rigify bone rolls differ between chains and between rigs,
    and a wrong guess drives a tail sideways when it should lift while every printed
    number still looks plausible.
    """
    bone = rig.pose.bones[bone_name]
    tip = rig.pose.bones[tip_name]
    original = bone.rotation_mode
    bone.rotation_mode = "XYZ"
    best, axis_index, axis_sign = 0.0, 0, 1.0
    start = (rig.matrix_world @ tip.tail)[component]
    for index in range(3):
        for direction in (1.0, -1.0):
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.rotation_euler[index] = probe * direction
            bpy.context.view_layer.update()
            moved = (rig.matrix_world @ tip.tail)[component] - start
            if moved > best:
                best, axis_index, axis_sign = moved, index, direction
    bone.rotation_euler = (0.0, 0.0, 0.0)
    bone.rotation_mode = original
    bpy.context.view_layer.update()
    return axis_index, axis_sign


def measured_head_yaw(rig, mesh, head_group="DEF-spine.011", neck_group="DEF-spine.010"):
    """How far the head points off the body's axis, read from the mesh it deforms.

    Generated meshes inherit the pose of their source image; this fox's head sits about
    32 degrees to its own left, which is in the asset rather than the rig.
    """
    import math

    from mathutils import Vector

    groups = {g.name: g.index for g in mesh.vertex_groups}

    def centroid(name):
        index = groups.get(name)
        if index is None:
            return None
        points = [mesh.matrix_world @ v.co for v in mesh.data.vertices
                  if any(g.group == index and g.weight >= 0.5 for g in v.groups)]
        return sum(points, Vector()) / len(points) if points else None

    head, neck = centroid(head_group), centroid(neck_group)
    if head is None or neck is None:
        return 0.0
    offset = head - neck
    return math.radians(math.degrees(math.atan2(offset.x, -offset.y)))


def audit_cycle(bpy, rig, scene, *, body_height, leg_lengths, axis):
    """Measure what was actually authored, so the run can check its own work.

    Reports on the evaluated deform chain rather than the controls that were keyed: a
    control can carry a perfect curve while the mesh does nothing, which is exactly how
    keying IK targets in FK mode fails.
    """
    import math

    from mathutils import Vector

    frames = list(range(scene.frame_start, scene.frame_end + 1))
    tracks = {limb: [] for limb in LIMBS}
    torso_z, leans, first, last = [], [], None, None
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        # Read the evaluated copy, not the authoring one. In background mode the
        # depsgraph flushes synchronously and the two agree, but in a live GUI session
        # the authoring bones can lag a frame -- which had this audit report 10.6 degrees
        # of foreleg rake against the 6.2 the very same cycle measured headless, and fail
        # a cycle that was in fact correct.
        evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        pose = evaluated.pose.bones
        for limb in LIMBS:
            paw = evaluated.matrix_world @ pose[PAW[limb]].tail
            tracks[limb].append((paw[axis], paw.z))
        torso_z.append((evaluated.matrix_world @ pose["torso"].head).z)
        top = evaluated.matrix_world @ pose[RAKE_CHAIN[0]].tail
        bottom = evaluated.matrix_world @ pose[RAKE_CHAIN[1]].tail
        leans.append(math.degrees(math.atan2(-(bottom[axis] - top[axis]),
                                             max(top.z - bottom.z, 1e-6))))
        sample = Vector((tracks["FL"][-1][0], tracks["FL"][-1][1], torso_z[-1]))
        first = sample if first is None else first
        last = sample

    measured = {"body_bob_pct_height": (max(torso_z) - min(torso_z)) / body_height,
                "front_rake_excursion_deg": max(leans) - min(leans),
                "loop_gap": (first - last).length,
                "limbs": {}}
    for limb in LIMBS:
        fore = [p[0] for p in tracks[limb]]
        up = [p[1] for p in tracks[limb]]
        floor = min(up)
        measured["limbs"][limb] = {
            "stride_pct_leg": (max(fore) - min(fore)) / leg_lengths[limb[0]],
            "lift_pct_height": (max(up) - floor) / body_height,
            "planted_fraction": sum(1 for z in up if z - floor < body_height * 0.003)
            / len(up),
        }
    return measured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait", choices=("walk", "trot", "scamper"), default="walk")
    parser.add_argument("--frames", type=int, default=None,
                        help="Cycle length; defaults to the gait's own natural cadence")
    parser.add_argument("--name", default=None, help="Action name (default: Quad<Gait>)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiply stride, lift and bob; 1.0 matches the reference walk")
    parser.add_argument("--head-yaw", default="0",
                        help="Degrees of yaw correction for a head that does not face "
                             "forward, SPLIT across neck and head so no single deform "
                             "bone drags its vertices far enough to crush the muzzle; "
                             "'auto' measures the offset from the mesh, 0 disables")
    parser.add_argument("--neck-offset", default=None,
                        help="Static neck rotation 'x,y,z' in degrees, added to the "
                             "cycle; use with --head-offset to keep a hand-made fix")
    parser.add_argument("--head-offset", default=None,
                        help="Static head rotation 'x,y,z' in degrees, added to the cycle")
    parser.add_argument("--tail-swing", type=float, default=None,
                        help="Degrees of tail side-to-side swing (default 7)")
    parser.add_argument("--spend-reach", action="store_true",
                        help="Grow the stride to fill the reach the body drop creates; "
                             "without it a bigger lift just marches on the spot")
    parser.add_argument("--reach-fill", type=float, default=0.9,
                        help="Fraction of available reach to stride into (default 0.9)")
    parser.add_argument("--body-sway", type=float, default=None,
                        help="Lateral roll as a fraction of the vertical bob "
                             "(default 0.35; 1.0 gives the Bee Gees strut)")
    parser.add_argument("--body-drop", type=float, default=None,
                        help="Lower the body by this much to buy IK reach; off by "
                             "default because it distorts forelegs with no headroom")
    parser.add_argument("--shoulder-share", type=float, default=None,
                        help="Fraction of the foreleg's reach carried by the shoulder "
                             "rather than by bending the limb; higher keeps the lower "
                             "foreleg more upright")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Do not check the authored cycle against known failures")
    parser.add_argument("--out", default=None, help="Save the result to this .blend")
    args = parser.parse_args(argv if argv is not None else _script_args())

    import bpy
    from mathutils import Vector

    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    rig = find_rig(list(bpy.data.objects))
    metarig = next((o for o in bpy.data.objects
                    if o.type == "ARMATURE" and o is not rig), None)
    mesh = max((o for o in bpy.data.objects
                if o.type == "MESH" and not o.name.startswith("WGT-")),
               key=lambda o: len(o.data.vertices), default=None)
    if mesh is None:
        raise RuntimeError("no mesh found to measure body height from")

    required = (list(FOOT_IK.values()) + list(BODY_CHAIN) + list(SHOULDER.values()))
    missing = [n for n in required if n not in rig.pose.bones]
    if missing:
        raise RuntimeError(f"rig {rig.name!r} is missing controls {missing}; "
                           f"this script targets the Rigify basic_quadruped rig")

    front_length = chain_length(rig, metarig, FRONT_CHAIN)
    hind_length = chain_length(rig, metarig, HIND_CHAIN)
    body_height = rest_extent([v.co for v in mesh.data.vertices])
    frames = args.frames or GAITS[args.gait]["frames"]

    # Work out the drop first, then let the stride grow into the reach it creates.
    fore_geo = reach_geometry(rig, metarig, "front_thigh.L", "front_shin.L")
    hind_geo = reach_geometry(rig, metarig, "thigh.L", "shin.L")
    drop_now = args.body_drop or 0.0
    stride_override = None
    if args.spend_reach:
        fill = args.reach_fill
        fore_room = max_half_stride(**dict(fore_geo,
                                           vertical_drop=fore_geo["vertical_drop"] - drop_now))
        hind_room = max_half_stride(**dict(hind_geo,
                                           vertical_drop=hind_geo["vertical_drop"] - drop_now))
        stride_override = {"front_stride": 2.0 * fill * fore_room,
                           "hind_stride": 2.0 * fill * hind_room}

    plan = gait_plan(
        frames, args.gait,
        front_leg_length=front_length,
        hind_leg_length=hind_length,
        body_height=body_height,
        scale=args.scale,
        **({"tail_swing_deg": args.tail_swing} if args.tail_swing is not None else {}),
        **({"sway_per_bob": args.body_sway} if args.body_sway is not None else {}),
        stride_override=stride_override,
    )

    # Every foot rides on IK so it plants; the shoulders then absorb the forelegs' reach.
    # IK controls are only obeyed in IK mode -- keying them in FK authors dead curves.
    switched = {}
    for name in LEG_PARENTS:
        bone = rig.pose.bones.get(name)
        if bone is not None and "IK_FK" in bone:
            switched[name] = round(bone["IK_FK"], 3)
            bone["IK_FK"] = 0.0

    action_name = args.name or f"Quad{args.gait.capitalize()}"
    if action_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[action_name])
    action = bpy.data.actions.new(action_name)
    animation = rig.animation_data or rig.animation_data_create()
    animation.action = action
    try:  # Blender 4.4+ slotted actions need a slot bound before curves land
        slot = action.slots.new(id_type='OBJECT', name=rig.name)
        animation.action_slot = slot
    except Exception:  # noqa: S110 - older Blender has no slots, and needs none
        pass

    # Clear every channel of the CONTROL bones, but never touch the rig's internal
    # machinery. Rigify stores real state on MCH- bones -- this quadruped keeps roughly
    # 70 degrees of rotation on MCH-front_foot_parent.L/R that the foreleg IK chain
    # depends on -- so a blanket reset silently breaks the front legs before a single
    # keyframe is written: elbow hyperextended to 180 degrees and wrist folded to 118
    # against 141 and 174 at rest. It presents as a bad walk, which is where three
    # rounds of fixing the animation went.
    #
    # Clearing all channels of the controls still matters: zeroing `rotation_euler` alone
    # is a no-op on a quaternion-mode bone, so leftovers survive into the next run.
    internal = ("MCH-", "DEF-", "ORG-", "VIS_", "WGT-")
    for bone in rig.pose.bones:
        if bone.name.startswith(internal):
            continue
        bone.location = Vector((0.0, 0.0, 0.0))
        bone.scale = Vector((1.0, 1.0, 1.0))
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)

    # A limb already near full extension cannot stride without locking; lower the body
    # by exactly what the requested stride needs rather than quietly clipping the gait.
    fore_reach = fore_geo
    half_stride = plan["sizes"]["front_stride"] * 0.5
    # Reported, not applied. Lowering the body is the textbook way to buy IK headroom,
    # but on this rig a 0.02 drop on its own hyperextends the foreleg elbow to 176
    # degrees and folds the wrist to 118 -- worse than the over-reach it was meant to
    # cure. Opt in with --body-drop once you have checked what it does to your rig.
    body_drop = args.body_drop if args.body_drop is not None else 0.0
    suggested_drop = body_drop_for_reach(**fore_reach, half_stride=half_stride)

    axis, sign = forward_axis(rig)
    lateral = 1 - axis  # the remaining horizontal axis
    front_axis, front_sign = swing_axis(bpy, rig, FRONT_FK["FL"][0])
    # Calibrated separately: the sign that swings a thigh forward does not carry over to
    # the shoulder, and reusing it rotated the scapula the wrong way, deepening the very
    # forward rake the shoulder exists to absorb (lean widened to -0.9..17.5 degrees).
    shoulder_axis, shoulder_sign = swing_axis(
        bpy, rig, SHOULDER["FL"], tip_name="ORG-front_thigh.L")

    tail_present = [b for b in TAIL_CHAIN if b in rig.pose.bones]
    tail_yaw = axis_for(bpy, rig, tail_present[0], tail_present[-1], 0) if tail_present \
        else None
    tail_pitch = axis_for(bpy, rig, tail_present[0], tail_present[-1], 2) if tail_present \
        else None
    head_present = [b for b in HEAD_CHAIN if b in rig.pose.bones]
    head_yaw_axis = axis_for(bpy, rig, "head", "head", 0) if "head" in rig.pose.bones \
        else None
    head_pitch_axis = axis_for(bpy, rig, "head", "head", 2) if "head" in rig.pose.bones \
        else None

    import math as _math

    # Split the yaw correction between neck and head. Putting all of it on `head` drags
    # the head vertices through the neck's influence and visibly squashes the muzzle --
    # a 33 degree correction flattened this fox's face. Hand-correcting the same rig
    # took 17 degrees on the neck and 23 on the head, which is the ratio used here.
    if args.head_yaw == "auto":
        total_yaw = -measured_head_yaw(rig, mesh)
    else:
        total_yaw = -_math.radians(float(args.head_yaw))
    yaw_split = {"neck": total_yaw * 0.43, "head": total_yaw * 0.57}

    def parse_offset(text):
        if not text:
            return (0.0, 0.0, 0.0)
        parts = [float(v) for v in text.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"offset needs three comma-separated degrees, got {text!r}")
        return tuple(_math.radians(v) for v in parts)

    static_offset = {"neck": parse_offset(args.neck_offset),
                     "head": parse_offset(args.head_offset)}

    def to_local(bone, world_delta):
        rest = bone.bone.matrix_local.to_3x3()
        return rest.inverted() @ world_delta

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames + 1

    # Anchor each foot to where it actually rests, not to `bone.matrix_local`. That is
    # the bone's rest matrix with its parent ignored, and Rigify parents each front foot
    # IK control to an MCH bone carrying ~70 degrees of rotation -- so the two differ,
    # and driving from the rest matrix placed the front feet somewhere the rig never
    # rests, hyperextending the elbow to 180 degrees and folding the wrist to 114 on
    # every frame. Read with the basis cleared, so it is the rig's own neutral.
    neutral = {}
    for limb in LIMBS:
        bone = rig.pose.bones[FOOT_IK[limb]]
        bone.location = Vector((0.0, 0.0, 0.0))
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    for limb in LIMBS:
        neutral[limb] = rig.pose.bones[FOOT_IK[limb]].matrix.translation.copy()

    torso = rig.pose.bones["torso"]

    def key_rotation(bone, frame):
        channel = ("rotation_quaternion" if bone.rotation_mode == "QUATERNION"
                   else "rotation_euler")
        bone.keyframe_insert(channel, frame=frame)

    # The body is posed first and the depsgraph flushed, because each foot IK control
    # hangs off an MCH parent that moves with the body. The feet are then pinned in
    # *world* space via `pose_bone.matrix`, which makes Blender solve the parent maths
    # itself. Writing `location` from a rest-space conversion instead silently skews the
    # offset through any rotated parent -- Rigify gives the front feet an
    # MCH-front_foot_parent carrying ~70 degrees, so the forelegs splayed and the paws
    # twisted outward while the (unrotated) hind legs looked correct.
    for index, body in enumerate(plan["body"]):
        frame = body["frame"]
        delta = Vector((0.0, 0.0, 0.0))
        delta.z = body["up"] - body_drop
        delta[lateral] = body["lateral"]
        torso.location = to_local(torso, delta)
        torso.keyframe_insert("location", frame=frame)
        bpy.context.view_layer.update()

        for limb in LIMBS:
            bone = rig.pose.bones[FOOT_IK[limb]]
            key = plan["feet"][limb][index]
            foot_delta = Vector((0.0, 0.0, 0.0))
            foot_delta[axis] = sign * key["fore_aft"]
            foot_delta.z = key["up"]
            # Translate the control; do not pin its world orientation. Forcing the foot
            # back to its rest orientation every frame holds the paw at a fixed world
            # angle while the forearm above it tilts, so the wrist snaps to a right angle
            # -- measured at 93-101 degrees against 173.7 at rest, which reads as the paw
            # hinging out in front of the leg. Leaving the rotation at its identity basis
            # lets the foot ride with the limb, as it does when an animator drags the
            # control by hand.
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            placed = bone.matrix.copy()
            placed.translation = neutral[limb] + foot_delta
            bone.matrix = placed
            bone.keyframe_insert("location", frame=frame)
            key_rotation(bone, frame)

        tail_plan = plan["tail"][index]["segments"]
        for segment, bone_name in enumerate(tail_present):
            bone = rig.pose.bones[bone_name]
            if bone.rotation_mode == "QUATERNION":
                bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
            if tail_yaw:
                bone.rotation_euler[tail_yaw[0]] = (
                    tail_yaw[1] * tail_plan[segment]["swing"])
            if tail_pitch and tail_pitch[0] != tail_yaw[0]:
                bone.rotation_euler[tail_pitch[0]] = (
                    tail_pitch[1] * tail_plan[segment]["lift"])
            bone.keyframe_insert("rotation_euler", frame=frame)

        head_plan = plan["head"][index]
        for bone_name in head_present:
            bone = rig.pose.bones[bone_name]
            if bone.rotation_mode == "QUATERNION":
                bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
            angles = head_plan[bone_name]
            offset = static_offset[bone_name]
            bone.rotation_euler = offset
            if head_yaw_axis:
                bone.rotation_euler[head_yaw_axis[0]] = (
                    offset[head_yaw_axis[0]]
                    + head_yaw_axis[1] * angles["yaw"] + yaw_split[bone_name])
            if head_pitch_axis and head_pitch_axis[0] != head_yaw_axis[0]:
                bone.rotation_euler[head_pitch_axis[0]] = (
                    offset[head_pitch_axis[0]] + head_pitch_axis[1] * angles["nod"])
            bone.keyframe_insert("rotation_euler", frame=frame)

        # The shoulder carries part of each foreleg's reach so the wrist does not have
        # to lead. Without this the lower foreleg rakes forward as the IK target moves.
        for limb in FRONT_LIMBS:
            bone = rig.pose.bones[SHOULDER[limb]]
            if bone.rotation_mode == "QUATERNION":
                bone.rotation_mode = "XYZ"
            bone.rotation_euler = (0.0, 0.0, 0.0)
            bone.rotation_euler[shoulder_axis] = shoulder_sign * shoulder_angle(
                plan["feet"][limb][index]["fore_aft"], front_length,
                *( (args.shoulder_share,) if args.shoulder_share is not None else () ))
            bone.keyframe_insert("rotation_euler", frame=frame)

    report = {
        "action": action_name, "gait": args.gait,
        "frames": [scene.frame_start, scene.frame_end],
        "sizes": {k: round(v, 4) for k, v in plan["sizes"].items()},
        "body_bob": round(max(b["up"] for b in plan["body"])
                          - min(b["up"] for b in plan["body"]), 4),
        "forward_axis": ("XYZ"[axis], sign),
        "fore_reach": {k: round(v, 4) for k, v in fore_reach.items()},
        "fore_extension_at_rest_pct": round(
            100 * ((fore_reach["vertical_drop"] ** 2
                    + fore_reach["horizontal_at_rest"] ** 2) ** 0.5)
            / fore_reach["chain_length"], 1),
        "fore_headroom_before_drop": round(max_half_stride(**fore_reach), 4),
        "half_stride_requested": round(half_stride, 4),
        "body_drop_applied": round(body_drop, 4),
        "body_drop_suggested": round(suggested_drop, 4),
        "ik_fk_switched_from": switched,
        "front_swing_axis": ("XYZ"[front_axis], front_sign),
        "shoulder_axis": ("XYZ"[shoulder_axis], shoulder_sign),
        "shoulder_share": args.shoulder_share,
        "head_yaw_split_deg": {k: round(_math.degrees(v), 2)
                               for k, v in yaw_split.items()},
        "tail_bones": tail_present,
        "driven": sorted(list(FOOT_IK.values()) + list(SHOULDER.values())
                         + tail_present + head_present + ["torso"]),
    }
    problems = []
    if not args.skip_audit:
        measured = audit_cycle(
            bpy, rig, scene, body_height=body_height,
            leg_lengths={"F": front_length, "B": hind_length}, axis=axis)
        problems = check_gait(measured, args.gait)
        report["audit"] = {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in measured.items() if k != "limbs"}
        report["audit"]["limbs"] = {
            limb: {k: round(v, 4) for k, v in stats.items()}
            for limb, stats in measured["limbs"].items()}
        report["audit"]["problems"] = problems

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        report["saved"] = str(out)
    print("I2L_GAIT_REPORT " + json.dumps(report))
    if problems:
        raise RuntimeError("authored cycle failed its audit:\n  - "
                           + "\n  - ".join(problems))
    return 0


def _script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


if __name__ == "__main__":
    raise SystemExit(main())
