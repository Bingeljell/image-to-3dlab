#!/usr/bin/env python3
"""Author a looping idle for the rigged fox — breathing, ear flicks, a lazy tail.

An idle is not a slow gait. Nothing cycles through contact phases; the feet stay planted
and every motion hangs off one slow **breath**. This is the animation a character screen
shows by default, so it has to survive being watched for a long time, which means:

* **Breathing is the spine, not the whole body.** The ribcage lifts and the belly follows
  a beat later; translating the entire rig up and down reads as bobbing, not breathing.
* **Nothing is exactly in phase.** Head, ears and tail each lag the breath by a different
  amount. Motion that shares one phase reads as mechanical however subtle it is.
* **An ear flick breaks the loop.** A purely sinusoidal idle looks dead after two cycles.
  One sharp asymmetric flick per loop, decaying fast, is what makes it read as alive —
  the eye catches the irregularity, not the rhythm.
* **Amplitudes are tiny.** Degrees, not tens of degrees. If an idle is obvious it is
  wrong; it should be noticeable only when it stops.

The feet are deliberately untouched, so this animation is unaffected by the foot-sliding
that limits the gait cycles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

TEMPLATE = '''
import bpy, json, math

FRAMES = {frames}
BREATH = math.radians({breath})
HEAD = math.radians({head})
EAR = math.radians({ear})
TAIL = math.radians({tail})
FLICK = math.radians({flick})
FLICK_AT = {flick_at}
HEAD_YAW = math.radians({head_yaw})
SHIFT = {shift}
EAR_TURN = math.radians({ear_turn})
TURN_AT = {ear_turn_at}
TURN_HOLD = {ear_turn_hold}

arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    raise RuntimeError("no armature in the scene — build the rig first")
arm = arms[0]
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

from mathutils import Matrix
missing = []

def put(name, axis, value, frame):
    pb = arm.pose.bones.get(name)
    if pb is None:
        missing.append(name)
        return
    setattr(pb.rotation_euler, axis, value)
    pb.keyframe_insert("rotation_euler", frame=frame)

for frame in range(1, FRAMES + 2):
    t = (frame - 1) / FRAMES
    bpy.context.scene.frame_set(frame)
    breath = math.sin(2.0 * math.pi * t)

    # Ribcage leads, belly follows a beat later — the lag is what makes it breathing
    # rather than a uniform swell.
    put("spine_02", "x", -BREATH * breath, frame)
    put("spine_01", "x", BREATH * 0.45 * math.sin(2.0 * math.pi * t - 0.55), frame)

    # The head rides the breath, lagging further still, plus a slow independent drift so
    # it never returns to exactly the same place.
    head_rot = HEAD * math.sin(2.0 * math.pi * t - 0.9) + HEAD * 0.35 * math.sin(4.0 * math.pi * t + 1.7)
    put("head", "x", head_rot, frame)
    put("jaw", "x", HEAD * 0.15 * math.sin(2.0 * math.pi * t - 1.2), frame)

    # The neck carries the static yaw correction for a head generated turned; applied as
    # a world rotation expressed in bone space, since a bone's axes run along the bone.
    neck = arm.pose.bones.get("neck")
    if neck is not None:
        rest = neck.bone.matrix_local.to_3x3()
        world = Matrix.Rotation(HEAD_YAW, 3, "Z")
        e = (rest.inverted() @ world @ rest).to_euler("XYZ")
        neck.rotation_euler = (e.x + HEAD * 0.4 * math.sin(2.0 * math.pi * t - 0.7), e.y, e.z)
        neck.keyframe_insert("rotation_euler", frame=frame)

    # One sharp ear flick per loop, decaying fast, deliberately on one ear only. A purely
    # sinusoidal idle reads as dead after two cycles; the eye catches the irregularity.
    phase = (t - FLICK_AT) % 1.0
    flick = math.exp(-phase * 26.0) * math.sin(phase * 58.0) if phase < 0.35 else 0.0
    put("ear_L", "x", EAR * math.sin(2.0 * math.pi * t - 0.4) + FLICK * flick, frame)

    # One ear swivels back to track a sound behind the fox, then returns. Independent
    # ear direction is the single most characteristically alert thing a fox does, and
    # because only one ear moves it reads as attention rather than as a pose change.
    # The turn is a world-up rotation expressed in bone space, so the ear pivots at its
    # base instead of shearing; amplitude is kept well inside anatomical range.
    tp = (t - TURN_AT) % 1.0
    if tp < TURN_HOLD:
        e = tp / TURN_HOLD
        # ease in, hold, ease out
        env = min(1.0, e / 0.3) if e < 0.3 else (1.0 if e < 0.7 else max(0.0, (1.0 - e) / 0.3))
    else:
        env = 0.0
    ear_r = arm.pose.bones.get("ear_R")
    if ear_r is not None:
        base = EAR * math.sin(2.0 * math.pi * t - 0.15) + FLICK * 0.25 * flick
        rest_r = ear_r.bone.matrix_local.to_3x3()
        swivel = Matrix.Rotation(EAR_TURN * env, 3, "Z")
        er = (rest_r.inverted() @ swivel @ rest_r).to_euler("XYZ")
        ear_r.rotation_euler = (er.x + base, er.y, er.z)
        ear_r.keyframe_insert("rotation_euler", frame=frame)
    else:
        missing.append("ear_R")

    # A lazy tail, slower than the breath and drifting sideways as well, so it never
    # traces the same arc twice within the loop.
    put("tail_01", "z", TAIL * math.sin(2.0 * math.pi * t - 1.1), frame)
    put("tail_02", "z", TAIL * 0.8 * math.sin(2.0 * math.pi * t - 1.9), frame)
    put("tail_01", "x", TAIL * 0.3 * math.sin(4.0 * math.pi * t), frame)

    # A weight shift far smaller than a breath: enough to avoid looking frozen, small
    # enough that the planted feet do not visibly slide.
    arm.location.z = SHIFT * breath
    arm.keyframe_insert("location", frame=frame)

bpy.ops.object.mode_set(mode="OBJECT")

action = arm.animation_data.action


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


curves = _fcurves(action)
for fc in curves:
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"

scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = FRAMES
scn.frame_set(1)

# hide_render, never hide_viewport: the latter drops the armature from the viewport
# dependency graph and silently freezes the mesh while renders keep animating.
for o in bpy.data.objects:
    if o.name.startswith("JOINT_"):
        o.hide_render = True
        o.hide_set(True)
arm.hide_render = True
arm.hide_set(True)

print(json.dumps({{
    "frames": FRAMES,
    "fcurves": len(curves),
    "keyframes": sum(len(fc.keyframe_points) for fc in curves),
    "missing_bones": sorted(set(missing)),
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=96, help="one full breath, ~4s at 24fps")
    parser.add_argument("--breath", type=float, default=2.4, help="ribcage rise, degrees")
    parser.add_argument("--head", type=float, default=2.0)
    parser.add_argument("--ear", type=float, default=2.5)
    parser.add_argument("--tail", type=float, default=3.5)
    parser.add_argument("--flick", type=float, default=13.0, help="ear flick amplitude")
    parser.add_argument("--flick-at", type=float, default=0.62, help="0-1 position in the loop")
    parser.add_argument("--head-yaw", type=float, default=-32.0)
    parser.add_argument("--shift", type=float, default=0.004, help="vertical weight shift")
    parser.add_argument(
        "--ear-turn", type=float, default=-62.0,
        help="How far one ear swivels back to track a sound behind the fox. Negative "
             "turns the ear rearward given this subject faces -Y; positive swings it "
             "forward across the face, which is wrong. Kept inside anatomical range",
    )
    parser.add_argument("--ear-turn-at", type=float, default=0.30, help="0-1 position in the loop")
    parser.add_argument("--ear-turn-hold", type=float, default=0.34, help="fraction of the loop it lasts")
    args = parser.parse_args()

    print(send(TEMPLATE.format(
        frames=args.frames, breath=args.breath, head=args.head, ear=args.ear,
        tail=args.tail, flick=args.flick, flick_at=args.flick_at,
        head_yaw=args.head_yaw, shift=args.shift,
        ear_turn=args.ear_turn, ear_turn_at=args.ear_turn_at,
        ear_turn_hold=args.ear_turn_hold,
    ), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
