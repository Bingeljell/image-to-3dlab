#!/usr/bin/env python3
"""Author a looping quadruped walk cycle on the rigged fox in the live Blender scene.

A quadruped walk is a **four-beat lateral gait**: the legs do not move in diagonal
pairs (that is a trot). Each foot lands a quarter-cycle after the one before it, in the
order back-left, front-left, back-right, front-right, so the animal always has at least
two feet down. Encoding that as a quarter-cycle phase offset per leg is what makes the
result read as a walk rather than a bounce.

Each leg's motion is two superimposed parts:

* **swing** — the upper bone (upper arm or thigh) rocks forward and back through the
  whole cycle, a cosine;
* **lift** — the lower bone (forearm or shin) folds only while the foot is off the
  ground, a half-wave rectified sine. A leg that bends while bearing weight looks
  broken, which is why the fold is clamped to the airborne half.

The body carries a vertical bob at *twice* stride frequency (the hips rise as each
diagonal support pair passes under), plus a small counter-sway on the spine, tail and
head so the fox does not look welded to a rail.

The animation is authored so frame 1 and frame `frames + 1` are identical, then the
scene end is set to `frames`, which makes playback loop seamlessly.
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
SW_FRONT = math.radians({swing_front})
SW_BACK = math.radians({swing_back})
BEND_FRONT = math.radians({bend_front})
BEND_BACK = math.radians({bend_back})
PAW = math.radians({paw})
BOB = {bob}
SWAY = math.radians({sway})

arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    raise RuntimeError("no armature in the scene — build the rig first")
arm = arms[0]
arm.hide_viewport = False
arm.hide_render = False

# Legs land a quarter-cycle apart, in lateral sequence.
PHASE = {{"backL": 0.0, "frontL": 0.25, "backR": 0.5, "frontR": 0.75}}
CHAIN = {{
    "frontL": ("frontL_upperarm", "frontL_forearm", "frontL_paw", SW_FRONT, BEND_FRONT),
    "frontR": ("frontR_upperarm", "frontR_forearm", "frontR_paw", SW_FRONT, BEND_FRONT),
    "backL": ("backL_thigh", "backL_shin", "backL_paw", SW_BACK, BEND_BACK),
    "backR": ("backR_thigh", "backR_shin", "backR_paw", SW_BACK, BEND_BACK),
}}

if arm.animation_data and arm.animation_data.action:
    arm.animation_data.action = None
arm.animation_data_clear()

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.0, 0.0, 0.0)
    pb.location = (0.0, 0.0, 0.0)

base_z = arm.location.z
missing = []

for frame in range(1, FRAMES + 2):
    t = (frame - 1) / FRAMES
    bpy.context.scene.frame_set(frame)

    for leg, phase in PHASE.items():
        upper, lower, paw, swing_amp, bend_amp = CHAIN[leg]
        theta = 2.0 * math.pi * (t + phase)
        swing = swing_amp * math.cos(theta)
        lift = max(0.0, math.sin(theta))       # airborne half only
        for name, value in ((upper, swing), (lower, bend_amp * lift), (paw, -PAW * lift)):
            pb = arm.pose.bones.get(name)
            if pb is None:
                missing.append(name)
                continue
            pb.rotation_euler.x = value
            pb.keyframe_insert("rotation_euler", frame=frame)

    # Hips rise twice per stride, as each diagonal support pair passes under the body.
    arm.location.z = base_z + BOB * math.cos(4.0 * math.pi * t)
    arm.keyframe_insert("location", frame=frame)

    body = {{
        "spine_01": ("z", SWAY * math.sin(2.0 * math.pi * t)),
        "spine_02": ("z", -SWAY * 0.6 * math.sin(2.0 * math.pi * t)),
        "neck": ("x", SWAY * 0.5 * math.cos(4.0 * math.pi * t)),
        "head": ("x", -SWAY * 0.4 * math.cos(4.0 * math.pi * t)),
        "tail_01": ("z", -SWAY * 1.6 * math.sin(2.0 * math.pi * t)),
        "tail_02": ("z", -SWAY * 1.2 * math.sin(2.0 * math.pi * t - 0.6)),
        "ear_L": ("x", SWAY * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
        "ear_R": ("x", SWAY * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
    }}
    for name, (axis, value) in body.items():
        pb = arm.pose.bones.get(name)
        if pb is None:
            missing.append(name)
            continue
        setattr(pb.rotation_euler, axis, value)
        pb.keyframe_insert("rotation_euler", frame=frame)

bpy.ops.object.mode_set(mode="OBJECT")

action = arm.animation_data.action


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


curves = _fcurves(action)
for fcurve in curves:
    for kp in fcurve.keyframe_points:
        kp.interpolation = "BEZIER"

scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = FRAMES          # frame FRAMES+1 duplicates frame 1, so this loops
scn.frame_set(1)

for o in bpy.data.objects:
    if o.name.startswith("JOINT_"):
        o.hide_viewport = True

print(json.dumps({{
    "frames": FRAMES,
    "fcurves": len(curves),
    "keyframes": sum(len(fc.keyframe_points) for fc in curves),
    "missing_bones": sorted(set(missing)),
    "frame_range": [scn.frame_start, scn.frame_end],
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=32, help="length of one full stride")
    parser.add_argument("--swing-front", type=float, default=24.0)
    parser.add_argument("--swing-back", type=float, default=20.0)
    parser.add_argument("--bend-front", type=float, default=34.0)
    parser.add_argument("--bend-back", type=float, default=40.0)
    parser.add_argument("--paw", type=float, default=16.0)
    parser.add_argument("--bob", type=float, default=0.012)
    parser.add_argument("--sway", type=float, default=5.0)
    args = parser.parse_args()

    code = TEMPLATE.format(
        frames=args.frames,
        swing_front=args.swing_front,
        swing_back=args.swing_back,
        bend_front=args.bend_front,
        bend_back=args.bend_back,
        paw=args.paw,
        bob=args.bob,
        sway=args.sway,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
