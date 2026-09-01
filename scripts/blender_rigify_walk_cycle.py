"""Author a trot cycle on the Rigify-generated 'rig' armature in the live Blender scene.

Bpy layer only -- the actual pose math lives in rigify_walk_pose.py so it stays testable.
Assumes IK_FK is already set to 1.0 on the leg parent bones (front_thigh_parent.L/R,
thigh_parent.L/R); this script does not touch that switch, since flipping it mid-session
without the user's awareness would silently change how their manual posing behaves too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigify_walk_pose import sample
from blender_joint_markers import send


def build_code(frames: int, gait: str, swing_front: float, swing_back: float,
                bend_front: float, bend_back: float, low_fold: float,
                base_bend: float, spine_sway: float, action_name: str) -> str:
    poses = sample(frames, gait, swing_front, swing_back, bend_front, bend_back,
                    low_fold, base_bend, spine_sway)
    return f'''
import bpy, math, json

poses = json.loads({json.dumps(json.dumps(poses))})

arm = bpy.data.objects.get("rig")
if arm is None:
    raise RuntimeError("no object named 'rig' in the scene")

ik_fk_bones = ["front_thigh_parent.L", "front_thigh_parent.R", "thigh_parent.L", "thigh_parent.R"]
not_fk = [b for b in ik_fk_bones if arm.pose.bones.get(b) and arm.pose.bones[b].get("IK_FK", 1.0) < 0.99]
if not_fk:
    print(json.dumps({{"warning": "these legs are not in FK mode, rotations will be ignored by the deform bones", "bones": not_fk}}))

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")

if arm.animation_data and arm.animation_data.action:
    arm.animation_data.action = None
for stale in [a for a in bpy.data.actions if a.name.split(".")[0] == "{action_name}"]:
    bpy.data.actions.remove(stale)

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
    for name in driven_names:
        pb = arm.pose.bones.get(name)
        if pb is None:
            missing.append(name)
            continue
        deg = pose.get(name, 0.0)
        pb.rotation_euler = (math.radians(deg), 0.0, 0.0)
        pb.keyframe_insert("rotation_euler", frame=frame)

action = arm.animation_data.action
action.name = "{action_name}"
scene.frame_set(1)
bpy.ops.object.mode_set(mode="OBJECT")

print(json.dumps({{
    "frames": len(poses),
    "action": action.name,
    "driven_bones": sorted(driven_names),
    "missing_bones": sorted(set(missing)),
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=28)
    parser.add_argument("--gait", choices=("trot", "walk"), default="trot")
    parser.add_argument("--swing-front", type=float, default=30.0)
    parser.add_argument("--swing-back", type=float, default=30.0)
    parser.add_argument("--bend-front", type=float, default=35.0)
    parser.add_argument("--bend-back", type=float, default=35.0)
    parser.add_argument("--low-fold", type=float, default=30.0)
    parser.add_argument("--base-bend", type=float, default=8.0)
    parser.add_argument("--spine-sway", type=float, default=4.0)
    parser.add_argument("--name", default="RigifyWalk")
    args = parser.parse_args()

    code = build_code(args.frames, args.gait, args.swing_front, args.swing_back,
                       args.bend_front, args.bend_back, args.low_fold, args.base_bend,
                       args.spine_sway, args.name)
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
