"""Author a headbutt/charge clip on the Rigify-generated 'rig' armature.

Bpy layer only -- pose math lives in rigify_headbutt_pose.py so it stays testable.
Assumes IK_FK is already 1.0 on the leg parent bones (same requirement as the Rigify
walk cycle); this script doesn't touch that switch itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigify_headbutt_pose import sample
from blender_joint_markers import send


def build_code(frames: int, action_name: str) -> str:
    poses = sample(frames)
    return f'''
import bpy, math, json

poses = json.loads({json.dumps(json.dumps(poses))})

arm = bpy.data.objects.get("rig")
if arm is None:
    raise RuntimeError("no object named 'rig' in the scene")

ik_fk_bones = ["front_thigh_parent.L", "front_thigh_parent.R", "thigh_parent.L", "thigh_parent.R"]
not_fk = [b for b in ik_fk_bones if arm.pose.bones.get(b) and arm.pose.bones[b].get("IK_FK", 1.0) < 0.99]
if not_fk:
    print(json.dumps({{"warning": "legs not in FK mode, rotations will be ignored", "bones": not_fk}}))

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
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--name", default="RigifyHeadbutt")
    args = parser.parse_args()

    code = build_code(args.frames, args.name)
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
