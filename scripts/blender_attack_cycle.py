"""Author a slam-attack clip on a rigged quadruped in the live Blender scene.

All the animation maths lives in `attack_pose.py` so it can be imported and tested; this
file is only the `bpy` layer that inserts the keys. That split is deliberate -- the walk
cycle keeps its curves inside a template string, which means nothing about its timing can
be checked except by watching it, and a wind-up that leaned the wrong way survived
review here until a test caught it.

The clip does NOT loop like a gait. It starts and ends at rest so it can be triggered,
played once, and blended back into idle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_pose import sample  # noqa: E402
from blender_joint_markers import send  # noqa: E402


def build_code(frames: int, action_name: str, toe: float) -> str:
    """Blender-side script. Poses arrive precomputed, so nothing is derived in here."""
    poses = sample(frames)
    # Stone toes barely articulate; the walk's fox-tuned toe roll reads as rubbery on a
    # brute, so the toe follows the paw at a fraction of its angle rather than its own.
    return f'''
import bpy, math, json

poses = json.loads({json.dumps(json.dumps(poses))})
TOE_FOLLOW = {toe}

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")

# Detaching the old action is not enough: it survives in bpy.data.actions, so the new
# one is created as "Attack.001", both get exported, and a viewer plays whichever comes
# first -- silently showing the previous take. Delete any same-named action outright.
if arm.animation_data and arm.animation_data.action:
    arm.animation_data.action = None
for stale in [a for a in bpy.data.actions if a.name.split(".")[0] == "{action_name}"]:
    bpy.data.actions.remove(stale)

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
    for bone, degrees in pose.items():
        pb = arm.pose.bones.get(bone)
        if pb is None:
            missing.append(bone)
            continue
        pb.rotation_euler = (math.radians(degrees), 0.0, 0.0)
        driven.add(bone)
    for side in ("L", "R"):
        paw = pose.get("front" + side + "_paw", 0.0)
        toe = arm.pose.bones.get("front" + side + "_toe")
        if toe is not None:
            toe.rotation_euler = (math.radians(paw * TOE_FOLLOW), 0.0, 0.0)
            driven.add("front" + side + "_toe")
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_euler", frame=frame)

action = arm.animation_data.action
action.name = "{action_name}"
scene.frame_set(1)
bpy.ops.object.mode_set(mode="OBJECT")

print(json.dumps({{
    "frames": len(poses),
    "action": action.name,
    "driven_bones": sorted(driven),
    "missing_bones": sorted(set(missing)),
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--name", default="Attack", help="action name in the export")
    parser.add_argument(
        "--toe",
        type=float,
        default=0.35,
        help=(
            "how much the toe follows the paw, 0-1. The walk's toe roll is tuned for a "
            "fox's flexible paw and reads as rubbery on a stone brute; 0 locks the toe"
        ),
    )
    args = parser.parse_args()
    print(send(build_code(args.frames, args.name, args.toe), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
