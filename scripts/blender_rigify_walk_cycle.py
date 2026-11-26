"""Author a gait on a Rigify armature in the live Blender scene.

Run (live): python scripts/blender_rigify_walk_cycle.py --help

Bpy layer only -- the actual pose math lives in rigify_walk_pose.py so it stays testable.
Assumes IK_FK is already set to 1.0 on the leg parent bones (front_thigh_parent.L/R,
thigh_parent.L/R); this script does not touch that switch, since flipping it mid-session
without the user's awareness would silently change how their manual posing behaves too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animation_cli import read_name_map
from blender_joint_markers import send
from blender_worker_source import build_worker_code
from rigify_walk_pose import sample


def build_code(
    frames: int,
    gait: str,
    swing_front: float,
    swing_back: float,
    bend_front: float,
    bend_back: float,
    low_fold: float,
    base_bend: float,
    spine_sway: float,
    action_name: str,
    *,
    rig_name: str = "rig",
    bone_names: dict[str, str] | None = None,
) -> str:
    poses = sample(
        frames,
        gait,
        swing_front,
        swing_back,
        bend_front,
        bend_back,
        low_fold,
        base_bend,
        spine_sway,
    )
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_rigify_walk_cycle_worker.py"),
        {
            "poses": poses,
            "action_name": action_name,
            "rig_name": rig_name,
            "bone_names": bone_names,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--rig-name", default="rig", help="armature object to animate")
    parser.add_argument(
        "--bone-map",
        dest="bone_names",
        type=read_name_map,
        help="JSON file mapping default bone names to your rig's bone names",
    )
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

    code = build_code(
        args.frames,
        args.gait,
        args.swing_front,
        args.swing_back,
        args.bend_front,
        args.bend_back,
        args.low_fold,
        args.base_bend,
        args.spine_sway,
        args.name,
        rig_name=args.rig_name,
        bone_names=args.bone_names,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
