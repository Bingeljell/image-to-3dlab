"""Author a slam-attack clip on a rigged quadruped in the live Blender scene.

Run (live): python scripts/blender_attack_cycle.py --help

The animation curves live in `attack_pose.py`. The remote worker applies the sampled
poses and adds the toe keys.

The clip does NOT loop like a gait. It starts and ends at rest so it can be triggered,
played once, and blended back into idle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animation_cli import read_name_map
from attack_pose import sample
from blender_joint_markers import send
from blender_worker_source import build_worker_code


def build_code(
    frames: int,
    action_name: str,
    toe: float,
    *,
    rig_name: str | None = None,
    bone_names: dict[str, str] | None = None,
) -> str:
    """Bundle the sampled attack for the remote Blender worker."""
    poses = sample(frames)
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_attack_cycle_worker.py"),
        {
            "poses": poses,
            "action_name": action_name,
            "toe": toe,
            "rig_name": rig_name,
            "bone_names": bone_names,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--rig-name", help="armature object to animate")
    parser.add_argument(
        "--bone-map",
        dest="bone_names",
        type=read_name_map,
        help="JSON file mapping default bone names to your rig's bone names",
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--name", default="Attack", help="action name in the export")
    parser.add_argument(
        "--toe",
        type=float,
        default=0.35,
        help=("how much the toe follows the paw, 0-1. Set it to 0 to lock the toe."),
    )
    args = parser.parse_args()
    code = build_code(
        args.frames,
        args.name,
        args.toe,
        rig_name=args.rig_name,
        bone_names=args.bone_names,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
