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
from attack_pose import sample
from blender_joint_markers import send
from blender_worker_source import build_worker_code


def build_code(
    frames: int,
    action_name: str,
    toe: float,
) -> str:
    """Bundle the sampled attack for the remote Blender worker."""
    poses = sample(frames)
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_attack_cycle_worker.py"),
        {
            "poses": poses,
            "action_name": action_name,
            "toe": toe,
        },
    )


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
        help=("how much the toe follows the paw, 0-1. Set it to 0 to lock the toe."),
    )
    args = parser.parse_args()
    code = build_code(
        args.frames,
        args.name,
        args.toe,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
