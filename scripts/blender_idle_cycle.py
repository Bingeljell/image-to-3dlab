#!/usr/bin/env python3
"""Author a looping idle for the rigged fox — breathing, ear flicks, a lazy tail.

Run (live): python scripts/blender_idle_cycle.py --help

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
from blender_worker_source import build_worker_code


def build_code(arguments: dict[str, object]) -> str:
    """Bundle idle controls for the remote Blender worker."""
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_idle_cycle_worker.py"), arguments
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--frames", type=int, default=96, help="one full breath, ~4s at 24fps"
    )
    parser.add_argument(
        "--breath", type=float, default=2.4, help="ribcage rise, degrees"
    )
    parser.add_argument("--head", type=float, default=2.0)
    parser.add_argument("--ear", type=float, default=2.5)
    parser.add_argument("--tail", type=float, default=3.5)
    parser.add_argument("--flick", type=float, default=13.0, help="ear flick amplitude")
    parser.add_argument(
        "--flick-at", type=float, default=0.62, help="0-1 position in the loop"
    )
    parser.add_argument("--head-yaw", type=float, default=-32.0)
    parser.add_argument(
        "--shift", type=float, default=0.004, help="vertical weight shift"
    )
    parser.add_argument(
        "--ear-turn",
        type=float,
        default=-62.0,
        help="How far one ear swivels back to track a sound behind the character. Negative "
        "turns the ear rearward if the rig faces -Y. Positive swings it "
        "forward across the face.",
    )
    parser.add_argument(
        "--ear-turn-at", type=float, default=0.30, help="0-1 position in the loop"
    )
    parser.add_argument(
        "--ear-turn-hold",
        type=float,
        default=0.34,
        help="fraction of the loop it lasts",
    )
    args = parser.parse_args()

    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in ("host", "port") and value is not None
    }
    print(send(build_code(arguments), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
