#!/usr/bin/env python3
"""Build a quadruped armature from placed joint markers and bind the mesh to it.

Run (live): python scripts/blender_build_rig.py --help

Reads the `JOINT_*` empties left by `blender_joint_markers.py`, constructs a bone
hierarchy, and binds with Blender's automatic (bone heat) weighting.

**The point of this script is the verification, not the binding.** Heat weighting
diffuses influence *across the mesh surface*, so an ear is far from a leg even when the
two are close in straight-line space. It requires a clean manifold surface, and when it
fails Blender falls back silently to *envelope* weights — crude capsules that grab any
vertex inside them, through the air, ignoring the surface. The visible symptom is a rig
that looks fine until you move a leg and the ears move with it.

Blender reports that failure as a warning rather than an exception, so a bind that
"succeeded" proves nothing. Instead this script measures, for every leg bone, how far
its influenced vertices reach up the body — and fails loudly if a leg reaches the head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_rig_weights_worker as worker
from blender_joint_markers import send
from blender_worker_source import build_worker_code

SKELETON = worker.SKELETON
EAR_BONES = worker.EAR_BONES


def build_code(ear_length: float, weight_threshold: float) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_rig_weights_worker.py"),
        {"operation": "build", "ear_length": ear_length, "weight_threshold": weight_threshold},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--ear-length", type=float, default=0.09,
        help="Ear bones are extrapolated from the head through the ear base, so an "
             "over-long value pushes their tips outside the mesh",
    )
    parser.add_argument("--weight-threshold", type=float, default=0.15)
    args = parser.parse_args()
    print(send(build_code(args.ear_length, args.weight_threshold), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
