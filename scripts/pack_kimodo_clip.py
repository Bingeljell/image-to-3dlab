#!/usr/bin/env python3
"""Shrink a Kimodo motion clip to what `kimodo_retarget.py` reads, about 12x smaller.

    python scripts/pack_kimodo_clip.py walk.npz image_to_3dlab/motions/walk.npz

Kimodo's `.npz` carries local and global rotation matrices, every joint's position on every
frame, foot contacts and more: ~1 MB for a 5 s clip. The retarget needs only three things:
each joint's world rotation per frame, the hips track (how far the body travels) and one
frame's joint positions (to recover SOMA's T-pose and hip height). Rotations are kept as
half-precision quaternions, under 0.1 degree off the originals (measured: 0.08); the
positions stay full precision because the hips can travel metres.

`kimodo_retarget.load_motion` reads both this and Kimodo's own format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import SOMA_INDEX, matrices_to_quaternions


def pack(global_rot_mats: np.ndarray, posed_joints: np.ndarray, fps: float = 30.0) -> dict:
    """Arrays for `np.savez_compressed`, from Kimodo's [T,77,3,3] and [T,77,3]."""
    if global_rot_mats.ndim == 5:  # [samples, T, J, 3, 3] -> first sample
        global_rot_mats, posed_joints = global_rot_mats[0], posed_joints[0]
    return {
        "quats": matrices_to_quaternions(global_rot_mats).astype(np.float16),
        "joints0": posed_joints[0].astype(np.float32),
        "hips": posed_joints[:, SOMA_INDEX["Hips"]].astype(np.float32),
        "fps": np.float32(fps),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="Kimodo .npz (kimodo_gen output)")
    parser.add_argument("target", type=Path, help="packed .npz to write")
    args = parser.parse_args(argv)
    data = np.load(args.source)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.target, **pack(data["global_rot_mats"], data["posed_joints"]))
    before, after = args.source.stat().st_size, args.target.stat().st_size
    print(f"{args.target}: {before / 1024:.0f} KB -> {after / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
