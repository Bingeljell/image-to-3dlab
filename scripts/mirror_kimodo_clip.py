#!/usr/bin/env python3
"""Mirror a packed Kimodo clip left to right: a strafe left becomes a strafe right.

    python scripts/mirror_kimodo_clip.py image_to_3dlab/motions/strafe_left_b.npz strafe_right.npz

Kimodo writes its 77-joint SOMA skeleton Z-up, where x is the side axis, so a mirror is
x -> -x on the hips and the rest pose, every joint's world rotation turned into its
reflection (w, x, -y, -z), and the left and right joints swapped. Mirroring twice gives
the clip back exactly (bar half-precision rounding), which is what its test checks.

The clip is packed (see pack_kimodo_clip.py): `quats` [T, 77, 4], `joints0` [77, 3],
`hips` [T, 3], `fps`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import SOMA77


def opposites(names: list[str] | None = None) -> np.ndarray:
    """For each joint, the index of the same joint on the other side (itself when central)."""
    names = list(names or SOMA77)
    index = {name: i for i, name in enumerate(names)}
    out = np.arange(len(names))
    for name, i in index.items():
        if "Left" in name:
            out[i] = index[name.replace("Left", "Right")]
        elif "Right" in name:
            out[i] = index[name.replace("Right", "Left")]
    return out


MIRROR = opposites()


def mirror(quats: np.ndarray, joints0: np.ndarray, hips: np.ndarray):
    """(quats, joints0, hips) turned into their left-right mirror."""
    turned = np.asarray(quats)[:, MIRROR].astype(np.float32)
    turned[..., 2] *= -1            # a mirrored rotation is (w, x, -y, -z)
    turned[..., 3] *= -1
    rest = np.asarray(joints0)[MIRROR].astype(np.float32)
    rest[:, 0] *= -1
    track = np.asarray(hips).astype(np.float32)
    track[:, 0] *= -1
    return turned, rest, track


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="packed .npz to mirror")
    parser.add_argument("target", type=Path, help="packed .npz to write")
    args = parser.parse_args(argv)
    data = np.load(args.source)
    quats, joints0, hips = mirror(data["quats"], data["joints0"], data["hips"])
    args.target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.target, quats=quats.astype(np.float16), joints0=joints0, hips=hips,
                        fps=np.float32(data["fps"]))
    print(f"{args.target}: mirrored from {args.source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
