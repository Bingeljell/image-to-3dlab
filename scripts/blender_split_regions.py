#!/usr/bin/env python3
"""Split a generated mesh into per-region material slots, each with its own texture.

Run (live): python scripts/blender_split_regions.py model.glb out_dir/

**2048 is the cap per *material*, not per model.** The whole fox currently shares one
2048 atlas, so at 101k faces every triangle gets ~41 texels and the head — where detail
matters most — receives a small share of a fixed budget. Giving the head, body and tail
their own material slot and their own 2048 map triples the budget outright, without
changing a single generation setting. Game characters normally ship this way.

Measured on the moss fox with the default cuts:

    head   35,824 faces   41 -> 117 texels/triangle   2.8x
    body   36,827 faces   41 -> 114 texels/triangle   2.8x
    tail   28,647 faces   41 -> 146 texels/triangle   3.5x

How it works: each region is unwrapped independently so it fills the whole 0..1 UV
square, then the original albedo is baked into a fresh 2048 map per region. Regions
overlap in UV space, which is fine and is the point — they sample different images.

The source material must sample through an explicit UV Map node pointing at the
*original* layout, because the bake target uses the *active* layout. Without that the
bake reads from the new UVs it is writing into and produces garbage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_split_regions_worker as worker
from blender_joint_markers import send
from blender_worker_source import build_worker_code

REGION_NAMES = worker.REGION_NAMES
classify_faces = worker.classify_faces
region_counts = worker.region_counts
texel_density = worker.texel_density


def blender_code(
    asset: Path, out_dir: Path, size: int, head_cut: float, tail_cut: float,
) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_split_regions_worker.py"),
        {
            "asset_path": str(asset), "out_dir": str(out_dir), "size": size,
            "head_cut": head_cut, "tail_cut": tail_cut,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--head-cut", type=float, default=0.25,
        help="Position on the front-back axis above which faces are head",
    )
    parser.add_argument(
        "--tail-cut", type=float, default=-0.10,
        help="Position below which faces are tail",
    )
    args = parser.parse_args()

    if args.head_cut <= args.tail_cut:
        raise SystemExit("--head-cut must be greater than --tail-cut")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    code = blender_code(
        args.asset.expanduser().resolve(),
        args.out_dir.expanduser().resolve(),
        args.size, args.head_cut, args.tail_cut,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
