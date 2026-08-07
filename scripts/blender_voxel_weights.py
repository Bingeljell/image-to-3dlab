#!/usr/bin/env python3
"""Weight a generated mesh via a watertight voxel proxy, then transfer the result back.

Bone heat weighting diffuses influence across the mesh *surface*, which is what keeps a
leg bone from grabbing an ear that happens to be nearby in straight-line space. It needs
a clean manifold surface to solve on. TRELLIS output is not that: the 101k moss fox has
16,467 boundary edges and 226 components, and binding it directly produces the warning
"Bone Heat Weighting: failed to find solution for one or more bones" and **21 empty
vertex groups** — every weight zero.

The fix is to never solve on the raw mesh:

1. voxel-remesh a copy into a single watertight manifold blob,
2. heat-weight that proxy, which now solves cleanly,
3. transfer the weights back onto the real mesh by nearest-surface interpolation,
4. bind the real mesh to the armature with the transferred groups.

The proxy is throwaway — it only ever carries weights, never geometry, so its ugliness
does not matter. Only the transfer needs to be accurate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_voxel_weights_code import build_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--voxel-size", type=float, default=0.012,
        help="Proxy resolution as a fraction of world units; smaller is finer and slower",
    )
    parser.add_argument("--weight-threshold", type=float, default=0.15)
    parser.add_argument(
        "--keep-proxy", action="store_true",
        help="Leave the voxel proxy in the scene for inspection",
    )
    args = parser.parse_args()

    from blender_joint_markers import send
    code = build_code(args.voxel_size, args.weight_threshold, args.keep_proxy)
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
