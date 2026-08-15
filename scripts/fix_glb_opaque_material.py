#!/usr/bin/env python3
"""Make an existing TRELLIS GLB opaque and single-sided without rebaking it."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.trellis_backend import read_glb, write_glb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    target = args.output or args.asset
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.asset, args.output)

    gltf, chunks, json_index, version = read_glb(target)
    materials = gltf.get("materials", [])
    if not materials:
        raise ValueError(f"{target} has no materials")
    changes = 0
    for material in materials:
        if material.get("alphaMode", "OPAQUE") != "OPAQUE":
            material["alphaMode"] = "OPAQUE"
            changes += 1
        if material.get("doubleSided", False):
            material["doubleSided"] = False
            changes += 1
    write_glb(target, gltf, chunks, json_index, version)
    print(f"{target}: {changes} material flag change(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
