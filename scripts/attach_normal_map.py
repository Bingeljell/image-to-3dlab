#!/usr/bin/env python3
"""Attach a baked normal map to a GLB's material.

Baking produces a PNG, which does nothing on its own — the glTF material has to reference
it as `normalTexture` before any renderer will use it. This is the step that turns a baked
image into visible surface detail.

Kept as a separate pure-Python step rather than folded into the Blender bake, so it
matches how the rest of this repo treats GLB post-processing: Blender does rigging and
rendering, everything that operates *on the file* is headless Python and re-runnable.

The map is written with `Non-Color` semantics in mind — a normal map encodes directions,
not colour, so it must never be colour-managed. glTF handles that automatically for
`normalTexture`, which is one reason to put it in the right slot rather than abuse
another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="GLB to modify")
    parser.add_argument("normal_map", type=Path, help="baked normal map PNG")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="glTF normalTexture scale. Below 1 softens the effect; above 1 exaggerates "
             "it, which usually reads as noise",
    )
    args = parser.parse_args()

    scene = trimesh.load(args.asset, process=False)
    geometries = list(scene.geometry.values())
    if not geometries:
        raise SystemExit(f"no geometry in {args.asset}")

    normal = Image.open(args.normal_map).convert("RGB")

    attached = 0
    for mesh in geometries:
        material = mesh.visual.material
        if material is None:
            continue
        material.normalTexture = normal
        # trimesh's PBRMaterial does not model normalTexture.scale, so record the
        # requested value where the exporter can still see it if it grows support.
        if args.scale != 1.0:
            material.normalScale = args.scale
        attached += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)

    check = trimesh.load(args.output, process=False)
    got = sum(
        1 for m in check.geometry.values()
        if getattr(m.visual.material, "normalTexture", None) is not None
    )
    print(f"attached to {attached} material(s); {got} survived the round-trip")
    print(f"normal map {normal.size[0]}x{normal.size[1]} -> {args.output}")
    if got == 0:
        raise SystemExit("normalTexture did not survive export — check the glTF writer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
