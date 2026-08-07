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

import numpy as np
import trimesh
from PIL import Image

# A tangent-space normal map encodes "no deviation from the surface" as this colour:
# x and y at zero (128 after the 0..255 encoding) and z pointing straight out (255).
FLAT_NORMAL = (128, 128, 255)


def soften(image: Image.Image, strength: float) -> Image.Image:
    """Blend a normal map toward flat.

    glTF has a `normalTexture.scale` for exactly this, but trimesh does not model it, so
    the value would be silently dropped on export. Blending the pixels instead survives
    any writer.

    Baked at full strength this map reads crunchy — every triangle edge catches light and
    the cream muzzle picks up dark speckling. Strength is the dose: 1.0 is the raw bake,
    0.0 is no effect at all.
    """
    if strength < 0.0:
        raise ValueError("strength must not be negative")
    if strength == 1.0:
        return image
    pixels = np.asarray(image.convert("RGB")).astype(np.float32)
    flat = np.array(FLAT_NORMAL, dtype=np.float32)
    blended = flat + (pixels - flat) * strength
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="GLB to modify")
    parser.add_argument("normal_map", type=Path, help="baked normal map PNG")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Dose of the effect. 1.0 is the raw bake, which usually reads crunchy; "
             "0.4-0.6 is typically the useful range. Applied by blending the map toward "
             "flat, because trimesh drops glTF's normalTexture.scale on export",
    )
    args = parser.parse_args()

    scene = trimesh.load(args.asset, process=False)
    geometries = list(scene.geometry.values())
    if not geometries:
        raise SystemExit(f"no geometry in {args.asset}")

    normal = soften(Image.open(args.normal_map).convert("RGB"), args.strength)

    attached = 0
    for mesh in geometries:
        material = mesh.visual.material
        if material is None:
            continue
        material.normalTexture = normal
        attached += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)

    check = trimesh.load(args.output, process=False)
    got = sum(
        1 for m in check.geometry.values()
        if getattr(m.visual.material, "normalTexture", None) is not None
    )
    print(f"attached to {attached} material(s); {got} survived the round-trip")
    print(f"normal map {normal.size[0]}x{normal.size[1]} at strength "
          f"{args.strength} -> {args.output}")
    if got == 0:
        raise SystemExit("normalTexture did not survive export — check the glTF writer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
