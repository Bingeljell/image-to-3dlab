#!/usr/bin/env python3
"""Grade a generated GLB's albedo toward the colour of its source concept art.

TRELLIS bakes the moss fox as a cool grass green, while the concept art is a warm
yellow-olive: the source leads red over green by about +17, every render by -7 to -20.
The shift is present in the baked albedo itself, so it is neither a material-mode nor a
lighting artefact and can be corrected without regenerating anything.

The correction runs in CIE LAB and touches **only the a/b (chroma) channels**, leaving
L (lightness) alone. That matters: the concept art is *lit* and the albedo is *unlit*,
so their lightness legitimately differs and matching it would flatten the cream-versus-
green structure that the asset depends on. Matching chroma alone moves the hue without
disturbing which parts are light and which are dark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from skimage import color


def _foreground(image: Image.Image) -> np.ndarray:
    """Return the subject's RGB pixels, dropping transparent and near-black padding."""
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    rgb = rgba[..., :3].reshape(-1, 3)
    alpha = rgba[..., 3].reshape(-1)
    # Atlas gutters are unfilled black; concept art is matted with real alpha.
    keep = (alpha > 200) & (rgb.mean(axis=1) >= 32)
    return rgb[keep]


def _chroma_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = color.rgb2lab((rgb / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return lab[:, 1:].mean(axis=0), lab[:, 1:].std(axis=0)


def grade(atlas: Image.Image, source: Image.Image, strength: float) -> Image.Image:
    src_mean, src_std = _chroma_stats(_foreground(source))
    dst_mean, dst_std = _chroma_stats(_foreground(atlas))

    rgba = np.asarray(atlas.convert("RGBA")).astype(np.float32)
    height, width = rgba.shape[:2]
    lab = color.rgb2lab(rgba[..., :3] / 255.0)

    # Rescale chroma to the source's spread, then recentre on the source's mean.
    scale = np.where(dst_std > 1e-4, src_std / dst_std, 1.0)
    corrected = (lab[..., 1:] - dst_mean) * scale + src_mean
    lab[..., 1:] += (corrected - lab[..., 1:]) * strength

    graded = np.clip(color.lab2rgb(lab) * 255.0, 0, 255)
    out = np.dstack([graded, rgba[..., 3:]]).astype(np.uint8)
    return Image.fromarray(out.reshape(height, width, 4), mode="RGBA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="GLB whose albedo should be graded")
    parser.add_argument("source", type=Path, help="Concept art the colour comes from")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="0 leaves the albedo untouched, 1 matches the source fully (default: 1.0)",
    )
    parser.add_argument(
        "--dump-atlas", type=Path, default=None,
        help="Also write the graded atlas as a PNG, for inspection",
    )
    args = parser.parse_args()

    scene = trimesh.load(args.asset, process=False)
    geometries = list(scene.geometry.values())
    if not geometries:
        raise SystemExit(f"no geometry in {args.asset}")

    source = Image.open(args.source)
    for mesh in geometries:
        material = mesh.visual.material
        atlas = getattr(material, "baseColorTexture", None)
        if atlas is None:
            raise SystemExit(f"{args.asset} has no baseColorTexture to grade")
        graded = grade(atlas, source, args.strength)
        material.baseColorTexture = graded
        if args.dump_atlas is not None:
            graded.save(args.dump_atlas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)

    before = _foreground(atlas).mean(axis=0)
    after = _foreground(graded).mean(axis=0)
    reference = _foreground(source).mean(axis=0)
    print(f"source    meanRGB={reference.round(1)}  R-G={reference[0] - reference[1]:+.1f}")
    print(f"before    meanRGB={before.round(1)}  R-G={before[0] - before[1]:+.1f}")
    print(f"after     meanRGB={after.round(1)}  R-G={after[0] - after[1]:+.1f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
