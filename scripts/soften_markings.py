#!/usr/bin/env python3
"""Reduce the contrast of flat painted markings in a conditioning image.

**The problem this exists for.** TRELLIS infers shape partly from shading, so a hard dark
line on a light body reads as a shadow, and a shadow implies a crease -- so it carves
one. Verified on Flicker: strip every texture, render plain grey, and its forehead V and
shoulder chevrons are still there as physical cracks with ragged lips. No texture work
fixes that; the fix has to happen before generation.

The irony is that the cleaner and more graphic the artwork, the worse this gets.

**What it does.** Lightens pixels inside a luminance band toward the body's own level,
leaving hue alone. The band matters: on Flicker the body sits near 200, the flat markings
near 60-80, and the **eyes near 26**. Eyes are genuinely recessed and their sockets are
correct geometry, so anything below ``--low`` is left untouched -- soften the paint, keep
the anatomy.

This is a generation-input change, so judging it needs a regeneration run. Look at the
image before spending one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance 0..255 from an HxWx3 array."""
    return rgb.astype(np.float32) @ _LUMA


def band_weight(
    lum: np.ndarray, low: float, high: float, edge: float = 12.0
) -> np.ndarray:
    """1.0 inside [low, high], ramping to 0 across `edge` luminance units at each end.

    Below `low` is protected (eyes, claws); above `high` is already body colour.

    `edge` is a fixed width, deliberately not a fraction of the band. A proportional
    ramp made the transition scale with the band, and with low=40/high=150 that meant
    full strength was only reached around luminance 79 -- straddling exactly the 58-79
    range Flicker's markings occupy, so they would have been two-thirds softened while
    appearing fully softened. A fixed edge keeps the protected zone tight around `low`
    wherever the band is set.
    """
    if not 0.0 <= low < high:
        raise ValueError("require 0 <= low < high")
    if edge <= 0:
        raise ValueError("edge must be positive")
    ramp_in = np.clip((lum - low) / edge, 0.0, 1.0)
    ramp_out = np.clip((high - lum) / edge, 0.0, 1.0)
    return np.minimum(ramp_in, ramp_out).astype(np.float32)


def soften(
    image: Image.Image,
    lighten: float,
    low: float = 40.0,
    high: float = 150.0,
    feather: float = 2.0,
    edge: float = 12.0,
) -> Image.Image:
    """Lighten banded pixels toward the subject's body level by `lighten` (0..1).

    0 leaves the image untouched; 1 takes the markings all the way to body colour and
    erases them. Hue is preserved by scaling RGB uniformly.
    """
    if not 0.0 <= lighten <= 1.0:
        raise ValueError("lighten must be in 0..1")

    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    lum = luminance(rgb)

    subject = alpha > 200
    body = lum[subject & (lum >= high)]
    target = float(np.median(body)) if body.size else float(high)

    weight = band_weight(lum, low, high, edge) * subject
    if feather > 0:
        weight = np.asarray(
            Image.fromarray((weight * 255).astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(feather)
            )
        ).astype(np.float32) / 255.0

    new_lum = lum + (target - lum) * lighten * weight
    scale = np.where(lum > 1e-3, new_lum / np.maximum(lum, 1e-3), 1.0)
    out = np.clip(rgb * scale[..., None], 0, 255)
    return Image.fromarray(
        np.dstack([out, alpha]).astype(np.uint8), mode="RGBA"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lighten", type=float, default=0.5,
                   help="0 leaves markings alone, 1 erases them (default: 0.5)")
    p.add_argument("--low", type=float, default=40.0,
                   help="protect anything darker than this -- eyes, claws (default: 40)")
    p.add_argument("--high", type=float, default=150.0,
                   help="anything brighter is already body colour (default: 150)")
    p.add_argument("--feather", type=float, default=2.0)
    p.add_argument("--edge", type=float, default=12.0,
                   help="luminance width of the protected ramp above --low (default: 12)")
    args = p.parse_args()

    src = Image.open(args.image)
    out = soften(src, args.lighten, args.low, args.high, args.feather, args.edge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)

    before = luminance(np.asarray(src.convert("RGB")).astype(np.float32))
    after = luminance(np.asarray(out.convert("RGB")).astype(np.float32))
    band = (before >= args.low) & (before <= args.high)
    print(f"markings in band: {int(band.sum())} px, "
          f"mean luminance {before[band].mean():.0f} -> {after[band].mean():.0f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
