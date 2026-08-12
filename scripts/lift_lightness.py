#!/usr/bin/env python3
"""Brighten a GLB's albedo without shifting its colour.

The exact complement of ``colour_match_albedo.py``: that one grades chroma and refuses
to touch lightness, this one moves lightness and refuses to touch chroma. Together they
are a complete two-knob grade.

**The brightening happens in linear light, not in LAB.** The first version of this
multiplied LAB's L channel, on the reasoning that an RGB multiply would drift the hue.
That reasoning was backwards, and a test caught it: holding a/b fixed while raising L
*desaturates*, because chroma in LAB is absolute while saturation is chroma relative to
lightness. Scaling linear RGB leaves every channel ratio exactly intact, so hue and
saturation survive untouched -- which is what the docstring claimed all along.

**Prefer fixing the lighting first.** Measured 2026-08-11 on the thorn-knot: raising the
render lights and lifting the texture reached the same overall brightness, but the lit
version kept the crevices between the coils dark while the lifted texture washed them
out, because a gain raises the shadow texels that were doing the shading work. The form
flattened and the eye lost separation from the body. A generated albedo is de-lit, and a
de-lit bark albedo sitting near 0.12 mean is *correct*, not broken.

So reach for this only when the lighting is not yours to change -- a fixed dim scene, or
an engine preset -- and expect to pay for it in flatness. Keep the gain modest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

# sRGB transfer function constants, so the gain is applied to light and not to
# gamma-encoded values -- doubling a gamma-encoded texel is not doubling its brightness.
_SRGB_KNEE = 0.04045
_LINEAR_KNEE = 0.0031308


def _to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= _SRGB_KNEE, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= _LINEAR_KNEE, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055
    )


def lift(atlas: Image.Image, gain: float) -> Image.Image:
    """Multiply the albedo's brightness by `gain`, preserving hue and saturation."""
    if gain < 0:
        raise ValueError("gain must be non-negative")
    rgba = np.asarray(atlas.convert("RGBA")).astype(np.float32) / 255.0
    lit = _to_linear(rgba[..., :3]) * gain
    out = np.clip(_to_srgb(np.clip(lit, 0.0, 1.0)) * 255.0, 0, 255)
    alpha = rgba[..., 3:] * 255.0
    return Image.fromarray(
        np.dstack([out, alpha]).astype(np.uint8), mode="RGBA"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--gain", type=float, default=1.3)
    a = p.parse_args()

    scene = trimesh.load(a.asset, process=False)
    for mesh in scene.geometry.values():
        mat = mesh.visual.material
        atlas = getattr(mat, "baseColorTexture", None)
        if atlas is None:
            raise SystemExit(f"{a.asset} has no baseColorTexture")
        before = np.asarray(atlas.convert("RGB")).astype(np.float32)
        mat.baseColorTexture = lift(atlas, a.gain)
        after = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(a.output)
    bm = before[before.max(axis=2) > 8]
    am = after[after.max(axis=2) > 8]
    print(f"gain={a.gain}  mean {bm.mean():.1f} -> {am.mean():.1f}   "
          f"peak {bm.max():.0f} -> {am.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
