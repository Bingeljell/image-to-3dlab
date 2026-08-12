#!/usr/bin/env python3
"""Give a generated asset a surface: normal relief and roughness variation.

TRELLIS emits a base colour texture and nothing else. Its metallic-roughness image is
near-constant (red channel identically 0, roughness ~0.93 everywhere) and the material
never references it, so a generated asset arrives with a correct silhouette, correct
colour, and no surface at all: every part of it is uniformly matte, so nothing catches
light and the whole mass reads as one lump.

Two of the three fixes are derivable from the albedo alone and live here. The third,
ambient occlusion, is real geometric information and has to be baked in Blender --
see ``blender_bake_ao.py``.

**These two are inventions, not measurements.** A normal map derived from albedo
luminance turns every dark mark into a groove, whether or not it is one; a light stain
on flat bark becomes a raised ridge. That is usually flattering on organic subjects and
wrong on painted detail such as an eye, which is why ``--gloss-mask`` exists and why the
strength knobs default low. Judge them by eye, never by the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance in 0..1 from an HxWx3 array of 0..255 values."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights) / 255.0


def _blur(height: np.ndarray, radius: int) -> np.ndarray:
    """Cheap separable box blur, used to keep only relief coarser than `radius`."""
    if radius < 1:
        return height
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(height, ((0, 0), (radius, radius)), mode="edge")
    out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, padded)


def normal_from_height(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (HxWx3, uint8) from a 0..1 height field.

    Central differences give the surface gradient; the normal is the vector
    perpendicular to it. Encoded in glTF's convention: +X right, +Y up, +Z out,
    mapped from -1..1 into 0..255, so flat ground is the familiar (128, 128, 255).
    """
    if height.ndim != 2:
        raise ValueError("height must be a 2-D array")

    # np.gradient returns d/drow, d/dcol; v is flipped because image rows run downward
    # while the tangent-space Y axis runs up.
    dy, dx = np.gradient(height.astype(np.float32))
    nx = -dx * strength
    ny = dy * strength
    nz = np.ones_like(nx)

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    stacked = np.dstack([nx / length, ny / length, nz / length])
    return np.clip((stacked * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def roughness_from_luminance(
    lum: np.ndarray, low: float = 0.55, high: float = 0.95
) -> np.ndarray:
    """Map luminance to roughness in [low, high], darkest pixels being roughest.

    Crevices and damp bark are dark and scatter light; exposed ridges are lighter and
    catch a sheen. Inverting luminance is a crude stand-in for that, but it produces
    variation where there was a single constant, which is the point.
    """
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("require 0 <= low <= high <= 1")
    lo, hi = float(lum.min()), float(lum.max())
    spread = (lum - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(lum)
    return (high - spread * (high - low)).astype(np.float32)


def pack_metallic_roughness(
    roughness: np.ndarray, occlusion: np.ndarray | None = None
) -> Image.Image:
    """glTF packs occlusion in R, roughness in G, metallic in B. Metallic stays 0.

    One image serves as both ``metallicRoughnessTexture`` and ``occlusionTexture``,
    which is the standard ORM layout and saves shipping a second 2048 map. With no
    occlusion supplied R stays 0; a renderer only reads R when the material actually
    declares an occlusionTexture, so leaving it dark is safe.
    """
    h, w = roughness.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 1] = np.clip(roughness * 255.0, 0, 255).astype(np.uint8)
    if occlusion is not None:
        if occlusion.shape != roughness.shape:
            raise ValueError("occlusion and roughness must be the same size")
        out[..., 0] = np.clip(occlusion * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def load_occlusion(path: Path, size: tuple[int, int], strength: float) -> np.ndarray:
    """Read a baked AO map as 0..1, resized to the atlas, with strength dialled back.

    ``strength`` 0 leaves the surface fully lit, 1 uses the bake as-is. Blending toward
    white rather than scaling keeps the open areas open instead of greying everything.
    """
    ao = np.asarray(Image.open(path).convert("L").resize(size, Image.LANCZOS))
    ao = ao.astype(np.float32) / 255.0
    return 1.0 - (1.0 - ao) * strength


def apply_gloss(
    roughness: np.ndarray, mask: np.ndarray, gloss_roughness: float
) -> np.ndarray:
    """Force `roughness` down to `gloss_roughness` wherever `mask` (0..1) is set.

    A generated asset is one material, so an eye ends up exactly as matte as bark -- and
    a matte eye is a dead eye, because the specular highlight is what reads as wet. This
    is the smallest change that brings one back.
    """
    if mask.shape != roughness.shape:
        raise ValueError("mask and roughness must be the same size")
    if not 0.0 <= gloss_roughness <= 1.0:
        raise ValueError("gloss_roughness must be in 0..1")
    return roughness * (1.0 - mask) + gloss_roughness * mask


def build_maps(
    albedo: Image.Image,
    normal_strength: float,
    detail_radius: int,
    rough_low: float,
    rough_high: float,
    occlusion: np.ndarray | None = None,
    gloss_mask: np.ndarray | None = None,
    gloss_roughness: float = 0.12,
) -> tuple[Image.Image, Image.Image]:
    """Derive (normal map, ORM map) from a base colour texture and optional baked AO."""
    rgb = np.asarray(albedo.convert("RGB"))
    lum = luminance(rgb)

    # Subtracting a blurred copy keeps fine relief (bark cracks) and discards the broad
    # light-to-dark drift across a coil, which is form, not surface, and would otherwise
    # dish the whole shape.
    height = lum - _blur(lum, detail_radius) + 0.5

    if gloss_mask is not None:
        # Painted detail is not relief. Deriving height from an eye's paint embosses its
        # iris into bumps and makes the one smooth thing on the model look crusty, so
        # flatten the masked region back to neutral before the gradient is taken.
        height = height * (1.0 - gloss_mask) + 0.5 * gloss_mask

    rough = roughness_from_luminance(lum, rough_low, rough_high)
    if gloss_mask is not None:
        rough = apply_gloss(rough, gloss_mask, gloss_roughness)

    normal = Image.fromarray(normal_from_height(height, normal_strength), mode="RGB")
    return normal, pack_metallic_roughness(rough, occlusion)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", type=Path, help="GLB to give a surface to")
    p.add_argument("output", type=Path, help="GLB to write")
    p.add_argument("--normal-strength", type=float, default=6.0)
    p.add_argument("--detail-radius", type=int, default=6,
                   help="relief coarser than this many texels is treated as form, not surface")
    p.add_argument("--rough-low", type=float, default=0.55)
    p.add_argument("--rough-high", type=float, default=0.95)
    p.add_argument("--ao", type=Path, default=None,
                   help="baked AO map from blender_bake_ao.py, packed into the R channel")
    p.add_argument("--ao-strength", type=float, default=1.0,
                   help="0 ignores the bake, 1 uses it as baked (default: 1.0)")
    p.add_argument("--gloss-mask", type=Path, default=None,
                   help="UV mask from feature_mask.py; masked texels go glossy and flat")
    p.add_argument("--gloss-roughness", type=float, default=0.12,
                   help="roughness inside the gloss mask (default: 0.12, wet)")
    p.add_argument("--dump-maps", type=Path, default=None,
                   help="directory to also write the maps into, for inspection")
    args = p.parse_args()

    scene = trimesh.load(args.asset, process=False)
    for mesh in scene.geometry.values():
        material = mesh.visual.material
        albedo = getattr(material, "baseColorTexture", None)
        if albedo is None:
            raise SystemExit(f"{args.asset} has no baseColorTexture to derive from")

        occlusion = None
        if args.ao is not None:
            occlusion = load_occlusion(args.ao, albedo.size, args.ao_strength)

        gloss = None
        if args.gloss_mask is not None:
            gloss = np.asarray(
                Image.open(args.gloss_mask).convert("L").resize(albedo.size, Image.LANCZOS)
            ).astype(np.float32) / 255.0

        normal, rough = build_maps(
            albedo, args.normal_strength, args.detail_radius,
            args.rough_low, args.rough_high, occlusion, gloss, args.gloss_roughness,
        )
        material.normalTexture = normal
        material.metallicRoughnessTexture = rough
        material.metallicFactor = 0.0
        material.roughnessFactor = 1.0
        if occlusion is not None:
            material.occlusionTexture = rough

        if args.dump_maps is not None:
            args.dump_maps.mkdir(parents=True, exist_ok=True)
            normal.save(args.dump_maps / "normal.png")
            rough.save(args.dump_maps / "metallic_roughness.png")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)
    print(f"wrote {args.output}  (normal strength {args.normal_strength}, "
          f"roughness {args.rough_low}-{args.rough_high})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
