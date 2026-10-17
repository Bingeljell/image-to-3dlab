#!/usr/bin/env python3
"""Apply a reproducible living-organic material recipe to an existing GLB.

This is deliberately a *fast finishing* operation. It changes texture maps and material
parameters while leaving every position and triangle untouched. The original GLB remains the
master; the output is a derived candidate that can be recreated from the JSON recipe.

The treatment has three spatial lanes:

* body: non-metallic, rough, and chroma-shifted toward living wood without globally matching
  the source image's painted lightness;
* moss: a UV mask derived from upward-facing mesh normals, optionally intersected with the
  green chroma evidence already present in TRELLIS's atlas;
* eye: a supplied UV mask recoloured amber while preserving a dark pupil, with low roughness.

The eye mask should come from ``feature_mask.py`` after a visible point is raycast with
``blender_pick_pixel.py``. A hand-drawn screen mask can later feed the same recipe once it is
projected to UV space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from skimage import color


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_colour(value: str) -> np.ndarray:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"colour must be #RRGGBB, got {value!r}")
    try:
        return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255
    except ValueError as exc:
        raise ValueError(f"colour must be #RRGGBB, got {value!r}") from exc


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    if high <= low:
        raise ValueError("smoothstep high must exceed low")
    unit = np.clip((values - low) / (high - low), 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def rasterise_face_values(
    uv: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    size: tuple[int, int],
    blur: float = 2.0,
) -> Image.Image:
    """Rasterise scalar per-face values into the mesh's UV atlas."""
    width, height = size
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for face, value in zip(faces, values):
        shade = int(np.clip(value, 0.0, 1.0) * 255)
        if shade <= 1:
            continue
        points = [
            (float(uv[index, 0]) * (width - 1), (1.0 - float(uv[index, 1])) * (height - 1))
            for index in face
        ]
        draw.polygon(points, fill=shade)
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return image


def moss_mask(
    mesh: trimesh.Trimesh,
    size: tuple[int, int],
    up_axis: int = 1,
    threshold: float = 0.15,
    softness: float = 0.45,
    blur: float = 2.0,
) -> Image.Image:
    """Build a spatial moss mask from upward-facing triangles in glTF's Y-up space."""
    if mesh.visual.uv is None:
        raise ValueError("mesh has no UV coordinates")
    weights = smoothstep(threshold, threshold + softness, mesh.face_normals[:, up_axis])
    return rasterise_face_values(mesh.visual.uv, mesh.faces, weights, size, blur)


def colourise_preserving_lightness(
    rgb: np.ndarray,
    target: np.ndarray,
    amount: np.ndarray | float,
) -> np.ndarray:
    """Move LAB chroma toward ``target`` while preserving the image's L channel."""
    lab = color.rgb2lab(np.clip(rgb, 0.0, 1.0))
    target_lab = color.rgb2lab(target.reshape(1, 1, 3))[0, 0]
    weight = np.asarray(amount, dtype=np.float32)
    if weight.ndim == 2:
        weight = weight[..., None]
    lab[..., 1:] += (target_lab[1:] - lab[..., 1:]) * weight
    return np.clip(color.lab2rgb(lab), 0.0, 1.0)


def moss_chroma_evidence(
    rgb: np.ndarray,
    hue_threshold: float = 0.175,
    hue_softness: float = 0.030,
    saturation_threshold: float = 0.25,
    saturation_softness: float = 0.25,
) -> np.ndarray:
    """Return moss-like atlas evidence from yellow-green hue and useful saturation.

    The generated Snag atlas is dark, but it still distinguishes warmer bark pixels from
    greener pixels. This extracts that distinction without changing luminance. It is only a
    supporting signal: face direction continues to supply the spatial prior.
    """
    hsv = color.rgb2hsv(np.clip(rgb, 0.0, 1.0))
    hue = smoothstep(hue_threshold, hue_threshold + hue_softness, hsv[..., 0])
    saturation = smoothstep(
        saturation_threshold,
        saturation_threshold + saturation_softness,
        hsv[..., 1],
    )
    return hue * saturation


def apply_recipe(
    mesh: trimesh.Trimesh,
    eye_mask_image: Image.Image,
    body_colour: np.ndarray,
    moss_colour: np.ndarray,
    eye_colour: np.ndarray,
    body_strength: float,
    moss_strength: float,
    eye_strength: float,
    lightness_gamma: float,
    eye_lightness: float,
    rough_low: float,
    rough_high: float,
    eye_roughness: float,
    moss_up_threshold: float,
    moss_up_softness: float,
    moss_blur: float,
    moss_chroma_strength: float = 0.0,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    material = mesh.visual.material
    atlas = getattr(material, "baseColorTexture", None)
    if atlas is None:
        raise ValueError("mesh has no baseColorTexture")
    rgba = np.asarray(atlas.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    valid = (rgb.max(axis=2) > 8 / 255.0).astype(np.float32)

    eye = np.asarray(eye_mask_image.convert("L").resize(atlas.size, Image.Resampling.LANCZOS),
                     dtype=np.float32) / 255.0
    eye *= valid
    moss_image = moss_mask(
        mesh, atlas.size, threshold=moss_up_threshold,
        softness=moss_up_softness, blur=moss_blur,
    )
    moss = np.asarray(moss_image, dtype=np.float32) / 255.0
    if not 0 <= moss_chroma_strength <= 1:
        raise ValueError("moss chroma strength must be in [0, 1]")
    chroma = moss_chroma_evidence(rgb)
    moss *= (1.0 - moss_chroma_strength) + moss_chroma_strength * chroma
    moss *= valid * (1.0 - eye)
    moss_image = Image.fromarray(np.clip(moss * 255, 0, 255).astype(np.uint8), "L")

    # Body warmth, then spatial green moss. Both preserve L, so painted crevices stay dark.
    finished = colourise_preserving_lightness(rgb, body_colour, body_strength * valid)
    finished = colourise_preserving_lightness(
        finished, moss_colour, moss_strength * moss,
    )
    finished = colourise_preserving_lightness(
        finished, eye_colour, eye_strength * eye,
    )

    lab = color.rgb2lab(finished)
    if lightness_gamma <= 0:
        raise ValueError("lightness gamma must be positive")
    lab[..., 0] = 100.0 * np.power(np.clip(lab[..., 0] / 100.0, 0.0, 1.0), lightness_gamma)

    # Lift only iris midtones/highlights. The smooth gate leaves a dark pupil dark.
    iris_gate = smoothstep(10.0, 28.0, lab[..., 0]) * eye
    lab[..., 0] += iris_gate * eye_lightness * (1.0 - lab[..., 0] / 100.0)
    finished = np.clip(color.lab2rgb(lab), 0.0, 1.0)
    finished *= valid[..., None]

    output_rgba = np.dstack([finished, rgba[..., 3]])
    base = Image.fromarray(np.clip(output_rgba * 255, 0, 255).astype(np.uint8), "RGBA")

    luminance = finished @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lo, hi = float(luminance[valid > 0].min()), float(luminance[valid > 0].max())
    spread = np.clip((luminance - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    roughness = rough_high - spread * (rough_high - rough_low)
    # Damp moss remains rough but not chalky; the eye is the only intentionally glossy region.
    roughness = roughness * (1.0 - moss * 0.25) + 0.78 * moss * 0.25
    roughness = roughness * (1.0 - eye) + eye_roughness * eye
    orm = np.zeros((*roughness.shape, 3), dtype=np.uint8)
    orm[..., 1] = np.clip(roughness * 255, 0, 255).astype(np.uint8)
    # B is metalness and remains zero everywhere, including the reflective dielectric eye.
    metallic_roughness = Image.fromarray(orm, "RGB")
    return base, metallic_roughness, moss_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eye-mask", type=Path, required=True)
    parser.add_argument("--body-colour", default="#6f4521")
    parser.add_argument("--moss-colour", default="#55742e")
    parser.add_argument("--eye-colour", default="#d98208")
    parser.add_argument("--body-strength", type=float, default=0.52)
    parser.add_argument("--moss-strength", type=float, default=0.72)
    parser.add_argument("--eye-strength", type=float, default=0.96)
    parser.add_argument("--lightness-gamma", type=float, default=0.94)
    parser.add_argument("--eye-lightness", type=float, default=22.0)
    parser.add_argument("--rough-low", type=float, default=0.62)
    parser.add_argument("--rough-high", type=float, default=0.94)
    parser.add_argument("--eye-roughness", type=float, default=0.18)
    parser.add_argument("--moss-up-threshold", type=float, default=0.15)
    parser.add_argument("--moss-up-softness", type=float, default=0.45)
    parser.add_argument("--moss-blur", type=float, default=2.0)
    parser.add_argument(
        "--moss-chroma-strength",
        type=float,
        default=0.0,
        help="intersect the directional mask with green evidence in the original atlas (0-1)",
    )
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()

    for name in ("body_strength", "moss_strength", "eye_strength", "moss_chroma_strength"):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 <= args.rough_low <= args.rough_high <= 1:
        raise SystemExit("require 0 <= rough-low <= rough-high <= 1")
    if not 0 <= args.eye_roughness <= 1:
        raise SystemExit("--eye-roughness must be in [0, 1]")

    scene = trimesh.load(args.asset, process=False)
    geometries = list(scene.geometry.values())
    if len(geometries) != 1:
        raise SystemExit(f"expected one textured geometry, found {len(geometries)}")
    mesh = geometries[0]
    eye_mask_image = Image.open(args.eye_mask)
    base, mr, generated_moss_mask = apply_recipe(
        mesh,
        eye_mask_image,
        parse_colour(args.body_colour),
        parse_colour(args.moss_colour),
        parse_colour(args.eye_colour),
        args.body_strength,
        args.moss_strength,
        args.eye_strength,
        args.lightness_gamma,
        args.eye_lightness,
        args.rough_low,
        args.rough_high,
        args.eye_roughness,
        args.moss_up_threshold,
        args.moss_up_softness,
        args.moss_blur,
        args.moss_chroma_strength,
    )
    material = mesh.visual.material
    material.baseColorTexture = base
    material.metallicRoughnessTexture = mr
    material.metallicFactor = 0.0
    material.roughnessFactor = 1.0
    material.alphaMode = "OPAQUE"
    material.doubleSided = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        base.save(args.dump_dir / "basecolor.png")
        mr.save(args.dump_dir / "metallic_roughness.png")
        generated_moss_mask.save(args.dump_dir / "moss_mask.png")
        eye_mask_image.resize(base.size, Image.Resampling.NEAREST).save(args.dump_dir / "eye_mask.png")

    recipe = {
        "schema_version": 1,
        "operation": "living_organic_material",
        "input": {"path": str(args.asset), "sha256": _sha256(args.asset)},
        "eye_mask": {"path": str(args.eye_mask), "sha256": _sha256(args.eye_mask)},
        "parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"asset", "output", "eye_mask", "dump_dir", "recipe"}
        },
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
    }
    recipe["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in recipe["parameters"].items()
    }
    recipe_path = args.recipe or args.output.with_suffix(".recipe.json")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"recipe {recipe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
