#!/usr/bin/env python3
"""Paint the source image's markings back onto a generated mesh's texture.

**Why this exists.** High-contrast painted markings in a conditioning image get built as
*geometry* -- a dark line reads as a shadow, a shadow implies a crease, so the generator
carves one, and the crease then tears (see `docs/conditioning-images.md`). Softening the
markings before generation stops the carving, but because a single image drives both the
shape and the paint, it also removes the markings from the texture. Erase them completely
and you get the geometry we want and a blank creature.

So the markings have to go back on *as paint*, which is what they always were. Paint can
be added to a finished texture; a carved groove cannot be uncarved.

**How.** Every vertex is projected into the source image using the same square-crop,
orthographic geometry TRELLIS itself conditions on (`scripts/project_labels.py` does this
and is already validated against the silhouette). Those per-vertex image coordinates are
then rasterised across each UV triangle, giving every texel in the atlas a position in the
source image. Sample there, decide how strongly that pixel reads as a marking, and blend.

Interpolating the *projected coordinates* across a triangle is exact for an orthographic
projection, which is why this does not need a per-texel ray cast.

**What it deliberately does not do.** It only paints what the source view can see. A
three-quarter reference leaves the far side unpainted; `--mirror` handles that for
bilaterally symmetric subjects by sampling the mirrored position instead.

Validate before trusting it. ``--validate`` replaces the albedo with the projected source
colours outright: the asset should come out looking like the source photo wrapped onto it.
If that is scrambled or offset, the yaw is wrong and no marking blend will save it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scripts.project_labels import crop_box, project

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def uv_to_texel(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """UV coordinates in 0..1 to float texel coordinates, honouring the V flip.

    glTF puts UV origin at the top-left of the image while the V axis runs upward, so
    the row index is ``(1 - v)``. Getting this backwards mirrors every marking
    vertically, which looks plausible on a roughly symmetric creature -- hence the
    explicit test.
    """
    return np.stack(
        [uv[:, 0] * (width - 1), (1.0 - uv[:, 1]) * (height - 1)], axis=1
    ).astype(np.float32)


def marking_strength(
    rgb: np.ndarray, low: float = 40.0, high: float = 150.0, edge: float = 12.0
) -> np.ndarray:
    """How strongly a sampled source colour reads as a painted marking, 0..1.

    Deliberately the same luminance band as `soften_markings.py`, because this is its
    inverse: whatever that lightened on the way in, this restores on the way out. Pixels
    darker than ``low`` are eyes and claws -- they are real features the generator got
    right, and repainting them would double-darken already-correct geometry.
    """
    lum = rgb.astype(np.float32) @ _LUMA
    ramp_in = np.clip((lum - low) / edge, 0.0, 1.0)
    ramp_out = np.clip((high - lum) / edge, 0.0, 1.0)
    return np.minimum(ramp_in, ramp_out).astype(np.float32)


def barycentric_fill(
    triangle: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texels covered by one triangle, with their barycentric weights.

    Returns (cols, rows, weights) where weights is (n, 3). Samples at pixel centres and
    keeps anything touching the triangle, with a small tolerance so that adjacent
    triangles do not leave a hairline of unpainted texels along their shared edge.
    """
    x0 = max(int(np.floor(triangle[:, 0].min())), 0)
    x1 = min(int(np.ceil(triangle[:, 0].max())), width - 1)
    y0 = max(int(np.floor(triangle[:, 1].min())), 0)
    y1 = min(int(np.ceil(triangle[:, 1].max())), height - 1)
    if x1 < x0 or y1 < y0:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, empty_i, np.empty((0, 3), dtype=np.float32)

    cols, rows = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float32),
        np.arange(y0, y1 + 1, dtype=np.float32),
        indexing="xy",
    )
    px, py = cols.ravel(), rows.ravel()

    (ax, ay), (bx, by), (cx, cy) = triangle
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denominator) < 1e-12:  # degenerate triangle in UV space
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, empty_i, np.empty((0, 3), dtype=np.float32)

    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
    w2 = 1.0 - w0 - w1

    tolerance = -1e-3
    inside = (w0 >= tolerance) & (w1 >= tolerance) & (w2 >= tolerance)
    return (
        px[inside].astype(np.int32),
        py[inside].astype(np.int32),
        np.stack([w0[inside], w1[inside], w2[inside]], axis=1).astype(np.float32),
    )


def rasterize(
    texel_uv: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the atlas by interpolating per-vertex values across each UV triangle.

    Returns (values, covered) where values is (height, width, channels).
    """
    channels = vertex_values.shape[1]
    out = np.zeros((height, width, channels), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)

    for face in faces:
        cols, rows, weights = barycentric_fill(texel_uv[face], width, height)
        if cols.size == 0:
            continue
        out[rows, cols] = weights @ vertex_values[face]
        covered[rows, cols] = True
    return out, covered


def strength_from_pair(
    original: np.ndarray, erased: np.ndarray, high: float = 150.0
) -> np.ndarray:
    """Recover the exact marking mask by comparing the two conditioning images.

    **Why not just threshold on darkness.** `marking_strength` calls any mid-dark pixel a
    marking, but the artwork is a *rendered* image: its own shadows -- under the chin, the
    shaded flank -- sit in the same luminance band as its paint. Thresholding paints those
    shadows into the albedo, and the result looks like crumpled paper. This is the same
    paint-versus-shading confusion that made softening tear the ears, one level up.

    We do not have to guess, because softening is what removed the markings in the first
    place. It lightened each pixel toward the body level by ``weight``:

        erased = original + (target - original) * weight

    so ``weight`` inverts exactly. Whatever softening treated as paint -- and nothing else,
    including anything a ``--protect`` mask spared -- comes back.
    """
    original_lum = original.astype(np.float32) @ _LUMA
    erased_lum = erased.astype(np.float32) @ _LUMA
    body = original_lum[original_lum >= high]
    target = float(np.median(body)) if body.size else float(high)
    headroom = target - original_lum
    weight = np.where(
        np.abs(headroom) > 1e-3, (erased_lum - original_lum) / np.where(np.abs(headroom) > 1e-3, headroom, 1.0), 0.0
    )
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def marking_ratio(
    original: np.ndarray, erased: np.ndarray, floor: float = 8.0
) -> np.ndarray:
    """Per-channel multiplier that turns unmarked body colour into marked colour.

    **Why a ratio rather than the colour itself.** The artwork is a rendered image, so its
    pixels carry its own lighting -- highlights along a flank, falloff into shadow.
    Painting those absolute colours into the albedo bakes a second lighting pass into a
    texture that the renderer will then light again, which is what made the surface look
    like crumpled paper and left the markings a flat dead grey.

    The erased conditioning image is exactly this artwork *with the markings removed*, so
    dividing one by the other cancels the lighting and leaves only what the marking did:
    a darkening factor. Multiplying the generated albedo by that keeps the generated
    shading intact and applies the marking on top, which is what paint actually does.

    Clipped to at most 1 because markings only ever darken; a ratio above 1 would be
    noise in the near-black cores.
    """
    original = original.astype(np.float32)
    erased = erased.astype(np.float32)
    return np.clip(original / np.maximum(erased, floor), 0.0, 1.0)


def dark_core_strength(rgb: np.ndarray, low: float = 40.0, edge: float = 12.0) -> np.ndarray:
    """Full strength for pixels *darker* than ``low`` -- the cores of the markings.

    Softening deliberately protects everything below ``low`` so it does not lighten eyes
    and claws. The side effect is that the darkest heart of each marking is protected too,
    so it is neither erased before generation nor recovered by `strength_from_pair` --
    which is why repainting from the pair alone leaves grey markings with black missing.

    Painting these back is safe even though it also covers the eyes: the eyes are already
    dark in the generated albedo, so writing the source's dark colour over them changes
    nothing visible. Erasing them would have been destructive; painting them is not.
    """
    lum = rgb.astype(np.float32) @ _LUMA
    return np.clip((low - lum) / edge + 1.0, 0.0, 1.0).astype(np.float32)


def combine_strength(pair: np.ndarray, cores: np.ndarray) -> np.ndarray:
    """Union of the two masks: whatever softening removed, plus the cores it spared."""
    return np.clip(np.maximum(pair, cores), 0.0, 1.0)


def sample_image(pixels: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample an image at normalised coordinates.

    Sampling happens **per texel**, using coordinates interpolated across the triangle,
    rather than per vertex. That distinction is the whole point: a vertex-sampled colour
    interpolated across a face blurs every marking edge to the width of a triangle, which
    turns knife-edge artwork into grey smudge. Interpolating the *coordinates* instead is
    exact for an orthographic projection and keeps the source's full resolution.
    """
    height, width = pixels.shape[:2]
    x = np.clip((u * width).astype(np.int32), 0, width - 1)
    y = np.clip((v * height).astype(np.int32), 0, height - 1)
    return pixels[y, x]


def dilate_into_gutter(
    values: np.ndarray, covered: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bleed painted texels outward into the unpainted gutter around each UV island.

    A UV atlas is a patchwork of islands with unused space between them. Paint that stops
    exactly at an island's edge tears when the renderer filters across that boundary --
    it blends painted texels with untouched gutter, which shows up as ragged, papery
    edges along every seam. Every baker pads its islands for this reason; ours has to too.

    Nearest-neighbour fill, so a marking bleeds its own colour outward rather than
    averaging with whatever is adjacent in atlas space (which is usually an unrelated
    part of the body).
    """
    from scipy import ndimage

    if radius <= 0 or covered.all() or not covered.any():
        return values, covered
    distance, indices = ndimage.distance_transform_edt(
        ~covered, return_indices=True, return_distances=True
    )
    grown = covered | (distance <= radius)
    filled = values[indices[0], indices[1]]
    return np.where(grown[..., None], filled, values), grown


def blend_markings(
    albedo: np.ndarray, colours: np.ndarray, strength: np.ndarray, amount: float = 1.0
) -> np.ndarray:
    """Lerp the albedo toward the projected source colour, by strength.

    The marking's own colour comes from the source rather than a constant, so a marking
    that shades from charcoal to grey stays that way instead of flattening to one tone.
    """
    weight = np.clip(strength * amount, 0.0, 1.0)[..., None]
    return np.clip(albedo * (1.0 - weight) + colours * weight, 0, 255)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if getattr(mesh.visual, "uv", None) is None:
        raise ValueError(f"{path} has no UV coordinates -- cannot paint its texture")
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh", type=Path, help="the generated .glb to paint")
    parser.add_argument("source", type=Path, help="the original artwork, markings intact")
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", default="z", choices=("z", "x"))
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="degrees to turn the projection, for a three-quarter source")
    parser.add_argument("--flip-h", action="store_true")
    parser.add_argument("--flip-v", action="store_true")
    parser.add_argument("--flip-depth", action="store_true")
    parser.add_argument("--depth-buffer", type=int, default=0,
                        help="bin count for the depth-buffer occlusion test (0 = ray cast)")
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    parser.add_argument("--amount", type=float, default=1.0,
                        help="how far to push the albedo toward the source marking colour")
    parser.add_argument("--low", type=float, default=40.0)
    parser.add_argument("--high", type=float, default=150.0)
    parser.add_argument("--erased", type=Path, default=None,
                        help="the softened conditioning image the mesh was generated from. "
                             "Strongly preferred: comparing it against the source recovers "
                             "the exact marking mask, instead of guessing from darkness and "
                             "mistaking the artwork's own shadows for paint")
    parser.add_argument("--dilate", type=int, default=6,
                        help="texels to bleed the paint into the gutter around each UV "
                             "island. Zero leaves ragged, papery seams wherever the "
                             "renderer filters across an island edge")
    parser.add_argument("--no-dark-cores", action="store_true",
                        help="do not repaint pixels darker than --low. Those are the cores "
                             "of the markings, which softening spares to protect eyes and "
                             "claws; without them the markings come back grey")
    parser.add_argument("--mirror", action="store_true",
                        help="also paint the far side by sampling the mirrored position; "
                             "only valid for bilaterally symmetric subjects")
    parser.add_argument("--validate", action="store_true",
                        help="replace the albedo with the raw projected colours, to check "
                             "the projection lines up before trusting a marking blend")
    parser.add_argument("--dump-strength", type=Path, default=None,
                        help="write the marking-strength atlas as a PNG, for inspection")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    source = Image.open(args.source).convert("RGBA")

    _, visible, u, v, _ = project(
        mesh, source, args.axis, args.flip_h, args.flip_v, args.flip_depth,
        args.depth_buffer, args.depth_tolerance, yaw=args.yaw,
    )
    if args.mirror:
        mirrored = mesh.copy()
        mirrored.vertices[:, 0] = -mirrored.vertices[:, 0]
        _, back_visible, back_u, back_v, _ = project(
            mirrored, source, args.axis, args.flip_h, args.flip_v, args.flip_depth,
            args.depth_buffer, args.depth_tolerance, yaw=args.yaw,
        )
        take = back_visible & ~visible
        u = np.where(take, back_u, u)
        v = np.where(take, back_v, v)
        visible = visible | back_visible

    texture = mesh.visual.material.baseColorTexture.convert("RGB")
    width, height = texture.size
    albedo = np.array(texture).astype(np.float32)

    # Rasterise the projected COORDINATES, not the sampled colours -- see sample_image.
    texel_uv = uv_to_texel(np.asarray(mesh.visual.uv), width, height)
    per_vertex = np.stack(
        [u.astype(np.float32), v.astype(np.float32), visible.astype(np.float32)], axis=1
    )
    baked, covered = rasterize(texel_uv, mesh.faces, per_vertex, width, height)

    box = crop_box(source)
    crop = source.crop(box)
    pixels = np.array(crop.convert("RGB"))
    alpha = np.array(crop.convert("RGBA"))[..., 3]

    if args.erased:
        erased = np.array(Image.open(args.erased).convert("RGBA").crop(box).convert("RGB"))
        ratio = marking_ratio(pixels, erased)
        source_strength = strength_from_pair(pixels, erased, args.high)
        if not args.no_dark_cores:
            source_strength = combine_strength(
                source_strength, dark_core_strength(pixels, args.low)
            )
    else:
        ratio = None
        source_strength = None

    # Pad the coordinate map into the gutter BEFORE sampling, so bled texels carry a real
    # projected position rather than a colour averaged across an island boundary.
    baked, covered = dilate_into_gutter(baked, covered, args.dilate)

    projected_colour = sample_image(pixels, baked[..., 0], baked[..., 1]).astype(np.float32)
    on_subject = sample_image(alpha[..., None], baked[..., 0], baked[..., 1])[..., 0] > 204
    seen = covered & (baked[..., 2] > 0.5) & on_subject

    if args.validate:
        painted = np.where(seen[..., None], projected_colour, albedo)
    else:
        if source_strength is not None:
            strength = sample_image(
                source_strength[..., None], baked[..., 0], baked[..., 1]
            )[..., 0] * seen
        else:
            strength = marking_strength(projected_colour, args.low, args.high) * seen
        if args.dump_strength:
            Image.fromarray((strength * 255).astype(np.uint8), "L").save(args.dump_strength)
        if ratio is not None:
            # Transfer the marking as a multiplier so the generated shading survives.
            projected_ratio = sample_image(ratio, baked[..., 0], baked[..., 1])
            target = albedo * projected_ratio
        else:
            target = projected_colour
        painted = blend_markings(albedo, target, strength, args.amount)
        print(f"marking texels: {int((strength > 0.5).sum()):,} of {seen.sum():,} covered")

    mesh.visual.material.baseColorTexture = Image.fromarray(painted.astype(np.uint8), "RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
