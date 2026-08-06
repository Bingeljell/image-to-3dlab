#!/usr/bin/env python3
"""Project a 2D image onto a generated mesh as per-vertex colours.

A generated mesh has no idea what its parts are -- it is one undifferentiated blob,
because the generator works in occupancy-per-point of space and never has a concept of
"hair" or "leaf" to lose. That missing part information is what blocks foliage wind,
cloth, hair, and multi-material effects, all at once.

This recovers it from 2D. Paint a flat-colour mask over the *source* image (leaves
green, body red), and this transfers those labels onto the mesh: every vertex is
projected back into the image and samples the colour painted there.

The projection mirrors what TRELLIS conditions on. Its preprocessing removes the
background, takes the subject's alpha bounding box, and crops to a SQUARE centred on
it, so the mesh's own bounding square maps onto the image's subject square. The view
is treated as orthographic, which is a close enough approximation to validate against
the silhouette.

Validate before painting anything: pass the original image as --image, and the mesh
should come out looking like itself. If it looks scrambled, the projection is wrong
and no mask will save it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

# Which mesh axes face the camera. The GLB is Y-up; "front" is the axis the
# conditioning view looks down. Signs are resolved empirically -- see --flip flags.
AXES = {
    "z": (0, 1, 2),  # horizontal=X, vertical=Y, depth=Z
    "x": (2, 1, 0),  # horizontal=Z, vertical=Y, depth=X
}


def subject_bbox(image: Image.Image, threshold: int = 204) -> tuple[int, int, int, int]:
    """Bounding box of the subject, matching how TRELLIS finds it.

    Uses alpha when present (TRELLIS skips background removal in that case, so this
    reproduces its crop exactly). Falls back to "not near-white" for opaque images,
    which approximates what background removal would have produced.
    """
    array = np.array(image)
    if image.mode == "RGBA" and not np.all(array[:, :, 3] == 255):
        mask = array[:, :, 3] > threshold
    else:
        rgb = array[:, :, :3].astype(np.int32)
        mask = rgb.sum(axis=2) < (threshold * 3)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        raise ValueError("could not find a subject in the image")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def square_crop(image: Image.Image) -> Image.Image:
    """Crop to the square TRELLIS conditions on: centred on the subject bbox."""
    x0, y0, x1, y1 = subject_bbox(image)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0)
    box = (
        int(cx - size // 2),
        int(cy - size // 2),
        int(cx + size // 2),
        int(cy + size // 2),
    )
    return image.crop(box)


def project(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    axis: str,
    flip_h: bool,
    flip_v: bool,
    flip_depth: bool,
    depth_buffer: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (colours, visible) for every vertex."""
    h_axis, v_axis, d_axis = AXES[axis]
    vertices = mesh.vertices
    horizontal = vertices[:, h_axis]
    vertical = vertices[:, v_axis]
    depth = vertices[:, d_axis] * (-1.0 if flip_depth else 1.0)

    # The mesh's bounding square maps onto the image's subject square, which is how
    # TRELLIS framed the subject in the first place.
    lo = np.array([horizontal.min(), vertical.min()])
    hi = np.array([horizontal.max(), vertical.max()])
    center = (lo + hi) * 0.5
    extent = float(max(hi - lo))

    u = (horizontal - center[0]) / extent + 0.5
    v = 0.5 - (vertical - center[1]) / extent  # image rows run downward
    if flip_h:
        u = 1.0 - u
    if flip_v:
        v = 1.0 - v

    crop = square_crop(image)
    pixels = np.array(crop.convert("RGB"))
    height, width = pixels.shape[:2]
    px = np.clip((u * width).astype(np.int32), 0, width - 1)
    py = np.clip((v * height).astype(np.int32), 0, height - 1)
    colours = pixels[py, px]

    # A vertex landing off the subject sampled the backdrop, not a label. Without this
    # the background's colour becomes a bogus label -- white "paint" smeared over every
    # silhouette edge. Treat those as unlabelled instead.
    cropped = np.array(crop)
    if crop.mode == "RGBA" and not np.all(cropped[:, :, 3] == 255):
        foreground = cropped[:, :, 3] > 204
    else:
        foreground = cropped[:, :, :3].astype(np.int32).sum(axis=2) < (204 * 3)
    on_subject = foreground[py, px]

    # Depth test: a vertex on the back of the head would otherwise sample the chest.
    # Bin vertices by pixel and keep only those at (or near) the frontmost depth in
    # their bin. This is topology-independent, which matters on a fragmented mesh.
    bins = np.clip((u * depth_buffer).astype(np.int64), 0, depth_buffer - 1) + (
        np.clip((v * depth_buffer).astype(np.int64), 0, depth_buffer - 1) * depth_buffer
    )
    nearest = np.full(depth_buffer * depth_buffer, -np.inf)
    np.maximum.at(nearest, bins, depth)
    span = float(depth.max() - depth.min()) or 1.0
    visible = (depth >= (nearest[bins] - depth_tolerance * span)) & on_subject

    return colours, visible


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="generated .glb")
    parser.add_argument("image", type=Path, help="source image, or a painted mask")
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument("--axis", choices=tuple(AXES), default="z")
    parser.add_argument("--flip-h", action="store_true")
    parser.add_argument("--flip-v", action="store_true")
    parser.add_argument("--flip-depth", action="store_true")
    parser.add_argument(
        "--hidden-colour",
        default="20,20,20",
        help="R,G,B given to vertices the view cannot see (unlabelled)",
    )
    parser.add_argument("--depth-buffer", type=int, default=512)
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.02,
        help="fraction of total depth a vertex may sit behind the frontmost and still count",
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh.expanduser().resolve(), force="mesh")
    image = Image.open(args.image.expanduser().resolve())

    colours, visible = project(
        mesh,
        image,
        args.axis,
        args.flip_h,
        args.flip_v,
        args.flip_depth,
        args.depth_buffer,
        args.depth_tolerance,
    )

    hidden = np.array([int(c) for c in args.hidden_colour.split(",")], dtype=np.uint8)
    colours = colours.copy()
    colours[~visible] = hidden

    rgba = np.concatenate(
        [colours.astype(np.uint8), np.full((len(colours), 1), 255, np.uint8)], axis=1
    )
    # Drop the original texture: the point is to see the projected colours alone.
    painted = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=rgba,
        process=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    painted.export(output)

    print(
        f"PROJECT:: vertices {len(mesh.vertices)} visible {int(visible.sum())} "
        f"({100.0 * visible.mean():.1f}%) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
