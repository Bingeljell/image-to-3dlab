#!/usr/bin/env python3
"""Build a UV-space mask for one feature of a mesh, from a point in 3D.

A generated asset is one undifferentiated material, so an eye is exactly as matte as
bark -- and a matte eye is a dead eye, because gloss is what reads as wet. Fixing that
needs to know which texels are the eye, and the atlas cannot tell you: grading warms the
bark toward the same amber, so a colour threshold shatters into a hundred fragments
scattered across the map.

Geometry can tell you. Given a centre and radius in mesh-local coordinates -- obtained
by finding the feature in a render and casting a ray back through it -- this selects the
faces inside that sphere and paints their UV triangles into a mask.

``--front-only`` keeps just the faces pointing at the camera, so a sphere that reaches
through a thin shell does not also mask the inside surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def select_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
    radius: float,
    normals: np.ndarray | None = None,
    facing: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean per-face mask: centroid within `radius` of `centre`, optionally front-facing."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    centroids = vertices[faces].mean(axis=1)
    inside = np.linalg.norm(centroids - np.asarray(centre, dtype=float), axis=1) <= radius
    if normals is not None and facing is not None:
        inside &= normals @ np.asarray(facing, dtype=float) > 0
    return inside


def rasterise_uv(
    uv: np.ndarray, faces: np.ndarray, selected: np.ndarray, size: int, dilate: int = 2
) -> Image.Image:
    """Paint the selected faces' UV triangles white on black, at `size` x `size`.

    UV origin is bottom-left; image origin is top-left, hence the v flip. `dilate`
    widens the mask by a few texels so bilinear sampling at the edge does not pick up
    unmasked neighbours.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for tri in faces[selected]:
        pts = [
            (float(uv[i, 0]) * (size - 1), (1.0 - float(uv[i, 1])) * (size - 1))
            for i in tri
        ]
        draw.polygon(pts, fill=255)
    if dilate > 0:
        arr = np.asarray(img)
        grown = arr.copy()
        for shift in range(1, dilate + 1):
            for axis in (0, 1):
                grown = np.maximum(grown, np.roll(arr, shift, axis=axis))
                grown = np.maximum(grown, np.roll(arr, -shift, axis=axis))
        img = Image.fromarray(grown, mode="L")
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", type=Path)
    p.add_argument("output", type=Path, help="mask PNG to write")
    p.add_argument("--centre", type=float, nargs=3, required=True,
                   help="feature centre in mesh-local coordinates")
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--front-only", action="store_true",
                   help="keep only faces whose normal points along --facing")
    p.add_argument("--facing", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    args = p.parse_args()

    scene = trimesh.load(args.asset, process=False)
    mesh = next(iter(scene.geometry.values()))
    selected = select_faces(
        mesh.vertices, mesh.faces, np.array(args.centre), args.radius,
        mesh.face_normals if args.front_only else None,
        args.facing if args.front_only else None,
    )
    mask = rasterise_uv(mesh.visual.uv, mesh.faces, selected, args.size, args.dilate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.output)
    covered = float(np.asarray(mask).mean() / 255.0)
    print(f"{int(selected.sum())} of {len(mesh.faces)} faces -> "
          f"{covered * 100:.3f}% of the atlas; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
