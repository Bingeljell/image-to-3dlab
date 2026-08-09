#!/usr/bin/env python3
"""Delete disconnected junk from a generated mesh, keeping the texture intact.

Measured on the 3/4 moss fox hero (2026-08-09): 93.6% of the surface is one
connected body, and the remaining 6.4% -- 6,509 faces -- is scattered across ~693
separate pieces, 498 of them under ten faces. Those specks are extraction debris.
They contribute nothing visually, they inflate every hole and component count, and
they bury the ~92 large tears that are the defect actually worth fixing.

**Why this exists rather than reusing `fill_holes.py`'s welding.** That script welds
by position and exports the welded mesh, which collapses the vertices glTF splits at
every UV seam -- so it silently discards UVs, the material and the 2048 base-colour
texture. (`moss_fox_hero_101k_filled.glb` is 1.9 MB against the hero's 12 MB for
exactly this reason.) Component finding *does* need position welding, or it counts UV
islands instead of geometry. The resolution: weld only to compute the mask, and apply
that mask to the ORIGINAL faces. Welding merges vertices, never faces, so face indices
survive the round trip and the texture comes through untouched.

Judge the result with a backface-culled grey render, never a textured one --
`scripts/blender_render_asset.py --culled`. See `scripts/mesh_health.py` to measure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _weld_index(vertices: np.ndarray) -> np.ndarray:
    """Map each vertex to a canonical index for coincident positions.

    Quantising at 1e-6 of the asset's own size matches `mesh_health.weld_by_position`,
    so the two tools agree about what "connected" means.
    """
    scale = float(np.ptp(vertices, axis=0).max())
    quantised = np.round(vertices / (scale * 1e-6)).astype(np.int64)
    _, inverse = np.unique(quantised, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def loose_face_mask(
    vertices: np.ndarray, faces: np.ndarray, min_faces: int
) -> np.ndarray:
    """Boolean mask over the ORIGINAL faces: True to keep.

    A face is kept when it belongs to a connected component of at least `min_faces`
    triangles. Degenerate faces (two corners welding to the same position) are always
    dropped -- they carry no area and break adjacency.
    """
    faces = np.asarray(faces)
    welded = _weld_index(np.asarray(vertices))[faces]

    non_degenerate = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 0] != welded[:, 2])
    )
    original_index = np.flatnonzero(non_degenerate)

    surface = trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=welded[non_degenerate], process=False
    )
    components = trimesh.graph.connected_components(
        surface.face_adjacency, nodes=np.arange(len(surface.faces))
    )

    mask = np.zeros(len(faces), dtype=bool)
    for component in components:
        if len(component) >= min_faces:
            mask[original_index[component]] = True
    return mask


def prune(mesh: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    """Return `mesh` with loose components removed, preserving UVs and material."""
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces)
    pruned = mesh.copy()
    pruned.update_faces(mask)
    pruned.remove_unreferenced_vertices()
    return pruned


def describe(vertices: np.ndarray, faces: np.ndarray, min_faces: int) -> list[dict]:
    """What each dropped component is, so a real body part cannot vanish unnoticed.

    Face count alone is a poor junk test: a leaf card is small but legitimate. The
    span figure -- a component's bounding-box diagonal as a fraction of the whole
    asset's -- distinguishes a speck from a genuine thin part.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    welded = _weld_index(vertices)[faces]
    non_degenerate = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 0] != welded[:, 2])
    )
    original_index = np.flatnonzero(non_degenerate)
    surface = trimesh.Trimesh(
        vertices=vertices, faces=welded[non_degenerate], process=False
    )
    components = trimesh.graph.connected_components(
        surface.face_adjacency, nodes=np.arange(len(surface.faces))
    )
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))

    dropped = []
    for component in components:
        if len(component) >= min_faces:
            continue
        corners = vertices[faces[original_index[component]].ravel()]
        span = float(np.linalg.norm(np.ptp(corners, axis=0)))
        dropped.append(
            {
                "faces": len(component),
                "span": span / diagonal,
                "centre": corners.mean(axis=0),
            }
        )
    dropped.sort(key=lambda item: item["faces"], reverse=True)
    return dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument(
        "--min-faces",
        type=int,
        default=100,
        help=(
            "drop connected components smaller than this. The default keeps six "
            "components on the hero fox and removes ~688 specks."
        ),
    )
    parser.add_argument(
        "--report",
        type=int,
        default=10,
        help="list this many of the largest dropped components, so a real part "
        "disappearing is visible rather than silent",
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh.expanduser().resolve(), force="mesh", process=False)
    before_faces, before_vertices = len(mesh.faces), len(mesh.vertices)
    had_uv = getattr(mesh.visual, "uv", None) is not None

    dropped = describe(mesh.vertices, mesh.faces, args.min_faces)
    pruned = prune(mesh, args.min_faces)

    print(f"faces     {before_faces:>9,} -> {len(pruned.faces):>9,}")
    print(f"vertices  {before_vertices:>9,} -> {len(pruned.vertices):>9,}")
    print(f"dropped {len(dropped)} components, {sum(d['faces'] for d in dropped):,} faces")
    if dropped:
        print("\nlargest dropped (span = bbox diagonal as fraction of the asset's):")
        for item in dropped[: args.report]:
            centre = ", ".join(f"{value:+.3f}" for value in item["centre"])
            print(f"  {item['faces']:>5,} faces  span {item['span']:.4f}  at ({centre})")

    kept_uv = getattr(pruned.visual, "uv", None) is not None
    print(f"\nUVs: {'preserved' if kept_uv else 'ABSENT'} (input {'had' if had_uv else 'had no'} UVs)")
    if had_uv and not kept_uv:
        raise SystemExit("refusing to write: the texture would be lost")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pruned.export(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
