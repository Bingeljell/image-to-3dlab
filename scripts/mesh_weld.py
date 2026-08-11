#!/usr/bin/env python3
"""Weld vertices that share coordinates exactly, and drop the faces that collapse.

**Why this exists.** `o_voxel/postprocess.py` calls `repair_non_manifold_edges()`
immediately before every `simplify()`. That repair works by *splitting vertices* -- its own
docstring says so: "This creates duplicate vertices with the same coordinates." QEM edge
collapse cannot collapse across a duplicate pair, because they are topologically distinct
points, so the simplifier tears the surface open around every seam the repair introduced.

Measured on the thorn-knot Snag: entering step 7 the mesh is at 7.8% faces touching a
boundary, and one `simplify()` call takes it to 44.7%. The same call on the same target,
run against a welded mesh, gives 8.4%.

**Exact matching, deliberately.** The duplicates are bit-identical because they were made
by splitting one vertex, so `np.unique` on the raw coordinates finds them precisely. A
tolerance weld would additionally merge genuinely distinct nearby surfaces -- exactly the
fusing of adjacent coils that made voxel remeshing unusable on this subject.

Note this is the opposite of what `remove_loose_parts.py` does with welding: there it is a
throwaway used only to *compute* a component mask, because welding a glTF collapses its UV
seams. Here there are no UVs yet -- `uv_unwrap` runs after all of this -- so welding is
safe and is the point.
"""

from __future__ import annotations

import numpy as np


def weld_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge exactly-coincident vertices and drop faces that become degenerate.

    Returns (vertices, faces) with duplicates removed and every remaining face still
    referencing three distinct vertices.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be an (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an (M, 3) array")
    if len(faces) and (faces.max() >= len(vertices) or faces.min() < 0):
        raise ValueError("face indices out of range")

    unique, index, inverse = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )
    remapped = inverse.reshape(-1)[faces] if len(faces) else faces.reshape(0, 3)

    # Welding can make two corners of a face the same point; such a face has no area and
    # would otherwise survive as a degenerate sliver.
    if len(remapped):
        keep = (
            (remapped[:, 0] != remapped[:, 1])
            & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2])
        )
        remapped = remapped[keep]

    # Preserve the original ordering of first appearance rather than np.unique's sort,
    # so the result stays comparable with the input for debugging.
    order = np.argsort(index)
    relabel = np.empty(len(unique), dtype=np.int64)
    relabel[order] = np.arange(len(unique))
    return vertices[np.sort(index)], relabel[remapped]


def duplicate_count(vertices: np.ndarray) -> int:
    """How many vertices are exact duplicates of an earlier one."""
    vertices = np.asarray(vertices)
    return len(vertices) - len(np.unique(vertices, axis=0))
