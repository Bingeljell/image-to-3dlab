#!/usr/bin/env python3
"""Close small holes in a generated mesh, leaving large openings alone.

This WORKS, and visibly so (verified 2026-08-06). On the moss fox it takes boundary
edges from 34,789 to 18,736, and in a backface-culled render the hind legs go from
near-invisible wisps to solid limbs.

A caution about how that was nearly missed: a centroid-based "inward-facing area"
metric said the fill made things *worse* (44.8% -> 53.7% after normal recalculation,
versus 44.6% -> 21.1% without it), and that reading was briefly written up as a failed
experiment. The metric is worthless -- it asks whether a face points away from the
whole body's centroid, which is meaningless for concave regions and for anything
off-centre. **Judge this by rendering with backface culling, never by that metric.**

Two different kinds of hole reach our output, and only one of them should be patched.

**Small holes are an extraction artefact.** TRELLIS converts a sparse voxel grid to
triangles with a dual-grid method: each intersected edge becomes a quad spanning four
neighbouring voxels, emitted only when all four exist. At the boundary of the active
voxel set one is always missing, so quads are dropped. Upstream repairs this during
decode with `fill_holes()`; our Mac port disables that call (Metal `cumesh` segfaults,
and `cumesh` is not installed), so they survive into the export. These *should* be
filled -- they are missing by accident.

**Large openings are missing evidence.** The back of a head the model never saw is not
an artefact, and stretching a membrane across it would look worse than the hole. Those
need multi-view input, not patching. Hence `--max-perimeter`: holes above it are left.

`trimesh.repair.fill_holes` only closes triangular and quad holes -- about 20% of them
here. This traces every boundary loop and fans it from its own centroid, which handles
arbitrary size and non-planar loops.

Measure with `scripts/mesh_health.py`. Note both must merge by POSITION only: glTF
splits vertices at UV seams, so naive counting measures UV islands, not geometry.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def weld(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices by position, ignoring UVs and normals."""
    scale = float(np.ptp(mesh.vertices, axis=0).max())
    quantised = np.round(mesh.vertices / (scale * 1e-6)).astype(np.int64)
    _, index, inverse = np.unique(
        quantised, axis=0, return_index=True, return_inverse=True
    )
    vertices = mesh.vertices[index]
    faces = inverse[mesh.faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return vertices, faces[keep]


def weld_index(vertices: np.ndarray) -> np.ndarray:
    """Map each vertex to a canonical index for coincident positions."""
    scale = float(np.ptp(vertices, axis=0).max())
    quantised = np.round(vertices / (scale * 1e-6)).astype(np.int64)
    _, inverse = np.unique(quantised, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def fill(mesh: trimesh.Trimesh, max_perimeter: float) -> trimesh.Trimesh:
    """Close small holes, preserving the original vertices, UVs and material.

    The earlier implementation welded by position and exported the welded mesh, which
    discarded UVs -- glTF splits a vertex at every UV seam, so welding collapses them
    and the baked texture no longer has coordinates to sample. But welding is still
    *necessary* to find boundaries, or seam duplicates read as open edges and the tool
    tries to fill the UV islands rather than the geometry.

    The resolution is the same one `remove_loose_parts.py` uses: weld only to ANALYSE,
    and emit patches referring to the ORIGINAL vertex indices. Existing geometry is
    untouched and patch triangles are appended, so the texture keeps lining up.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    uv = getattr(mesh.visual, "uv", None)
    uv = None if uv is None else np.asarray(uv)

    welded = weld_index(vertices)
    welded_faces = welded[faces]
    non_degenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )

    # A welded edge carried by exactly one face is a boundary. Because it belongs to
    # exactly one face, the original directed edge behind it is unambiguous -- which
    # is what lets the patch be wound against real geometry rather than welded copies.
    original_edge: dict[tuple[int, int], tuple[int, int]] = {}
    counts: dict[tuple[int, int], int] = {}
    for face, welded_face in zip(faces[non_degenerate], welded_faces[non_degenerate]):
        for i in range(3):
            u, v = int(face[i]), int(face[(i + 1) % 3])
            wu, wv = int(welded_face[i]), int(welded_face[(i + 1) % 3])
            key = (wu, wv) if wu < wv else (wv, wu)
            counts[key] = counts.get(key, 0) + 1
            original_edge[(wu, wv)] = (u, v)

    welded_vertices = np.zeros((welded.max() + 1, 3), dtype=np.float64)
    welded_vertices[welded] = vertices

    loops = boundary_loops(welded_vertices, welded_faces[non_degenerate])
    scale = float(np.ptp(vertices, axis=0).max())
    limit = scale * max_perimeter

    extra_vertices: list[np.ndarray] = []
    extra_uv: list[np.ndarray] = []
    extra_faces: list[list[int]] = []
    next_index = len(vertices)
    filled = skipped = 0

    for loop in loops:
        ring = welded_vertices[loop]
        perimeter = float(
            np.linalg.norm(np.diff(ring, axis=0, append=ring[:1]), axis=1).sum()
        )
        if perimeter > limit:
            skipped += 1
            continue

        centre = ring.mean(axis=0)
        triangles = []
        rim_original: list[int] = []
        for i in range(len(loop)):
            wa, wb = loop[i], loop[(i + 1) % len(loop)]
            if (wa, wb) in original_edge:
                # The existing face runs a->b, so the patch runs b->a.
                a, b = original_edge[(wa, wb)]
                triangles.append([b, a, next_index])
            elif (wb, wa) in original_edge:
                a, b = original_edge[(wb, wa)]
                triangles.append([b, a, next_index])
            else:
                triangles = []
                break
            rim_original.extend((a, b))
        if not triangles:
            skipped += 1
            continue

        extra_vertices.append(centre.reshape(1, 3))
        if uv is not None:
            # Take the UV of the rim vertex nearest the centre rather than averaging.
            # A loop that straddles a UV seam has rim coordinates on opposite sides of
            # the atlas, and their mean lands somewhere unrelated -- a patch sampling
            # the wrong part of the texture. Borrowing one neighbour's UV keeps the
            # patch sampling a plausible nearby colour.
            rim = np.array(sorted(set(rim_original)), dtype=np.int64)
            nearest = rim[np.argmin(np.linalg.norm(vertices[rim] - centre, axis=1))]
            extra_uv.append(uv[nearest].reshape(1, 2))
        extra_faces.extend(triangles)
        next_index += 1
        filled += 1

    fill.last_run = {"filled": filled, "skipped": skipped, "loops": len(loops)}

    if not extra_faces:
        return mesh.copy()

    combined = trimesh.Trimesh(
        vertices=np.vstack([vertices, *extra_vertices]),
        faces=np.vstack([faces, np.array(extra_faces, dtype=np.int64)]),
        process=False,
    )
    if uv is not None:
        combined.visual = trimesh.visual.TextureVisuals(
            uv=np.vstack([uv, *extra_uv]),
            material=getattr(mesh.visual, "material", None),
        )
    return combined


def boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    """Trace open boundary edges into closed loops."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ordered = np.sort(edges, axis=1)
    unique, counts = np.unique(ordered, axis=0, return_counts=True)
    boundary = unique[counts == 1]

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    seen: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for start in list(adjacency):
        for first in adjacency[start]:
            if (start, first) in seen or (first, start) in seen:
                continue
            loop = [start]
            previous, current = start, first
            seen.add((start, first))
            # Walk the boundary until it returns to the start. A pinch point can have
            # more than two boundary edges; take the first unvisited neighbour.
            while current != start:
                loop.append(current)
                nxt = None
                for candidate in adjacency[current]:
                    if candidate == previous:
                        continue
                    if (current, candidate) in seen or (candidate, current) in seen:
                        continue
                    nxt = candidate
                    break
                if nxt is None:
                    break
                seen.add((current, nxt))
                previous, current = current, nxt
            if current == start and len(loop) >= 3:
                loops.append(loop)
    return loops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument(
        "--max-perimeter",
        type=float,
        default=0.15,
        help=(
            "skip holes whose perimeter exceeds this fraction of the asset's size. "
            "Large openings are missing evidence, not artefacts, and patching them "
            "stretches a membrane across e.g. the back of a head."
        ),
    )
    args = parser.parse_args()

    original = trimesh.load(
        args.mesh.expanduser().resolve(), force="mesh", process=False
    )
    had_uv = getattr(original.visual, "uv", None) is not None

    def boundary_count(mesh: trimesh.Trimesh) -> int:
        """Count open edges after welding by POSITION.

        Counting on raw indices measures UV islands, not geometry: glTF splits a
        vertex at every seam, so on the textured hero it reports ~83,000 "boundary
        edges" against a true 6,877, and the number goes UP after a successful fill.
        Now that this tool preserves UVs, its own report was the first casualty.
        """
        faces = weld_index(np.asarray(mesh.vertices))[np.asarray(mesh.faces)]
        edges = np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int((counts == 1).sum())

    patched = fill(original, args.max_perimeter)
    stats = fill.last_run

    print(f"boundary loops found: {stats['loops']}")
    print(
        f"filled {stats['filled']} holes "
        f"({len(patched.faces) - len(original.faces)} new faces), "
        f"skipped {stats['skipped']} too large"
    )
    print(f"boundary edges: {boundary_count(original)} -> {boundary_count(patched)}")

    kept_uv = getattr(patched.visual, "uv", None) is not None
    print(f"UVs: {'preserved' if kept_uv else 'ABSENT'} "
          f"(input {'had' if had_uv else 'had no'} UVs)")
    if had_uv and not kept_uv:
        raise SystemExit("refusing to write: the texture would be lost")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    patched.export(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
