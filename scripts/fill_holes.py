#!/usr/bin/env python3
"""Close small holes in a generated mesh, leaving large openings alone.

SUPERSEDED (2026-08-06) — this approach does not work post-export. It closes holes as
advertised (34,789 -> 18,736 boundary edges on the moss fox) but makes the mesh WORSE
overall: normal recalculation afterwards goes from 44.6% -> 21.1% inward WITHOUT the
fill, but 44.8% -> 53.7% WITH it, and a backface-culled render confirms more gaps, not
fewer. That held both for naive patches and for patches wound to match their
neighbours, so it is not simply an orientation bug.

The likely reason is that centroid-fan patches over non-planar rims introduce folded
or self-intersecting geometry, which defeats the ray casting that decides "outside".

Kept as a documented dead end, and because the boundary-loop tracer and the
small-vs-large hole distinction below are still sound. See docs/open-questions.md.

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

    original = trimesh.load(args.mesh.expanduser().resolve(), force="mesh")
    vertices, faces = weld(original)
    scale = float(np.ptp(vertices, axis=0).max())
    limit = scale * args.max_perimeter

    loops = boundary_loops(vertices, faces)
    print(f"boundary loops found: {len(loops)}")

    # Every directed edge of the existing surface, so patches can be wound to match.
    directed = set()
    for tri in faces:
        directed.add((int(tri[0]), int(tri[1])))
        directed.add((int(tri[1]), int(tri[2])))
        directed.add((int(tri[2]), int(tri[0])))

    new_vertices = [vertices]
    new_faces = [faces]
    filled = skipped = added = 0
    next_index = len(vertices)

    for loop in loops:
        ring = vertices[loop]
        perimeter = float(np.linalg.norm(np.diff(ring, axis=0, append=ring[:1]), axis=1).sum())
        if perimeter > limit:
            skipped += 1
            continue
        # Fan from the loop's own centroid: robust for non-planar and concave loops,
        # where a simple ear-clip on the ring would fold over itself.
        centre = ring.mean(axis=0)
        new_vertices.append(centre.reshape(1, 3))

        # Wind each patch triangle to agree with the face it abuts. A boundary edge
        # belongs to exactly one face, so the patch must traverse that edge in the
        # OPPOSITE direction. Skipping this injects randomly-oriented faces, which
        # measurably defeats normal recalculation afterwards -- filling holes then
        # made a mesh *more* inside-out, not less.
        triangles = []
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            if (a, b) in directed:  # the existing face runs a->b, so the patch runs b->a
                triangles.append([b, a, next_index])
            else:
                triangles.append([a, b, next_index])
        new_faces.append(np.array(triangles, dtype=np.int64))
        next_index += 1
        filled += 1
        added += len(triangles)

    patched = trimesh.Trimesh(
        vertices=np.vstack(new_vertices),
        faces=np.vstack(new_faces),
        process=False,
    )

    def boundary_count(mesh: trimesh.Trimesh) -> int:
        edges = np.sort(
            np.vstack(
                [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
            ),
            axis=1,
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int((counts == 1).sum())

    before = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    print(f"filled {filled} holes ({added} new faces), skipped {skipped} too large")
    print(f"boundary edges: {boundary_count(before)} -> {boundary_count(patched)}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    patched.export(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
