#!/usr/bin/env python3
"""Measure how torn a generated mesh is, and gate post-processing on it.

**The number that would have saved a week.** The share of faces touching an open edge.
A closed mesh is 0%. A surface with a few tears is 1-3%. The thorn-knot Snag measured
**40.9%** -- at that level the average patch is two or three triangles wide, so it is not
a surface with holes in it, it is a mesh of ribbons. No amount of welding, wrapping,
remeshing, culling or smoothing repairs that, and a full day was spent proving it one
method at a time.

Run this on a decode BEFORE spending anything on finishing. Above the gate, regenerate
rather than repair.

Note what this measures that component and boundary-edge counts do not. Deleting 3,265
small components changed the Snag's appearance not at all, because 79% of its faces are
in one connected web -- a lace, not a sheet. Counting components said "one big object,
some debris". This says "the big object is full of holes", which is the truth.

**Welding first is mandatory**, not a nicety: glTF splits vertices at every UV seam, so
an unwelded mesh reports every UV chart boundary as a tear and the number is meaningless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def boundary_stats(faces: np.ndarray) -> tuple[float, float]:
    """Return (percent of edges on a boundary, percent of faces touching one).

    An edge shared by exactly one face is a boundary. `faces` must already be welded --
    see the module docstring.
    """
    if len(faces) == 0:
        raise ValueError("mesh has no faces")
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    on_boundary = counts == 1
    # edges were built face-major, three per face, so this reshape re-groups them
    per_face = on_boundary[inverse].reshape(3, -1).T
    return (
        100.0 * on_boundary.sum() / len(counts),
        100.0 * per_face.any(axis=1).sum() / len(faces),
    )


def verdict(pct_faces: float, gate: float = 10.0) -> str:
    """Turn the number into the decision it exists to make."""
    if pct_faces < gate:
        return "PASS - safe to post-process"
    if pct_faces < 25.0:
        return "MARGINAL - torn; fix generation before spending on finishing"
    return "FAIL - a mesh of ribbons; regenerate, do not repair"


def measure(path: Path) -> dict:
    scene = trimesh.load(path, process=False)
    if isinstance(scene, trimesh.Scene):
        geom = max(scene.geometry.values(), key=lambda g: len(g.faces))
    else:
        geom = scene
    mesh = trimesh.Trimesh(
        vertices=np.asarray(geom.vertices), faces=np.asarray(geom.faces), process=False
    )
    mesh.merge_vertices()
    pct_edges, pct_faces = boundary_stats(np.asarray(mesh.faces))
    return {
        "faces": len(mesh.faces),
        "pct_boundary_edges": pct_edges,
        "pct_faces_touching_boundary": pct_faces,
        "verdict": verdict(pct_faces),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("meshes", type=Path, nargs="+")
    p.add_argument("--gate", type=float, default=10.0,
                   help="percent of faces touching a boundary above which to refuse (default: 10)")
    args = p.parse_args()

    worst = 0.0
    for path in args.meshes:
        r = measure(path)
        worst = max(worst, r["pct_faces_touching_boundary"])
        print(f"{path.name}")
        print(f"   faces {r['faces']:>9,}   boundary edges {r['pct_boundary_edges']:5.1f}%"
              f"   FACES TOUCHING BOUNDARY {r['pct_faces_touching_boundary']:5.1f}%"
              f"   {verdict(r['pct_faces_touching_boundary'], args.gate)}")
    return 0 if worst < args.gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
