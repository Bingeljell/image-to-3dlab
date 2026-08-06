#!/usr/bin/env python3
"""Measure the structural health of a generated mesh, and optionally repair it.

Every mesh problem this project has hit was invisible until measured: see-through
holes, interior speckle, and failed bone-heat weighting are all the same underlying
defect, and none of them announce themselves as errors. This turns "the back of his
head looks weird" into a number.

**The measurement trap this exists to prevent:** glTF splits a vertex at every UV and
normal seam, and `trimesh.merge_vertices()` will not merge vertices whose UVs differ.
Counting anything after it measures UV islands, not geometry. Everything here welds by
POSITION only. (This tool shipped with that very bug in its first version, which is
how seriously to take it.)

The numbers that matter, and what they tell you:

- **components** -- is the surface connected? Ours are, largely: one body holding
  ~99-100% of faces.
- **boundary edges** -- how much open hole there is.
- **dangling boundary verts** -- boundary vertices with only ONE boundary edge. These
  are torn ends, not rims. Hole filling needs closed rims, so a high count means
  patching is not viable; the geometry needs reconstruction instead.
- **non-manifold edges** -- edges shared by 3+ faces: overlapping or intersecting
  sheets. Breaks inside/outside reasoning, which is why normal repair fails.
- **volume** -- negative means the surface is inside-out overall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def weld_by_position(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge coincident vertices by POSITION only.

    `trimesh.merge_vertices()` will not merge vertices whose UVs differ, and glTF
    splits a vertex at every UV seam -- so using it here measures UV islands rather
    than geometry. That mistake produced a wrong diagnosis that stood for several
    sessions; this function exists so it cannot recur.
    """
    scale = float(np.ptp(mesh.vertices, axis=0).max())
    quantised = np.round(mesh.vertices / (scale * 1e-6)).astype(np.int64)
    _, index, inverse = np.unique(
        quantised, axis=0, return_index=True, return_inverse=True
    )
    faces = inverse[mesh.faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return trimesh.Trimesh(
        vertices=mesh.vertices[index], faces=faces[keep], process=False
    )


def measure(mesh: trimesh.Trimesh) -> dict:
    """Structural stats, after welding coincident vertices by position."""
    welded = weld_by_position(mesh)

    edges = welded.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())

    # A boundary vertex with one edge is a dangling end, not part of a closed rim.
    # Hole filling needs rims, so this is the number that says whether it is viable.
    degree: dict[int, int] = {}
    for a, b in unique[counts == 1]:
        degree[int(a)] = degree.get(int(a), 0) + 1
        degree[int(b)] = degree.get(int(b), 0) + 1
    dangling = sum(1 for d in degree.values() if d == 1)

    # Outward-facing check: for a closed surface the signed volume is positive when
    # normals point out. Large regions facing inward show up as a negative or
    # near-zero volume relative to the convex hull.
    try:
        volume = float(welded.volume)
    except Exception:
        volume = float("nan")

    return {
        "vertices": len(welded.vertices),
        "faces": len(welded.faces),
        "components": int(welded.body_count),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "dangling_boundary_verts": dangling,
        "watertight": bool(welded.is_watertight),
        "winding_consistent": bool(welded.is_winding_consistent),
        "volume": volume,
    }


def report(name: str, stats: dict) -> None:
    print(f"=== {name}")
    print(f"  vertices            {stats['vertices']:>10,}")
    print(f"  faces               {stats['faces']:>10,}")
    print(f"  components          {stats['components']:>10,}")
    print(f"  boundary edges      {stats['boundary_edges']:>10,}")
    print(f"  non-manifold edges  {stats['non_manifold_edges']:>10,}")
    print(f"  dangling bnd verts  {stats['dangling_boundary_verts']:>10,}")
    print(f"  watertight          {str(stats['watertight']):>10}")
    print(f"  winding consistent  {str(stats['winding_consistent']):>10}")
    print(f"  volume              {stats['volume']:>10.5f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--weld-sweep",
        action="store_true",
        help="try increasing weld tolerances to see whether components are merely unwelded",
    )
    parser.add_argument(
        "--repair",
        type=Path,
        default=None,
        help="write a repaired copy here (weld, fix winding/normals, fill holes)",
    )
    parser.add_argument(
        "--weld",
        type=float,
        default=0.0,
        help="weld tolerance for --repair, as a fraction of the asset's size",
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh.expanduser().resolve(), force="mesh")
    before = measure(mesh)
    report(args.mesh.name, before)

    scale = float(np.ptp(mesh.vertices, axis=0).max())

    if args.weld_sweep:
        print()
        print("=== weld sweep (tolerance as fraction of asset size)")
        print(f"  {'tolerance':>10}  {'components':>12}  {'boundary':>12}  {'faces':>10}")
        for fraction in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008):
            probe = mesh.copy()
            if fraction > 0:
                # Quantise positions onto a grid, then weld: coincident-after-rounding
                # vertices merge, which is what welding "within a tolerance" means.
                step = scale * fraction
                probe.vertices = np.round(probe.vertices / step) * step
            probe = weld_by_position(probe)
            edges = probe.edges_sorted
            _, counts = np.unique(edges, axis=0, return_counts=True)
            print(
                f"  {fraction:>10.4f}  {probe.body_count:>12,}  "
                f"{int((counts == 1).sum()):>12,}  {len(probe.faces):>10,}"
            )

    if args.repair:
        repaired = mesh.copy()
        if args.weld > 0:
            step = scale * args.weld
            repaired.vertices = np.round(repaired.vertices / step) * step
        repaired.merge_vertices()
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(repaired)
        trimesh.repair.fix_inversion(repaired)
        trimesh.repair.fix_normals(repaired)
        trimesh.repair.fill_holes(repaired)

        print()
        report("repaired", measure(repaired))
        output = args.repair.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        repaired.export(output)
        print(f"\nwrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
