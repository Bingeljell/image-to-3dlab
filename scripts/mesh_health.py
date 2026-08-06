#!/usr/bin/env python3
"""Measure the structural health of a generated mesh, and optionally repair it.

Every mesh problem this project has hit was invisible until measured: see-through
holes, interior speckle, and failed bone-heat weighting are all the same underlying
defect, and none of them announce themselves as errors. This turns "the back of his
head looks weird" into "this mesh has 26,000 components".

One measurement trap it avoids: glTF splits vertices at UV and normal seams, so
`is_watertight` on a round-tripped GLB is *always* False and means nothing. Vertices
are welded by position before anything is counted.

The weld sweep answers a specific question: are the components genuinely separate
surfaces, or are they adjacent shards left unwelded by the extractor? If a small
tolerance collapses the count, the fix is cheap. If it does not, the geometry really
is disconnected and needs reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def measure(mesh: trimesh.Trimesh) -> dict:
    """Structural stats, after welding coincident vertices."""
    welded = mesh.copy()
    welded.merge_vertices()

    edges = welded.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())

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
            probe.merge_vertices()
            probe.update_faces(probe.nondegenerate_faces())
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
