#!/usr/bin/env python3
"""Measure how accurate MtlBVH's unsigned_distance actually is, against exact ground truth.

    vendor/trellis-mac/.venv/bin/python scripts/measure_bvh_precision.py

**Why this matters.** Narrow-band dual contouring decides "does the surface cross this voxel
edge?" by comparing distances against `eps = band * scale / resolution`. At the demo's
band=1 and resolution 1024, eps is **0.00098** — about one voxel. Any error in the distance
of that order flips the sign test, the voxel reports no crossing, and the kernel strands its
vertex at the voxel centre. We measure 22-29% of vertices stranded that way, and quads
connecting grid centres are exactly the lattice the port produces.

The Python is identical to the CUDA reference and so are the constants, including a
band-membership threshold of `0.87 * cell_size` — √3/2, the exact half-diagonal of a voxel,
with zero tolerance. So the suspicion is that `MtlBVH` returns distances less precise than
`cuBVH`, and the algorithm has no margin to absorb it.

**The test.** A sphere has an analytic distance field: for a point p and radius r centred at
the origin, the distance to the surface is exactly `abs(|p| - r)`. Sample points in a shell
around it, ask the BVH, and compare. Error at or above ~1e-3 confirms the theory and gives
the number a fix should be sized against.

Run backgrounded with a timeout: a cold Metal context has hung here before, at 0% CPU with
no output and no error.
"""

from __future__ import annotations

import argparse


def sphere_ground_truth(points, radius: float):
    """Exact distance from each point to the surface of a sphere centred on the origin."""
    import numpy as np

    return np.abs(np.linalg.norm(points, axis=1) - radius)


def error_report(measured, exact, eps: float) -> dict:
    """Absolute error, and what it means for a sign test with a margin of `eps`.

    `sign_flip_risk` is the share of samples whose error exceeds eps — those are the ones
    where the crossing test can land on the wrong side.
    """
    import numpy as np

    err = np.abs(measured - exact)
    return {
        "samples": int(err.size),
        "mean_abs_error": float(err.mean()),
        "p50": float(np.percentile(err, 50)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
        "eps": eps,
        "error_over_eps_p99": float(np.percentile(err, 99) / eps) if eps else 0.0,
        "sign_flip_risk": float((err > eps).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--subdivisions", type=int, default=6)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument(
        "--eps", type=float, default=0.00097942,
        help="the real eps from a band=1, resolution=1024 run",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    import trimesh
    from mtlbvh import MtlBVH

    sphere = trimesh.creation.icosphere(subdivisions=args.subdivisions, radius=args.radius)
    print(f"sphere: {len(sphere.faces):,} faces, radius {args.radius}", flush=True)

    # Sample in a thin shell around the surface, which is where the algorithm actually asks.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(args.samples, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    offset = rng.uniform(-0.01, 0.01, size=(args.samples, 1))
    points = direction * (args.radius + offset)

    bvh = MtlBVH(
        torch.tensor(sphere.vertices, dtype=torch.float32, device="mps"),
        torch.tensor(sphere.faces, dtype=torch.int32, device="mps"),
    )
    print("bvh built", flush=True)
    measured = bvh.unsigned_distance(
        torch.tensor(points, dtype=torch.float32, device="mps")
    )[0].cpu().numpy()
    print("distances measured", flush=True)

    # Ground truth is analytic: for a sphere centred on the origin the distance to the
    # surface is exactly abs(|p| - r). The icosphere is a polyhedron so it sits fractionally
    # inside the ideal sphere, but at subdivision 6 that faceting error is ~3e-5 - two orders
    # below eps - so it cannot account for anything we are looking for.
    # (trimesh's exact closest_point would need rtree, which this venv lacks.)
    exact = sphere_ground_truth(points, args.radius)

    stats = error_report(measured, exact, args.eps)
    for key, value in stats.items():
        print(f"  {key:22s} {value}")

    print()
    if stats["p99"] >= args.eps:
        print(f"  ** p99 error {stats['p99']:.6f} >= eps {args.eps:.6f}")
        print("  ** The distance error is the size of the band. Crossing tests will flip.")
    else:
        print(f"  BVH is precise relative to eps ({stats['p99']:.2e} vs {args.eps:.2e}).")
        print("  The lattice comes from somewhere else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
