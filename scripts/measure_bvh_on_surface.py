#!/usr/bin/env python3
"""Check MtlBVH on an invariant that scales with the real production mesh.

Every sampled triangle centroid lies exactly on a triangle stored in the BVH, so its
unsigned distance must be zero apart from floating-point noise. This catches traversal
branches being silently dropped on multi-million-face meshes; a small analytic sphere does
not create a deep enough BVH to exercise that failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decode", type=Path)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument(
        "--eps",
        type=float,
        default=0.00097942,
        help="band=1 cell epsilon from the 1024-resolution fox remesh",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from mtlbvh import MtlBVH

    payload = torch.load(args.decode, map_location="cpu", weights_only=False)
    vertices = payload["vertices"].float().contiguous()
    faces = payload["faces"].int().contiguous()
    sample_count = min(args.samples, len(faces))
    sample_ids = torch.linspace(0, len(faces) - 1, sample_count).long()
    sampled_faces = faces[sample_ids].long()
    centroids = vertices[sampled_faces].mean(dim=1).contiguous()

    print(
        f"mesh: {len(faces):,} faces, {len(vertices):,} vertices; "
        f"querying {sample_count:,} exact centroids",
        flush=True,
    )
    bvh = MtlBVH(vertices.to("mps"), faces.to("mps"))
    print("bvh built", flush=True)
    distances = bvh.unsigned_distance(centroids.to("mps"))[0].cpu().numpy()
    print("distances measured", flush=True)

    stats = {
        "p50": float(np.percentile(distances, 50)),
        "p90": float(np.percentile(distances, 90)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(distances.max()),
        "over_eps": float((distances > args.eps).mean()),
        "eps": args.eps,
    }
    for name, value in stats.items():
        print(f"  {name:12s} {value}")

    if stats["p99"] > args.eps:
        print("FAIL: production-scale BVH drops or misorders nearest-triangle traversal")
        return 1
    print("PASS: exact surface queries are accurate relative to the remesh cell epsilon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
