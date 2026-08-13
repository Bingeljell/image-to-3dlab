#!/usr/bin/env python3
"""Cut a region out of a mesh at full density, so it can be judged by eye.

    python scripts/crop_mesh.py big.ply head.glb --region head
    python scripts/crop_mesh.py big.ply slab.glb --axis z --from 0.45 --to 0.55

**Why not just decimate.** A 15-million-face mesh rendered whole has more triangles than the
screen has pixels, so a chain-link cage and a solid surface look identical — the very
distinction we need. Cropping keeps every original triangle and only reduces how many are in
frame, which is the difference between a view that can answer the question and one that
cannot.

Fractions are of the mesh's own bounding box, so `--region head` means the top slice
regardless of the model's units or origin. Faces are selected by centroid, so a face
straddling the boundary is included once rather than split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Fractional (low, high) bounds per axis. Y is up in glTF.
REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "head": {"y": (0.70, 1.00)},
    "top": {"y": (0.50, 1.00)},
    "bottom": {"y": (0.00, 0.50)},
    "torso": {"y": (0.35, 0.70)},
    "front": {"z": (0.50, 1.00)},
    "core": {"x": (0.35, 0.65), "y": (0.35, 0.65), "z": (0.35, 0.65)},
}
AXES = {"x": 0, "y": 1, "z": 2}


def face_mask(vertices, faces, bounds: dict[str, tuple[float, float]]):
    """Boolean mask of faces whose centroid falls inside the fractional bounds.

    Centroid rather than "any vertex inside": the latter pulls in a shell of faces hanging
    off the cut plane, which reads as fringe and muddies exactly the judgement being made.
    """
    import numpy as np

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    centroids = vertices[faces].mean(axis=1)
    frac = (centroids - lo) / span

    keep = np.ones(len(faces), dtype=bool)
    for axis, (low, high) in bounds.items():
        i = AXES[axis]
        keep &= (frac[:, i] >= low) & (frac[:, i] <= high)
    return keep


def resolve_bounds(args) -> dict[str, tuple[float, float]]:
    if args.region:
        if args.region not in REGIONS:
            raise ValueError(f"unknown region {args.region!r}; have {sorted(REGIONS)}")
        return REGIONS[args.region]
    if args.axis:
        return {args.axis: (args.start, args.end)}
    raise ValueError("give --region or --axis")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--region", choices=sorted(REGIONS))
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, default=1.0)
    args = parser.parse_args()

    import trimesh

    mesh = trimesh.load(args.source, process=False)
    if hasattr(mesh, "geometry"):
        mesh = mesh.dump(concatenate=True)
    print(f"loaded {len(mesh.faces):,} faces from {args.source.name}")

    keep = face_mask(mesh.vertices, mesh.faces, resolve_bounds(args))
    if not keep.any():
        raise SystemExit("the region contains no faces — check the bounds")

    cropped = mesh.submesh([keep], append=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cropped.export(args.output)
    print(f"kept {int(keep.sum()):,} faces ({keep.mean():.1%}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
