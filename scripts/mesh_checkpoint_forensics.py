#!/usr/bin/env python3
"""Measure exact-position topology in an o_voxel `.pt` geometry checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--export-glb", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vertices = payload["vertices"].numpy()
    faces_stored = payload["faces"].numpy().astype(np.int64, copy=False)

    unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces_stored]
    sorted_faces = np.sort(faces, axis=1)
    _, keep, face_counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True
    )
    duplicate_groups = int((face_counts > 1).sum())
    duplicate_copies = int(np.maximum(face_counts - 1, 0).sum())
    faces = faces[np.sort(keep)]

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    sorted_edges = np.sort(edges, axis=1)
    _, edge_inverse, edge_counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    orientation_sum = np.bincount(edge_inverse, weights=direction)
    paired = edge_counts == 2

    print(f"checkpoint: {args.checkpoint}")
    print(f"stored vertices: {len(vertices):,}")
    print(f"stored faces: {len(faces_stored):,}")
    print(f"unique positions: {len(unique_vertices):,}")
    print(f"faces after geometric dedupe: {len(faces):,}")
    print(f"duplicate geometric face groups: {duplicate_groups:,}")
    print(f"extra duplicate face copies: {duplicate_copies:,}")
    print(f"boundary/open edges: {int((edge_counts == 1).sum()):,}")
    print(f"non-manifold edges: {int((edge_counts > 2).sum()):,}")
    print(f"same-direction manifold pairs: {int((paired & (orientation_sum != 0)).sum()):,}")

    if args.export_glb:
        import trimesh

        args.export_glb.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(
            vertices=unique_vertices, faces=faces, process=False
        ).export(args.export_glb)
        print(f"wrote {args.export_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
