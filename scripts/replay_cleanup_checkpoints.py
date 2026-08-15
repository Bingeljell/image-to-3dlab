#!/usr/bin/env python3
"""Replay o_voxel cleanup on an exact mesh checkpoint, saving every substage."""

from __future__ import annotations

import argparse
from pathlib import Path


def save(mesh, path: Path) -> None:
    import torch

    vertices, faces = mesh.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vertices": vertices.detach().cpu().contiguous(),
            "faces": faces.detach().cpu().contiguous(),
        },
        path,
    )
    print(
        f"{path}: {mesh.num_vertices:,} vertices, {mesh.num_faces:,} faces",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()

    import torch
    import cumesh

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mesh = cumesh.CuMesh()
    mesh.init(payload["vertices"], payload["faces"])
    del payload

    stages = (
        ("remove_duplicates", mesh.remove_duplicate_faces),
        ("repair_nonmanifold", mesh.repair_non_manifold_edges),
        ("remove_small_components", lambda: mesh.remove_small_connected_components(1e-5)),
        ("fill_holes", lambda: mesh.fill_holes(max_hole_perimeter=3e-2)),
    )
    for tag, operation in stages:
        operation()
        save(mesh, Path(f"{args.output_prefix}_{tag}.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
