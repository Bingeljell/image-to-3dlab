#!/usr/bin/env python3
"""Voxel-remesh a mesh into a closed surface and bake its albedo back on.

Run (live): python scripts/blender_solidify_bake.py model.glb closed.glb

SUPERSEDED (2026-08-06). This was written to fix a "confetti mesh" of tens of thousands
of disconnected shards. That diagnosis was a measurement error: `merge_vertices` does
not merge across UV seams, so the component count was really a count of UV islands.
Measured position-only, the meshes are single connected surfaces -- so this script was
solving a problem that did not exist. See `docs/open-questions.md` question 1.

Kept because the remesh-and-rebake machinery may still be useful, and so the dead end
is documented rather than rediscovered. Do not reach for it to "fix topology".
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from blender_worker_source import build_worker_code


def blender_code(
    asset: Path,
    output: Path,
    voxel_size: float,
    texture_size: int,
    extrusion: float,
    min_component_verts: int,
) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_solidify_bake_worker.py"),
        {
            "asset_path": str(asset), "output_path": str(output), "voxel_size": voxel_size,
            "texture_size": texture_size, "extrusion": extrusion,
            "min_component_verts": min_component_verts,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="remesh voxel size; <=0 derives 0.2%% of the asset's largest dimension",
    )
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument(
        "--min-component-verts",
        type=int,
        default=200,
        help="drop remeshed components smaller than this many vertices (specks)",
    )
    parser.add_argument(
        "--extrusion",
        type=float,
        default=0.0,
        help="bake cage extrusion; <=0 derives 6x the voxel size",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    code = blender_code(
        asset,
        output,
        args.voxel_size,
        args.texture_size,
        args.extrusion,
        args.min_component_verts,
    )
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        # Remesh plus a full-resolution bake is slow; give it room.
        connection.settimeout(3600)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
    print(b"".join(chunks).decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
