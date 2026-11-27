#!/usr/bin/env python3
"""Recalculate outward-facing normals on a generated mesh and re-export it.

Run (live): python scripts/blender_fix_normals.py model.glb fixed.glb

Generated meshes come out substantially inside-out: measured by inward-facing area,
60% on one Nikita run and 23% on another. Blender's ray-cast heuristic (Edit Mode >
Mesh > Normals > Recalculate Outside, Shift+N) fixes most of it -- 60% -> 19% and
23% -> 15%. The residue is genuine concavity (armpits, between the legs, inside the
ears) which legitimately faces inward relative to the centroid.

Why this matters even though our own renders look fine: glTF marks these materials
doubleSided, and Blender shades backfaces identically, flipping the normal for
lighting automatically. So a preview render is *blind* to normal direction. Engines
are not -- SceneKit and RealityKit commonly cull backfaces, where an inside-out mesh
renders inverted or disappears. Check it with the viewport's Overlays > Face
Orientation: blue is outward, red is flipped.

`trimesh.repair.fix_winding`/`fix_normals` do not work here; they bail on meshes with
this many boundary edges, leaving the inward fraction unchanged.

This does NOT fix holes. Missing geometry -- such as the opening where the back of a
head should be -- is a separate problem; see docs/open-questions.md.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_stage_worker.py")


def blender_code(asset: Path, output: Path) -> str:
    return build_worker_code(WORKER_PATH, {
        "action": "fix_normals", "parameters": {"asset": str(asset), "output": str(output)},
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    code = blender_code(args.asset.expanduser().resolve(), output)

    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(1800)
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
