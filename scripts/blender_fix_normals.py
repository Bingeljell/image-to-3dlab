#!/usr/bin/env python3
"""Recalculate outward-facing normals on a generated mesh and re-export it.

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


def blender_code(asset: Path, output: Path) -> str:
    return f'''
import bpy
import numpy as np

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for coll in list(bpy.data.collections):
    bpy.data.collections.remove(coll)

bpy.ops.import_scene.gltf(filepath={str(asset)!r})
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
for o in bpy.data.objects:
    o.select_set(False)
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
mesh = obj.data


def inward():
    """Area fraction facing away from the centroid. A crude but usable proxy: some
    residue is real concavity, so treat it as an upper bound rather than an error."""
    count = len(mesh.polygons)
    normals = np.zeros(count * 3, dtype=np.float32)
    centres = np.zeros(count * 3, dtype=np.float32)
    areas = np.zeros(count, dtype=np.float32)
    mesh.polygons.foreach_get("normal", normals)
    mesh.polygons.foreach_get("center", centres)
    mesh.polygons.foreach_get("area", areas)
    normals = normals.reshape(-1, 3)
    centres = centres.reshape(-1, 3)
    facing = ((centres - centres.mean(axis=0)) * normals).sum(axis=1)
    return 100.0 * areas[facing < 0].sum() / areas.sum()


before = inward()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
after = inward()

bpy.ops.export_scene.gltf(
    filepath={str(output)!r}, export_format="GLB", use_selection=False
)
print("NORMALS:: inward area %.1f%% -> %.1f%%" % (before, after))
result = {{"before": before, "after": after}}
'''


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
