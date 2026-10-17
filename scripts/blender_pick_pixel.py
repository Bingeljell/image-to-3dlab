#!/usr/bin/env python3
"""Raycast one render pixel into the asset currently staged in live Blender.

This is the bridge between a visible feature and ``feature_mask.py``. Render an asset with
``blender_render_asset.py``, note a feature's pixel, then run for example:

    python scripts/blender_pick_pixel.py 455 410 --view profile_yneg

The printed ``gltf_local`` position can be passed directly to ``feature_mask.py --centre``.
``mesh_local`` is also reported for Blender debugging, but Blender changes glTF's axis order
on import and that coordinate must not be used as the glTF-space mask centre.
The script only reads and moves the scene camera; it does not edit the mesh or save the blend.
"""

from __future__ import annotations

import argparse
import json
import socket

VIEWS = {
    "front_xneg": "(-distance, 0, target.z)",
    "front_three_quarter": "(-distance * 0.9, -distance * 0.45, target.z)",
    "rear_xpos": "(distance, 0, target.z)",
    "profile_yneg": "(0, -distance, target.z)",
    "profile_ypos": "(0, distance, target.z)",
}


def build_code(x: float, y: float, width: int, height: int, view: str) -> str:
    location = VIEWS[view]
    return f'''
import bpy, json
from mathutils import Vector

scene = bpy.context.scene
camera = scene.camera
meshes = [obj for obj in scene.objects if obj.type == "MESH"]
corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
lo = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
hi = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
center = (lo + hi) * 0.5
size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
distance = size * 2.25
target = center + Vector((0, 0, size * 0.03))
camera.location = Vector({location})
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
projection = camera.calc_matrix_camera(
    depsgraph, x={width}, y={height},
    scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y,
)
inverse = (projection @ camera.matrix_world.inverted()).inverted()
ndc_x = 2.0 * ({x} + 0.5) / {width} - 1.0
ndc_y = 1.0 - 2.0 * ({y} + 0.5) / {height}
near = inverse @ Vector((ndc_x, ndc_y, -1.0, 1.0))
far = inverse @ Vector((ndc_x, ndc_y, 1.0, 1.0))
near /= near.w
far /= far.w
origin = camera.matrix_world.translation
direction = (far.xyz - near.xyz).normalized()
hit, location, normal, face_index, obj, matrix = scene.ray_cast(depsgraph, origin, direction)
result = {{"hit": bool(hit), "pixel": [{x}, {y}], "view": {view!r}}}
if hit:
    mesh_local = obj.matrix_world.inverted() @ location
    result.update({{
        "object": obj.name,
        "face_index": int(face_index),
        "world": list(location),
        "mesh_local": list(mesh_local),
        "gltf_local": [mesh_local.x, mesh_local.z, -mesh_local.y],
        "normal_world": list(normal),
    }})
print("PICK_RESULT " + json.dumps(result))
result
'''


def send(code: str, host: str, port: int) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(60)
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
    return b"".join(chunks).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--view", choices=tuple(VIEWS), default="profile_yneg")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()
    print(send(build_code(args.x, args.y, args.width, args.height, args.view), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
