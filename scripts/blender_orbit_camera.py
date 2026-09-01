#!/usr/bin/env python3
"""Point an orbit camera at a target and (optionally) render -- the Blender-side
twin of Worklings' `OrbitCamera` Swift struct (Sources/Worklings/
DungeonStageCameraTool.swift), so a dungeon-stage shot can be framed and
re-framed in Blender without hand-writing a new camera-math script each time.

Talks to a running Blender over the execute_code RPC (see blender_pick_pixel.py
for the wire format).

Position/target are in the SceneKit convention this project's game code uses
(Y up, Z toward camera) -- converted to Blender's Z-up the same
handedness-preserving way as blender_import_character.py:
(x, y, z) -> (x, -z, y).

Example -- reproduce the Cache Warren's locked shot exactly:

    python scripts/blender_orbit_camera.py \\
        --target -1.92 -0.10 2.29 --azimuth 59.7 --elevation 39.7 --radius 27.95 \\
        --fov 32 --render /tmp/preview.png

Dolly in on the same target/angles to check character scale at a tighter crop:

    python scripts/blender_orbit_camera.py --target -1.92 -0.10 2.29 \\
        --azimuth 59.7 --elevation 39.7 --radius 14 --render /tmp/closer.png
"""

from __future__ import annotations

import argparse
import json
import socket


def send(code: str, host: str, port: int, timeout: int = 60) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(timeout)
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


CODE_TEMPLATE = '''
import bpy, json, math
from mathutils import Vector

cam_name = {cam_name!r}
target_sk = {target!r}
azimuth_deg = {azimuth!r}
elevation_deg = {elevation!r}
radius = {radius!r}
fov_deg = {fov!r}
render_path = {render_path!r}
resolution = {resolution!r}

def sk_to_blender(x, y, z):
    return (x, -z, y)

cam_obj = bpy.data.objects.get(cam_name)
if cam_obj is None:
    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
cam_obj.data.lens_unit = "FOV"
cam_obj.data.sensor_fit = "VERTICAL"  # SceneKit's fieldOfView is vertical; Blender's `angle` defaults horizontal
cam_obj.data.angle = math.radians(fov_deg)

target = Vector(sk_to_blender(*target_sk))
az = math.radians(azimuth_deg)
el = math.radians(elevation_deg)
offset = Vector((
    radius * math.cos(el) * math.sin(az),
    -radius * math.cos(el) * math.cos(az),
    radius * math.sin(el),
))
cam_obj.location = target + offset
direction = (target - cam_obj.location).normalized()
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam_obj

result = {{"camera": cam_obj.name, "location": list(cam_obj.location)}}

if render_path:
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    result["rendered"] = render_path

print("ORBIT_RESULT " + json.dumps(result))
result
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-name", default="stagePreviewCam")
    parser.add_argument("--target", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--azimuth", type=float, required=True)
    parser.add_argument("--elevation", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--fov", type=float, default=32.0, help="Vertical FOV in degrees (default: 32, matching DungeonStageScene.cameraFieldOfView)")
    parser.add_argument("--render", default=None, help="If given, render a still to this path")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    code = CODE_TEMPLATE.format(
        cam_name=args.camera_name,
        target=tuple(args.target),
        azimuth=args.azimuth,
        elevation=args.elevation,
        radius=args.radius,
        fov=args.fov,
        render_path=args.render,
        resolution=tuple(args.resolution),
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
