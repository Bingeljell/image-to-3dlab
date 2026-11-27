#!/usr/bin/env python3
"""Point an orbit camera at a target and (optionally) render -- the Blender-side
twin of Worklings' `OrbitCamera` Swift struct (Sources/Worklings/
DungeonStageCameraTool.swift), so a dungeon-stage shot can be framed and
re-framed in Blender without hand-writing a new camera-math script each time.

Run (live): python scripts/blender_orbit_camera.py --help

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
import math
import socket
from pathlib import Path

if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_query_worker.py")


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


def camera_pose(
    target: tuple[float, float, float], azimuth: float, elevation: float, radius: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the Blender camera position and direction toward a SceneKit target."""
    x, y, z = target
    az = math.radians(azimuth)
    el = math.radians(elevation)
    dx = radius * math.cos(el) * math.sin(az)
    dy = -radius * math.cos(el) * math.cos(az)
    dz = radius * math.sin(el)
    location = (x + dx, -z + dy, y + dz)
    return location, (x - location[0], -z - location[1], y - location[2])


def build_code(cam_name, location, toward, fov, render_path, resolution) -> str:
    return build_worker_code(WORKER_PATH, {
        "action": "orbit_camera",
        "parameters": {
            "cam_name": cam_name, "location": location, "toward": toward,
            "fov": fov, "render_path": render_path, "resolution": resolution,
        },
    })


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

    location, toward = camera_pose(tuple(args.target), args.azimuth, args.elevation, args.radius)
    code = build_code(
        cam_name=args.camera_name,
        location=location,
        toward=toward,
        fov=args.fov,
        render_path=args.render,
        resolution=tuple(args.resolution),
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
