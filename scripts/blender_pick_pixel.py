#!/usr/bin/env python3
"""Raycast one render pixel into the asset currently staged in live Blender.

Run (live): python scripts/blender_pick_pixel.py 455 410 --view profile_yneg

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
from pathlib import Path

if __package__:
    from .blender_scene_query_worker import VIEW_OFFSETS as VIEWS
    from .blender_worker_source import build_worker_code
else:
    from blender_scene_query_worker import VIEW_OFFSETS as VIEWS
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_query_worker.py")


def pixel_ndc(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Convert top-left pixel coordinates to normalized camera coordinates."""
    return 2.0 * (x + 0.5) / width - 1.0, 1.0 - 2.0 * (y + 0.5) / height


def build_code(x: float, y: float, width: int, height: int, view: str) -> str:
    if view not in VIEWS:
        raise KeyError(view)
    return build_worker_code(WORKER_PATH, {
        "action": "pick_pixel",
        "parameters": {
            "pixel": [x, y], "width": width, "height": height, "view": view,
            "ndc": pixel_ndc(x, y, width, height),
        },
    })


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
