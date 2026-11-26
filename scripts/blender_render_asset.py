#!/usr/bin/env python3
"""Import a GLB into the local Blender MCP scene and render cardinal previews.

Run (live): python scripts/blender_render_asset.py model.glb renders/"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

# Render environments. "dark" is the near-black studio that flatters matte
# assets; "studio" lifts the world and enables ray-traced reflections so a
# metallic (pbr-mode) surface has something to reflect instead of black.
ENVIRONMENTS = {
    "dark": {"world_color": (0.035, 0.035, 0.035), "raytracing": False},
    "studio": {"world_color": (0.22, 0.19, 0.15), "raytracing": True},
}


def blender_code(
    asset: Path,
    output_dir: Path,
    label: str,
    env: str,
    culled: bool = False,
    recalc_normals: bool = False,
) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_render_asset_worker.py"),
        {
            "asset_path": str(asset),
            "output_dir": str(output_dir),
            "label": label,
            "culled": culled,
            "recalc_normals": recalc_normals,
            **ENVIRONMENTS[env],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="asset")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--env",
        choices=tuple(ENVIRONMENTS),
        default="dark",
        help="dark flatters matte assets; studio lifts the world for metallic (pbr) assets",
    )
    parser.add_argument(
        "--culled",
        action="store_true",
        help="the honest test: plain grey, backface culling on. Predicts what "
        "SceneKit/RealityKit will show, where a textured doubleSided render hides holes",
    )
    parser.add_argument(
        "--recalc-normals",
        action="store_true",
        help="Recalculate Outside before rendering. Culling without this keeps the "
        "wrong faces; but it has also torn solid regions open here, so render both ways",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "type": "execute_code",
        "params": {
            "code": blender_code(
                asset,
                output_dir,
                args.label,
                args.env,
                culled=args.culled,
                recalc_normals=args.recalc_normals,
            )
        },
    }
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(300)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks = []
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
