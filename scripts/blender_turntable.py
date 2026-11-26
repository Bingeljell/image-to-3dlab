#!/usr/bin/env python3
"""Render a seamlessly looping 360-degree turntable of a GLB via the Blender MCP scene.

Run (live): python scripts/blender_turntable.py model.glb turntable.mp4

Renders a PNG sequence (the model spinning under fixed lights) and muxes it into an
MP4 with ffmpeg. The loop is seamless because the last frame stops one step short of
a full turn, so frame N wraps onto frame 0 without a duplicate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Reuse the render environments from the still-preview helper so a turntable and a
# preview of the same asset are lit identically.
from blender_render_asset import ENVIRONMENTS
from blender_worker_source import build_worker_code


def blender_code(
    asset: Path,
    frame_dir: Path,
    label: str,
    env: str,
    frames: int,
    resolution: int,
    zoom: float,
    darken_backfaces: bool,
) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_turntable_worker.py"),
        {
            "asset_path": str(asset),
            "frame_dir": str(frame_dir),
            "label": label,
            "frames": frames,
            "resolution": resolution,
            "zoom": zoom,
            "darken_backfaces": darken_backfaces,
            **ENVIRONMENTS[env],
        },
    )


def send(host: str, port: int, code: str, timeout: float) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, help="destination .mp4")
    parser.add_argument("--label", default="turntable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="dark")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument(
        "--darken-backfaces",
        action="store_true",
        help=(
            "shade backfacing polygons near-black so shell gaps read as shadow. Off by "
            "default: TRELLIS meshes have large inverted-normal regions, so this can "
            "blacken outward-facing surfaces too."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="camera tightening factor; >1 fills more of the frame",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="keep the intermediate PNG sequence instead of deleting it",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".frames_{args.label}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    code = blender_code(
        asset,
        frame_dir,
        args.label,
        args.env,
        args.frames,
        args.resolution,
        args.zoom,
        args.darken_backfaces,
    )
    # Rendering the whole sequence happens inside one blocking call, so allow for it.
    print(send(args.host, args.port, code, timeout=1800))

    rendered = sorted(frame_dir.glob(f"{args.label}_*.png"))
    if len(rendered) != args.frames:
        print(
            f"error: expected {args.frames} frames, found {len(rendered)}",
            file=sys.stderr,
        )
        return 1

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / f"{args.label}_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # x264 needs even dimensions; scale defensively for odd --resolution values.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )

    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
