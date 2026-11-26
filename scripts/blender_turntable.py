#!/usr/bin/env python3
"""Render a seamlessly looping 360-degree turntable of a GLB via the Blender MCP scene.

Run (live): python scripts/blender_turntable.py model.glb turntable.mp4

Renders a PNG sequence (the model spinning under fixed lights) and muxes it into an
MP4 with ffmpeg. The loop is seamless because the last frame stops one step short of
a full turn, so frame N wraps onto frame 0 without a duplicate.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Reuse the render environments from the still-preview helper so a turntable and a
# preview of the same asset are lit identically.
from blender_render_asset import ENVIRONMENTS
from blender_socket import execute_code
from blender_video import encode_frames
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
    return execute_code(code, host, port, timeout)


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
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}_frames_", dir=output.parent))

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
    try:
        print(send(args.host, args.port, code, timeout=1800))
        encode_frames(frame_dir, args.label, args.frames, args.fps, output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}. Frames kept in {frame_dir}", file=sys.stderr)
        return 1

    if args.keep_frames:
        print(f"Frames kept in {frame_dir}")
    else:
        shutil.rmtree(frame_dir)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
