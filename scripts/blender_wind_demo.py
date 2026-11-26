#!/usr/bin/env python3
"""Animate labelled foliage with shader-style wind and render it to MP4.

Run (live): python scripts/blender_wind_demo.py model.glb labelled.glb wind.mp4

Foliage in a game is almost never rigged or simulated. It is moved by a vertex shader:
a small program that nudges each point every frame by a wave, scaled by how floppy that
point is. Trunk rigid, leaf tips loose. No bones, no solver, effectively free.

This does the same arithmetic ahead of time so the result can be rendered locally,
which proves the whole chain -- painted mask, projected labels, derived stiffness,
visible motion -- without needing a game engine set up. An engine would run the same
formula live from the same data.

Stiffness is not painted. It falls out of the labels: a foliage vertex's distance to
the nearest *body* vertex, normalised. Zero where the tail meets the hip, one at the
tips, so the tail bends rather than sliding rigidly.

Takes two meshes. Appearance comes from the original textured GLB; labels come from the
GLB written by project_labels.py. They are matched by position rather than by index,
because exporting through trimesh does not preserve vertex order.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from blender_render_asset import ENVIRONMENTS
from blender_socket import execute_code
from blender_video import encode_frames
from blender_worker_source import build_worker_code


def blender_code(
    mesh: Path,
    labels: Path,
    frame_dir: Path,
    label: str,
    env: str,
    resolution: int,
    frames: int,
    zoom: float,
    azimuth: float,
    frequency: float,
    wave: float,
    gust: float,
    bend: float,
    axis_y: float,
    axis_z: float,
    cluster_size: float,
    jitter: float,
    flutter: float,
    *,
    body_name: str = "FOX",
    labels_name: str = "LABELS",
) -> str:
    settings = ENVIRONMENTS[env]
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_wind_demo_worker.py"),
        {
            "mesh_path": str(mesh),
            "labels_path": str(labels),
            "body_name": body_name,
            "labels_name": labels_name,
            "frame_dir": str(frame_dir),
            "label": label,
            "resolution": resolution,
            "frames": frames,
            "zoom": zoom,
            "azimuth": azimuth,
            "frequency": frequency,
            "wave": wave,
            "gust": gust,
            "bend": bend,
            "axis_y": axis_y,
            "axis_z": axis_z,
            "cluster_size": cluster_size,
            "jitter": jitter,
            "flutter": flutter,
            "world_color": settings["world_color"],
            "raytracing": settings["raytracing"],
        },
    )


def send(host: str, port: int, code: str, timeout: float) -> str:
    return execute_code(code, host, port, timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="original textured .glb")
    parser.add_argument("labels", type=Path, help="labelled .glb from project_labels.py")
    parser.add_argument("output", type=Path, help="destination .mp4")
    parser.add_argument("--label", default="wind")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="dark")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--azimuth", type=float, default=25.0)
    parser.add_argument(
        "--bend",
        type=float,
        default=22.0,
        help="peak bend angle in degrees at the floppiest tips",
    )
    parser.add_argument(
        "--axis-y", type=float, default=1.0, help="bend axis: horizontal sweep component"
    )
    parser.add_argument(
        "--axis-z", type=float, default=0.35, help="bend axis: vertical lift component"
    )
    parser.add_argument(
        "--cluster-size",
        type=float,
        default=0.07,
        help="clump size for independent flutter, as a fraction of the asset",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.8, help="how independently clumps flutter (0-1)"
    )
    parser.add_argument(
        "--flutter", type=float, default=0.012, help="high-frequency tip shimmer"
    )
    parser.add_argument("--frequency", type=float, default=2.0, help="sway cycles per loop")
    parser.add_argument("--wave", type=float, default=5.0, help="how much the wave travels")
    parser.add_argument("--gust", type=float, default=0.45, help="depth of the slow gust")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}_frames_", dir=output.parent))

    code = blender_code(
        args.mesh.expanduser().resolve(),
        args.labels.expanduser().resolve(),
        frame_dir,
        args.label,
        args.env,
        args.resolution,
        args.frames,
        args.zoom,
        args.azimuth,
        args.frequency,
        args.wave,
        args.gust,
        args.bend,
        args.axis_y,
        args.axis_z,
        args.cluster_size,
        args.jitter,
        args.flutter,
    )
    try:
        print(send(args.host, args.port, code, timeout=3600))
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
