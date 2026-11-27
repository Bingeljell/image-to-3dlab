#!/usr/bin/env python3
"""Render a GLB under an orbiting key light, holding the camera still.

Run (live): python scripts/blender_light_orbit.py model.glb frames/f_

A normal map is invisible in a static render — it changes how a surface *responds* to
light, not what colour it is. Orbiting the camera does not reveal it either, because the
lighting relative to the surface never changes. **Only moving the light does.**

So this holds the camera fixed and sweeps the key light through a full circle. Surface
detail that exists only in the normal map appears as travelling highlights and shadows
across the leaves and fur; geometry that is genuinely flat stays flat.

The fill and rim are held static and dim, so the sweep reads clearly rather than being
washed out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_stage_worker.py")


def key_light_rotation(frame: int, frames: int, elevation: float) -> tuple[float, float, float]:
    """Euler rotation for the key light at `frame`, as a full azimuth sweep.

    A Blender sun light is directional — only its rotation matters, never its position.
    Returned in radians as (x, y, z); the sweep is on z so elevation stays constant and
    the highlight travels horizontally, which is what makes relief read.

    Frame 1 and frame `frames + 1` give the same azimuth, so the loop is seamless.
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    t = (frame - 1) / frames
    return (math.radians(elevation), 0.0, 2.0 * math.pi * t)


def blender_code(
    asset: Path,
    out: Path,
    frames: int,
    elevation: float,
    res: int,
    rotations: str,
) -> str:
    return build_worker_code(WORKER_PATH, {
        "action": "light_orbit",
        "parameters": {
            "asset": str(asset), "out": str(out), "resolution": res,
            "rotations": json.loads(rotations),
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out_prefix", type=Path, help="frame path prefix, e.g. dir/f_")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--elevation", type=float, default=52.0)
    parser.add_argument("--res", type=int, default=800)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rotations = [
        list(key_light_rotation(f, args.frames, args.elevation))
        for f in range(1, args.frames + 1)
    ]
    code = blender_code(
        args.asset.expanduser().resolve(),
        args.out_prefix.expanduser().resolve(),
        args.frames, args.elevation, args.res, json.dumps(rotations),
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
