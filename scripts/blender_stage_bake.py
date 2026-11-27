#!/usr/bin/env python3
"""Render a full frame sequence of a named Action at the locked dungeon-stage
camera angle (docs/design/bake-spec.md section 10 in the worklings repo).

Run (live): python scripts/blender_stage_bake.py --help

Reusable across characters: every `.blend` built to the section-6 rig contract
already carries `CAM_TARGET` (the turntable empty) and an ortho camera parented
to it named `QuickCam2` (or `Camera`), so this script only ever swaps which
Action drives `rig` and which azimuth `CAM_TARGET` is turned to -- it never
re-authors the camera rig itself.

Talks to a running Blender over the execute_code RPC (see blender_pick_pixel.py
for the wire format) rather than running headless, so it renders whatever
`.blend` is currently open in that Blender instance. Switching files is the
caller's job (bpy.ops.wm.open_mainfile from another snippet, or just opening it
by hand) -- this script never opens or saves a file itself, so it can't clobber
unsaved work.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_stage_worker.py")


def build_code(
    action_name: str,
    az: float,
    el: float,
    ortho_scale: float | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
    resolution: int,
    output_dir: str,
    label: str,
) -> str:
    return build_worker_code(WORKER_PATH, {
        "action": "stage_bake",
        "parameters": {
            "action_name": action_name, "az": az, "el": el, "ortho_scale": ortho_scale,
            "frame_start": frame_start, "frame_end": frame_end, "frame_step": frame_step,
            "resolution": resolution, "output_dir": str(output_dir), "label": label,
        },
    })


def send(code: str, host: str, port: int) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(120)
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
    parser.add_argument("action", help="Action name already authored on the 'rig' armature")
    parser.add_argument("--label", required=True, help="Character label used in output filenames, e.g. tempest-ram")
    parser.add_argument("--az", type=float, required=True, help="CAM_TARGET Z rotation in degrees (35=stageFaceBL, 245=stageFaceTR)")
    parser.add_argument("--el", type=float, default=-28.0, help="CAM_TARGET X rotation in degrees (locked dungeon-stage value)")
    parser.add_argument("--ortho-scale", type=float, default=None, help="Override the camera's ortho scale; omit to use whatever is authored in the file")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1, help="1 = every native frame, 2 = half native fps, etc.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square render size; fast-preview default per the bake-spec pickup note")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    code = build_code(
        args.action, args.az, args.el, args.ortho_scale,
        args.frame_start, args.frame_end, args.frame_step,
        args.resolution, args.output_dir, args.label,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
