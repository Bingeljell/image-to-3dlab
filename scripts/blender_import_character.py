#!/usr/bin/env python3
"""Append a rigged character (armature + skinned mesh) into the currently
live Blender scene, cleaned up and positioned -- reusable instead of
hand-writing a fresh append/cleanup script per character per session.

Run (live): python scripts/blender_import_character.py --help

Talks to a running Blender over the execute_code RPC (see blender_pick_pixel.py
for the wire format). Built during the 2026-08-28 Worklings dungeon-room
blockout session, where appending the Tempest Ram and Forest Flicker each
pulled in unrelated dependencies (the source file's own camera rig, a
leftover voxel-weighting proxy, decorative effect props) that had to be
found and deleted by hand. This script finds them automatically: anything
that gets pulled in besides the requested armature and mesh is either a
`WGT-` bone-shape widget (kept, but render-hidden -- Rigify doesn't set
that itself) or a stray dependency (deleted).

Position/facing use the SceneKit convention this project's game code is
written in (see Sources/Worklings/DungeonStage3D.swift): Y-up, X right, Z
toward the camera. That gets converted to Blender's Z-up internally as
(x, y, z) -> (x, -z, y) -- a proper rotation (handedness-preserving), not a
naive axis swap; the naive version mirrors the whole scene left-right (found
the hard way in this same session).

Example -- import the Tempest Ram at the party floor's billboard corner,
facing the foe platform, posed on its idle action:

    python scripts/blender_import_character.py \\
        ~/projects/worklings-blender-work/tempest-ram-rigify-natural-walk.blend \\
        --prefix Ram --position -3 0.0 8.5 --facing 0 0.4 -2 \\
        --action RamIdle_Breathe_Paw --frame 5 --save
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

WORKER_PATH = Path(__file__).with_name("blender_import_character_worker.py")


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


def build_code(
    *, path: str, prefix: str, armature_name: str, mesh_name: str,
    position: tuple[float, float, float], facing: tuple[float, float, float] | None,
    action_name: str | None, frame: int | None, do_save: bool,
) -> str:
    return build_worker_code(WORKER_PATH, {
        "path": path, "prefix": prefix, "armature_name": armature_name,
        "mesh_name": mesh_name, "position": position, "facing": facing,
        "action_name": action_name, "frame": frame, "do_save": do_save,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_blend", help="Path to the .blend file containing the rigged character")
    parser.add_argument("--prefix", required=True, help="Name prefix for the imported objects, e.g. Ram -> Ram_rig/Ram_mesh/Ram_root")
    parser.add_argument("--armature-name", default="rig", help="Armature object name in the source file (default: rig)")
    parser.add_argument("--mesh-name", default="geometry_0", help="Mesh object name in the source file (default: geometry_0)")
    parser.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                         help="Placement in SceneKit (x, y, z) convention -- Y up, Z toward camera")
    parser.add_argument("--facing", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                         help="Optional SceneKit-space point to face toward (yaw only)")
    parser.add_argument("--action", default=None, help="Action name to assign and pose on, if any")
    parser.add_argument("--frame", type=int, default=None, help="Scene frame to set after assigning the action")
    parser.add_argument("--save", action="store_true", help="Save the currently open .blend after importing")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    code = build_code(
        path=args.source_blend,
        prefix=args.prefix,
        armature_name=args.armature_name,
        mesh_name=args.mesh_name,
        position=tuple(args.position),
        facing=tuple(args.facing) if args.facing else None,
        action_name=args.action,
        frame=args.frame,
        do_save=args.save,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
