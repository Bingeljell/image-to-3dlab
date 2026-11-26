#!/usr/bin/env python3
"""Inspect a Blender scene or an unopened .blend file's contents.

Run (live): python scripts/blender_inspect.py --help

Talks to a running Blender over the execute_code RPC (see blender_pick_pixel.py
for the wire format), same as every other blender_*.py helper in this repo.
Written because a 2026-08-28 session (blocking out the Worklings dungeon room
and importing rigged characters into it) kept hand-writing throwaway RPC
scripts just to answer "what's actually in this file/scene" -- this replaces
that ad hoc loop with one reusable command.

Three modes:

    # What's in the currently open scene (objects, dims, scale, hide_render)?
    python scripts/blender_inspect.py

    # Peek inside a .blend file WITHOUT opening it in the live Blender --
    # read-only, never touches the running scene.
    python scripts/blender_inspect.py --file path/to/character.blend

    # Full detail on one object already in the live scene.
    python scripts/blender_inspect.py --object rig
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

WORKER_PATH = Path(__file__).with_name("blender_scene_query_worker.py")


def send(code: str, host: str, port: int, timeout: int = 30) -> str:
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


def build_inspect_code(path: str | None = None, name: str | None = None) -> str:
    if path:
        request = {"action": "inspect_file", "parameters": {"path": path}}
    elif name:
        request = {"action": "inspect_object", "parameters": {"name": name}}
    else:
        request = {"action": "inspect_scene", "parameters": {}}
    return build_worker_code(WORKER_PATH, request)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="Peek inside this .blend file's index without opening it")
    parser.add_argument("--object", help="Full detail on this object in the currently live scene")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    if args.file and args.object:
        parser.error("--file and --object are mutually exclusive")

    code = build_inspect_code(args.file, args.object)
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
