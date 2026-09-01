#!/usr/bin/env python3
"""Inspect a Blender scene or an unopened .blend file's contents.

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


LIVE_SCENE_CODE = '''
import bpy, json
objs = []
for o in bpy.context.scene.objects:
    objs.append({
        "name": o.name,
        "type": o.type,
        "dimensions": [round(v, 4) for v in o.dimensions],
        "scale": [round(v, 4) for v in o.scale],
        "location": [round(v, 4) for v in o.location],
        "hide_render": o.hide_render,
        "parent": o.parent.name if o.parent else None,
    })
result = {
    "filepath": bpy.data.filepath,
    "is_dirty": bpy.data.is_dirty,
    "object_count": len(objs),
    "objects": objs,
    "actions": [a.name for a in bpy.data.actions],
    "materials": [m.name for m in bpy.data.materials],
}
print("INSPECT_RESULT " + json.dumps(result))
result
'''

FILE_PEEK_CODE_TEMPLATE = '''
import bpy, json
path = {path!r}
with bpy.data.libraries.load(path, link=False, assets_only=False) as (data_from, data_to):
    result = {{
        "path": path,
        "objects": list(data_from.objects),
        "actions": list(data_from.actions),
        "meshes": list(data_from.meshes),
        "armatures": list(data_from.armatures),
        "materials": list(data_from.materials),
    }}
    # Loading nothing into data_to -- this only reads the library's index,
    # it never actually appends anything into the running Blender.
print("INSPECT_RESULT " + json.dumps(result))
result
'''

OBJECT_DETAIL_CODE_TEMPLATE = '''
import bpy, json
name = {name!r}
obj = bpy.data.objects.get(name)
if obj is None:
    result = {{"error": f"no object named {{name!r}} in the live scene"}}
else:
    result = {{
        "name": obj.name,
        "type": obj.type,
        "dimensions": list(obj.dimensions),
        "scale": list(obj.scale),
        "location": list(obj.location),
        "rotation_euler_deg": [d * 57.29577951308232 for d in obj.rotation_euler],
        "hide_render": obj.hide_render,
        "parent": obj.parent.name if obj.parent else None,
        "children": [c.name for c in obj.children],
        "modifiers": [(m.name, m.type) for m in getattr(obj, "modifiers", [])],
        "materials": [m.name for m in obj.data.materials] if hasattr(obj.data, "materials") else [],
        "active_action": obj.animation_data.action.name if (obj.animation_data and obj.animation_data.action) else None,
    }}
print("INSPECT_RESULT " + json.dumps(result))
result
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="Peek inside this .blend file's index without opening it")
    parser.add_argument("--object", help="Full detail on this object in the currently live scene")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    if args.file and args.object:
        parser.error("--file and --object are mutually exclusive")

    if args.file:
        code = FILE_PEEK_CODE_TEMPLATE.format(path=args.file)
    elif args.object:
        code = OBJECT_DETAIL_CODE_TEMPLATE.format(name=args.object)
    else:
        code = LIVE_SCENE_CODE

    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
