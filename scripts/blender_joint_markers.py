#!/usr/bin/env python3
"""Spawn named joint markers on a mesh in Blender, and read their placed positions back.

The rigging plan calls for the user to place joints by hand in Blender while the agent,
which cannot see the viewport, reads the results back over the MCP socket. Markers are
pre-placed at anatomically plausible positions rather than parked at the origin: 23
overlapping empties are miserable to untangle, whereas nudging a roughly-right marker
is quick and is the same learning exercise.

Positions are expressed as fractions of the imported mesh's bounding box, measured in
Blender space *after* the glTF importer's Y-up to Z-up conversion. In that space X is
left/right, Z is up, and the subject's front is at minimum Y.

    spawn   import the asset and create the markers
    read    print every marker's placed world position as JSON
    save    write the placed positions to a JSON file
    load    recreate markers from a JSON file

**Always `save` after placing.** Marker positions are the only genuinely manual step in
the rigging lane, and they live nowhere but the Blender scene until written out. Several
scripts here clear the scene before doing their work; running one of those against
unsaved markers destroys the placement irrecoverably, autosave included. It has happened.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

# name -> (x, y, z) as fractions of the mesh bounding box.
# x: 0 left, 1 right.  y: 0 front (muzzle), 1 back (tail tip).  z: 0 ground, 1 ear tips.
QUADRUPED_MARKERS: dict[str, tuple[float, float, float]] = {
    "pelvis": (0.50, 0.62, 0.55),
    "spine_mid": (0.50, 0.50, 0.57),
    "chest": (0.50, 0.36, 0.57),
    "neck_base": (0.50, 0.28, 0.66),
    "head": (0.50, 0.18, 0.76),
    "jaw": (0.50, 0.10, 0.70),
    # L is the *character's* left, which is +X (x > 0.5) given that the subject's front
    # is at minimum Y. On screen that puts the L markers on the viewer's right whenever
    # the subject faces the camera — correct, and the opposite of what looks right.
    "ear_L": (0.62, 0.24, 0.93),
    "ear_R": (0.38, 0.24, 0.93),
    "tail_base": (0.50, 0.70, 0.55),
    "tail_mid": (0.50, 0.82, 0.50),
    "tail_tip": (0.50, 0.93, 0.42),
    # Front and back legs are named for the anatomy they actually have. A quadruped's
    # front leg is an arm (shoulder/elbow/wrist) and its back leg is a leg
    # (hip/knee/ankle); the animal walks on its fingers and toes, so the visible
    # mid-leg bend is a wrist in front and an ankle behind — not a knee in either case.
    # Calling both "knee" was actively misleading, so the names differ by limb.
    "frontL_shoulder": (0.34, 0.32, 0.55),
    "frontL_elbow": (0.34, 0.32, 0.40),
    "frontL_wrist": (0.34, 0.32, 0.18),
    "frontL_paw": (0.34, 0.32, 0.03),
    "frontR_shoulder": (0.66, 0.32, 0.55),
    "frontR_elbow": (0.66, 0.32, 0.40),
    "frontR_wrist": (0.66, 0.32, 0.18),
    "frontR_paw": (0.66, 0.32, 0.03),
    "backL_hip": (0.34, 0.66, 0.55),
    "backL_knee": (0.34, 0.68, 0.40),
    "backL_ankle": (0.34, 0.70, 0.18),
    "backL_paw": (0.34, 0.70, 0.03),
    "backR_hip": (0.66, 0.66, 0.55),
    "backR_knee": (0.66, 0.68, 0.40),
    "backR_ankle": (0.66, 0.70, 0.18),
    "backR_paw": (0.66, 0.70, 0.03),
}

PREFIX = "JOINT_"


def spawn_code(asset: Path) -> str:
    return f'''
import bpy
import json
from mathutils import Vector

asset_path = {str(asset)!r}
markers = {json.dumps(QUADRUPED_MARKERS)}
prefix = {PREFIX!r}

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for prior in list(bpy.data.collections):
    bpy.data.collections.remove(prior)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=asset_path)
meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
if not meshes:
    raise RuntimeError("no mesh imported")

lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for obj in meshes:
    for corner in obj.bound_box:
        world = obj.matrix_world @ Vector(corner)
        for axis in range(3):
            lo[axis] = min(lo[axis], world[axis])
            hi[axis] = max(hi[axis], world[axis])
size = hi - lo

# The mesh is reference geometry here; locking selection stops a stray click in the
# viewport from grabbing 101k faces instead of the marker the user meant to move.
for obj in meshes:
    obj.hide_select = True

collection = bpy.data.collections.new("JOINT_MARKERS")
bpy.context.scene.collection.children.link(collection)

radius = max(size) * 0.018
for name, (fx, fy, fz) in markers.items():
    empty = bpy.data.objects.new(prefix + name, None)
    empty.empty_display_type = "SPHERE"
    empty.empty_display_size = radius
    empty.location = (
        lo.x + size.x * fx,
        lo.y + size.y * fy,
        lo.z + size.z * fz,
    )
    empty.show_name = True
    collection.objects.link(empty)

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "MATERIAL"

print(json.dumps({{
    "spawned": len(markers),
    "bbox_min": [round(v, 4) for v in lo],
    "bbox_max": [round(v, 4) for v in hi],
}}))
'''


def read_code() -> str:
    return f'''
import bpy
import json

prefix = {PREFIX!r}
placed = {{}}
for obj in bpy.data.objects:
    if obj.name.startswith(prefix):
        loc = obj.matrix_world.translation
        placed[obj.name[len(prefix):]] = [round(loc.x, 5), round(loc.y, 5), round(loc.z, 5)]
print(json.dumps(placed, indent=2, sort_keys=True))
'''


def send(code: str, host: str, port: int) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(300)
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


def load_code(markers: dict[str, list[float]]) -> str:
    return f'''
import bpy, json
markers = json.loads({json.dumps(json.dumps(markers))})
coll = bpy.data.collections.get("JOINT_MARKERS")
if coll is None:
    coll = bpy.data.collections.new("JOINT_MARKERS")
    bpy.context.scene.collection.children.link(coll)
radius = 0.018
restored = []
for name, loc in markers.items():
    full = "JOINT_" + name
    obj = bpy.data.objects.get(full)
    if obj is None:
        obj = bpy.data.objects.new(full, None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = radius
        obj.show_name = True
        coll.objects.link(obj)
    obj.location = loc
    restored.append(name)
print(json.dumps({{"restored": len(restored)}}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["spawn", "read", "save", "load"])
    parser.add_argument("--file", type=Path, help="JSON path for save/load")
    parser.add_argument("--asset", type=Path, help="GLB to import (spawn mode)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    if args.mode == "spawn":
        if args.asset is None:
            raise SystemExit("spawn requires --asset")
        code = spawn_code(args.asset.expanduser().resolve())
    elif args.mode == "load":
        if args.file is None:
            raise SystemExit("load requires --file")
        code = load_code(json.loads(args.file.read_text()))
    else:
        code = read_code()

    response = send(code, args.host, args.port)

    if args.mode == "save":
        if args.file is None:
            raise SystemExit("save requires --file")
        placed = json.loads(json.loads(response[response.find(chr(123)):])
                            ["result"]["result"])
        if not placed:
            raise SystemExit("no JOINT_ markers in the scene — nothing to save")
        args.file.parent.mkdir(parents=True, exist_ok=True)
        args.file.write_text(json.dumps(placed, indent=2, sort_keys=True) + chr(10))
        print(f"saved {len(placed)} markers to {args.file}")
        return 0

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
