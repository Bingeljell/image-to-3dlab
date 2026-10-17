#!/usr/bin/env python3
"""Append a rigged character (armature + skinned mesh) into the currently
live Blender scene, cleaned up and positioned -- reusable instead of
hand-writing a fresh append/cleanup script per character per session.

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


CODE_TEMPLATE = '''
import bpy, json, math
from mathutils import Vector

path = {path!r}
prefix = {prefix!r}
armature_name = {armature_name!r}
mesh_name = {mesh_name!r}
position = {position!r}
facing = {facing!r}
action_name = {action_name!r}
frame = {frame!r}
do_save = {do_save!r}

def sk_to_blender(x, y, z):
    return (x, -z, y)

before = set(bpy.data.objects.keys())
requested_actions = [action_name] if action_name else []
with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
    data_to.objects = [n for n in data_from.objects if n in (armature_name, mesh_name)]
    data_to.actions = [n for n in data_from.actions if n in requested_actions]

imported = [o for o in data_to.objects if o is not None]
for obj in imported:
    bpy.context.collection.objects.link(obj)

armature_obj = next((o for o in imported if o.type == "ARMATURE"), None)
mesh_obj = next((o for o in imported if o.type == "MESH"), None)
if armature_obj is None or mesh_obj is None:
    raise RuntimeError(f"expected an armature named {{armature_name!r}} and a mesh named {{mesh_name!r}} in {{path!r}}, found {{[o.name for o in imported]}}")

armature_obj.name = f"{{prefix}}_rig"
mesh_obj.name = f"{{prefix}}_mesh"

# Auto-cleanup: anything the append pulled in besides the two objects we
# asked for is either a bone-shape widget (keep, hide from render) or an
# unrelated dependency from the source file (camera rigs, effect props,
# weighting proxies) that has no business in the target scene.
after = set(bpy.data.objects.keys())
new_names = after - before - {{armature_obj.name, mesh_obj.name}}
removed = []
widgets_hidden = []
for name in new_names:
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    if name.startswith("WGT-"):
        obj.hide_render = True
        widgets_hidden.append(name)
    else:
        bpy.data.objects.remove(obj, do_unlink=True)
        removed.append(name)

root = bpy.data.objects.new(f"{{prefix}}_root", None)
root.empty_display_size = 0.5
bpy.context.collection.objects.link(root)
for o in (armature_obj, mesh_obj):
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()

root.location = sk_to_blender(*position)

if facing is not None:
    target = Vector(sk_to_blender(*facing))
    to_target = target - Vector(root.location)
    to_target.z = 0
    if to_target.length > 0.0001:
        root.rotation_euler = (0, 0, math.atan2(to_target.x, to_target.y))

if action_name:
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise RuntimeError(f"action {{action_name!r}} was not found in {{path!r}}")
    armature_obj.animation_data.action = action

if frame is not None:
    bpy.context.scene.frame_set(frame)

if do_save:
    bpy.ops.wm.save_mainfile()

result = {{
    "root": root.name,
    "armature": armature_obj.name,
    "mesh": mesh_obj.name,
    "location": list(root.location),
    "removed_strays": removed,
    "widgets_hidden": len(widgets_hidden),
    "saved": do_save,
}}
print("IMPORT_RESULT " + json.dumps(result))
result
'''


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

    code = CODE_TEMPLATE.format(
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
