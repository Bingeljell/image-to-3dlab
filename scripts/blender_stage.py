#!/usr/bin/env python3
"""Put assets side by side in the RUNNING Blender, lit, textured and ready to look at.

    python scripts/blender_stage.py A.glb B.glb --labels "old" "new"

**Why this exists.** The render helpers in this repo wipe the scene and swap in their own
materials -- `compare_to_source.py --grey` in particular replaces every material with plain
grey. Run one of those after staging a comparison and the user is left looking at grey
blobs wondering where the texture went. That happened repeatedly. This script is the
counterpart: it stages for *looking at*, not for rendering, and it is the last thing to run
before handing the viewport back.

What it guarantees:

* every asset normalised to the same height and spread along Y, so size differences do not
  masquerade as quality differences
* three-point lighting, so the surface is readable
* viewport set to **Material Preview**, so textures actually show
* **backface culling on**, because a double-sided preview hides a hollow mesh completely
  and that is exactly the defect that went unnoticed here for weeks
* nothing selected, so no orange highlight is mistaken for a property of the mesh

This drives the live GUI over the socket on 9876. It replaces whatever is in the scene.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def spread_offsets(count: int, gap: float = 1.25) -> list[float]:
    """Y offsets that keep the group centred on the origin.

    Centred rather than starting at zero so `view_all` frames the set symmetrically and
    a two-asset comparison sits either side of the middle instead of off to one edge.
    """
    if count <= 0:
        return []
    middle = (count - 1) / 2.0
    return [(index - middle) * gap for index in range(count)]


def labels_for(assets: list[str], labels: list[str] | None) -> list[str]:
    """Collection names: given labels, else the file stems, always unique and prefixed.

    The index prefix matters: Blender's outliner sorts alphabetically, so without it the
    left-to-right order in the viewport and the top-to-bottom order in the outliner
    disagree, which makes "the second one" ambiguous.
    """
    chosen = list(labels) if labels else [Path(a).stem for a in assets]
    if len(chosen) != len(assets):
        raise SystemExit(f"got {len(assets)} assets but {len(chosen)} labels")
    cleaned = []
    for index, name in enumerate(chosen):
        safe = "".join(character if character.isalnum() else "_" for character in name)
        cleaned.append(f"{chr(ord('A') + index)}_{safe}"[:60])
    return cleaned


def build_code(assets: list[str], names: list[str], offsets: list[float], culled: bool) -> str:
    payload = json.dumps(
        [{"path": str(Path(a).expanduser().resolve()), "name": n, "dy": d}
         for a, n, d in zip(assets, names, offsets)]
    )
    return f'''
import bpy, json
from mathutils import Vector

items = json.loads({payload!r})
culled = {culled!r}

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for collection in list(bpy.data.collections):
    bpy.data.collections.remove(collection)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

for item in items:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=item["path"])
    fresh = [o for o in bpy.data.objects if o not in before]
    collection = bpy.data.collections.new(item["name"])
    bpy.context.scene.collection.children.link(collection)
    meshes = []
    for obj in fresh:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
        if obj.type == "MESH":
            meshes.append(obj)
    if not meshes:
        continue
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    centre = (low + high) * 0.5
    size = max(high.x - low.x, high.y - low.y, high.z - low.z) or 1.0
    for obj in fresh:
        if obj.parent is None:
            obj.scale = (1.0 / size,) * 3
            obj.location = -Vector((centre.x, centre.y, low.z)) / size
            obj.location.y += item["dy"]
    for obj in meshes:
        for slot in obj.material_slots:
            if slot.material:
                slot.material.use_backface_culling = culled

for name, location, energy in (
    ("KEY", (-2.8, -3.8, 4.5), 900),
    ("FILL", (3.2, -1.8, 2.7), 500),
    ("RIM", (1.0, 3.5, 4.0), 700),
):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = 4.0
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = Vector(location)
    light.rotation_euler = (Vector((0, 0, 0.5)) - light.location).to_track_quat("-Z", "Y").to_euler()

bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = None

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces[0].shading.type = "MATERIAL"   # textures visible
        for region in area.regions:
            if region.type == "WINDOW":
                with bpy.context.temp_override(area=area, region=region):
                    bpy.ops.view3d.view_all()

print("STAGED:", ", ".join(i["name"] for i in items))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("assets", nargs="+", type=str)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--gap", type=float, default=1.25)
    parser.add_argument(
        "--double-sided",
        action="store_true",
        help="turn backface culling OFF. Hides hollow meshes -- only for showing what a "
             "double-sided preview flatters",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    names = labels_for(args.assets, args.labels)
    offsets = spread_offsets(len(args.assets), args.gap)
    print(send(build_code(args.assets, names, offsets, not args.double_sided), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
