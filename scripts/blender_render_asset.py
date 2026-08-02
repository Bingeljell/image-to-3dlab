#!/usr/bin/env python3
"""Import a GLB into the local Blender MCP scene and render cardinal previews."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

# Render environments. "dark" is the near-black studio that flatters matte
# assets; "studio" lifts the world and enables ray-traced reflections so a
# metallic (pbr-mode) surface has something to reflect instead of black.
ENVIRONMENTS = {
    "dark": {"world_color": (0.035, 0.035, 0.035), "raytracing": False},
    "studio": {"world_color": (0.22, 0.19, 0.15), "raytracing": True},
}


def blender_code(asset: Path, output_dir: Path, label: str, env: str) -> str:
    settings = ENVIRONMENTS[env]
    world_color = settings["world_color"]
    raytracing = settings["raytracing"]
    return f'''
import bpy
import math
from mathutils import Vector

asset_path = {str(asset)!r}
output_dir = {str(output_dir)!r}
label = {label!r}
collection_name = "IMG3D_" + label

# Remove assets imported by any previous run in this long-lived Blender session.
# Each run grounds its asset at the origin, so a leftover mesh from an earlier
# label would interpenetrate the new one (e.g. an old head fused into the new body).
for prior in list(bpy.data.collections):
    if prior.name.startswith("IMG3D_"):
        for obj in list(prior.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(prior)

collection = bpy.data.collections.new(collection_name)
bpy.context.scene.collection.children.link(collection)

# A freshly launched Blender ships a default Cube/Light/Camera that can occlude the
# asset or hijack the active camera. Remove them so previews are reproducible.
for _default in ("Cube", "Light", "Camera"):
    _obj = bpy.data.objects.get(_default)
    if _obj is not None:
        bpy.data.objects.remove(_obj, do_unlink=True)

# Free the now-unreferenced meshes/materials/images from earlier imports.
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=asset_path)
imported = [obj for obj in bpy.data.objects if obj not in before]
for obj in imported:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)

root = bpy.data.objects.new(collection_name + "_ROOT", None)
collection.objects.link(root)
for obj in imported:
    if obj.parent is None:
        obj.parent = root

# Blender's glTF importer already converts Y-up (glTF) to Z-up on import. Adding a
# further 90 degree rotation here double-rotated the mesh and laid it face-down.
bpy.context.view_layer.update()

mesh_objects = [obj for obj in imported if obj.type == "MESH"]
world_corners = [obj.matrix_world @ Vector(corner) for obj in mesh_objects for corner in obj.bound_box]
min_corner = Vector((min(v.x for v in world_corners), min(v.y for v in world_corners), min(v.z for v in world_corners)))
max_corner = Vector((max(v.x for v in world_corners), max(v.y for v in world_corners), max(v.z for v in world_corners)))
center = (min_corner + max_corner) * 0.5
root.location -= Vector((center.x, center.y, min_corner.z))
bpy.context.view_layer.update()

world_corners = [obj.matrix_world @ Vector(corner) for obj in mesh_objects for corner in obj.bound_box]
min_corner = Vector((min(v.x for v in world_corners), min(v.y for v in world_corners), min(v.z for v in world_corners)))
max_corner = Vector((max(v.x for v in world_corners), max(v.y for v in world_corners), max(v.z for v in world_corners)))
center = (min_corner + max_corner) * 0.5
size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)

camera_data = bpy.data.cameras.new(collection_name + "_CAMERA")
camera = bpy.data.objects.new(collection_name + "_CAMERA", camera_data)
collection.objects.link(camera)
camera_data.lens = 58
bpy.context.scene.camera = camera

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

for suffix, location, energy, scale in (
    ("KEY", (-2.8, -3.8, 4.5), 700, 4.0),
    ("FILL", (3.2, -1.8, 2.7), 420, 3.5),
    ("RIM", (1.0, 3.5, 4.0), 600, 3.0),
):
    light_data = bpy.data.lights.new(collection_name + "_" + suffix, "AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = scale
    light = bpy.data.objects.new(collection_name + "_" + suffix, light_data)
    collection.objects.link(light)
    light.location = Vector(location) * size
    aim(light, center)

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 900
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.world.color = {world_color}
try:
    scene.eevee.use_raytracing = {raytracing}
except AttributeError:
    pass

distance = size * 2.25
target = center + Vector((0, 0, size * 0.03))
views = {{
    "front_xneg": (-distance, 0, target.z),
    "front_three_quarter": (-distance * 0.9, -distance * 0.45, target.z),
    "rear_xpos": (distance, 0, target.z),
    "profile_yneg": (0, -distance, target.z),
    "profile_ypos": (0, distance, target.z),
}}
for name, location in views.items():
    camera.location = Vector(location)
    aim(camera, target)
    scene.render.filepath = output_dir + "/" + label + "_" + name + ".png"
    bpy.ops.render.render(write_still=True)

result = {{
    "objects": [obj.name for obj in mesh_objects],
    "materials": [slot.material.name for obj in mesh_objects for slot in obj.material_slots if slot.material],
    "bounds": [list(min_corner), list(max_corner)],
    "renders": [output_dir + "/" + label + "_" + name + ".png" for name in views],
}}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="asset")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--env",
        choices=tuple(ENVIRONMENTS),
        default="dark",
        help="dark flatters matte assets; studio lifts the world for metallic (pbr) assets",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "type": "execute_code",
        "params": {"code": blender_code(asset, output_dir, args.label, args.env)},
    }
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(300)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks = []
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
    print(b"".join(chunks).decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
