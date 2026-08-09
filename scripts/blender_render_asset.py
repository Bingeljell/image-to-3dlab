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


def blender_code(
    asset: Path,
    output_dir: Path,
    label: str,
    env: str,
    culled: bool = False,
    recalc_normals: bool = False,
) -> str:
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

# Start from a clean slate. A long-lived Blender session may still hold assets,
# lights, or a default cube from an earlier render or from other tooling (which may
# use different collection names). Because each asset is grounded at the origin, any
# survivor would occlude or interpenetrate the new one, so remove every object and
# collection up front; this script rebuilds the camera, lights, and world it needs.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for prior in list(bpy.data.collections):
    bpy.data.collections.remove(prior)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

collection = bpy.data.collections.new(collection_name)
bpy.context.scene.collection.children.link(collection)

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

# The honest test. glTF marks materials doubleSided, so a normal preview draws
# surfaces from behind and visually fills in every hole -- which is why the
# turntables, the wind demo and the cheers animation all looked fine over geometry
# that is substantially perforated. SceneKit and RealityKit cull backfaces, so the
# culled view is what the target engine will actually show.
if {recalc_normals!r}:
    # Culling WITHOUT fixing normals removes the correct faces and keeps the wrong
    # ones, which makes the mesh look MORE complete, not less. Note the converse
    # caution: on this asset Recalculate Outside has previously made a culled render
    # worse, tearing solid regions open, so render both ways before concluding.
    for obj in mesh_objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

if {culled!r}:
    # Plain grey, because texture disguises small gaps.
    grey = bpy.data.materials.new("HONEST_GREY_" + label)
    grey.use_nodes = True
    shader = grey.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.48, 0.48, 0.48, 1.0)
    shader.inputs["Roughness"].default_value = 0.62
    shader.inputs["Metallic"].default_value = 0.0
    grey.use_backface_culling = True
    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(grey)

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
# The look names are namespaced by view transform, so a session left on a different
# transform (by other tooling) makes the AgX look name invalid. Set both.
try:
    scene.view_settings.view_transform = "AgX"
except TypeError:
    pass
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    scene.view_settings.look = "Medium High Contrast"
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
    parser.add_argument(
        "--culled",
        action="store_true",
        help="the honest test: plain grey, backface culling on. Predicts what "
        "SceneKit/RealityKit will show, where a textured doubleSided render hides holes",
    )
    parser.add_argument(
        "--recalc-normals",
        action="store_true",
        help="Recalculate Outside before rendering. Culling without this keeps the "
        "wrong faces; but it has also torn solid regions open here, so render both ways",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "type": "execute_code",
        "params": {
            "code": blender_code(
                asset,
                output_dir,
                args.label,
                args.env,
                culled=args.culled,
                recalc_normals=args.recalc_normals,
            )
        },
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
