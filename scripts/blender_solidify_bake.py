#!/usr/bin/env python3
"""Voxel-remesh a mesh into a closed surface and bake its albedo back on.

SUPERSEDED (2026-08-06). This was written to fix a "confetti mesh" of tens of thousands
of disconnected shards. That diagnosis was a measurement error: `merge_vertices` does
not merge across UV seams, so the component count was really a count of UV islands.
Measured position-only, the meshes are single connected surfaces -- so this script was
solving a problem that did not exist. See `docs/open-questions.md` question 1.

Kept because the remesh-and-rebake machinery may still be useful, and so the dead end
is documented rather than rediscovered. Do not reach for it to "fix topology".
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def blender_code(
    asset: Path,
    output: Path,
    voxel_size: float,
    texture_size: int,
    extrusion: float,
    min_component_verts: int,
) -> str:
    return f'''
import bpy
from mathutils import Vector

asset_path = {str(asset)!r}
output_path = {str(output)!r}
voxel_size = {voxel_size}
texture_size = {texture_size}
extrusion = {extrusion}
min_component_verts = {min_component_verts}

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
imported = [o for o in bpy.data.objects if o not in before]
meshes = [o for o in imported if o.type == "MESH"]

# The importer may split the asset across objects; bake wants a single source.
for o in bpy.data.objects:
    o.select_set(False)
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
source = bpy.context.view_layer.objects.active
source.name = "SOURCE"

# Absolute voxel size is meaningless without knowing the asset's scale, so derive it
# from the bounding box when the caller passes a non-positive value.
corners = [source.matrix_world @ Vector(c) for c in source.bound_box]
lo = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
hi = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
if voxel_size <= 0:
    voxel_size = extent * 0.002

target = source.copy()
target.data = source.data.copy()
bpy.context.scene.collection.objects.link(target)
target.name = "TARGET"

for o in bpy.data.objects:
    o.select_set(False)
target.select_set(True)
bpy.context.view_layer.objects.active = target

remesh = target.modifiers.new("Remesh", "REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = voxel_size
remesh.adaptivity = 0.0
bpy.ops.object.modifier_apply(modifier=remesh.name)

# Voxelising welds nearby geometry into a handful of large closed bodies
# (torso, head, each limb) plus a lot of tiny specks. Full fusion into one body would
# need a voxel far coarser than the face can survive, so instead keep every substantial
# component and drop only the specks. Each survivor is closed, which is what kills the
# see-through artefact; the pieces do not need to be welded to each other for that.
import bmesh

bm = bmesh.new()
bm.from_mesh(target.data)
bm.verts.ensure_lookup_table()

component = [-1] * len(bm.verts)
sizes = []
current = 0
for seed in range(len(bm.verts)):
    if component[seed] != -1:
        continue
    stack = [seed]
    component[seed] = current
    size = 0
    while stack:
        index = stack.pop()
        size += 1
        for edge in bm.verts[index].link_edges:
            other = edge.other_vert(bm.verts[index]).index
            if component[other] == -1:
                component[other] = current
                stack.append(other)
    sizes.append(size)
    current += 1

keep = {{index for index, size in enumerate(sizes) if size >= min_component_verts}}
if not keep:  # threshold too aggressive; fall back to the largest component
    keep = {{max(range(len(sizes)), key=lambda i: sizes[i])}}
debris = [v for v in bm.verts if component[v.index] not in keep]
if debris:
    bmesh.ops.delete(bm, geom=debris, context="VERTS")
bm.to_mesh(target.data)
bm.free()
target.data.update()
components_kept = len(keep)
components_removed = current - components_kept

# The remeshed surface carries no usable UVs, so make a fresh atlas to bake into.
target.data.materials.clear()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.002)
bpy.ops.object.mode_set(mode="OBJECT")

baked = bpy.data.images.new("BAKED", texture_size, texture_size, alpha=False)
material = bpy.data.materials.new("BAKED_MAT")
material.use_nodes = True
nodes = material.node_tree.nodes
bsdf = nodes.get("Principled BSDF")
if bsdf is not None:
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
tex_node = nodes.new("ShaderNodeTexImage")
tex_node.image = baked
tex_node.select = True
nodes.active = tex_node
target.data.materials.append(material)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
try:
    scene.cycles.device = "GPU"
except Exception:
    pass
scene.cycles.samples = 1
scene.cycles.bake_type = "DIFFUSE"
scene.render.bake.use_pass_direct = False
scene.render.bake.use_pass_indirect = False
scene.render.bake.use_pass_color = True
scene.render.bake.use_selected_to_active = True
scene.render.bake.cage_extrusion = extrusion if extrusion > 0 else voxel_size * 6.0
scene.render.bake.max_ray_distance = (extrusion if extrusion > 0 else voxel_size * 6.0) * 2.0

for o in bpy.data.objects:
    o.select_set(False)
source.select_set(True)
target.select_set(True)
bpy.context.view_layer.objects.active = target
bpy.ops.object.bake(type="DIFFUSE")

# Pack the baked image so the glTF exporter embeds it rather than chasing a file path.
baked.pack()
if bsdf is not None:
    material.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

bpy.data.objects.remove(source, do_unlink=True)
for o in bpy.data.objects:
    o.select_set(False)
target.select_set(True)
bpy.context.view_layer.objects.active = target
bpy.ops.export_scene.gltf(
    filepath=output_path,
    export_format="GLB",
    use_selection=True,
    export_materials="EXPORT",
)

print(
    "SOLIDIFY:: voxel %.5f faces %d verts %d kept %d removed %d"
    % (voxel_size, len(target.data.polygons), len(target.data.vertices), components_kept, components_removed)
)

result = {{
    "voxel_size": voxel_size,
    "extent": extent,
    "faces": len(target.data.polygons),
    "verts": len(target.data.vertices),
    "components_kept": components_kept,
    "debris_components_removed": components_removed,
    "output": output_path,
}}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="remesh voxel size; <=0 derives 0.2%% of the asset's largest dimension",
    )
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument(
        "--min-component-verts",
        type=int,
        default=200,
        help="drop remeshed components smaller than this many vertices (specks)",
    )
    parser.add_argument(
        "--extrusion",
        type=float,
        default=0.0,
        help="bake cage extrusion; <=0 derives 6x the voxel size",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    code = blender_code(
        asset,
        output,
        args.voxel_size,
        args.texture_size,
        args.extrusion,
        args.min_component_verts,
    )
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        # Remesh plus a full-resolution bake is slow; give it room.
        connection.settimeout(3600)
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
    print(b"".join(chunks).decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
