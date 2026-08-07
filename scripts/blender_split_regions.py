#!/usr/bin/env python3
"""Split a generated mesh into per-region material slots, each with its own texture.

**2048 is the cap per *material*, not per model.** The whole fox currently shares one
2048 atlas, so at 101k faces every triangle gets ~41 texels and the head — where detail
matters most — receives a small share of a fixed budget. Giving the head, body and tail
their own material slot and their own 2048 map triples the budget outright, without
changing a single generation setting. Game characters normally ship this way.

Measured on the moss fox with the default cuts:

    head   35,824 faces   41 -> 117 texels/triangle   2.8x
    body   36,827 faces   41 -> 114 texels/triangle   2.8x
    tail   28,647 faces   41 -> 146 texels/triangle   3.5x

How it works: each region is unwrapped independently so it fills the whole 0..1 UV
square, then the original albedo is baked into a fresh 2048 map per region. Regions
overlap in UV space, which is fine and is the point — they sample different images.

The source material must sample through an explicit UV Map node pointing at the
*original* layout, because the bake target uses the *active* layout. Without that the
bake reads from the new UVs it is writing into and produces garbage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

REGION_NAMES = ("head", "body", "tail")


def classify_faces(
    face_axis_positions: list[float],
    head_cut: float,
    tail_cut: float,
) -> list[str]:
    """Assign each face to a region from its position along the front-back axis.

    Positions are the face centroid's coordinate on the axis running from tail to head,
    with the head at the positive end. `head_cut` and `tail_cut` are absolute positions
    on that axis, not fractions, so they can be read straight off the model's bounds.
    """
    if head_cut <= tail_cut:
        raise ValueError("head_cut must be greater than tail_cut")
    return [
        "head" if p > head_cut else "tail" if p < tail_cut else "body"
        for p in face_axis_positions
    ]


def region_counts(assignments: list[str]) -> dict[str, int]:
    return {name: assignments.count(name) for name in REGION_NAMES}


def texel_density(face_count: int, texture_size: int) -> float:
    """Texels available per triangle when `face_count` faces share one square atlas."""
    if face_count <= 0:
        return 0.0
    return (texture_size * texture_size) / face_count


TEMPLATE = '''
import bpy, json, time
from mathutils import Vector

ASSET = {asset!r}
OUT_DIR = {out_dir!r}
SIZE = {size}
HEAD_CUT = {head_cut}
TAIL_CUT = {tail_cut}
REGIONS = ["head", "body", "tail"]

t0 = time.time()

def _is_precious(o):
    return o.name.startswith("JOINT_") or o.type == "ARMATURE"

for obj in list(bpy.data.objects):
    if not _is_precious(obj):
        bpy.data.objects.remove(obj, do_unlink=True)

bpy.ops.import_scene.gltf(filepath=ASSET)
mesh = [o for o in bpy.data.objects if o.type == "MESH" and not _is_precious(o)][0]
bpy.context.view_layer.objects.active = mesh
me = mesh.data

src_uv = me.uv_layers.active
src_uv_name = src_uv.name
new_uv = me.uv_layers.new(name="region_uv")

src_mat = me.materials[0]
src_tex = None
for node in src_mat.node_tree.nodes:
    if node.type == "TEX_IMAGE" and node.image is not None:
        src_tex = node.image
        break
if src_tex is None:
    raise RuntimeError("source material has no image texture to bake from")

# One material per region, each sampling the ORIGINAL albedo through an explicit UV Map
# node. Without that node the shader would read the active (new) layout — the very one
# the bake writes into — and bake noise.
targets = {{}}
me.materials.clear()
for i, region in enumerate(REGIONS):
    mat = bpy.data.materials.new("moss_fox_" + region)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = src_tex
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = src_uv_name
    nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    img = bpy.data.images.new("albedo_" + region, width=SIZE, height=SIZE, alpha=False)
    bake_node = nt.nodes.new("ShaderNodeTexImage")
    bake_node.image = img
    nt.nodes.active = bake_node          # active node = bake destination
    targets[region] = img
    me.materials.append(mat)

# Assign each face to a region by its centroid along the front-back axis. glTF is Y-up
# so the importer maps the model's front-back onto Blender's Y, with the head at -Y.
counts = {{r: 0 for r in REGIONS}}
for poly in me.polygons:
    centre = sum((me.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)
    axis = -centre.y                     # positive toward the head
    region = "head" if axis > HEAD_CUT else "tail" if axis < TAIL_CUT else "body"
    poly.material_index = REGIONS.index(region)
    counts[region] += 1

# Unwrap each region on its own so it fills the whole 0..1 square. Regions overlap in UV
# space, which is correct: they sample different images.
me.uv_layers.active = new_uv
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="DESELECT")
bpy.ops.object.mode_set(mode="OBJECT")

for i, region in enumerate(REGIONS):
    for poly in me.polygons:
        poly.select = (poly.material_index == i)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")

scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.samples = 1
scn.render.bake.use_selected_to_active = False
scn.render.bake.margin = 8
scn.render.bake.use_clear = True

bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh

err = None
try:
    bpy.ops.object.bake(type="EMIT")     # EMIT bakes the colour straight through
except RuntimeError as exc:
    err = str(exc)

written = {{}}
if err is None:
    for region, img in targets.items():
        path = OUT_DIR + "/albedo_" + region + ".png"
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        written[region] = path

    # Point each material at its own freshly baked map, through the new UV layout.
    for i, region in enumerate(REGIONS):
        mat = me.materials[i]
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = targets[region]
        uvmap = nt.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = "region_uv"
        nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    while me.uv_layers.get(src_uv_name):
        me.uv_layers.remove(me.uv_layers[src_uv_name])

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=OUT_DIR + "/moss_fox_regions.glb",
        export_format="GLB", use_selection=True,
    )

print(json.dumps({{
    "faces_per_region": counts,
    "texture_size": SIZE,
    "written": written,
    "glb": None if err else OUT_DIR + "/moss_fox_regions.glb",
    "error": err,
    "seconds": round(time.time() - t0, 1),
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--head-cut", type=float, default=0.25,
        help="Position on the front-back axis above which faces are head",
    )
    parser.add_argument(
        "--tail-cut", type=float, default=-0.10,
        help="Position below which faces are tail",
    )
    args = parser.parse_args()

    if args.head_cut <= args.tail_cut:
        raise SystemExit("--head-cut must be greater than --tail-cut")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    code = TEMPLATE.format(
        asset=str(args.asset.expanduser().resolve()),
        out_dir=str(args.out_dir.expanduser().resolve()),
        size=args.size, head_cut=args.head_cut, tail_cut=args.tail_cut,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
