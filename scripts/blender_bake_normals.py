#!/usr/bin/env python3
"""Bake TRELLIS' discarded high-poly detail into a normal map for the low-poly mesh.

TRELLIS generates ~6.2 million triangles and the Mac port decimates to at most 200,000
before texture baking. That discarded 98% is real surface detail we already paid for.
A **normal map** captures it: the lighting engine then renders bumps that do not exist in
the geometry, so a 101k mesh lights like a multi-million-triangle one.

Unusually, this needs no sculpting step — the high-poly source already exists. Dump it
with `scripts/patch_trellis_highpoly.py` (`--dump-highpoly`), then bake with this.

Two details that decide whether the bake is usable:

* **Alignment.** The PLY comes straight from the generator; the GLB has been through the
  glTF exporter and Blender's Y-up to Z-up import conversion. Rather than guess the
  transform chain, the high-poly is aligned to the low-poly by bounding box, which is
  robust to any convention mismatch.
* **Ray distance.** Blender casts rays outward from the low-poly surface to find the
  high-poly. Too short and detail is missed; too long and rays hit the wrong surface
  across a gap — very visible on thin geometry like ears and leaf tips. Scaled from the
  model's size rather than hardcoded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

TEMPLATE = '''
import bpy, json, time
from mathutils import Vector

LOW = {low!r}
HIGH = {high!r}
OUT = {out!r}
SIZE = {size}
RAY = {ray}
DECIMATE_TO = {decimate_to}
SAMPLES = {samples}

t0 = time.time()
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for coll in list(bpy.data.collections):
    bpy.data.collections.remove(coll)

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=LOW)
low = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
low.name = "LOWPOLY"

before = set(bpy.data.objects)
bpy.ops.wm.ply_import(filepath=HIGH)
high = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
high.name = "HIGHPOLY"
high_tris_in = len(high.data.polygons)

# Optional decimation of the *bake source*. 6M triangles is slow to ray-cast against;
# a 1-2M source keeps nearly all the detail a 2048 map can represent.
if DECIMATE_TO and high_tris_in > DECIMATE_TO:
    bpy.context.view_layer.objects.active = high
    mod = high.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = DECIMATE_TO / high_tris_in
    bpy.ops.object.modifier_apply(modifier=mod.name)

def bbox(o):
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        for i in range(3):
            lo[i] = min(lo[i], w[i]); hi[i] = max(hi[i], w[i])
    return lo, hi

# Align by bounding box rather than by assuming a transform chain: the PLY is in raw
# generator space and the GLB has been through export plus a Y-up to Z-up import.
llo, lhi = bbox(low)
hlo, hhi = bbox(high)
lsize = lhi - llo; hsize = hhi - hlo
scale = min(
    (lsize[i] / hsize[i]) if hsize[i] > 1e-9 else 1.0
    for i in range(3)
)
high.scale = (scale, scale, scale)
bpy.context.view_layer.update()
hlo, hhi = bbox(high)
high.location = high.location + ((llo + lhi) / 2 - (hlo + hhi) / 2)
bpy.context.view_layer.update()
hlo, hhi = bbox(high)
align_err = max(abs((hlo[i] + hhi[i]) / 2 - (llo[i] + lhi[i]) / 2) for i in range(3))

# The bake target: a non-colour image wired into the low-poly's material.
img = bpy.data.images.new("NORMAL_BAKE", width=SIZE, height=SIZE, alpha=False,
                          float_buffer=False)
img.colorspace_settings.name = "Non-Color"

mat = low.data.materials[0] if low.data.materials else None
if mat is None:
    mat = bpy.data.materials.new("LowMat"); low.data.materials.append(mat)
mat.use_nodes = True
node = mat.node_tree.nodes.new("ShaderNodeTexImage")
node.image = img
mat.node_tree.nodes.active = node

scn = bpy.context.scene
scn.render.engine = "CYCLES"
try:
    scn.cycles.device = "GPU"
except Exception:
    pass
scn.cycles.samples = SAMPLES
scn.render.bake.use_selected_to_active = True
scn.render.bake.cage_extrusion = RAY
scn.render.bake.max_ray_distance = RAY
scn.render.bake.margin = 16
scn.render.bake.use_clear = True

bpy.ops.object.select_all(action="DESELECT")
high.select_set(True)
low.select_set(True)
bpy.context.view_layer.objects.active = low          # active = bake TARGET

err = None
try:
    bpy.ops.object.bake(type="NORMAL")
except RuntimeError as exc:
    err = str(exc)

if err is None:
    img.filepath_raw = OUT
    img.file_format = "PNG"
    img.save()

print(json.dumps({{
    "high_tris_in": high_tris_in,
    "high_tris_baked": len(high.data.polygons),
    "low_tris": len(low.data.polygons),
    "uniform_scale_applied": round(scale, 5),
    "alignment_error": round(align_err, 5),
    "ray_distance": RAY,
    "error": err,
    "written": None if err else OUT,
    "seconds": round(time.time() - t0, 1),
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low", type=Path, help="low-poly GLB (the bake target)")
    parser.add_argument("high", type=Path, help="high-poly PLY (the bake source)")
    parser.add_argument("out", type=Path, help="normal map PNG to write")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--ray", type=float, default=0.02,
        help="How far rays search for the high-poly surface, in world units. Too short "
             "misses detail; too long jumps a gap and bakes the wrong surface, which is "
             "very visible on thin geometry like ears and leaf tips",
    )
    parser.add_argument(
        "--decimate-to", type=int, default=1_500_000,
        help="Decimate the bake source to this many triangles first. 6M is slow to "
             "ray-cast against and a 2048 map cannot represent that much detail anyway. "
             "0 disables",
    )
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    code = TEMPLATE.format(
        low=str(args.low.expanduser().resolve()),
        high=str(args.high.expanduser().resolve()),
        out=str(args.out.expanduser().resolve()),
        size=args.size, ray=args.ray,
        decimate_to=args.decimate_to, samples=args.samples,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
