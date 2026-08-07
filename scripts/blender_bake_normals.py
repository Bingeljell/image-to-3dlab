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

# glTF is Y-up and Blender is Z-up, so the importer maps (x, y, z) -> (x, -z, y). The
# high-poly PLY never goes through that: it is written straight out of the generator and
# the PLY importer applies no conversion. Left uncorrected the two meshes sit 90 degrees
# apart, which a bounding-box fit cannot repair — it would simply squash one to match the
# other's dimensions and bake nonsense.
AXIS_PERMUTATIONS = {
    "none": (0, 1, 2),
    "gltf_to_blender": (0, 2, 1),
}


def axis_size_ratios(
    high_size: tuple[float, float, float],
    low_size: tuple[float, float, float],
    permutation: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Per-axis size ratio after permuting the high-poly's axes.

    All three land near 1.0 only when the permutation is right, which makes this a
    direct test for orientation rather than a guess.
    """
    permuted = tuple(high_size[i] for i in permutation)
    return tuple(
        (low_size[i] / permuted[i]) if permuted[i] > 1e-9 else 0.0 for i in range(3)
    )


def best_permutation(
    high_size: tuple[float, float, float],
    low_size: tuple[float, float, float],
) -> tuple[str, float]:
    """Pick the axis mapping whose size ratios are most uniform.

    Returns the name and the spread (max ratio / min ratio). A spread near 1.0 means the
    two meshes really are the same shape in that orientation; a large spread means
    neither candidate fits and the bake should not be trusted.
    """
    best_name, best_spread = None, float("inf")
    for name, perm in AXIS_PERMUTATIONS.items():
        ratios = axis_size_ratios(high_size, low_size, perm)
        if min(ratios) <= 0:
            continue
        spread = max(ratios) / min(ratios)
        if spread < best_spread:
            best_name, best_spread = name, spread
    if best_name is None:
        raise ValueError("no usable axis permutation; is one mesh degenerate?")
    return best_name, best_spread


TEMPLATE = '''
import bpy, json, math, time
from mathutils import Vector

LOW = {low!r}
HIGH = {high!r}
OUT = {out!r}
SIZE = {size}
RAY = {ray}
DECIMATE_TO = {decimate_to}
SAMPLES = {samples}
ROT_X = {rot_x}

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

# Orientation first, then fit. A bounding-box fit cannot repair a rotation — it would
# just squash the high-poly to match the low-poly's dimensions. The permutation is
# decided outside Blender and verified by per-axis size ratios, which land near 1.0
# only when the orientation is actually right.
if ROT_X:
    high.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.view_layer.update()

llo, lhi = bbox(low)
hlo, hhi = bbox(high)
lsize = lhi - llo; hsize = hhi - hlo
ratios = [
    (lsize[i] / hsize[i]) if hsize[i] > 1e-9 else 0.0
    for i in range(3)
]
spread = max(ratios) / min(ratios) if min(ratios) > 0 else 999.0
scale = sum(ratios) / 3.0
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
    "axis_ratios": [round(r, 4) for r in ratios],
    "orientation_spread": round(spread, 4),
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
    parser.add_argument(
        "--permutation", choices=sorted(AXIS_PERMUTATIONS), default="gltf_to_blender",
        help="Axis mapping from high-poly space to the low-poly's. The PLY comes "
             "straight from the generator while the GLB has been through the glTF "
             "exporter, so their Y and Z are swapped and the default corrects it",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    perm_name = args.permutation
    code = TEMPLATE.format(
        rot_x=(perm_name == "gltf_to_blender"),
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
