"""Delete every face never seen from outside the mesh, headless.

Run: Blender -b -P scripts/blender_visibility_cull.py -- <in.glb> <out.glb> [views] [res]

Renders the mesh from `views` directions spread over a sphere. Instead of colour, each
pixel encodes the index of the nearest face, so reading the image back tells us exactly
which faces are observable. Faces in no image are interior junk, enclosed fragments or
back-facing debris, and get deleted.

Why this and not a distance-based repair: see ``visibility_cull.py``. In short, every
method that reasons about proximity either fails to weld (the shards are not coincident)
or fuses neighbouring coils (voxel remesh). Visibility never asks about proximity at all.

Two details that matter and are easy to get wrong:

- **Backface culling stays OFF.** This mesh has inconsistent winding, so a legitimate
  outer face may have a flipped normal. Culling backfaces would delete it for a reason
  that has nothing to do with whether it is really on the outside.
- **Colour management must be neutral.** The pixels are data, not a picture. With a view
  transform applied, the encoded indices come back wrong and the cull silently deletes
  the wrong faces.
"""

import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.visibility_cull import (
    color_to_index,
    cull_report,
    fibonacci_directions,
    index_to_color,
)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: ... -- <in.glb> <out.glb> [views] [resolution]")
src, dst = argv[0], argv[1]
views = int(argv[2]) if len(argv) > 2 else 200
res = int(argv[3]) if len(argv) > 3 else 512

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit(f"no mesh in {src}")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
me = obj.data
face_count = len(me.polygons)
print(f"culling {obj.name}: {face_count} faces, {views} views at {res}px")

# --- paint each face with its own index -------------------------------------------
attr = me.color_attributes.new(name="fid", type="FLOAT_COLOR", domain="CORNER")
loop_total = np.empty(face_count, dtype=np.int32)
loop_start = np.empty(face_count, dtype=np.int32)
me.polygons.foreach_get("loop_total", loop_total)
me.polygons.foreach_get("loop_start", loop_start)

face_rgb = index_to_color(np.arange(face_count)).astype(np.float64) / 255.0
colors = np.ones((len(me.loops), 4), dtype=np.float64)
colors[:, :3] = np.repeat(face_rgb, loop_total, axis=0)
attr.data.foreach_set("color", colors.ravel())

mat = bpy.data.materials.new("fid")
mat.use_nodes = True
mat.use_backface_culling = False
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
col = nt.nodes.new("ShaderNodeVertexColor")
col.layer_name = "fid"
nt.links.new(col.outputs["Color"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

# The index shader is a temporary OVERRIDE, not a replacement. Clearing the original
# materials here and exporting afterwards ships a mesh with correct UVs and no albedo,
# which is useless to every downstream finishing step. Keep the originals, add the
# index material as an extra slot, and restore the assignment before export.
original_indices = np.empty(face_count, dtype=np.int32)
me.polygons.foreach_get("material_index", original_indices)
me.materials.append(mat)
fid_slot = len(me.materials) - 1
me.polygons.foreach_set(
    "material_index", np.full(face_count, fid_slot, dtype=np.int32)
)

# --- a camera that sees the whole subject, with no filtering anywhere -------------
bpy.context.view_layer.update()
corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
lo = Vector(min(c[i] for c in corners) for i in range(3))
hi = Vector(max(c[i] for c in corners) for i in range(3))
centre = (lo + hi) / 2.0
diag = max((hi - lo).length, 1e-6)

cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = diag * 1.05
cam_data.clip_start = 0.001
cam_data.clip_end = diag * 4.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)

sc = bpy.context.scene
sc.camera = cam
for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        sc.render.engine = engine
        break
    except TypeError:
        continue
sc.render.resolution_x = sc.render.resolution_y = res
sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.render.filter_size = 0.0            # any blur mixes neighbouring face IDs
sc.render.dither_intensity = 0.0       # dithering corrupts the encoded bytes
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.image_settings.color_depth = "8"
if hasattr(sc, "eevee"):
    for attr_name, value in (("taa_render_samples", 1), ("use_gtao", False)):
        if hasattr(sc.eevee, attr_name):
            setattr(sc.eevee, attr_name, value)
# The pixels are data. A view transform would re-map them and corrupt every index.
for name in ("Raw", "Standard"):
    try:
        sc.view_settings.view_transform = name
        break
    except TypeError:
        continue
sc.view_settings.look = "None"
sc.display_settings.display_device = "sRGB"

tmp = Path(bpy.app.tempdir) / "fid.png"
seen = np.zeros(face_count, dtype=bool)

for n, direction in enumerate(fibonacci_directions(views)):
    d = Vector(direction.tolist())
    cam.location = centre + d * diag * 1.5
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(str(tmp), check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    buf = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)

    rgba = buf.reshape(-1, 4)
    rgb = np.rint(rgba[:, :3] * 255.0).astype(np.int64)
    idx = color_to_index(rgb)
    idx = idx[(idx >= 0) & (idx < face_count)]
    seen[idx] = True
    if (n + 1) % 25 == 0:
        print(f"  {n + 1}/{views} views, {int(seen.sum())} faces seen so far")

print("CULL " + cull_report(seen))
# Saved so the same decision can be re-applied without re-rendering 400 views.
np.save(str(Path(dst).with_suffix(".visible.npy")), seen)

# --- put the original materials back before touching geometry ----------------------
me.polygons.foreach_set("material_index", original_indices)
me.materials.pop(index=fid_slot)
bpy.data.materials.remove(mat)

# --- delete what was never seen ----------------------------------------------------
bm = bmesh.new()
bm.from_mesh(me)
bm.faces.ensure_lookup_table()
doomed = [bm.faces[i] for i in np.flatnonzero(~seen)]
bmesh.ops.delete(bm, geom=doomed, context="FACES")
bm.to_mesh(me)
bm.free()
me.color_attributes.remove(me.color_attributes["fid"])
print(f"after cull: {len(me.polygons)} faces")

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=dst, use_selection=True, export_format="GLB")
print(f"SAVED {dst}")
