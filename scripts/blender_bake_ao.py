"""Bake an ambient-occlusion map from an asset's own geometry, headless.

Run: Blender -b -P scripts/blender_bake_ao.py -- <asset.glb> <ao_out.png> [samples] [size] [distance_frac]

Unlike the normal and roughness maps in ``surface_detail.py``, this is *measured*, not
invented: it asks where the geometry actually shadows itself. On a coiled subject that
is most of what sells the form, because every place a limb tucks under another should
darken and currently does not.

The bake reuses the asset's existing UVs, so the AO map lines up with the albedo atlas
and can be attached as ``occlusionTexture`` without repacking anything.
"""

import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: ... -- <asset.glb> <ao_out.png> [samples] [size] [distance_frac]")

asset, ao_out = argv[0], argv[1]
samples = int(argv[2]) if len(argv) > 2 else 24
size = int(argv[3]) if len(argv) > 3 else 2048
distance_frac = float(argv[4]) if len(argv) > 4 else 0.04

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=asset)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit(f"no mesh in {asset}")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
print(f"baking AO for {obj.name}: {len(obj.data.polygons)} faces, {size}px, {samples} samples")

if not obj.data.uv_layers:
    raise SystemExit("asset has no UVs; AO cannot be baked to a texture")

# The bake target is whichever Image Texture node is active in the material, so make one.
target = bpy.data.images.new("AO", width=size, height=size)
for slot in obj.material_slots:
    mat = slot.material
    if mat is None:
        continue
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = target
    node.select = True
    mat.node_tree.nodes.active = node

scene = bpy.context.scene
scene.render.engine = "CYCLES"
# MPS is not a Cycles device; CPU is the reliable path on Apple Silicon and an AO bake
# is cheap enough that it does not matter.
scene.cycles.device = "CPU"
scene.cycles.samples = samples
scene.render.bake.use_clear = True
scene.render.bake.margin = 8

# **The ray distance is the whole ballgame.** Left at its default the rays reach across
# the entire subject, so a coiled mass answers "am I buried inside the pile?" -- which is
# yes almost everywhere -- instead of "am I in a crevice?". The first bake here came back
# 68% occluded and rendered as mud. Scaling the distance to a fraction of the subject's
# own size keeps it as contact shading, which is what actually sells the form.
dims = obj.dimensions
diagonal = max((dims.x ** 2 + dims.y ** 2 + dims.z ** 2) ** 0.5, 1e-6)
if scene.world is None:
    scene.world = bpy.data.worlds.new("W")
scene.world.light_settings.distance = diagonal * distance_frac
print(f"AO ray distance = {scene.world.light_settings.distance:.4f} "
      f"({distance_frac:.3f} of the {diagonal:.3f} diagonal)")

bpy.ops.object.bake(type="AO")
target.filepath_raw = ao_out
target.file_format = "PNG"
target.save()
print(f"SAVED_AO {ao_out}")
