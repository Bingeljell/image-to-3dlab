"""Re-unwrap a generated mesh into coherent UV islands and re-bake its texture, headless.

Run with:
    blender --background --python scripts/blender_reunwrap_bake.py -- IN.glb OUT.glb [size]

**Why.** TRELLIS unwraps with xatlas per cluster, which has no idea what the parts of a
creature are. Measured on Flicker at 50k faces: **6,763 UV islands with a median size of
11 texels** -- roughly 3x3 pixels each. Nothing painted into that atlas can hold a crisp
edge, because the surface it lives on is diced into fragments smaller than the edge. That
is why marking projection produced soft, torn lines no matter how the blend was tuned, and
why widening the gutter padding only spilled colour onto neighbouring islands.

Face count helps a little (100k -> 50k roughly halves the island count) and the generator's
own `--uv-refine-iterations` / `--uv-cone-degrees` knobs made it *worse* when tested, so
the fix has to be a real re-unwrap.

**What this does.** Smart UV Project for large, coherent islands, then an emission bake
that reads the original albedo through the *original* UVs and writes it into the *new*
layout. Geometry is untouched -- only the UV layout and the texture change -- so this
composes with everything downstream, including `scripts/project_markings.py`.

Baking margin is deliberately generous: the whole point is islands big enough that a few
texels of bleed no longer reach a neighbour.

Headless on purpose. Driving a Cycles bake through the live Blender socket blocks the
GUI's handler and loses the reply.
"""

from __future__ import annotations

import math
import sys


def parse_args(argv: list[str]) -> tuple[str, str, int, float, float]:
    """Arguments after Blender's ``--`` separator.

    Kept import-safe and free of ``bpy`` so it can be tested without Blender.
    """
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 2:
        raise SystemExit(
            "usage: blender --background --python scripts/blender_reunwrap_bake.py "
            "-- IN.glb OUT.glb [size]"
        )
    size = int(argv[2]) if len(argv) > 2 else 2048
    if size not in (1024, 2048, 4096):
        raise SystemExit(f"atlas size must be 1024, 2048 or 4096, got {size}")
    # Degrees in, radians out. 89 merges hard; Blender's own default of 66 still left
    # 4,449 islands on a 50k-face creature.
    angle_degrees = float(argv[3]) if len(argv) > 3 else 89.0
    island_margin = float(argv[4]) if len(argv) > 4 else 0.0002
    if not 0.0 < angle_degrees <= 89.9:
        raise SystemExit(f"angle limit must be in (0, 89.9] degrees, got {angle_degrees}")
    if not 0.0 <= island_margin < 0.01:
        raise SystemExit(
            f"island margin is a fraction of the atlas per island and must stay tiny; "
            f"got {island_margin}"
        )
    return argv[0], argv[1], size, math.radians(angle_degrees), island_margin


def base_colour_image(material):
    """The image feeding a glTF material's Base Color, or None.

    The importer builds Principled BSDF <- Image Texture. Finding it by walking the link
    rather than by node name, because the importer's naming is not stable across versions.
    """
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        socket = node.inputs.get("Base Color")
        if socket and socket.is_linked:
            upstream = socket.links[0].from_node
            if upstream.type == "TEX_IMAGE" and upstream.image:
                return upstream.image
    for node in material.node_tree.nodes:  # fall back to any image in the tree
        if node.type == "TEX_IMAGE" and node.image:
            return node.image
    return None


def main() -> int:
    import bpy

    source, destination, size, angle_limit, island_margin = parse_args(list(sys.argv))

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh found in {source}")
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    material = obj.data.materials[0] if obj.data.materials else None
    original_image = base_colour_image(material)
    if original_image is None:
        raise SystemExit("could not find the base colour texture to re-bake")
    original_uv = obj.data.uv_layers.active.name
    print(f"REUNWRAP:: source uv={original_uv} texture={original_image.size[0]}px")

    # --- new UV layout ---------------------------------------------------------------
    new_uv = obj.data.uv_layers.new(name="reuv")
    obj.data.uv_layers.active = new_uv
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # angle_limit is how far face normals may diverge before Smart Project cuts a new
    # island: HIGHER merges more aggressively, which is what we want -- some stretch is a
    # fair trade for islands big enough to hold an edge.
    #
    # island_margin is a fraction of the whole atlas *per island*, so it must stay tiny.
    # At 0.005 with a few thousand islands the margins consume the entire atlas and every
    # island collapses -- measured at 1% coverage, median island 2 texels. The bake margin
    # set below provides the gutter instead.
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- bake target -----------------------------------------------------------------
    target = bpy.data.images.new("reuv_albedo", width=size, height=size, alpha=False)

    tree = material.node_tree
    nodes, links = tree.nodes, tree.links

    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = original_uv           # read through the ORIGINAL layout
    source_tex = nodes.new("ShaderNodeTexImage")
    source_tex.image = original_image
    source_tex.interpolation = "Closest"   # keep marking edges hard through the bake
    links.new(uv_node.outputs["UV"], source_tex.inputs["Vector"])

    emission = nodes.new("ShaderNodeEmission")
    links.new(source_tex.outputs["Color"], emission.inputs["Color"])
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    target_tex = nodes.new("ShaderNodeTexImage")
    target_tex.image = target             # unconnected: bake destination only
    nodes.active = target_tex
    for node in nodes:
        node.select = node is target_tex

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1          # emission bake needs exactly one sample
    except AttributeError:
        pass
    scene.render.bake.margin = 16         # gutter padding, now that islands are large
    scene.render.bake.use_clear = True

    print("REUNWRAP:: baking...")
    bpy.ops.object.bake(type="EMIT")

    # --- rewire to the baked texture and drop the old layout -------------------------
    principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(target_tex.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    for stale in (uv_node, source_tex, emission):
        nodes.remove(stale)

    old = obj.data.uv_layers.get(original_uv)
    if old is not None and len(obj.data.uv_layers) > 1:
        obj.data.uv_layers.remove(old)
    obj.data.uv_layers.active = obj.data.uv_layers[0]

    bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB")
    print(f"REUNWRAP:: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
