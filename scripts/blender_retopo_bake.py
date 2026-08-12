"""Quad-retopologise a generated mesh and transfer its texture onto the clean topology.

Run with:
    blender --background --python scripts/blender_retopo_bake.py -- IN.glb OUT.glb [faces] [size]

**Why.** Painting markings into a TRELLIS atlas cannot produce a crisp edge, and the cause
is not the unwrapper. Measured on Flicker (50k faces, 2048 atlas):

    TRELLIS xatlas          64.4% coverage   6,763 islands   median island 11 px
    + its own UV knobs      66.1%            9,052           13 px   (worse)
    Blender Smart UV        32.2%           10,943            6 px   (worse)

TRELLIS's own unwrap is the best of the three, and all of them are confetti. Generated
meshes are a chaotic triangle soup with no coherent surface flow, so there are no natural
charts for any unwrapper to find. **You cannot unwrap your way out of bad topology.**

So fix the topology. QuadriFlow rebuilds the surface as quads that follow its curvature,
which unwraps into large islands -- and then everything downstream, including
`scripts/project_markings.py`, works as designed.

**The bake is selected-to-active**, not a UV re-bake: remeshing throws the old UVs away, so
the texture has to be transferred from the original mesh onto the new one by ray casting
between the two surfaces. That is why this is a separate script from
`blender_reunwrap_bake.py`.

**What this costs.** Retopology is destructive to fine detail -- it resamples the surface.
Good for smooth subjects (Flicker's ceramic); expect it to damage foliage and fur, which
is consistent with remesh having previously destroyed leafy meshes here.

Headless on purpose: a Cycles bake through the live Blender socket blocks the GUI.
"""

from __future__ import annotations

import math
import sys


def parse_args(argv: list[str]) -> tuple[str, str, int, int, float, float]:
    """Arguments after Blender's ``--`` separator. Kept free of ``bpy`` so it is testable."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 2:
        raise SystemExit(
            "usage: blender --background --python scripts/blender_retopo_bake.py "
            "-- IN.glb OUT.glb [target_faces] [atlas_size] [angle_degrees] [voxel_fraction]"
        )
    target_faces = int(argv[2]) if len(argv) > 2 else 20000
    size = int(argv[3]) if len(argv) > 3 else 2048
    angle_degrees = float(argv[4]) if len(argv) > 4 else 89.0
    voxel_fraction = float(argv[5]) if len(argv) > 5 else 0.004
    if size not in (1024, 2048, 4096):
        raise SystemExit(f"atlas size must be 1024, 2048 or 4096, got {size}")
    if not 1000 <= target_faces <= 200000:
        raise SystemExit(
            f"target faces must be 1000..200000; too few loses the silhouette, too many "
            f"re-fragments the atlas. got {target_faces}"
        )
    if not 0.0 < angle_degrees <= 89.9:
        raise SystemExit(f"angle limit must be in (0, 89.9] degrees, got {angle_degrees}")
    if not 0.0005 <= voxel_fraction <= 0.05:
        raise SystemExit(
            f"voxel size is a fraction of the asset's largest dimension; too coarse melts "
            f"the subject, too fine runs out of memory. got {voxel_fraction}"
        )
    return argv[0], argv[1], target_faces, size, math.radians(angle_degrees), voxel_fraction


def base_colour_image(material):
    """The image feeding a glTF material's Base Color, or None."""
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
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image
    return None


def ray_distance(dimensions, fraction: float = 0.02) -> float:
    """How far the bake may search from the new surface to find the old one.

    Scaled to the asset rather than fixed: too small and the new surface misses the old
    one entirely, leaving the atlas empty; too large and a ray from the chest can reach
    the far side of the body and sample its colour.
    """
    return max(dimensions) * fraction


def main() -> int:
    import bpy

    source, destination, target_faces, size, angle_limit, voxel_fraction = parse_args(
        list(sys.argv)
    )

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh found in {source}")
    original = meshes[0]

    material = original.data.materials[0] if original.data.materials else None
    image = base_colour_image(material)
    if image is None:
        raise SystemExit("could not find the base colour texture to transfer")
    print(f"RETOPO:: in faces={len(original.data.polygons):,} texture={image.size[0]}px")

    # --- clean topology ---------------------------------------------------------------
    # An explicit copy, NOT bpy.ops.object.duplicate(): that can produce a linked
    # duplicate sharing mesh data, so clearing materials on the copy also strips them
    # from the original and the bake silently finds no target.
    retopo = original.copy()
    retopo.data = original.data.copy()
    retopo.name = "RETOPO"
    retopo.data.materials.clear()
    bpy.context.scene.collection.objects.link(retopo)

    bpy.ops.object.select_all(action="DESELECT")
    retopo.select_set(True)
    bpy.context.view_layer.objects.active = retopo

    # QuadriFlow refuses a mesh that is not manifold with consistent normals -- which
    # describes every asset here. Voxel remeshing rebuilds the surface as a watertight
    # manifold first, which also closes the see-through holes as a side effect. It
    # resamples the surface, so fine detail suffers; acceptable while the bar is visible
    # quality rather than exact geometry.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    voxel = max(retopo.dimensions) * voxel_fraction
    print(f"RETOPO:: voxel remesh at {voxel:.4f} ...")
    retopo.data.remesh_voxel_size = voxel
    retopo.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    print(f"RETOPO:: manifold faces={len(retopo.data.polygons):,}")

    print(f"RETOPO:: quadriflow to {target_faces:,} faces (slow)...")
    try:
        bpy.ops.object.quadriflow_remesh(
            target_faces=target_faces,
            use_preserve_boundary=False,
            use_mesh_symmetry=False,
        )
    except RuntimeError as exc:
        print(f"RETOPO:: quadriflow failed ({exc}); keeping the voxel mesh")
    print(f"RETOPO:: out faces={len(retopo.data.polygons):,}")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=0.0002)
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- emission material on the ORIGINAL, so the bake reads its albedo --------------
    emit = bpy.data.materials.new("EMIT_SRC")
    emit.use_nodes = True
    nodes, links = emit.node_tree.nodes, emit.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    original.data.materials.clear()
    original.data.materials.append(emit)

    # --- bake target on the RETOPO ----------------------------------------------------
    target_image = bpy.data.images.new("retopo_albedo", width=size, height=size, alpha=False)
    dst = bpy.data.materials.new("RETOPO_MAT")
    dst.use_nodes = True
    dnodes, dlinks = dst.node_tree.nodes, dst.node_tree.links
    dst_tex = dnodes.new("ShaderNodeTexImage")
    dst_tex.image = target_image
    retopo.data.materials.append(dst)
    # Order matters: Blender resolves the bake target from the ACTIVE material slot's
    # active+selected image node, so assign the material first, make its slot active,
    # and only then mark the node. Doing this before assignment silently bakes nowhere.
    retopo.active_material_index = 0
    for node in dnodes:
        node.select = False
    dst_tex.select = True
    dnodes.active = dst_tex

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
    except AttributeError:
        pass
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.margin = 16
    bake.use_clear = True
    bake.cage_extrusion = ray_distance(retopo.dimensions)
    bake.max_ray_distance = ray_distance(retopo.dimensions) * 2.0

    bpy.ops.object.select_all(action="DESELECT")
    original.select_set(True)          # source
    retopo.select_set(True)
    bpy.context.view_layer.objects.active = retopo   # destination
    print("RETOPO:: baking original -> retopo ...")
    bpy.ops.object.bake(type="EMIT")

    # --- ship the retopo mesh with the baked albedo -----------------------------------
    principled = dnodes.new("ShaderNodeBsdfPrincipled")
    doutput = next(n for n in dnodes if n.type == "OUTPUT_MATERIAL")
    dlinks.new(dst_tex.outputs["Color"], principled.inputs["Base Color"])
    dlinks.new(principled.outputs["BSDF"], doutput.inputs["Surface"])

    bpy.data.objects.remove(original, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    retopo.select_set(True)
    bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB", use_selection=True)
    print(f"RETOPO:: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
