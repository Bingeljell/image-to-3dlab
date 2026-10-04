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


def quadriflow_reduced(before: int, after: int, target_faces: int) -> bool:
    """Did QuadriFlow actually retopologise, or silently decline?

    It refuses with a Blender *warning* rather than an exception when its preconditions
    are not met, leaving the mesh untouched. Comparing counts is the only reliable signal.
    Generous tolerance: QuadriFlow approximates the target rather than hitting it exactly,
    so anything within 3x of what was asked for counts as having run.
    """
    if after == before:
        return False
    return after <= max(target_faces * 3, 1)


def parse_args(argv: list[str]) -> tuple[str, str, int, int, float, float, float, float, float]:
    """Arguments after Blender's ``--`` separator. Kept free of ``bpy`` so it is testable."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 2:
        raise SystemExit(
            "usage: blender --background --python scripts/blender_retopo_bake.py "
            "-- IN.glb OUT.glb [target_faces] [atlas_size] [angle_degrees] "
            "[voxel_fraction] [metallic] [roughness] [ior]"
        )
    target_faces = int(argv[2]) if len(argv) > 2 else 20000
    size = int(argv[3]) if len(argv) > 3 else 2048
    angle_degrees = float(argv[4]) if len(argv) > 4 else 89.0
    voxel_fraction = float(argv[5]) if len(argv) > 5 else 0.004
    # Surface response. Only base colour is baked, so the source's metallic-roughness map
    # is not carried over and these stand in for it. The right values depend on what the
    # source artwork is -- wet bark, dry stone, painted metal -- so they are tuned per
    # asset rather than fixed. The defaults are a neutral organic surface: a little sheen,
    # fairly rough, not plastic and not chrome.
    metallic = float(argv[6]) if len(argv) > 6 else 0.25
    roughness = float(argv[7]) if len(argv) > 7 else 0.65
    ior = float(argv[8]) if len(argv) > 8 else 1.45
    for name, value in (("metallic", metallic), ("roughness", roughness)):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be within 0..1, got {value}")
    if not 1.0 <= ior <= 3.0:
        raise SystemExit(f"ior must be within 1.0..3.0, got {ior}")
    # 256 and 512 are for the far LODs finish_props.py bakes, which are seen small.
    if size not in (256, 512, 1024, 2048, 4096):
        raise SystemExit(f"atlas size must be 256, 512, 1024, 2048 or 4096, got {size}")
    if not 1000 <= target_faces <= 200000:
        raise SystemExit(
            f"target faces must be 1000..200000; too few loses the silhouette, too many "
            f"re-fragments the atlas. got {target_faces}"
        )
    if not 0.0 < angle_degrees <= 89.9:
        raise SystemExit(f"angle limit must be in (0, 89.9] degrees, got {angle_degrees}")
    if voxel_fraction != 0.0 and not 0.0005 <= voxel_fraction <= 0.05:
        raise SystemExit(
            f"voxel size is a fraction of the asset's largest dimension; too coarse melts "
            f"the subject, too fine runs out of memory, and 0 skips the remesh. "
            f"got {voxel_fraction}"
        )
    return (argv[0], argv[1], target_faces, size, math.radians(angle_degrees),
            voxel_fraction, metallic, roughness, ior)


def decimate_ratio(target_faces: int, triangle_count: int) -> float:
    """The Decimate ratio that actually lands on `target_faces` triangles.

    **The trap.** Blender's COLLAPSE decimation applies its ratio to *triangles*, but the
    voxel remesh before it emits *quads*, and `len(mesh.polygons)` counts those. Dividing
    the target by the polygon count therefore asks for twice as many faces as intended,
    every time: the Pixal3D fox was asked for 40,000 and came out at 79,991 from 200,632
    quads (401,264 triangles), and the Snag was asked for 20,000 and came out at 39,361.
    Every face target this repo has ever set has been silently doubled.

    So the ratio is computed against the triangle count, which is what the modifier reads.
    """
    if triangle_count <= 0:
        return 1.0
    return min(1.0, target_faces / triangle_count)


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

    (source, destination, target_faces, size, angle_limit, voxel_fraction,
     metallic, roughness, ior) = parse_args(list(sys.argv))

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
    # Weld first, always. A glTF mesh arrives split along every UV and normal seam, and
    # unwelded it *measures* as broken: the shipped Snag reads 237,359 non-manifold edges
    # (43.7%) as loaded and 2,671 (0.63%) once welded — the same file. Every operation
    # that walks the surface, decimation included, sees the split version until this runs.
    bpy.ops.mesh.remove_doubles(threshold=1e-6)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"RETOPO:: welded faces={len(retopo.data.polygons):,}")

    if voxel_fraction:
        voxel = max(retopo.dimensions) * voxel_fraction
        print(f"RETOPO:: voxel remesh at {voxel:.4f} ...")
        retopo.data.remesh_voxel_size = voxel
        retopo.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        print(f"RETOPO:: manifold faces={len(retopo.data.polygons):,}")
    else:
        # Skipping the remesh is worth testing per asset. It was adopted because the raw
        # mesh appeared hopelessly non-manifold and shattered under decimation, but that
        # reading came from measuring it unwelded. The voxel pass costs the creases — it
        # resamples the surface — so where a welded mesh decimates cleanly, going direct
        # keeps detail the remesh throws away.
        print("RETOPO:: voxel remesh skipped (voxel_fraction=0); decimating the weld")

    # Recalculate normals AFTER the voxel remesh, not only before it. QuadriFlow requires
    # a manifold mesh whose face normals point consistently, and it refuses with a
    # *warning* rather than an exception when they do not -- so the try/except below never
    # fires, the operator silently does nothing, and what ships is the voxel mesh. On the
    # Snag that meant 1.3M triangles where 20,000 were asked for.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    before = len(retopo.data.polygons)
    print(f"RETOPO:: quadriflow to {target_faces:,} faces (slow)...")
    try:
        bpy.ops.object.quadriflow_remesh(
            target_faces=target_faces,
            use_preserve_boundary=False,
            use_mesh_symmetry=False,
        )
    except RuntimeError as exc:
        print(f"RETOPO:: quadriflow raised ({exc})")
    after = len(retopo.data.polygons)
    if not quadriflow_reduced(before, after, target_faces):
        # QuadriFlow refuses every mesh this pipeline produces. Measured 2026-09-20: it is
        # not manifoldness (zero non-manifold edges and vertices after the voxel remesh),
        # not component count (refused on a 5-component mesh), and not size (refused at
        # 58k faces); it accepts a primitive sphere and rejects these. Its error message
        # names manifoldness regardless, so it is not to be trusted as a diagnosis.
        #
        # Collapse decimation is the fallback. It loses the quad topology, which matters
        # for deformation but not for a display asset carrying a normal map. What makes it
        # work here is the ordering: decimating the *raw* mesh is what shattered thin
        # geometry, because that mesh has hundreds of thousands of non-manifold edges.
        # The voxel remesh above produces a watertight surface first, and decimating that
        # preserves the silhouette.
        print(f"RETOPO:: quadriflow declined ({before:,} unchanged); decimating instead")
        modifier = retopo.modifiers.new("retopo_decimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        # Against triangles, not polygons -- see decimate_ratio. The voxel remesh above
        # emits quads, so counting polygons here asks for twice the target.
        retopo.data.calc_loop_triangles()
        triangles = len(retopo.data.loop_triangles)
        modifier.ratio = decimate_ratio(target_faces, triangles)
        print(f"RETOPO:: decimating {triangles:,} triangles to {target_faces:,} "
              f"(ratio {modifier.ratio:.4f})")
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        after = len(retopo.data.polygons)

    print(f"RETOPO:: out faces={after:,}")
    if after >= before:
        raise SystemExit(
            f"RETOPO:: nothing reduced the mesh ({before:,} -> {after:,}, asked for "
            f"{target_faces:,}). Writing this would be worse than the input, so it fails "
            "rather than producing a file that looks retopologised and is not."
        )

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

    # Only base colour is baked, so the source's metallic-roughness *map* does not come
    # across and the material would otherwise ship mathematically flat -- metallic 0,
    # roughness 0.5, no map -- which reads as dead plastic under any light. Flat factors
    # are a poor substitute for the map but a large improvement on nothing, and the right
    # values are asset-specific, so they are a knob rather than a constant.
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    if "IOR" in principled.inputs:
        principled.inputs["IOR"].default_value = ior

    bpy.data.objects.remove(original, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    retopo.select_set(True)
    bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB", use_selection=True)
    print(f"RETOPO:: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
