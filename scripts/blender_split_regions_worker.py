"""Bake independent texture maps for the regions of a Blender mesh.

Run (internal): bundled by blender_split_regions.py.
"""

from __future__ import annotations

import json
import time

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


def assign_regions(mesh, head_cut: float, tail_cut: float, vector) -> dict[str, int]:
    """Set polygon material slots using centroids along Blender's negative Y axis."""
    positions = [
        -sum((mesh.vertices[index].co for index in poly.vertices), vector())[1] / len(poly.vertices)
        for poly in mesh.polygons
    ]
    assignments = classify_faces(positions, head_cut, tail_cut)
    for poly, region in zip(mesh.polygons, assignments, strict=True):
        poly.material_index = REGION_NAMES.index(region)
    return region_counts(assignments)


def run(arguments: dict[str, object], *, bpy=None, vector=None) -> dict:
    """Run in the private scene and restore the active window afterward."""
    if bpy is None:
        import bpy
    if vector is None:
        from mathutils import Vector as vector

    t0 = time.time()
    scene_name = "IMG3D regions"
    window = bpy.context.window
    previous_scene = window.scene
    render_scene = bpy.data.scenes.get(scene_name) or bpy.data.scenes.new(scene_name)
    window.scene = render_scene
    if render_scene.world is None:
        render_scene.world = bpy.data.worlds.new(render_scene.name + " World")
    for obj in list(render_scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(render_scene.collection.children):
        bpy.data.collections.remove(collection)
    result = bake_regions(bpy, vector, render_scene, t0, **arguments)
    window.scene = previous_scene
    return result


def bake_regions(
    bpy, Vector, render_scene, t0: float, *, asset_path: str, out_dir: str,
    size: int, head_cut: float, tail_cut: float,
) -> dict:
    output_path = f"{out_dir}/moss_fox_regions.glb"
    bpy.ops.import_scene.gltf(filepath=asset_path)
    mesh = [o for o in render_scene.objects if o.type == "MESH" and not o.name.startswith("JOINT_")][0]  # noqa: RUF015 - preserve missing-mesh error
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
    targets = {}
    me.materials.clear()
    for region in REGION_NAMES:
        mat = bpy.data.materials.new(f"moss_fox_{region}")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(n)
        out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]  # noqa: RUF015 - preserve missing-node error
        emit = nt.nodes.new("ShaderNodeEmission")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = src_tex
        uvmap = nt.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = src_uv_name
        nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

        img = bpy.data.images.new("albedo_" + region, width=size, height=size, alpha=False)
        bake_node = nt.nodes.new("ShaderNodeTexImage")
        bake_node.image = img
        nt.nodes.active = bake_node          # active node = bake destination
        targets[region] = img
        me.materials.append(mat)

    # Assign each face to a region by its centroid along the front-back axis. glTF is Y-up
    # so the importer maps the model's front-back onto Blender's Y, with the head at -Y.
    counts = assign_regions(me, head_cut, tail_cut, Vector)

    # Re-pack rather than re-unwrap. A fresh Smart UV Project on this mesh produces one
    # island per triangle and leaves ~63% of the atlas empty, which more than cancels the
    # density gained. The original layout already works, so copy it and simply pack each
    # region's existing islands to fill their own square: same islands, three times the area.
    for loop_index in range(len(me.loops)):
        new_uv.data[loop_index].uv = src_uv.data[loop_index].uv
    me.uv_layers.active = new_uv

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    for i in range(len(REGION_NAMES)):
        for poly in me.polygons:
            poly.select = (poly.material_index == i)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.uv.select_all(action="SELECT")
        # scale=True grows the islands to fill the square. Do NOT call average_islands_scale
        # first: this atlas has 11,340 islands of wildly differing sizes and normalising them
        # shrinks everything, dropping fill from the source layout's 97% to about 14%.
        # Margin has to be tiny for the same reason — with ~3,800 islands per region the
        # gutters dominate long before the content does.
        bpy.ops.uv.pack_islands(rotate=True, scale=True, margin=0.0002,
                                shape_method="AABB")
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

    written = {}
    if err is None:
        for region, img in targets.items():
            path = out_dir + "/albedo_" + region + ".png"
            img.filepath_raw = path
            img.file_format = "PNG"
            img.save()
            written[region] = path

        # Point each material at its own freshly baked map, through the new UV layout.
        for i, region in enumerate(REGION_NAMES):
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
            filepath=output_path,
            export_format="GLB", use_selection=True,
        )

    report = {
        "faces_per_region": counts,
        "texture_size": size,
        "written": written,
        "glb": None if err else output_path,
        "error": err,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report, indent=2))
    return report
