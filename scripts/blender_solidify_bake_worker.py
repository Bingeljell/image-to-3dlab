"""Remesh and bake a model in a private Blender scene.

Run (internal): bundled by blender_solidify_bake.py.
"""

from __future__ import annotations


def component_filter(vertex_count: int, neighbours, min_vertices: int) -> tuple[list[int], int, int]:
    """Return debris vertex indexes and counts of retained and removed components."""
    component = [-1] * vertex_count
    sizes = []
    for seed in range(vertex_count):
        if component[seed] != -1:
            continue
        current = len(sizes)
        stack = [seed]
        component[seed] = current
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            for other in neighbours(index):
                if component[other] == -1:
                    component[other] = current
                    stack.append(other)
        sizes.append(size)
    keep = {index for index, size in enumerate(sizes) if size >= min_vertices}
    if not keep:
        keep = {max(range(len(sizes)), key=sizes.__getitem__)}
    debris = [index for index, group in enumerate(component) if group not in keep]
    return debris, len(keep), len(sizes) - len(keep)


def run(arguments: dict[str, object], *, bpy=None, vector=None, bmesh=None) -> dict:
    """Run in the private scene and restore the active window afterward."""
    if bpy is None:
        import bpy
    if vector is None:
        from mathutils import Vector as vector
    if bmesh is None:
        import bmesh

    scene_name = "IMG3D solidify"
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
    result = bake_mesh(bpy, vector, render_scene, bmesh, **arguments)
    window.scene = previous_scene
    return result


def bake_mesh(
    bpy, Vector, render_scene, bmesh, *, asset_path: str, output_path: str,
    voxel_size: float, texture_size: int, extrusion: float, min_component_verts: int,
) -> dict:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=asset_path)
    imported = [o for o in render_scene.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]

    # The importer may split the asset across objects; bake wants a single source.
    for o in render_scene.objects:
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

    for o in render_scene.objects:
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
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bm.verts.ensure_lookup_table()
    def neighbours(index):
        vertex = bm.verts[index]
        return (edge.other_vert(vertex).index for edge in vertex.link_edges)

    debris, components_kept, components_removed = component_filter(
        len(bm.verts), neighbours, min_component_verts
    )
    if debris:
        bmesh.ops.delete(bm, geom=[bm.verts[index] for index in debris], context="VERTS")
    bm.to_mesh(target.data)
    bm.free()
    target.data.update()

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
    except Exception:  # noqa: S110 - Blender builds without GPU support keep their default.
        pass
    scene.cycles.samples = 1
    scene.cycles.bake_type = "DIFFUSE"
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = extrusion if extrusion > 0 else voxel_size * 6.0
    scene.render.bake.max_ray_distance = (extrusion if extrusion > 0 else voxel_size * 6.0) * 2.0

    for o in render_scene.objects:
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
    for o in render_scene.objects:
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
        f"SOLIDIFY:: voxel {voxel_size:.5f} faces {len(target.data.polygons)} "
        f"verts {len(target.data.vertices)} kept {components_kept} removed {components_removed}"
    )

    return {
        "voxel_size": voxel_size,
        "extent": extent,
        "faces": len(target.data.polygons),
        "verts": len(target.data.vertices),
        "components_kept": components_kept,
        "debris_components_removed": components_removed,
        "output": output_path,
    }
