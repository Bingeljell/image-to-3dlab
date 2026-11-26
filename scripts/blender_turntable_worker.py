"""Build a private Blender scene and render a turntable frame sequence.

Run (internal): bundled and sent to Blender by blender_turntable.py.
"""

from __future__ import annotations

import math

if __package__:
    from .blender_scene import clear_private_scene
elif __name__ != "_assetfurnace_worker":
    from blender_scene import clear_private_scene


def run(arguments: dict[str, object]) -> dict[str, object]:
    """Render in the turntable scene and restore the window even on failure."""
    import bpy
    from mathutils import Vector

    scene_name = "IMG3D turntable"
    previous_scene = bpy.context.window.scene
    render_scene = bpy.data.scenes.get(scene_name) or bpy.data.scenes.new(scene_name)
    bpy.context.window.scene = render_scene
    try:
        if render_scene.world is None:
            render_scene.world = bpy.data.worlds.new(render_scene.name + " World")
        clear_private_scene(bpy, render_scene)
        return render_frames(bpy, Vector, render_scene, **arguments)
    finally:
        bpy.context.window.scene = previous_scene


def render_frames(
    bpy,
    Vector,
    render_scene,
    *,
    asset_path: str,
    frame_dir: str,
    label: str,
    frames: int,
    resolution: int,
    zoom: float,
    darken_backfaces: bool,
    world_color: list[float],
    raytracing: bool,
) -> dict[str, object]:
    collection_name = "IMG3D_TT_" + label
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=asset_path)
    imported = [obj for obj in render_scene.objects if obj not in before]
    for obj in imported:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)

    # The turntable spins this empty; the mesh rides along as its child.
    spin = bpy.data.objects.new(collection_name + "_SPIN", None)
    collection.objects.link(spin)
    for obj in imported:
        if obj.parent is None:
            obj.parent = spin

    bpy.context.view_layer.update()

    mesh_objects = [obj for obj in imported if obj.type == "MESH"]

    # A TRELLIS mesh has open holes and inconsistent winding, so some views expose the
    # shell's inner surface. That
    # interior carries the same skin-toned texture, which is why a hole in the back of a
    # head reads as a floating eye and gaps in a sweater read as bare skin. Shading
    # backfaces near-black turns those gaps into plain shadow, which reads as creases
    # rather than artefacts -- a render-time fix that needs no change to the geometry.
    if darken_backfaces:
        for obj in mesh_objects:
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                tree = material.node_tree
                output = next(
                    (
                        n
                        for n in tree.nodes
                        if n.type == "OUTPUT_MATERIAL" and n.is_active_output
                    ),
                    None,
                )
                if output is None or not output.inputs["Surface"].is_linked:
                    continue
                front = output.inputs["Surface"].links[0].from_socket
                geometry = tree.nodes.new("ShaderNodeNewGeometry")
                dark = tree.nodes.new("ShaderNodeBsdfDiffuse")
                dark.inputs["Color"].default_value = (0.012, 0.012, 0.012, 1.0)
                mix = tree.nodes.new("ShaderNodeMixShader")
                tree.links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
                tree.links.new(front, mix.inputs[1])
                tree.links.new(dark.outputs["BSDF"], mix.inputs[2])
                tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])

    def bounds():
        corners = [
            obj.matrix_world @ Vector(c) for obj in mesh_objects for c in obj.bound_box
        ]
        lo = Vector(
            (
                min(v.x for v in corners),
                min(v.y for v in corners),
                min(v.z for v in corners),
            )
        )
        hi = Vector(
            (
                max(v.x for v in corners),
                max(v.y for v in corners),
                max(v.z for v in corners),
            )
        )
        return lo, hi

    # Ground the asset and centre it on the spin axis, otherwise it wobbles as it turns.
    min_corner, max_corner = bounds()
    center = (min_corner + max_corner) * 0.5
    spin.location -= Vector((center.x, center.y, min_corner.z))
    bpy.context.view_layer.update()

    min_corner, max_corner = bounds()
    center = (min_corner + max_corner) * 0.5
    size = max(
        max_corner.x - min_corner.x,
        max_corner.y - min_corner.y,
        max_corner.z - min_corner.z,
    )

    camera_data = bpy.data.cameras.new(collection_name + "_CAMERA")
    camera = bpy.data.objects.new(collection_name + "_CAMERA", camera_data)
    collection.objects.link(camera)
    camera_data.lens = 58
    bpy.context.scene.camera = camera

    def aim(obj, target):
        obj.rotation_euler = (
            (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
        )

    for suffix, location, energy, scale in (
        ("KEY", (-2.8, -3.8, 4.5), 700, 4.0),
        ("FILL", (3.2, -1.8, 2.7), 420, 3.5),
        ("RIM", (1.0, 3.5, 4.0), 600, 3.0),
    ):
        light_data = bpy.data.lights.new(collection_name + "_" + suffix, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = scale
        light = bpy.data.objects.new(collection_name + "_" + suffix, light_data)
        collection.objects.link(light)
        light.location = Vector(location) * size
        aim(light, center)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    # The look names are namespaced by view transform, so a session left on a different
    # transform (by other tooling) makes the AgX look name invalid. Set both.
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "Medium High Contrast"
    scene.world.color = world_color
    try:
        scene.eevee.use_raytracing = raytracing
    except AttributeError:
        pass

    # This model's forward axis is -Y, so park the camera there for a face-on frame 0.
    # `size` is the widest dimension, which for a T-pose figure is the arm span, so the
    # default preview distance leaves the body small in frame; --zoom pulls the camera in.
    distance = size * 2.25 / zoom
    target = center + Vector((0, 0, size * 0.03))
    camera.location = Vector((0, -distance, target.z))
    aim(camera, target)

    # Stop one step short of a full turn so the loop wraps without a duplicate frame.
    step = 2.0 * math.pi / frames
    written = []
    for index in range(frames):
        spin.rotation_euler = (0.0, 0.0, index * step)
        bpy.context.view_layer.update()
        path = f"{frame_dir}/{label}_{index:04d}.png"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)

    return {
        "frames": len(written),
        "size": size,
        "first": written[0],
        "last": written[-1],
    }
