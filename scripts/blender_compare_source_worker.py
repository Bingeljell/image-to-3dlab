"""Render transparent panels for the source comparison command.

Run (headless): blender --background --python-exit-code 1 --python scripts/blender_compare_source_worker.py -- '<request JSON>'
"""

from __future__ import annotations

import json
import math
import sys

ENVIRONMENTS = {
    "dark": {"world_color": (0.05, 0.05, 0.06), "raytracing": True},
    "studio": {"world_color": (0.32, 0.32, 0.34), "raytracing": True},
}


def orbit_position(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    target_z: float = 0.0,
) -> tuple[float, float, float]:
    """Camera position on a sphere around the asset.

    Azimuth 0 sits in front of the asset at -X, which is the convention the existing
    ``blender_render_asset`` views already use (``front_xneg``). Positive azimuth swings
    toward -Y, so the 3/4 front view those views call ``front_three_quarter`` is +27ish.
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = distance * math.cos(elevation)
    return (
        -horizontal * math.cos(azimuth),
        -horizontal * math.sin(azimuth),
        distance * math.sin(elevation) + target_z,
    )


def scene_bounds(meshes, vector):
    """Find the world-space bounds of the imported meshes."""
    corners = [obj.matrix_world @ vector(corner) for obj in meshes for corner in obj.bound_box]
    return (
        vector(tuple(min(corner[axis] for corner in corners) for axis in range(3))),
        vector(tuple(max(corner[axis] for corner in corners) for axis in range(3))),
    )


def run(
    asset_path: str,
    output_dir: str,
    label: str,
    angles: list[float],
    elevation_deg: float,
    env: str,
    grey_mode: bool,
    *,
    bpy=None,
    vector=None,
) -> dict:
    """Render one panel per angle and return their paths and the asset extent."""
    if bpy is None:
        import bpy
    if vector is None:
        from mathutils import Vector
    else:
        Vector = vector

    settings = ENVIRONMENTS[env]
    collection_name = "IMG3D_CMP_" + label

    # Headless Blender, factory-empty. This used to wipe whatever file was open on the
    # GUI socket, which is how a day of unsaved work disappeared.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=asset_path)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)

    root = bpy.data.objects.new(collection_name + "_ROOT", None)
    collection.objects.link(root)
    for obj in imported:
        if obj.parent is None:
            obj.parent = root
    bpy.context.view_layer.update()

    mesh_objects = [obj for obj in imported if obj.type == "MESH"]

    if grey_mode:
        # Strip every texture. What survives is shape, not paint.
        grey = bpy.data.materials.new("CMP_GREY_" + label)
        grey.use_nodes = True
        shader = grey.node_tree.nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
        shader.inputs["Roughness"].default_value = 0.55
        shader.inputs["Metallic"].default_value = 0.0
        grey.use_backface_culling = True
        for obj in mesh_objects:
            obj.data.materials.clear()
            obj.data.materials.append(grey)

    min_corner, max_corner = scene_bounds(mesh_objects, Vector)
    center = (min_corner + max_corner) * 0.5
    root.location -= Vector((center.x, center.y, min_corner.z))
    bpy.context.view_layer.update()

    min_corner, max_corner = scene_bounds(mesh_objects, Vector)
    center = (min_corner + max_corner) * 0.5
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)

    camera_data = bpy.data.cameras.new(collection_name + "_CAMERA")
    camera = bpy.data.objects.new(collection_name + "_CAMERA", camera_data)
    collection.objects.link(camera)
    camera_data.lens = 85  # long lens keeps perspective flat, closer to a reference render
    bpy.context.scene.camera = camera

    def aim(obj, target):
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

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
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 1100
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True   # the crop finds the subject through alpha
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "Medium High Contrast"
    scene.world.color = settings["world_color"]
    try:
        scene.eevee.use_raytracing = settings["raytracing"]
    except AttributeError:
        pass

    distance = size * 2.4
    target = center + Vector((0, 0, size * 0.03))
    written = []
    for azimuth in angles:
        camera.location = Vector(orbit_position(azimuth, elevation_deg, distance, target.z))
        aim(camera, target)
        kind = "grey" if grey_mode else "tex"
        path = f"{output_dir}/{label}_{kind}_az{round(azimuth)}.png"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)
    return {"renders": written, "size": size}


def main() -> None:
    request, = sys.argv[sys.argv.index("--") + 1:]
    run(**json.loads(request))


if __name__ == "__main__":
    main()
