"""Stage assets, render scene previews, and recalculate mesh normals inside Blender.

Run (internal): bundled and sent to Blender by the staging, light orbit and normal repair commands.
"""

from __future__ import annotations

import json
import math
import os


def _enter_private_scene(bpy, name):
    scene = bpy.data.scenes.get(name) or bpy.data.scenes.new(name)
    bpy.context.window.scene = scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new(scene.name + " World")
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for child in list(scene.collection.children):
        bpy.data.collections.remove(child)
    return scene


def _import_staged_asset(bpy, Vector, scene, item, culled):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=item["path"])
    fresh = [obj for obj in bpy.data.objects if obj not in before]
    collection = bpy.data.collections.new(item["name"])
    scene.collection.children.link(collection)
    meshes = []
    for obj in fresh:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
        if obj.type == "MESH":
            meshes.append(obj)
    if not meshes:
        return

    corners = [
        obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box
    ]
    low = Vector(tuple(min(v[i] for v in corners) for i in range(3)))
    high = Vector(tuple(max(v[i] for v in corners) for i in range(3)))
    centre = (low + high) * 0.5
    size = max(high.x - low.x, high.y - low.y, high.z - low.z) or 1.0
    for obj in fresh:
        if obj.parent is None:
            obj.scale = (1.0 / size,) * 3
            obj.location = -Vector((centre.x, centre.y, low.z)) / size
            obj.location.y += item["dy"]
    for obj in meshes:
        for slot in obj.material_slots:
            if slot.material:
                slot.material.use_backface_culling = culled


def stage_assets(bpy, items, culled):
    from mathutils import Vector

    scene = _enter_private_scene(bpy, "IMG3D stage")
    for item in items:
        _import_staged_asset(bpy, Vector, scene, item, culled)
    for name, location, energy in (
        ("KEY", (-2.8, -3.8, 4.5), 900),
        ("FILL", (3.2, -1.8, 2.7), 500),
        ("RIM", (1.0, 3.5, 4.0), 700),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = 4.0
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = Vector(location)
        light.rotation_euler = (
            (Vector((0, 0, 0.5)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = None
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces[0].shading.type = "MATERIAL"
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_all()
    names = [item["name"] for item in items]
    print("STAGED:", ", ".join(names))
    return {"staged": names}


def light_orbit(bpy, asset, out, resolution, rotations):
    previous_scene = bpy.context.window.scene
    try:
        scene = _enter_private_scene(bpy, "IMG3D light orbit")
        bpy.ops.import_scene.gltf(filepath=asset)
        if not any(obj.type == "MESH" for obj in scene.objects):
            raise RuntimeError("no mesh imported from " + asset)

        data = bpy.data.cameras.new("CAM")
        camera = bpy.data.objects.new("CAM", data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        camera.location = (2.15, -1.75, 0.10)
        camera.rotation_euler = (math.radians(88), 0.0, math.radians(51))
        camera.data.lens = 70

        def sun(name, energy, angle, rotation):
            data = bpy.data.lights.new(name, type="SUN")
            data.energy = energy
            data.angle = math.radians(angle)
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            obj.rotation_euler = rotation
            return obj

        key = sun("KEY", 5.0, 6, (0, 0, 0))
        sun("FILL", 0.55, 50, (math.radians(75), 0, math.radians(-80)))
        sun("RIM", 1.4, 10, (math.radians(103), 0, math.radians(200)))

        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
            0.03,
            0.033,
            0.037,
            1,
        )
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = scene.render.resolution_y = resolution
        scene.render.fps = 24
        scene.render.image_settings.file_format = "PNG"
        for frame, rotation in enumerate(rotations, start=1):
            key.rotation_euler = rotation
            scene.frame_set(frame)
            scene.render.filepath = f"{out}{frame:04d}"
            bpy.ops.render.render(write_still=True)
        payload = {"asset": asset, "frames": len(rotations), "written_to": out}
        print(json.dumps(payload))
        return payload
    finally:
        bpy.context.window.scene = previous_scene


def stage_bake(
    bpy,
    action_name,
    az,
    el,
    ortho_scale,
    frame_start,
    frame_end,
    frame_step,
    resolution,
    output_dir,
    label,
):
    arm = bpy.data.objects.get("rig")
    if arm is None:
        raise RuntimeError("no object named 'rig' in the scene")
    if arm.animation_data is None:
        raise RuntimeError("'rig' has no animation_data")

    action = bpy.data.actions.get(action_name)
    if action is None:
        raise RuntimeError(f"no action named {action_name!r} in this file")
    target = bpy.data.objects.get("CAM_TARGET")
    if target is None:
        raise RuntimeError("no CAM_TARGET empty in this file")
    camera = bpy.data.objects.get("QuickCam2") or bpy.data.objects.get("Camera")
    if camera is None or camera.data.type != "ORTHO":
        raise RuntimeError(
            "no ortho camera parented to CAM_TARGET found (expected 'QuickCam2')"
        )

    scene = bpy.context.scene
    previous_action = arm.animation_data.action
    previous_target_rotation = tuple(target.rotation_euler)
    previous_camera = scene.camera
    previous_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    previous_film = scene.render.film_transparent
    previous_view = scene.view_settings.view_transform
    try:
        arm.animation_data.action = action
        target.rotation_euler = (math.radians(el), 0.0, math.radians(az))
        if ortho_scale is not None:
            camera.data.ortho_scale = ortho_scale
        scene.camera = camera
        scene.render.resolution_x = scene.render.resolution_y = resolution
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        os.makedirs(output_dir, exist_ok=True)

        native_start, native_end = (
            int(action.frame_range[0]),
            int(action.frame_range[1]),
        )
        start = frame_start if frame_start is not None else native_start
        end = frame_end if frame_end is not None else native_end
        rendered = []
        for frame in range(start, end + 1, frame_step):
            scene.frame_set(frame)
            filename = f"{label}_{action.name}_az{az:g}_f{frame:03d}.png"
            path = os.path.join(output_dir, filename)
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            rendered.append(os.path.basename(path))
        payload = {
            "label": label,
            "action": action.name,
            "az": az,
            "el": el,
            "native_range": [native_start, native_end],
            "step": frame_step,
            "frame_count": len(rendered),
            "frames": rendered,
        }
        print("STAGE_BAKE " + json.dumps(payload))
        return payload
    finally:
        arm.animation_data.action = previous_action
        target.rotation_euler = previous_target_rotation
        scene.camera = previous_camera
        scene.render.resolution_x, scene.render.resolution_y = previous_resolution
        scene.render.film_transparent = previous_film
        scene.view_settings.view_transform = previous_view


def inward_area_percentage(normals, centres, areas):
    """Measure the area of faces pointing toward the average face centre."""
    import numpy as np

    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    centres = np.asarray(centres, dtype=np.float32).reshape(-1, 3)
    areas = np.asarray(areas, dtype=np.float32)
    facing = ((centres - centres.mean(axis=0)) * normals).sum(axis=1)
    return 100.0 * areas[facing < 0].sum() / areas.sum()


def _inward_mesh_area(mesh):
    import numpy as np

    count = len(mesh.polygons)
    normals = np.zeros(count * 3, dtype=np.float32)
    centres = np.zeros(count * 3, dtype=np.float32)
    areas = np.zeros(count, dtype=np.float32)
    mesh.polygons.foreach_get("normal", normals)
    mesh.polygons.foreach_get("center", centres)
    mesh.polygons.foreach_get("area", areas)
    return inward_area_percentage(normals, centres, areas)


def fix_normals(bpy, asset, output):
    previous_scene = bpy.context.window.scene
    try:
        scene = _enter_private_scene(bpy, "IMG3D normals")
        bpy.ops.import_scene.gltf(filepath=asset)
        meshes = [obj for obj in scene.objects if obj.type == "MESH"]
        for obj in scene.objects:
            obj.select_set(False)
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        mesh = obj.data

        before = _inward_mesh_area(mesh)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        after = _inward_mesh_area(mesh)

        bpy.ops.export_scene.gltf(
            filepath=output, export_format="GLB", use_selection=False, use_active_scene=True
        )
        print(f"NORMALS:: inward area {before:.1f}% -> {after:.1f}%")
        return {"before": before, "after": after}
    finally:
        bpy.context.window.scene = previous_scene


ACTIONS = {
    "stage_assets": stage_assets,
    "light_orbit": light_orbit,
    "stage_bake": stage_bake,
    "fix_normals": fix_normals,
}


def run(arguments):
    import bpy

    return ACTIONS[arguments["action"]](bpy, **arguments["parameters"])
