"""Inspect live Blender scenes, place joint markers, and position preview cameras.

Run (internal): bundled and sent to Blender by the inspection, marker and camera commands.
"""

from __future__ import annotations

import json
import math

if __package__:
    from .blender_scene import (
        import_gltf_objects,
        remove_scene_objects,
        scene_marker_objects,
    )
elif __name__ != "_assetfurnace_worker":
    from blender_scene import (
        import_gltf_objects,
        remove_scene_objects,
        scene_marker_objects,
    )


VIEW_OFFSETS = {
    "front_xneg": (-1, 0),
    "front_three_quarter": (-0.9, -0.45),
    "rear_xpos": (1, 0),
    "profile_yneg": (0, -1),
    "profile_ypos": (0, 1),
}


def inspect_scene(bpy):
    objects = [
        {
            "name": obj.name,
            "type": obj.type,
            "dimensions": [round(v, 4) for v in obj.dimensions],
            "scale": [round(v, 4) for v in obj.scale],
            "location": [round(v, 4) for v in obj.location],
            "hide_render": obj.hide_render,
            "parent": obj.parent.name if obj.parent else None,
        }
        for obj in bpy.context.scene.objects
    ]
    return {
        "filepath": bpy.data.filepath,
        "is_dirty": bpy.data.is_dirty,
        "object_count": len(objects),
        "objects": objects,
        "actions": [action.name for action in bpy.data.actions],
        "materials": [material.name for material in bpy.data.materials],
    }


def inspect_file(bpy, path):
    with bpy.data.libraries.load(path, link=False, assets_only=False) as (source, _):
        # Leave the destination untouched so no datablocks are appended.
        return {
            "path": path,
            "objects": list(source.objects),
            "actions": list(source.actions),
            "meshes": list(source.meshes),
            "armatures": list(source.armatures),
            "materials": list(source.materials),
        }


def inspect_object(bpy, name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"error": f"no object named {name!r} in the live scene"}
    return {
        "name": obj.name,
        "type": obj.type,
        "dimensions": list(obj.dimensions),
        "scale": list(obj.scale),
        "location": list(obj.location),
        "rotation_euler_deg": [math.degrees(v) for v in obj.rotation_euler],
        "hide_render": obj.hide_render,
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
        "modifiers": [
            (modifier.name, modifier.type) for modifier in getattr(obj, "modifiers", [])
        ],
        "materials": [material.name for material in obj.data.materials]
        if hasattr(obj.data, "materials")
        else [],
        "active_action": obj.animation_data.action.name
        if obj.animation_data and obj.animation_data.action
        else None,
    }


def marker_position(low, size, fraction):
    return tuple(
        origin + extent * part for origin, extent, part in zip(low, size, fraction)
    )


def _new_marker(bpy, collection, name, radius):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = radius
    obj.show_name = True
    collection.objects.link(obj)
    return obj


def spawn_markers(bpy, asset_path, markers, prefix, confirm):
    from mathutils import Vector

    scene = bpy.context.scene
    existing = [
        obj for obj in scene.objects if obj.type == "EMPTY" and obj.name.startswith(prefix)
    ]
    if existing and not confirm:
        names = ", ".join(obj.name for obj in existing)
        raise RuntimeError(
            f"joint markers already in this scene ({names}). Re-run with --yes to replace the markers only."
        )

    imported = import_gltf_objects(bpy, asset_path, require_mesh=True)
    collection = None
    staged = []
    try:
        meshes = [obj for obj in imported if obj.type == "MESH"]
        low = Vector((1e9, 1e9, 1e9))
        high = Vector((-1e9, -1e9, -1e9))
        for obj in meshes:
            obj.hide_select = True
            for corner in obj.bound_box:
                point = obj.matrix_world @ Vector(corner)
                for axis in range(3):
                    low[axis] = min(low[axis], point[axis])
                    high[axis] = max(high[axis], point[axis])
        size = high - low
        positions = {
            prefix + name: marker_position(low, size, fraction)
            for name, fraction in markers.items()
        }
        collection = bpy.data.collections.new("JOINT_MARKERS")
        scene.collection.children.link(collection)
        for name, position in positions.items():
            obj = _new_marker(bpy, collection, name, max(size) * 0.018)
            staged.append((obj, name))
            obj.location = position
    except Exception:
        for obj in imported + [obj for obj, _ in staged]:
            bpy.data.objects.remove(obj, do_unlink=True)
        if collection is not None:
            bpy.data.collections.remove(collection)
        raise

    remove_scene_objects(bpy, existing)
    for obj, name in staged:
        obj.name = name
        if obj.name != name:
            obj["assetfurnace_marker_name"] = name

    if bpy.context.screen is not None:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
    return {
        "spawned": len(markers),
        "bbox_min": [round(v, 4) for v in low],
        "bbox_max": [round(v, 4) for v in high],
    }


def read_markers(bpy, prefix):
    return {
        name: [round(v, 5) for v in obj.matrix_world.translation]
        for name, obj in scene_marker_objects(bpy.context.scene, prefix).items()
    }


def load_markers(bpy, markers, prefix):
    collection = bpy.data.collections.get("JOINT_MARKERS")
    if collection is None:
        collection = bpy.data.collections.new("JOINT_MARKERS")
        bpy.context.scene.collection.children.link(collection)
    for name, location in markers.items():
        full_name = prefix + name
        obj = bpy.data.objects.get(full_name)
        if obj is None:
            obj = _new_marker(bpy, collection, full_name, 0.018)
        obj.location = location
    return {"restored": len(markers)}


def pick_location(view, distance, target_z):
    x, y = VIEW_OFFSETS[view]
    return distance * x, distance * y, target_z


def pick_pixel(bpy, pixel, width, height, view, ndc):
    from mathutils import Vector

    scene = bpy.context.scene
    camera = scene.camera
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    corners = [
        obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box
    ]
    low = Vector(tuple(min(corner[i] for corner in corners) for i in range(3)))
    high = Vector(tuple(max(corner[i] for corner in corners) for i in range(3)))
    center = (low + high) * 0.5
    size = max(high.x - low.x, high.y - low.y, high.z - low.z)
    target = center + Vector((0, 0, size * 0.03))
    camera.location = Vector(pick_location(view, size * 2.25, target.z))
    camera.rotation_euler = (
        (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    inverse = (projection @ camera.matrix_world.inverted()).inverted()
    near = inverse @ Vector((*ndc, -1.0, 1.0))
    far = inverse @ Vector((*ndc, 1.0, 1.0))
    near /= near.w
    far /= far.w
    origin = camera.matrix_world.translation
    direction = (far.xyz - near.xyz).normalized()
    hit, location, normal, face_index, obj, _ = scene.ray_cast(
        depsgraph, origin, direction
    )
    result = {"hit": bool(hit), "pixel": pixel, "view": view}
    if hit:
        local = obj.matrix_world.inverted() @ location
        result.update(
            {
                "object": obj.name,
                "face_index": int(face_index),
                "world": list(location),
                "mesh_local": list(local),
                "gltf_local": [local.x, local.z, -local.y],
                "normal_world": list(normal),
            }
        )
    return result


def orbit_camera(bpy, cam_name, location, toward, fov, render_path, resolution):
    from mathutils import Vector

    camera = bpy.data.objects.get(cam_name)
    if camera is None:
        data = bpy.data.cameras.new(cam_name)
        camera = bpy.data.objects.new(cam_name, data)
        bpy.context.collection.objects.link(camera)
    camera.data.lens_unit = "FOV"
    # The field of view specified by the caller is vertical.
    camera.data.sensor_fit = "VERTICAL"
    camera.data.angle = math.radians(fov)
    camera.location = location
    direction = Vector(toward).normalized()
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.camera = camera
    result = {"camera": camera.name, "location": list(camera.location)}
    if render_path:
        scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        result["rendered"] = render_path
    return result


ACTIONS = {
    "inspect_scene": (inspect_scene, "INSPECT_RESULT "),
    "inspect_file": (inspect_file, "INSPECT_RESULT "),
    "inspect_object": (inspect_object, "INSPECT_RESULT "),
    "spawn_markers": (spawn_markers, ""),
    "read_markers": (read_markers, ""),
    "load_markers": (load_markers, ""),
    "pick_pixel": (pick_pixel, "PICK_RESULT "),
    "orbit_camera": (orbit_camera, "ORBIT_RESULT "),
}


def run(arguments):
    import bpy

    action = arguments["action"]
    function, prefix = ACTIONS[action]
    result = function(bpy, **arguments["parameters"])
    if action == "read_markers":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(prefix + json.dumps(result))
    return result
