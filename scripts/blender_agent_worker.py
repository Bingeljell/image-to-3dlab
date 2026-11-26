"""Execute the Blender scene agent's actions inside the live Blender session.

Run (internal): bundled and sent to Blender by blender_agent.py.
"""

from __future__ import annotations

import json
import math

RESULT_SENTINEL = "AGENT_RESULT "


def _bounds(objects):
    from mathutils import Vector

    corners = [
        obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box
    ]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    return lo, hi


def orbit_direction(azimuth, elevation):
    """Return a unit offset from the target. Zero azimuth faces the front (-Y)."""
    az = math.radians(azimuth)
    el = math.radians(elevation)
    return math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)


def describe_scene(bpy):
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    objects = []
    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "ARMATURE", "LIGHT", "CAMERA", "EMPTY"}:
            continue
        entry = {
            "name": obj.name,
            "type": obj.type,
            "location": [round(v, 4) for v in obj.matrix_world.translation],
        }
        if obj.type == "MESH":
            lo, hi = _bounds([obj])
            lo = [round(v, 4) for v in lo]
            hi = [round(v, 4) for v in hi]
            entry["bounds_min"] = lo
            entry["bounds_max"] = hi
            entry["dimensions"] = [round(hi[i] - lo[i], 4) for i in range(3)]
            entry["polygons"] = len(obj.data.polygons)
        if obj.type == "LIGHT":
            entry["light_type"] = obj.data.type
            entry["energy"] = round(obj.data.energy, 3)
        if obj.type == "CAMERA":
            entry["lens_mm"] = round(obj.data.lens, 2)
        objects.append(entry)
    world = scene.world
    return {
        "objects": objects,
        "active_camera": scene.camera.name if scene.camera else None,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "film_transparent": bool(scene.render.film_transparent),
        "engine": scene.render.engine,
        "world_strength": (
            round(world.node_tree.nodes["Background"].inputs[1].default_value, 3)
            if world and world.use_nodes and "Background" in world.node_tree.nodes
            else None
        ),
    }


def render_view(bpy, output_path, resolution, samples):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
    if scene.camera is None:
        return {"error": "no active camera; call set_camera first"}
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    return {
        "rendered": output_path,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    }


def _remove_objects(bpy, types):
    removed = []
    for obj in list(bpy.data.objects):
        if obj.type in types:
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    return removed


def import_asset(bpy, path, clear_scene):
    if clear_scene:
        _remove_objects(bpy, {"MESH"})
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [obj.name for obj in bpy.data.objects if obj.name not in before]
    meshes = [name for name in imported if bpy.data.objects[name].type == "MESH"]
    return {"imported": imported, "meshes": meshes}


def _flat(obj):
    small, _, large = sorted(obj.dimensions)
    return large > 1e-6 and small / large < 0.05


def set_camera(bpy, azimuth, elevation, distance_factor, lens_mm, target):
    from mathutils import Vector

    candidates = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if target:
        candidates = [obj for obj in candidates if obj.name == target]
    else:
        # Exclude ground planes when there is another subject to frame.
        subjects = [obj for obj in candidates if not _flat(obj)]
        if subjects:
            candidates = subjects
    if not candidates:
        suffix = " named " + repr(target) if target else ""
        return {"error": "no mesh to aim at" + suffix}

    lo, hi = _bounds(candidates)
    centre = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    distance = radius * distance_factor
    offset = Vector(orbit_direction(azimuth, elevation)) * distance
    scene = bpy.context.scene
    cam = bpy.data.objects.get("AGENT_CAM")
    if cam is None:
        cam = bpy.data.objects.new("AGENT_CAM", bpy.data.cameras.new("AGENT_CAM"))
        scene.collection.objects.link(cam)
    cam.data.lens = lens_mm
    cam.location = centre + offset
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    return {
        "camera": cam.name,
        "location": [round(v, 4) for v in cam.location],
        "aimed_at": [round(v, 4) for v in centre],
        "distance": round(distance, 4),
        "lens_mm": cam.data.lens,
    }


def set_light(bpy, role, azimuth, elevation, energy, size, color):
    from mathutils import Vector

    name = "AGENT_LIGHT_" + role.upper()
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        return {"error": "no mesh in the scene to light"}
    lo, hi = _bounds(meshes)
    centre = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    offset = Vector(orbit_direction(azimuth, elevation)) * radius * 3.0
    light = bpy.data.objects.get(name)
    if light is None:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
        bpy.context.scene.collection.objects.link(light)
    light.data.type = "AREA"
    light.data.size = size * radius
    light.data.energy = energy
    light.data.color = tuple(color)[:3]
    light.location = centre + offset
    light.rotation_euler = (centre - light.location).to_track_quat("-Z", "Y").to_euler()
    return {
        "light": name,
        "energy": light.data.energy,
        "location": [round(v, 4) for v in light.location],
    }


def set_world(bpy, strength, color):
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("AGENT_WORLD")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        return {"error": "world has no Background node"}
    rgb = tuple(color)[:3]
    background.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    background.inputs[1].default_value = strength
    return {"world_strength": strength, "world_color": list(rgb)}


def move_object(bpy, name, dx, dy, dz):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"error": "no object named " + name}
    obj.location = (obj.location.x + dx, obj.location.y + dy, obj.location.z + dz)
    return {"moved": obj.name, "location": [round(v, 4) for v in obj.location]}


def resolve_engine(engine, available):
    """Match a name such as `eevee` against the installed Blender's enum."""
    wanted = engine.strip().upper()
    for identifier in available:
        if (
            identifier == wanted
            or wanted in identifier
            or identifier.replace("BLENDER_", "") == wanted
        ):
            return identifier
    return None


def set_render_settings(bpy, engine, film_transparent, exposure):
    scene = bpy.context.scene
    available = [
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    match = resolve_engine(engine, available)
    if match is None:
        return {
            "error": "no engine matching "
            + engine.strip().upper()
            + "; this Blender has "
            + ", ".join(available)
        }
    scene.render.engine = match
    scene.render.film_transparent = film_transparent
    scene.view_settings.exposure = exposure
    return {
        "engine": scene.render.engine,
        "film_transparent": bool(scene.render.film_transparent),
        "exposure": round(scene.view_settings.exposure, 3),
        "available": available,
    }


def clear_scene(bpy, keep_lights):
    removed = _remove_objects(bpy, {"MESH"} if keep_lights else {"MESH", "LIGHT"})
    return {"removed": removed, "count": len(removed)}


def add_primitive(bpy, kind, name, location, size, rotation, scale):
    adders = {
        "cube": lambda: bpy.ops.mesh.primitive_cube_add(size=size),
        "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2.0),
        "cylinder": lambda: bpy.ops.mesh.primitive_cylinder_add(
            radius=size / 2.0, depth=size
        ),
        "cone": lambda: bpy.ops.mesh.primitive_cone_add(radius1=size / 2.0, depth=size),
        "torus": lambda: bpy.ops.mesh.primitive_torus_add(
            major_radius=size / 2.0, minor_radius=size / 6.0
        ),
        "plane": lambda: bpy.ops.mesh.primitive_plane_add(size=size),
    }
    if kind not in adders:
        return {
            "error": "unknown primitive " + kind + "; have " + ", ".join(sorted(adders))
        }
    if name in bpy.data.objects:
        return {
            "error": "an object named " + name + " already exists; pick another name"
        }
    adders[kind]()
    obj = bpy.context.active_object
    obj.name = name
    obj.location = tuple(location)
    obj.rotation_euler = tuple(math.radians(angle) for angle in rotation)
    obj.scale = tuple(scale)
    # Update the evaluated dimensions after changing the scale.
    bpy.context.view_layer.update()
    return {
        "created": obj.name,
        "kind": kind,
        "location": [round(v, 4) for v in obj.location],
        "dimensions": [round(v, 4) for v in obj.dimensions],
    }


def set_object_transform(bpy, name, location, rotation, scale):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"error": "no object named " + name}
    if location is not None:
        obj.location = tuple(location)
    if rotation is not None:
        obj.rotation_euler = tuple(math.radians(angle) for angle in rotation)
    if scale is not None:
        obj.scale = tuple(scale)
    bpy.context.view_layer.update()
    return {
        "name": obj.name,
        "location": [round(v, 4) for v in obj.location],
        "rotation_deg": [round(math.degrees(a), 2) for a in obj.rotation_euler],
        "scale": [round(v, 4) for v in obj.scale],
        "dimensions": [round(v, 4) for v in obj.dimensions],
    }


def set_material(bpy, targets, color, roughness, metallic):
    if not targets:
        return {"error": "set_material needs name (one object) or names (several)"}
    rgb = tuple(color)
    red, green, blue = rgb
    key = f"AGENT_MAT_{red:.3f}_{green:.3f}_{blue:.3f}_{roughness:.2f}_{metallic:.2f}"
    material = bpy.data.materials.get(key)
    if material is None:
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = metallic
    coloured = []
    missing = []
    for target in targets:
        obj = bpy.data.objects.get(target)
        if obj is None or obj.type != "MESH":
            missing.append(target)
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)
        coloured.append(obj.name)
    payload = {"coloured": coloured, "material": key, "color": list(rgb)}
    if missing:
        payload["not_found"] = missing
    return payload


def duplicate_grid(bpy, name, counts, spacing, prefix):
    source = bpy.data.objects.get(name)
    prefix = prefix or name + "_"
    total = counts[0] * counts[1] * counts[2]
    if source is None:
        return {"error": "no object named " + name}
    if total > 200:
        return {"error": "that is " + str(total) + " copies; the limit is 200"}
    created = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                if i == j == k == 0:
                    copy = source
                else:
                    copy = source.copy()
                    # Material slots belong to the mesh, so each copy needs its own.
                    copy.data = source.data.copy()
                    bpy.context.scene.collection.objects.link(copy)
                copy.name = f"{prefix}{i}_{j}_{k}"
                copy.location = (
                    source.location.x + i * spacing[0],
                    source.location.y + j * spacing[1],
                    source.location.z + k * spacing[2],
                )
                created.append(copy.name)
    return {"created": created, "count": len(created)}


def mirror_object(bpy, name, new_name, axis):
    source = bpy.data.objects.get(name)
    axis = axis.strip().lower()
    index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if source is None:
        return {"error": "no object named " + name}
    if index is None:
        return {"error": "axis must be x, y or z, not " + axis}
    if new_name in bpy.data.objects:
        return {"error": "an object named " + new_name + " already exists"}
    copy = source.copy()
    copy.data = source.data.copy()
    bpy.context.scene.collection.objects.link(copy)
    copy.name = new_name
    location = list(source.location)
    location[index] = -location[index]
    copy.location = tuple(location)
    scale = list(source.scale)
    scale[index] = -scale[index]
    copy.scale = tuple(scale)
    bpy.context.view_layer.update()
    return {
        "created": copy.name,
        "mirrored": source.name,
        "axis": axis,
        "location": [round(v, 4) for v in copy.location],
    }


def delete_object(bpy, name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"error": "no object named " + name}
    bpy.data.objects.remove(obj, do_unlink=True)
    return {"deleted": name}


ACTIONS = {
    action.__name__: action
    for action in (
        describe_scene,
        render_view,
        import_asset,
        set_camera,
        set_light,
        set_world,
        move_object,
        set_render_settings,
        clear_scene,
        add_primitive,
        set_object_transform,
        set_material,
        duplicate_grid,
        mirror_object,
        delete_object,
    )
}


def run(arguments):
    """Dispatch one action and emit the response read by the agent."""
    import bpy

    payload = ACTIONS[arguments["action"]](bpy, **arguments["parameters"])
    print(RESULT_SENTINEL + json.dumps(payload))
    return payload
