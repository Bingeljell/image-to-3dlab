"""Render one comparison panel in a fresh headless Blender scene.

Run (internal): launched in headless Blender by render_glb_comparison.py.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence


def bounding_sphere_radius(lo: Sequence[float], hi: Sequence[float]) -> float:
    """Radius of the sphere enclosing an axis-aligned box.

    Half the longest side is not enough: an elongated subject presents its diagonal at some
    azimuths, so framing on one axis crops it at those angles and not others.
    """
    half = [(hi[i] - lo[i]) / 2 for i in range(3)]
    return math.sqrt(sum(h * h for h in half)) or 1.0


def camera_distance(radius: float, half_angle: float, margin: float = 1.02) -> float:
    """Distance at which a sphere of ``radius`` fills a camera with this half field of view."""
    if half_angle <= 0 or half_angle >= math.pi / 2:
        raise ValueError(f"half_angle must be within (0, pi/2); got {half_angle}")
    return radius / math.sin(half_angle) * margin


def render_panel(
    src: str,
    dst: str,
    azimuth: float,
    elevation: float,
    panel_w: int,
    panel_h: int,
    clay: bool,
) -> None:
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in " + src)

    if clay:
        # Strip every material for a neutral clay render. Texture and geometry fail in
        # different ways, and a broken UV map makes a sound mesh look ruined -- so when the
        # question is about shape, the paint has to come off.
        neutral = bpy.data.materials.new("clay")
        neutral.use_nodes = True
        bsdf = neutral.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.60, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
        for o in meshes:
            o.data.materials.clear()
            o.data.materials.append(neutral)

    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for o in meshes:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = bounding_sphere_radius(lo, hi)

    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.04, 0.04, 0.045, 1)

    scene = bpy.context.scene
    scene.render.resolution_x = panel_w
    scene.render.resolution_y = panel_h

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    cam_data.sensor_fit = "AUTO"
    half = min(cam_data.angle_x, cam_data.angle_y) / 2
    dist = camera_distance(radius, half)

    a, e = math.radians(azimuth), math.radians(elevation)
    cam.location = (
        centre[0] + dist * math.cos(e) * math.sin(a),
        centre[1] - dist * math.cos(e) * math.cos(a),
        centre[2] + dist * math.sin(e),
    )
    cam.rotation_euler = (
        (Vector(centre) - cam.location).to_track_quat("-Z", "Y").to_euler()
    )

    for name, offset, energy in (
        ("key", (2.5, -2.0, 2.4), 6.0),
        ("fill", (-2.6, -1.4, 0.7), 2.2),
        ("rim", (0.4, 2.8, 1.8), 3.2),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy * (radius**2) * 40
        light.size = radius * 2
        obj = bpy.data.objects.new(name, light)
        obj.location = tuple(centre[i] + offset[i] * radius for i in range(3))
        obj.rotation_euler = (
            (Vector(centre) - obj.location).to_track_quat("-Z", "Y").to_euler()
        )
        bpy.context.collection.objects.link(obj)

    available = {
        i.identifier for i in type(scene.render).bl_rna.properties["engine"].enum_items
    }
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if candidate in available:
            scene.render.engine = candidate
            break

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = dst
    bpy.ops.render.render(write_still=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    render_panel(
        argv[0],
        argv[1],
        float(argv[2]),
        float(argv[3]),
        int(argv[4]),
        int(argv[5]),
        argv[6] == "1",
    )


if __name__ == "__main__":
    main()
