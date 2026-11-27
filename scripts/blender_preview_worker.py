"""Render four views of a model or four moments of a move in headless Blender.

Run (headless): blender --background --factory-startup --python scripts/blender_preview_worker.py -- --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

Point = tuple[float, float, float]
Shot = tuple[str, float, float | None]

RENDER_ENGINES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES")
LIGHTS = (
    ("key", (1.6, -2.2, 1.8), 6.0),
    ("fill", (-2.4, -1.6, 0.6), 2.4),
    ("rim", (0.4, 2.8, 1.8), 3.0),
)


def parse_shots(value: str) -> list[Shot]:
    shots = []
    for item in value.split(";"):
        name, azimuth, fraction = item.split("=")
        shots.append((name, float(azimuth), float(fraction) if fraction else None))
    return shots


def shot_frames(shots: Sequence[Shot], first: float, last: float) -> dict[str, float | None]:
    return {name: first + fraction * (last - first) if fraction is not None else None
            for name, _, fraction in shots}


def bounding_sphere(points: Iterable[Point]) -> tuple[Point, float]:
    """Enclose all sampled world-space vertices in one sphere."""
    remaining = iter(points)
    first = next(remaining, None)
    if first is None:
        raise ValueError("no mesh vertices to frame")
    low, high = list(first), list(first)
    for point in remaining:
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
    centre = tuple((lo + hi) / 2 for lo, hi in zip(low, high, strict=True))
    radius = math.dist(low, high) / 2 or 1.0
    return centre, radius


def camera_distance(radius: float, angle: float, width: int, height: int) -> float:
    """The camera's lens angle spans the longer side. Fit the subject to the shorter one."""
    narrow = math.atan(math.tan(angle / 2) * min(width, height) / max(width, height))
    return radius / math.sin(narrow) * 1.05


def camera_location(centre: Point, distance: float, azimuth: float) -> Point:
    angle, elevation = math.radians(azimuth), math.radians(8)
    plane = distance * math.cos(elevation)
    return (
        centre[0] + plane * math.sin(angle),
        centre[1] - plane * math.cos(angle),
        centre[2] + distance * math.sin(elevation),
    )


def _visible_meshes(scene):
    # Hidden helper meshes would wreck the common framing.
    for obj in scene.objects:
        if obj.type == "MESH" and not obj.visible_get():
            obj.hide_render = True
    return [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]


def _clay_material(bpy, meshes) -> None:
    neutral = bpy.data.materials.new("clay")
    neutral.use_nodes = True
    shader = neutral.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.62, 0.62, 0.60, 1.0)
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(neutral)


def _set_frame(scene, frame: float) -> None:
    whole = int(frame)
    scene.frame_set(whole, subframe=frame - whole)


def _sampled_vertices(bpy, scene, meshes, frames: Iterable[float | None]) -> Iterator[Point]:
    for frame in sorted({f for f in frames if f is not None}) or [None]:
        if frame is not None:
            _set_frame(scene, frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in meshes:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                matrix = evaluated.matrix_world
                for vertex in mesh.vertices:
                    yield matrix @ vertex.co
            finally:
                evaluated.to_mesh_clear()


def _setup_render(bpy, scene, width: int, height: int) -> None:
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.05, 0.05, 0.055, 1)
    background.inputs[1].default_value = 0.6
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    engines = type(scene.render).bl_rna.properties["engine"].enum_items
    available = {engine.identifier for engine in engines}
    scene.render.engine = next(engine for engine in RENDER_ENGINES if engine in available)
    scene.render.image_settings.file_format = "PNG"


def _add_lights(bpy, vector, scene, centre: Point, radius: float):
    rig = bpy.data.objects.new("Lights", None)
    scene.collection.objects.link(rig)
    rig.location = centre
    for name, offset, energy in LIGHTS:
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy * radius ** 2 * 40
        light.size = radius * 2
        obj = bpy.data.objects.new(name, light)
        obj.parent = rig
        obj.location = tuple(value * radius for value in offset)
        obj.rotation_euler = (-vector(obj.location)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(obj)
    return rig


def render_preview(bpy, vector, options: argparse.Namespace) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=options.source)
    scene = bpy.context.scene
    meshes = _visible_meshes(scene)
    if not meshes:
        raise SystemExit("no mesh in " + options.source)
    if options.clay == "1":
        _clay_material(bpy, meshes)

    # The scene's default frame range is longer than most imported actions.
    ranges = [action.frame_range for action in bpy.data.actions]
    first, last = (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else (scene.frame_start, scene.frame_end)
    fps = scene.render.fps / (scene.render.fps_base or 1)
    frames = shot_frames(options.shots, first, last)
    centre, radius = bounding_sphere(_sampled_vertices(bpy, scene, meshes, frames.values()))
    _setup_render(bpy, scene, options.width, options.height)

    camera_data = bpy.data.cameras.new("Cam")
    camera = bpy.data.objects.new("Cam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    distance = camera_distance(radius, camera_data.angle, options.width, options.height)
    lights = _add_lights(bpy, vector, scene, centre, radius)

    times = {}
    for name, azimuth, _ in options.shots:
        frame = frames[name]
        if frame is not None:
            _set_frame(scene, frame)
            times[name] = round((frame - first) / fps, 2)
        angle = options.front + azimuth
        camera.location = camera_location(centre, distance, angle)
        camera.rotation_euler = (vector(centre) - camera.location).to_track_quat("-Z", "Y").to_euler()
        # Turn the lights with the camera so the back is lit like the front.
        lights.rotation_euler = (0, 0, math.radians(angle))
        scene.render.filepath = str(options.output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    (options.output / "times.json").write_text(json.dumps(times), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="model file to render")
    parser.add_argument("output", type=Path, help="existing directory for images and times.json")
    parser.add_argument("front", type=float, help="front angle in degrees")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("clay", help="1 for matte clay, 0 for original paint")
    parser.add_argument("shots", type=parse_shots, help="semicolon-separated name=angle=fraction entries")
    options = parser.parse_args(argv)

    import bpy
    from mathutils import Vector

    render_preview(bpy, Vector, options)


if __name__ == "__main__":
    main()
