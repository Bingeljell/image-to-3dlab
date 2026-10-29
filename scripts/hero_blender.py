#!/usr/bin/env python3
"""Join a grafted head and body into one object, or render a check sheet, in a background Blender.

Run (headless): blender --background --python-exit-code 1 --python scripts/hero_blender.py -- join head_and_body.glb joined.glb

    <bpy python> scripts/hero_blender.py join grafted.glb joined.glb
    <bpy python> scripts/hero_blender.py check model.glb out_dir [--head-share 0.13]

Runs with any Python that has the `bpy` module (the auto-rig's venv has one), never in the
user's open Blender: an earlier tool wiped a day of unsaved scenes by working in the live
session. Each part keeps its own material, so the body's paint sheet can later be swapped
for a skin without touching the head's.

Two export traps this avoids: `use_selection` picked up an object selected in another
scene (which broke the rigger), and exporting animations took every action in the file.
Here the file is factory-empty and the export is `use_active_scene` with no animations.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _fresh():
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy


def join(source: Path, output: Path) -> int:
    bpy = _fresh()
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print(f"[hero-blender] no mesh in {source}", file=sys.stderr)
        return 1
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = "Character"
    for obj in list(bpy.context.scene.objects):
        if obj != joined:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.export_scene.gltf(filepath=str(output), use_active_scene=True,
                              export_format="GLB", export_animations=False)
    print(f"[hero-blender] joined {len(meshes)} parts -> {output} "
          f"({len(joined.data.polygons)} faces, {len(joined.data.materials)} materials)")
    return 0


def check(model: Path, out_dir: Path, head_share: float) -> int:
    """Close-ups of the head (front, 35 deg, side, back) and the whole figure, lit by size."""
    bpy = _fresh()
    from mathutils import Vector

    bpy.ops.import_scene.gltf(filepath=str(model))
    scene = bpy.context.scene
    mesh = max((o for o in scene.objects if o.type == "MESH"), key=lambda o: len(o.data.polygons))
    points = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    top = max(p.z for p in points)
    height = top - min(p.z for p in points)
    scene.world = bpy.data.worlds.new("world")
    scene.world.color = (0.05, 0.05, 0.05)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 700
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    lights = []
    for offset, power in (((-3, -3, 4), 900), ((3, -2, 2), 400), ((0, 4, 3), 500)):
        light = bpy.data.objects.new("light", bpy.data.lights.new("light", "AREA"))
        scene.collection.objects.link(light)
        lights.append((light, Vector(offset), power))
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for part, share, zoom in (("head", head_share, 2.2), ("body", 1.0, 1.25)):
        chosen = [p for p in points if p.z > top - height * share]
        centre = sum(chosen, Vector()) / len(chosen)
        size = max(max(c[i] for c in chosen) - min(c[i] for c in chosen) for i in range(3))
        for light, offset, power in lights:
            light.data.energy = power * size ** 2 / 4    # light falls off with distance squared
            light.data.size = 3 * size
            light.location = centre + offset * size
            light.rotation_euler = (centre - light.location).to_track_quat("-Z", "Y").to_euler()
        for name, angle in (("front", 0), ("three_quarter", 35), ("side", 90), ("back", 180)):
            if part == "body" and name in ("side", "back"):
                continue
            a = math.radians(angle)
            # In a factory-empty Blender an imported glTF faces -Y.
            camera.location = centre + Vector((size * zoom * math.sin(a), -size * zoom * math.cos(a), 0))
            camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
            path = out_dir / f"{part}_{name}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            shots.append(path)
    print(f"[hero-blender] rendered {len(shots)} check shots -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    joining = sub.add_parser("join")
    joining.add_argument("source", type=Path)
    joining.add_argument("output", type=Path)
    checking = sub.add_parser("check")
    checking.add_argument("model", type=Path)
    checking.add_argument("out_dir", type=Path)
    checking.add_argument("--head-share", type=float, default=0.13)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.command == "join":
        return join(args.source, args.output)
    return check(args.model, args.out_dir, args.head_share)


if __name__ == "__main__":
    raise SystemExit(main())
