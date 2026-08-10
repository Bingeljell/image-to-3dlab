"""Render a camera sweep past a line-up of assets, headless.

Run with:
    blender --background --python scripts/blender_lineup_sweep.py -- FRAMEDIR/f_ asset.glb[:Label] ...
    ffmpeg -framerate 30 -i FRAMEDIR/f_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4

Writes a PNG sequence, not a video: Blender 5.2 ships without FFMPEG output, so
`image_settings.file_format` accepts still formats only. Encoding is left to the caller.

Deliberately a standalone `--background` script rather than something driven through the
live Blender socket on 9876. A long render issued over that socket blocks the GUI's
handler, which locks up the user's session and gives no progress; a 210-frame attempt hit
a ten-minute timeout with nothing to show. Headless also means the user can keep working
in Blender while this runs.

Every asset is scaled to a common height so the line-up reads evenly -- these meshes
arrive normalised to a unit bounding box, so a tall thin creature and a squat one are
otherwise wildly different apparent sizes.
"""

import math
import sys

import bpy
from mathutils import Vector

SPACING = 1.7
FRAMES = 150
FPS = 30
YAW_DEGREES = -25.0


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.cameras, bpy.data.lights, bpy.data.actions):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_asset(path: str, label: str, index: int, count: int) -> list:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    fresh = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for obj in fresh:
        obj.name = label
        height = obj.dimensions.z or 1.0
        obj.scale = (1.0 / height,) * 3
        bpy.context.view_layer.update()
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        obj.location.z -= min(c.z for c in corners)
        obj.location.x = index * SPACING - (count - 1) * SPACING / 2
        obj.rotation_euler[2] = math.radians(YAW_DEGREES)
    return fresh


def build_environment() -> None:
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.055, 0.06, 0.075, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    floor.data.materials.append(mat)

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.20, 0.22, 0.27, 1.0)
    bg.inputs[1].default_value = 1.0

    # Metal renders black without something to reflect, so light from two sides.
    for name, energy, location, rotation in (
        ("Key", 1400, (-4.0, -7.0, 7.0), (math.radians(52), 0.0, math.radians(-28))),
        ("Rim", 800, (5.0, 6.0, 5.5), (math.radians(122), 0.0, math.radians(40))),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy
        light.size = 10.0
        obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = rotation


def build_camera(count: int) -> None:
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 40
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # A constant-speed dolly needs LINEAR keys; the default is BEZIER, which eases in
    # and out of every key and makes the sweep stutter. Set the preference before
    # inserting rather than rewriting fcurves afterwards: Blender 5.x moved actions to
    # slots and channelbags, so `action.fcurves` no longer exists and walking it raises.
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    span = (count - 1) * SPACING / 2
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        frame = i + 1
        # The camera leads and trails the line-up so the first and last asset are
        # seen head-on rather than only in profile as it arrives or leaves.
        cam.location = (-span - 1.8 + (2 * span + 3.6) * t, -7.2, 1.5)
        cam.keyframe_insert("location", frame=frame)
        target.location = (-span + (2 * span) * t, 0.0, 0.55)
        target.keyframe_insert("location", frame=frame)


def configure_render(output: str) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in
               bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines
                           else "BLENDER_EEVEE")
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAMES
    # Blender 5.2 ships no FFMPEG output -- image_settings offers still formats only.
    # Render a PNG sequence and let the caller encode it (see the module docstring).
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    output, specs = argv[0], argv[1:]

    clear_scene()
    for index, spec in enumerate(specs):
        path, _, label = spec.partition(":")
        import_asset(path, label or f"asset_{index}", index, len(specs))
    build_environment()
    build_camera(len(specs))
    configure_render(output)
    print(f"LINEUP:: {len(specs)} assets, {FRAMES} frames, engine "
          f"{bpy.context.scene.render.engine}")
    bpy.ops.render.render(animation=True)
    print(f"LINEUP:: wrote {output}")


if __name__ == "__main__":
    main()
