#!/usr/bin/env python3
"""Render a seamlessly looping 360-degree turntable of a GLB via the Blender MCP scene.

Renders a PNG sequence (the model spinning under fixed lights) and muxes it into an
MP4 with ffmpeg. The loop is seamless because the last frame stops one step short of
a full turn, so frame N wraps onto frame 0 without a duplicate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Reuse the render environments from the still-preview helper so a turntable and a
# preview of the same asset are lit identically.
from blender_render_asset import ENVIRONMENTS


def blender_code(
    asset: Path,
    frame_dir: Path,
    label: str,
    env: str,
    frames: int,
    resolution: int,
    zoom: float,
    darken_backfaces: bool,
) -> str:
    settings = ENVIRONMENTS[env]
    return f'''
import bpy
import math
from mathutils import Vector

asset_path = {str(asset)!r}
frame_dir = {str(frame_dir)!r}
label = {label!r}
frames = {frames}
collection_name = "IMG3D_TT_" + label

# Same clean-slate policy as the preview renderer: a long-lived Blender session may
# still hold a previous asset, and every asset is grounded at the origin.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for prior in list(bpy.data.collections):
    bpy.data.collections.remove(prior)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

collection = bpy.data.collections.new(collection_name)
bpy.context.scene.collection.children.link(collection)

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=asset_path)
imported = [obj for obj in bpy.data.objects if obj not in before]
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

# A TRELLIS mesh is an open shard soup, so gaps expose the shell's inner surface. That
# interior carries the same skin-toned texture, which is why a hole in the back of a
# head reads as a floating eye and gaps in a sweater read as bare skin. Shading
# backfaces near-black turns those gaps into plain shadow, which reads as creases
# rather than artefacts -- a render-time fix that needs no change to the geometry.
if {darken_backfaces}:
    for obj in mesh_objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            tree = material.node_tree
            output = next(
                (n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
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
    corners = [obj.matrix_world @ Vector(c) for obj in mesh_objects for c in obj.bound_box]
    lo = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    hi = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return lo, hi

# Ground the asset and centre it on the spin axis, otherwise it wobbles as it turns.
min_corner, max_corner = bounds()
center = (min_corner + max_corner) * 0.5
spin.location -= Vector((center.x, center.y, min_corner.z))
bpy.context.view_layer.update()

min_corner, max_corner = bounds()
center = (min_corner + max_corner) * 0.5
size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)

camera_data = bpy.data.cameras.new(collection_name + "_CAMERA")
camera = bpy.data.objects.new(collection_name + "_CAMERA", camera_data)
collection.objects.link(camera)
camera_data.lens = 58
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
scene.render.resolution_x = {resolution}
scene.render.resolution_y = {resolution}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.world.color = {settings["world_color"]}
try:
    scene.eevee.use_raytracing = {settings["raytracing"]}
except AttributeError:
    pass

# This model's forward axis is -Y, so park the camera there for a face-on frame 0.
# `size` is the widest dimension, which for a T-pose figure is the arm span, so the
# default preview distance leaves the body small in frame; --zoom pulls the camera in.
distance = size * 2.25 / {zoom}
target = center + Vector((0, 0, size * 0.03))
camera.location = Vector((0, -distance, target.z))
aim(camera, target)

# Stop one step short of a full turn so the loop wraps without a duplicate frame.
step = 2.0 * math.pi / frames
written = []
for index in range(frames):
    spin.rotation_euler = (0.0, 0.0, index * step)
    bpy.context.view_layer.update()
    path = frame_dir + "/" + label + "_%04d" % index + ".png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    written.append(path)

result = {{"frames": len(written), "size": size, "first": written[0], "last": written[-1]}}
'''


def send(host: str, port: int, code: str, timeout: float) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(timeout)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
    return b"".join(chunks).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, help="destination .mp4")
    parser.add_argument("--label", default="turntable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="dark")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument(
        "--darken-backfaces",
        action="store_true",
        help=(
            "shade backfacing polygons near-black so shell gaps read as shadow. Off by "
            "default: TRELLIS meshes have large inverted-normal regions, so this can "
            "blacken outward-facing surfaces too."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="camera tightening factor; >1 fills more of the frame",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="keep the intermediate PNG sequence instead of deleting it",
    )
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".frames_{args.label}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    code = blender_code(
        asset,
        frame_dir,
        args.label,
        args.env,
        args.frames,
        args.resolution,
        args.zoom,
        args.darken_backfaces,
    )
    # Rendering the whole sequence happens inside one blocking call, so allow for it.
    print(send(args.host, args.port, code, timeout=1800))

    rendered = sorted(frame_dir.glob(f"{args.label}_*.png"))
    if len(rendered) != args.frames:
        print(
            f"error: expected {args.frames} frames, found {len(rendered)}",
            file=sys.stderr,
        )
        return 1

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / f"{args.label}_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # x264 needs even dimensions; scale defensively for odd --resolution values.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )

    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
