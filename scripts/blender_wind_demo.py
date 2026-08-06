#!/usr/bin/env python3
"""Animate labelled foliage with shader-style wind and render it to MP4.

Foliage in a game is almost never rigged or simulated. It is moved by a vertex shader:
a small program that nudges each point every frame by a wave, scaled by how floppy that
point is. Trunk rigid, leaf tips loose. No bones, no solver, effectively free.

This does the same arithmetic ahead of time so the result can be rendered locally,
which proves the whole chain -- painted mask, projected labels, derived stiffness,
visible motion -- without needing a game engine set up. An engine would run the same
formula live from the same data.

Stiffness is not painted. It falls out of the labels: a foliage vertex's distance to
the nearest *body* vertex, normalised. Zero where the tail meets the hip, one at the
tips, so the tail bends rather than sliding rigidly.

Takes two meshes. Appearance comes from the original textured GLB; labels come from the
GLB written by project_labels.py. They are matched by position rather than by index,
because exporting through trimesh does not preserve vertex order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from blender_render_asset import ENVIRONMENTS


def blender_code(
    mesh: Path,
    labels: Path,
    frame_dir: Path,
    label: str,
    env: str,
    resolution: int,
    frames: int,
    zoom: float,
    azimuth: float,
    frequency: float,
    wave: float,
    gust: float,
    bend: float,
    axis_y: float,
    axis_z: float,
    cluster_size: float,
    jitter: float,
    flutter: float,
) -> str:
    settings = ENVIRONMENTS[env]
    return f'''
import bpy, math
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

mesh_path = {str(mesh)!r}
labels_path = {str(labels)!r}
frame_dir = {str(frame_dir)!r}
label = {label!r}
frames = {frames}

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for coll in list(bpy.data.collections):
    bpy.data.collections.remove(coll)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass


def load(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in added if o.type == "MESH"]
    for o in bpy.data.objects:
        o.select_set(False)
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


body_obj = load(mesh_path)
body_obj.name = "FOX"
label_obj = load(labels_path)
label_obj.name = "LABELS"

# Read the label colours. glTF may land them on points or on face corners.
label_mesh = label_obj.data
attribute = label_mesh.color_attributes[0]
count = len(attribute.data)
raw = np.zeros(count * 4, dtype=np.float32)
attribute.data.foreach_get("color", raw)
raw = raw.reshape(-1, 4)[:, :3]
if attribute.domain == "CORNER":
    per_vertex = np.zeros((len(label_mesh.vertices), 3), dtype=np.float32)
    loop_vertex = np.zeros(len(label_mesh.loops), dtype=np.int32)
    label_mesh.loops.foreach_get("vertex_index", loop_vertex)
    per_vertex[loop_vertex] = raw
    raw = per_vertex

# glTF stores colour linearly; the palette was authored as sRGB. Only the winning
# channel matters here, so compare channels rather than converting.
is_foliage = (raw[:, 1] > raw[:, 0]) & (raw[:, 1] >= raw[:, 2])
is_flower = (raw[:, 2] > raw[:, 0]) & (raw[:, 2] > raw[:, 1])
moves = is_foliage | is_flower

label_points = np.zeros(len(label_mesh.vertices) * 3, dtype=np.float32)
label_mesh.vertices.foreach_get("co", label_points)
label_points = label_points.reshape(-1, 3)

# Match the textured mesh to the labelled one by position: exporting through trimesh
# does not preserve vertex order, so indices cannot be trusted between the two.
lookup = KDTree(len(label_points))
for index, point in enumerate(label_points):
    lookup.insert(Vector(point), index)
lookup.balance()

fox_mesh = body_obj.data
base = np.zeros(len(fox_mesh.vertices) * 3, dtype=np.float32)
fox_mesh.vertices.foreach_get("co", base)
base = base.reshape(-1, 3)

moving = np.zeros(len(base), dtype=bool)
for index, point in enumerate(base):
    _, nearest, _ = lookup.find(Vector(point))
    moving[index] = moves[nearest]

# Stiffness: distance to the nearest rigid (body) vertex, normalised. This is what
# makes the tail bend instead of sliding -- it is anchored where it meets the hip and
# free at the tips, and it costs nothing to derive because the labels already say
# which vertices are rigid.
rigid_points = label_points[~moves]
anchors = KDTree(len(rigid_points))
for index, point in enumerate(rigid_points):
    anchors.insert(Vector(point), index)
anchors.balance()

stiffness = np.zeros(len(base), dtype=np.float32)
# Keep the anchor position too, not just the distance: bending has to pivot about the
# point where the foliage attaches, and each frond has its own attachment point.
anchor_of = base.copy()
for index in np.nonzero(moving)[0]:
    position, _, distance = anchors.find(Vector(base[index]))
    stiffness[index] = distance
    anchor_of[index] = position
if stiffness.max() > 0:
    stiffness /= stiffness.max()
stiffness = stiffness * stiffness * (3.0 - 2.0 * stiffness)  # ease the falloff

print("WIND:: moving %d of %d vertices, max stiffness %.3f" % (int(moving.sum()), len(base), float(stiffness.max())))

extent = float(np.ptp(base, axis=0).max())
# Phase varies with position so the wind travels across the tail as a wave rather than
# moving every leaf in lockstep.
phase = (base[:, 0] + base[:, 2]) / max(extent, 1e-6) * {wave}

# Smoothly varying phase alone still moves neighbouring fronds almost together, which
# reads as one sheet rather than a bushy mass. Give each small clump its own offset by
# hashing a coarse grid of positions, so clumps flutter independently while staying
# coherent within themselves.
cell = np.floor(base / max(extent * {cluster_size}, 1e-6)).astype(np.int64)
clump = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
phase = phase + (clump % 997) / 997.0 * (2.0 * math.pi) * {jitter}

# Bending, not sliding. Translating every vertex along one direction is a shear: the
# tail slides bodily, its tip traces no arc, and the volume never turns to show its
# depth -- which is what makes a mesh with real thickness read as a flat cutout.
# Rotating about the attachment point instead preserves length and sweeps the mass
# through an arc.
offset_from_anchor = base - anchor_of
axis = np.array([0.0, {axis_y}, {axis_z}], dtype=np.float32)
axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
axis_dot = offset_from_anchor @ axis
axis_cross = np.cross(np.broadcast_to(axis, offset_from_anchor.shape), offset_from_anchor)
max_angle = math.radians({bend})

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = {resolution}
scene.render.resolution_y = {resolution}
scene.render.image_settings.file_format = "PNG"
try:
    scene.view_settings.view_transform = "AgX"
except TypeError:
    pass
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    scene.view_settings.look = "Medium High Contrast"
scene.world.color = {settings["world_color"]}
try:
    scene.eevee.use_raytracing = {settings["raytracing"]}
except AttributeError:
    pass

lo = base.min(axis=0)
hi = base.max(axis=0)
centre = Vector(((lo + hi) * 0.5).tolist())
size = float(max(hi - lo))

camera_data = bpy.data.cameras.new("CAM")
camera = bpy.data.objects.new("CAM", camera_data)
scene.collection.objects.link(camera)
camera_data.lens = 60
scene.camera = camera

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

for suffix, location, energy, radius in (
    ("KEY", (-2.8, -3.8, 4.5), 700, 4.0),
    ("FILL", (3.2, -1.8, 2.7), 420, 3.5),
    ("RIM", (1.0, 3.5, 4.0), 600, 3.0),
):
    light_data = bpy.data.lights.new(suffix, "AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = radius
    light = bpy.data.objects.new(suffix, light_data)
    scene.collection.objects.link(light)
    light.location = Vector(location) * size
    aim(light, centre)

angle = math.radians({azimuth})
distance = size * 1.9 / {zoom}
camera.location = centre + Vector(
    (-math.sin(angle) * distance, -math.cos(angle) * distance, size * 0.10)
)
aim(camera, centre)

label_obj.hide_render = True

flat = np.zeros(base.size, dtype=np.float32)
written = []
for index in range(frames):
    t = index / frames
    # Two waves at different rates so the motion does not read as a single metronome,
    # plus a slow gust that swells and fades across the loop. All multiples of the
    # loop length, so the last frame meets the first.
    swing = (
        np.sin(2.0 * math.pi * ({frequency} * t) + phase) * 0.65
        + np.sin(2.0 * math.pi * ({frequency} * 1.7 * t) + phase * 1.4) * 0.35
    )
    strength = 1.0 + {gust} * math.sin(2.0 * math.pi * t)
    angle = (swing * strength * max_angle * stiffness).astype(np.float32)

    # Rodrigues rotation of each vertex about its own anchor, by its own angle.
    cos_a = np.cos(angle)[:, None]
    sin_a = np.sin(angle)[:, None]
    rotated = (
        offset_from_anchor * cos_a
        + axis_cross * sin_a
        + np.broadcast_to(axis, offset_from_anchor.shape) * axis_dot[:, None] * (1.0 - cos_a)
    )
    moved = anchor_of + rotated

    # A little high-frequency flutter on the tips, on top of the bend, so the surface
    # shimmers instead of moving as one rigid fan.
    flutter = np.sin(2.0 * math.pi * ({frequency} * 3.1 * t) + phase * 2.3)
    moved[:, 1] += flutter * stiffness * extent * {flutter} * strength

    flat[:] = moved.reshape(-1)
    fox_mesh.vertices.foreach_set("co", flat)
    fox_mesh.update()

    path = frame_dir + "/" + label + "_%04d" % index + ".png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    written.append(path)

result = {{"frames": len(written)}}
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
    parser.add_argument("mesh", type=Path, help="original textured .glb")
    parser.add_argument("labels", type=Path, help="labelled .glb from project_labels.py")
    parser.add_argument("output", type=Path, help="destination .mp4")
    parser.add_argument("--label", default="wind")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="dark")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--azimuth", type=float, default=25.0)
    parser.add_argument(
        "--bend",
        type=float,
        default=22.0,
        help="peak bend angle in degrees at the floppiest tips",
    )
    parser.add_argument(
        "--axis-y", type=float, default=1.0, help="bend axis: horizontal sweep component"
    )
    parser.add_argument(
        "--axis-z", type=float, default=0.35, help="bend axis: vertical lift component"
    )
    parser.add_argument(
        "--cluster-size",
        type=float,
        default=0.07,
        help="clump size for independent flutter, as a fraction of the asset",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.8, help="how independently clumps flutter (0-1)"
    )
    parser.add_argument(
        "--flutter", type=float, default=0.012, help="high-frequency tip shimmer"
    )
    parser.add_argument("--frequency", type=float, default=2.0, help="sway cycles per loop")
    parser.add_argument("--wave", type=float, default=5.0, help="how much the wave travels")
    parser.add_argument("--gust", type=float, default=0.45, help="depth of the slow gust")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".frames_{args.label}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    code = blender_code(
        args.mesh.expanduser().resolve(),
        args.labels.expanduser().resolve(),
        frame_dir,
        args.label,
        args.env,
        args.resolution,
        args.frames,
        args.zoom,
        args.azimuth,
        args.frequency,
        args.wave,
        args.gust,
        args.bend,
        args.axis_y,
        args.axis_z,
        args.cluster_size,
        args.jitter,
        args.flutter,
    )
    print(send(args.host, args.port, code, timeout=3600))

    rendered = sorted(frame_dir.glob(f"{args.label}_*.png"))
    if len(rendered) != args.frames:
        print(f"error: expected {args.frames} frames, found {len(rendered)}", file=sys.stderr)
        return 1

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(frame_dir / f"{args.label}_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-crf", "18",
            "-movflags", "+faststart",
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
