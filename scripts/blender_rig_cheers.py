#!/usr/bin/env python3
"""Rig a T-pose figure and animate a "cheers" toast, then render it to MP4.

Blender's automatic ("bone heat") weighting needs a connected surface and fails on a
TRELLIS mesh, which is a soup of disconnected shards -- that failure is what made the
earlier fox walk cycle stiff. Generic proximity weighting replaces it but is blind to
anatomy, so a leg bone grabs whatever is physically nearest, including the face.

This script sidesteps both by exploiting what we know: the figure is in a T-pose, so
the arms run along +/-X and the body along Z. Weights are therefore assigned
analytically from a vertex's position rather than inferred from the mesh, which stays
predictable no matter how fragmented the geometry is. A held prop (the beer mug) is
bound rigidly to the hand bone so it travels with the arm instead of deforming.

Landmarks default to the values measured on the Nikita asset; pass overrides for
another figure.
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
    asset: Path,
    frame_dir: Path,
    label: str,
    env: str,
    resolution: int,
    zoom: float,
    frames: int,
    shoulder_x: float,
    elbow_x: float,
    wrist_x: float,
    prop_x: float,
    arm_z: float,
    export_glb: str,
    lift_sign: float,
    bend_sign: float,
    upright: float,
    right_sign: float,
    azimuth: float,
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
shoulder_x = {shoulder_x}
elbow_x = {elbow_x}
wrist_x = {wrist_x}
prop_x = {prop_x}
arm_z = {arm_z}
export_glb = {export_glb!r}

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for prior in list(bpy.data.collections):
    bpy.data.collections.remove(prior)
try:
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
except RuntimeError:
    pass

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=asset_path)
imported = [o for o in bpy.data.objects if o not in before]
meshes = [o for o in imported if o.type == "MESH"]
for o in bpy.data.objects:
    o.select_set(False)
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
body.name = "BODY"

# Bake the import transform in so vertex coordinates match the measured landmarks.
for o in bpy.data.objects:
    o.select_set(False)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

corners = [body.matrix_world @ Vector(c) for c in body.bound_box]
lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
height = hi.z - lo.z

# Vertical landmarks as fractions of height, measured from the asset's own profile.
pelvis_z = lo.z + height * 0.50
spine_z = lo.z + height * 0.62
chest_z = lo.z + height * 0.72
neck_z = lo.z + height * 0.83
head_z = lo.z + height * 0.88
head_top_z = hi.z

armature_data = bpy.data.armatures.new("RIG")
rig = bpy.data.objects.new("RIG", armature_data)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")

def bone(name, head, tail, parent=None, connect=False):
    b = armature_data.edit_bones.new(name)
    b.head = Vector(head)
    b.tail = Vector(tail)
    if parent is not None:
        b.parent = parent
        b.use_connect = connect
    return b

pelvis = bone("pelvis", (0, 0, pelvis_z), (0, 0, spine_z))
spine = bone("spine", (0, 0, spine_z), (0, 0, chest_z), pelvis, True)
chest = bone("chest", (0, 0, chest_z), (0, 0, neck_z), spine, True)
neck = bone("neck", (0, 0, neck_z), (0, 0, head_z), chest, True)
head = bone("head", (0, 0, head_z), (0, 0, head_top_z), neck, True)

# The mug side is +X; mirror the same chain on -X so the far arm is also driveable.
arms = {{}}
for side, sign in (("L", 1.0), ("R", -1.0)):
    shoulder = bone(
        "shoulder_" + side,
        (sign * shoulder_x * 0.35, 0, chest_z + (neck_z - chest_z) * 0.55),
        (sign * shoulder_x, 0, arm_z),
        chest,
    )
    upper = bone(
        "upperarm_" + side,
        (sign * shoulder_x, 0, arm_z),
        (sign * elbow_x, 0, arm_z),
        shoulder,
        True,
    )
    fore = bone(
        "forearm_" + side,
        (sign * elbow_x, 0, arm_z),
        (sign * wrist_x, 0, arm_z),
        upper,
        True,
    )
    hand = bone(
        "hand_" + side,
        (sign * wrist_x, 0, arm_z),
        (sign * (wrist_x + (hi.x - wrist_x) * 0.9), 0, arm_z),
        fore,
        True,
    )
    arms[side] = (shoulder.name, upper.name, fore.name, hand.name)

bpy.ops.object.mode_set(mode="OBJECT")

group_names = [
    "pelvis", "spine", "chest", "neck", "head",
    "shoulder_L", "upperarm_L", "forearm_L", "hand_L",
    "shoulder_R", "upperarm_R", "forearm_R", "hand_R",
]
groups = {{name: body.vertex_groups.new(name=name) for name in group_names}}


def smoothstep(edge0, edge1, x):
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Half-thickness of the band around the arm line that counts as arm rather than torso.
arm_band = (hi.z - lo.z) * 0.075
shoulder_blend = (elbow_x - shoulder_x) * 0.55

for vert in body.data.vertices:
    x, y, z = vert.co
    ax = abs(x)
    side = "L" if x >= 0 else "R"

    # The held prop is rigid and is far taller than the arm is thick, so it must be
    # bound to the hand *before* the arm-band test below -- otherwise its top and
    # bottom fall outside the band, stay pinned to the chest while its middle follows
    # the hand, and the mug stretches into taffy as the arm lifts.
    if ax >= prop_x:
        groups["hand_" + side].add([vert.index], 1.0, "REPLACE")
        continue

    # How much this vertex belongs to the arm at all: it must be far enough out along
    # X *and* close enough to the arm's height, otherwise the upper torso would be
    # dragged along by the shoulder.
    lateral = smoothstep(shoulder_x - shoulder_blend, shoulder_x + shoulder_blend, ax)
    vertical = 1.0 - smoothstep(arm_band * 0.6, arm_band, abs(z - arm_z))
    arm_w = lateral * vertical

    if arm_w > 0.001:
        # Share each arm vertex between the two bones that straddle it, so the elbow
        # bends smoothly instead of creasing.
        to_fore = smoothstep(
            elbow_x - (elbow_x - shoulder_x) * 0.35,
            elbow_x + (wrist_x - elbow_x) * 0.35,
            ax,
        )
        to_hand = smoothstep(wrist_x - (wrist_x - elbow_x) * 0.30, wrist_x, ax)
        parts = [
            ("upperarm_" + side, 1.0 - to_fore),
            ("forearm_" + side, to_fore * (1.0 - to_hand)),
            ("hand_" + side, to_hand),
        ]
        for name, weight in parts:
            if weight > 0.0:
                groups[name].add([vert.index], weight * arm_w, "REPLACE")

    body_w = 1.0 - arm_w
    if body_w > 0.001:
        # Torso and legs never move in this shot, but they still need an owner so the
        # armature does not leave them unweighted (which would collapse them to origin).
        to_spine = smoothstep(pelvis_z, spine_z, z)
        to_chest = smoothstep(spine_z, chest_z, z)
        to_neck = smoothstep(chest_z, neck_z, z)
        to_head = smoothstep(neck_z, head_z, z)
        chain = [
            ("pelvis", (1.0 - to_spine)),
            ("spine", to_spine * (1.0 - to_chest)),
            ("chest", to_chest * (1.0 - to_neck)),
            ("neck", to_neck * (1.0 - to_head)),
            ("head", to_head),
        ]
        for name, weight in chain:
            if weight > 0.0:
                groups[name].add([vert.index], weight * body_w, "REPLACE")

modifier = body.modifiers.new("Armature", "ARMATURE")
modifier.object = rig
body.parent = rig

# --- animation -----------------------------------------------------------------
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = frames - 1

bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for pbone in rig.pose.bones:
    pbone.rotation_mode = "XYZ"


def key(bone_name, frame, rotation):
    pbone = rig.pose.bones[bone_name]
    pbone.rotation_euler = rotation
    pbone.keyframe_insert(data_path="rotation_euler", frame=frame)


R = math.radians
# A pose bone's local Y runs along the bone, so rotating about Y only twists it.
# Local X is the swing that lifts the arm in the vertical plane; local Z swings it
# forward and back across the body. LIFT_SIGN/BEND_SIGN flip if the rig comes out
# mirrored, which depends on the bone roll Blender picks.
LIFT = {lift_sign}
BEND = {bend_sign}
upright = {upright}
# Swinging the arm *forward* is a yaw about the vertical, which does not tip the mug
# at all -- only lift and elbow bend do. So a toast raised out in front with a gently
# bent elbow needs far less counter-rotation at the wrist than one folded up beside
# the head, which is what kept the wrist from breaking.
BEATS = [
    # frame, upperarm lift, upperarm forward, forearm bend, chest twist, head lift
    (0,    0,    0,     0,    0,   0),
    (14,  10,   44,    16,    3,  -3),
    (30,  18,   76,    30,    6,  -7),   # mug out in front, arm gently bent
    (40,  15,   70,    25,    8,  -5),   # small wind-back
    (52,  27,   90,    35,   -3, -11),   # the toast: push it out and up
    (66,  20,   79,    31,    5,  -7),   # settle
    (80,  19,   78,    30,    4,  -6),
    (89,  19,   78,    30,    4,  -6),
]
for frame, ua_lift, ua_fwd, fa_bend, ch_twist, hd_lift in BEATS:
    key("upperarm_L", frame, (R(ua_lift) * LIFT, 0, R(ua_fwd) * BEND))
    key("forearm_L", frame, (R(fa_bend) * LIFT, 0, 0))
    # The mug inherits every rotation up the chain, so without this it tips over and
    # pours as the elbow bends. Cancelling the accumulated lift keeps it world-upright.
    key("hand_L", frame, (R(-(ua_lift + fa_bend)) * LIFT * upright, 0, 0))
    key("chest", frame, (0, R(ch_twist), 0))
    key("head", frame, (R(hd_lift) * LIFT, 0, 0))

# The free arm has to come down out of the T-pose or the whole thing reads as a
# scarecrow rather than someone raising a drink. The -X chain is mirrored, so its
# lift axis runs the other way; RIGHT flips if it swings up instead of down.
RIGHT = {right_sign}
REST = [
    (0,    0,   0),
    (16, -46, -14),
    (30, -68, -22),
    (52, -74, -30),
    (66, -70, -24),
    (89, -70, -24),
]
for frame, ua_drop, fa_curl in REST:
    key("upperarm_R", frame, (R(ua_drop) * LIFT * RIGHT, 0, 0))
    key("forearm_R", frame, (R(fa_curl) * LIFT * RIGHT, 0, 0))

# Blender 4.4+ moved f-curves behind slotted actions (layers -> strips -> channelbags);
# older builds expose action.fcurves directly. Support both.
def iter_fcurves(action):
    if hasattr(action, "fcurves") and len(action.fcurves):
        for fcurve in action.fcurves:
            yield fcurve
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                for fcurve in channelbag.fcurves:
                    yield fcurve

for fcurve in iter_fcurves(rig.animation_data.action):
    for kp in fcurve.keyframe_points:
        kp.interpolation = "BEZIER"
        kp.easing = "EASE_IN_OUT"

bpy.ops.object.mode_set(mode="OBJECT")

if export_glb:
    for o in bpy.data.objects:
        o.select_set(False)
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=export_glb,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
    )

# --- camera, lights, render ------------------------------------------------------
size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
center = (lo + hi) * 0.5

camera_data = bpy.data.cameras.new("CAM")
camera = bpy.data.objects.new("CAM", camera_data)
bpy.context.scene.collection.objects.link(camera)
camera_data.lens = 72
scene.camera = camera

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

for suffix, location, energy, scale in (
    ("KEY", (-2.8, -3.8, 4.5), 700, 4.0),
    ("FILL", (3.2, -1.8, 2.7), 420, 3.5),
    ("RIM", (1.0, 3.5, 4.0), 600, 3.0),
):
    light_data = bpy.data.lights.new(suffix, "AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = scale
    light = bpy.data.objects.new(suffix, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = Vector(location) * size
    aim(light, center)

try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = {resolution}
scene.render.resolution_y = {resolution}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.view_settings.look = "AgX - Medium High Contrast"
scene.world.color = {settings["world_color"]}
try:
    scene.eevee.use_raytracing = {settings["raytracing"]}
except AttributeError:
    pass

# Frame the upper body: the toast happens above the waist, so the legs are dead space.
focus = Vector((0.0, 0.0, lo.z + height * 0.74))
distance = size * 1.5 / {zoom}
# A toast raised straight ahead points at the lens from a head-on camera, so orbit
# round to a three-quarter view where the gesture actually reads.
azimuth = math.radians({azimuth})
camera.location = focus + Vector(
    (-math.sin(azimuth) * distance, -math.cos(azimuth) * distance, height * 0.04)
)
aim(camera, focus)

written = []
for index in range(frames):
    scene.frame_set(index)
    path = frame_dir + "/" + label + "_%04d" % index + ".png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    written.append(path)

print("RIG:: frames %d height %.3f arm_z %.3f" % (len(written), height, arm_z))
result = {{"frames": len(written), "height": height}}
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
    parser.add_argument("--label", default="cheers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="dark")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--export-glb", type=Path, default=None)
    # Landmarks measured from the asset's own vertical/lateral profile.
    parser.add_argument("--shoulder-x", type=float, default=0.12)
    parser.add_argument("--elbow-x", type=float, default=0.215)
    parser.add_argument("--wrist-x", type=float, default=0.30)
    parser.add_argument("--prop-x", type=float, default=0.305)
    parser.add_argument("--arm-z", type=float, default=0.247)
    parser.add_argument(
        "--lift-sign",
        type=float,
        default=1.0,
        help="flip to -1 if the arm swings down instead of up",
    )
    parser.add_argument(
        "--bend-sign",
        type=float,
        default=-1.0,
        help="flip to 1 if the arm swings backward instead of forward",
    )
    parser.add_argument(
        "--upright",
        type=float,
        default=1.0,
        help="how strongly the hand counter-rotates to keep the prop upright (0..1)",
    )
    parser.add_argument(
        "--right-sign",
        type=float,
        default=1.0,
        help="flip to -1 if the free arm swings up instead of down",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=35.0,
        help="camera orbit in degrees; 0 is head-on, positive swings toward the mug side",
    )
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".frames_{args.label}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    export_glb = str(args.export_glb.expanduser().resolve()) if args.export_glb else ""
    if args.export_glb:
        args.export_glb.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    code = blender_code(
        asset,
        frame_dir,
        args.label,
        args.env,
        args.resolution,
        args.zoom,
        args.frames,
        args.shoulder_x,
        args.elbow_x,
        args.wrist_x,
        args.prop_x,
        args.arm_z,
        export_glb,
        args.lift_sign,
        args.bend_sign,
        args.upright,
        args.right_sign,
        args.azimuth,
    )
    print(send(args.host, args.port, code, timeout=3600))

    rendered = sorted(frame_dir.glob(f"{args.label}_*.png"))
    if len(rendered) != args.frames:
        print(
            f"error: expected {args.frames} frames, found {len(rendered)}",
            file=sys.stderr,
        )
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
