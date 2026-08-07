#!/usr/bin/env python3
"""Render a GLB under an orbiting key light, holding the camera still.

A normal map is invisible in a static render — it changes how a surface *responds* to
light, not what colour it is. Orbiting the camera does not reveal it either, because the
lighting relative to the surface never changes. **Only moving the light does.**

So this holds the camera fixed and sweeps the key light through a full circle. Surface
detail that exists only in the normal map appears as travelling highlights and shadows
across the leaves and fur; geometry that is genuinely flat stays flat.

The fill and rim are held static and dim, so the sweep reads clearly rather than being
washed out.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send


def key_light_rotation(frame: int, frames: int, elevation: float) -> tuple[float, float, float]:
    """Euler rotation for the key light at `frame`, as a full azimuth sweep.

    A Blender sun light is directional — only its rotation matters, never its position.
    Returned in radians as (x, y, z); the sweep is on z so elevation stays constant and
    the highlight travels horizontally, which is what makes relief read.

    Frame 1 and frame `frames + 1` give the same azimuth, so the loop is seamless.
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    t = (frame - 1) / frames
    return (math.radians(elevation), 0.0, 2.0 * math.pi * t)


TEMPLATE = '''
import bpy, json, math

ASSET = {asset!r}
OUT = {out!r}
FRAMES = {frames}
ELEV = {elevation}
RES = {res}

# Preserve anything that represents manual work. Joint markers are hand-placed and
# exist nowhere but the scene until saved, and the armature is built from them; clearing
# either has already destroyed ~40 minutes of placement once, unrecoverably.
def _is_precious(o):
    return o.name.startswith("JOINT_") or o.type == "ARMATURE"

for obj in list(bpy.data.objects):
    if not _is_precious(obj):
        bpy.data.objects.remove(obj, do_unlink=True)
for coll in list(bpy.data.collections):
    if coll.name != "JOINT_MARKERS" and not coll.objects:
        bpy.data.collections.remove(coll)

bpy.ops.import_scene.gltf(filepath=ASSET)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("no mesh imported from " + ASSET)

scn = bpy.context.scene
cam_data = bpy.data.cameras.new("CAM"); cam = bpy.data.objects.new("CAM", cam_data)
scn.collection.objects.link(cam); scn.camera = cam
cam.location = (2.15, -1.75, 0.10)
cam.rotation_euler = (math.radians(88), 0.0, math.radians(51))
cam.data.lens = 70

def sun(name, energy, angle, rot):
    d = bpy.data.lights.new(name, type="SUN")
    d.energy = energy; d.angle = angle
    o = bpy.data.objects.new(name, d); scn.collection.objects.link(o)
    o.rotation_euler = rot
    return o

key = sun("KEY", 5.0, math.radians(6), (0, 0, 0))      # tight angle = crisp relief
sun("FILL", 0.55, math.radians(50), (math.radians(75), 0, math.radians(-80)))
sun("RIM", 1.4, math.radians(10), (math.radians(103), 0, math.radians(200)))

scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.033, 0.037, 1)

scn.render.engine = "BLENDER_EEVEE"
scn.render.resolution_x = RES; scn.render.resolution_y = RES
scn.render.fps = 24
scn.render.image_settings.file_format = "PNG"

rotations = json.loads({rotations!r})
for i, rot in enumerate(rotations, start=1):
    key.rotation_euler = rot
    scn.frame_set(i)
    scn.render.filepath = OUT + "%04d" % i
    bpy.ops.render.render(write_still=True)

print(json.dumps({{"asset": ASSET, "frames": len(rotations), "written_to": OUT}}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out_prefix", type=Path, help="frame path prefix, e.g. dir/f_")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--elevation", type=float, default=52.0)
    parser.add_argument("--res", type=int, default=800)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rotations = [
        list(key_light_rotation(f, args.frames, args.elevation))
        for f in range(1, args.frames + 1)
    ]
    import json as _json

    code = TEMPLATE.format(
        asset=str(args.asset.expanduser().resolve()),
        out=str(args.out_prefix.expanduser().resolve()),
        frames=args.frames, elevation=args.elevation, res=args.res,
        rotations=_json.dumps(rotations),
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
