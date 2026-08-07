#!/usr/bin/env python3
"""Build a quadruped armature from placed joint markers and bind the mesh to it.

Reads the `JOINT_*` empties left by `blender_joint_markers.py`, constructs a bone
hierarchy, and binds with Blender's automatic (bone heat) weighting.

**The point of this script is the verification, not the binding.** Heat weighting
diffuses influence *across the mesh surface*, so an ear is far from a leg even when the
two are close in straight-line space. It requires a clean manifold surface, and when it
fails Blender falls back silently to *envelope* weights — crude capsules that grab any
vertex inside them, through the air, ignoring the surface. The visible symptom is a rig
that looks fine until you move a leg and the ears move with it.

Blender reports that failure as a warning rather than an exception, so a bind that
"succeeded" proves nothing. Instead this script measures, for every leg bone, how far
its influenced vertices reach up the body — and fails loudly if a leg reaches the head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

# (bone, head marker, tail marker, parent bone, connected)
#
# The foot is two segments, not one. A single wrist->paw bone pivots the whole foot as a
# peg, so lifting it raises the toe instead of dropping it. With the paw joint at the
# *heel* and a toe segment ahead of it, the foot rolls the way a foot does: heel plants,
# foot flattens, toes push off.
SKELETON: list[tuple[str, str, str, str | None, bool]] = [
    ("spine_01", "pelvis", "spine_mid", None, False),
    ("spine_02", "spine_mid", "chest", "spine_01", True),
    ("neck", "chest", "neck_base", "spine_02", True),
    ("head", "neck_base", "head", "neck", True),
    ("jaw", "head", "jaw", "head", False),
    ("tail_01", "tail_base", "tail_mid", "spine_01", False),
    ("tail_02", "tail_mid", "tail_tip", "tail_01", True),
]
for _side in ("L", "R"):
    SKELETON += [
        (f"front{_side}_upperarm", f"front{_side}_shoulder", f"front{_side}_elbow", "spine_02", False),
        (f"front{_side}_forearm", f"front{_side}_elbow", f"front{_side}_wrist", f"front{_side}_upperarm", True),
        (f"front{_side}_paw", f"front{_side}_wrist", f"front{_side}_paw", f"front{_side}_forearm", True),
        (f"front{_side}_toe", f"front{_side}_paw", f"front{_side}_toe", f"front{_side}_paw", True),
        (f"back{_side}_thigh", f"back{_side}_hip", f"back{_side}_knee", "spine_01", False),
        (f"back{_side}_shin", f"back{_side}_knee", f"back{_side}_ankle", f"back{_side}_thigh", True),
        (f"back{_side}_paw", f"back{_side}_ankle", f"back{_side}_paw", f"back{_side}_shin", True),
        (f"back{_side}_toe", f"back{_side}_paw", f"back{_side}_toe", f"back{_side}_paw", True),
    ]

# Ears have only a base marker, so their bone is extrapolated outward from the head
# along the head->ear direction; that makes the bone lie along the ear and pivot at its
# base, which is how an ear actually moves.
EAR_BONES = [("ear_L", "ear_L"), ("ear_R", "ear_R")]


def build_code(ear_length: float, weight_threshold: float) -> str:
    return f'''
import bpy, json
from mathutils import Vector

# Parsed rather than interpolated as a literal: a None parent renders as JSON `null`,
# which is not valid Python, so the table has to cross as a string.
skeleton = json.loads({json.dumps(json.dumps(SKELETON))})
ear_bones = json.loads({json.dumps(json.dumps(EAR_BONES))})
EAR_LEN = {ear_length}
THRESH = {weight_threshold}

markers = {{o.name[6:]: o.matrix_world.translation.copy()
           for o in bpy.data.objects if o.name.startswith("JOINT_")}}
missing = sorted({{m for b in skeleton for m in (b[1], b[2])}} - set(markers))
if missing:
    raise RuntimeError("missing markers: " + ", ".join(missing))

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if len(mesh_objs) != 1:
    raise RuntimeError("expected exactly one mesh, found %d" % len(mesh_objs))
mesh = mesh_objs[0]
mesh.hide_select = False

for o in list(bpy.data.objects):
    if o.type == "ARMATURE":
        bpy.data.objects.remove(o, do_unlink=True)
for vg in list(mesh.vertex_groups):
    mesh.vertex_groups.remove(vg)
if mesh.parent is not None:
    mesh.parent = None
for mod in list(mesh.modifiers):
    if mod.type == "ARMATURE":
        mesh.modifiers.remove(mod)

arm_data = bpy.data.armatures.new("FoxRig")
arm = bpy.data.objects.new("FoxRig", arm_data)
bpy.context.scene.collection.objects.link(arm)

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones
for name, head_m, tail_m, parent, connect in skeleton:
    bone = eb.new(name)
    bone.head = markers[head_m]
    bone.tail = markers[tail_m]
    if (bone.tail - bone.head).length < 1e-5:
        bone.tail = bone.head + Vector((0, 0, 1e-3))
head_c = markers["head"]
for name, marker in ear_bones:
    bone = eb.new(name)
    base = markers[marker]
    direction = (base - head_c)
    direction = direction.normalized() if direction.length > 1e-6 else Vector((0, 0, 1))
    bone.head = base
    bone.tail = base + direction * EAR_LEN
for name, _h, _t, parent, connect in skeleton:
    if parent:
        eb[name].parent = eb[parent]
        eb[name].use_connect = connect
for name, _m in ear_bones:
    eb[name].parent = eb["head"]
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
heat_error = None
try:
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
except RuntimeError as exc:
    heat_error = str(exc)

# Verification. For every bone, find the vertices it actually influences and report
# how high up the body they reach. A leg bone whose influence reaches the head is the
# envelope-fallback signature.
groups = {{vg.index: vg.name for vg in mesh.vertex_groups}}
zs = [v.co.z for v in mesh.data.vertices]
z_lo, z_hi = min(zs), max(zs)
span = z_hi - z_lo
head_z = markers["head"].z
influence = {{}}
for v in mesh.data.vertices:
    for g in v.groups:
        if g.weight >= THRESH:
            name = groups.get(g.group)
            if name:
                rec = influence.setdefault(name, [0, -1e9, 1e9])
                rec[0] += 1
                rec[1] = max(rec[1], v.co.z)
                rec[2] = min(rec[2], v.co.z)

report = {{}}
for name in [b[0] for b in skeleton] + [b[0] for b in ear_bones]:
    rec = influence.get(name)
    if rec is None:
        report[name] = {{"verts": 0, "max_z": None, "reaches_head": False}}
    else:
        report[name] = {{
            "verts": rec[0],
            "max_z": round(rec[1], 4),
            "reaches_head": bool(rec[1] > head_z - span * 0.05),
        }}

leg_bones = [n for n in report if ("front" in n or "back" in n)]
contaminated = sorted(n for n in leg_bones if report[n]["reaches_head"])
empty = sorted(n for n in report if report[n]["verts"] == 0)

# Heat weighting is *expected* to fail on this mesh — that is why the voxel-proxy
# transfer exists. Leaving the scene in this state means the armature animates and
# the mesh does not follow, which looks like a broken rig rather than a missing step.
if empty:
    print("!!! MESH IS NOT USABLY WEIGHTED — run scripts/blender_voxel_weights.py now")

print(json.dumps({{
    "bones": len(arm_data.bones),
    "heat_error": heat_error,
    "vertex_groups": len(mesh.vertex_groups),
    "empty_groups": empty,
    "legs_reaching_head": contaminated,
    "verdict": "ENVELOPE FALLBACK" if contaminated else ("EMPTY GROUPS" if empty else "CLEAN"),
    "report": report,
}}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--ear-length", type=float, default=0.09,
        help="Ear bones are extrapolated from the head through the ear base, so an "
             "over-long value pushes their tips outside the mesh",
    )
    parser.add_argument("--weight-threshold", type=float, default=0.15)
    args = parser.parse_args()
    print(send(build_code(args.ear_length, args.weight_threshold), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
