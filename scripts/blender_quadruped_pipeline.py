#!/usr/bin/env python3
"""Bind, capture and animate fitted Rigify quadrupeds with tunable profiles and audit reports.

Run (headless): blender --background fitted.blend --python-exit-code 1 --python scripts/blender_quadruped_pipeline.py -- bind --mesh Creature --metarig metarig --out bound.blend (all steps: docs/quadruped-pipeline.md)

Run inside Blender, preferably in a background process on a saved input blend.
See docs/quadruped-pipeline.md for the staged workflow and visual-review gates.
No automatic skeleton fitting, mesh repair, or promise of perfect skinning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FEET = {
    "FL": "front_foot_ik.L",
    "FR": "front_foot_ik.R",
    "BL": "foot_ik.L",
    "BR": "foot_ik.R",
}
DEFAULT_CONTROLS = dict(
    FEET, torso="torso", chest="chest", hips="hips", neck="neck", head="head"
)
DEFAULT_TAIL = ["spine.003", "spine.002", "spine.001", "spine"]
INTERNAL = ("ORG-", "MCH-", "DEF-", "VIS_")


def settings(overrides):
    """Validate documented motion knobs; lengths are fractions of support height."""
    gait = overrides.get("gait", "trot")
    if gait not in {"walk", "trot"}:
        raise ValueError("gait must be walk or trot (scurry is a tuned trot profile)")
    result = {
        "gait": gait,
        "frames": 28 if gait == "walk" else 16,
        "fps": 24,
        "duty": 0.65 if gait == "walk" else 0.46,
        "stride": 0.35,
        "lift": 0.10,
        "curl_degrees": 16.0,
        "bob": 0.02,
        "chest_motion": 1.0,
        "hip_motion": 1.0,
        "neck_motion": 1.0,
        "head_compensation": 1.0,
        "body_roll_degrees": 0.35,
        "tail_pitch_degrees": 0.65,
        "tail_yaw_degrees": 0.9,
        "tail_lag": 0.45,
        "transition_hold": 12,
        "transition_blend": 16,
        "transition_cycles": 3,
    }
    unknown = set(overrides) - set(result)
    if unknown:
        raise ValueError(f"Unknown motion knobs: {sorted(unknown)}")
    result.update(overrides)
    for key, value in result.items():
        if key == "gait":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
        if value < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in [
        "frames",
        "fps",
        "transition_hold",
        "transition_blend",
        "transition_cycles",
    ]:
        if int(result[key]) != result[key] or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(result[key])
    if result["frames"] < 8 or not 0.1 <= result["duty"] <= 0.9:
        raise ValueError("frames must be >=8; duty must be in [0.1, 0.9]")
    return result


def phases(gait):
    return (
        {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}
        if gait == "trot"
        else {"BL": 0.0, "FL": 0.25, "BR": 0.5, "FR": 0.75}
    )


def foot(u, limb, cfg):
    """Common-speed stance and C1 Hermite recovery; returns forward/lift/curl."""
    p = (u - phases(cfg["gait"])[limb]) % 1
    stride, duty = cfg["stride"], cfg["duty"]
    if p < duty:
        return stride * (0.5 - p / duty), 0.0, 0.0
    t = (p - duty) / (1 - duty)
    tangent = -stride * (1 - duty) / duty
    x = (2 * t**3 - 3 * t * t + 1) * (-stride / 2) + (t**3 - 2 * t * t + t) * tangent
    x += (-2 * t**3 + 3 * t * t) * (stride / 2) + (t**3 - t * t) * tangent
    arc = math.sin(math.pi * t) ** 2
    return x, cfg["lift"] * arc, math.radians(cfg["curl_degrees"]) * arc


def smooth(value):
    t = max(0.0, min(1.0, value))
    return t * t * t * (t * (6 * t - 15) + 10)


def transition_length(cfg):
    return (
        2 * cfg["transition_hold"]
        + 4 * cfg["transition_blend"]
        + cfg["transition_cycles"] * cfg["frames"]
    )


def envelope(frame, cfg):
    hold, blend, period = cfg["transition_hold"], cfg["transition_blend"], cfg["frames"]
    stop = hold + 2 * blend + cfg["transition_cycles"] * period
    posture = smooth((frame - hold) / blend) * (
        1 - smooth((frame - stop - blend) / blend)
    )
    gait = smooth((frame - hold - blend) / blend) * (1 - smooth((frame - stop) / blend))
    return posture, gait, 1 + (frame - hold - 2 * blend) % period


def same_file(output: Path, opened: Path) -> bool:
    """True when both paths are the same file once links and .. are resolved."""
    return Path(output).expanduser().resolve() == Path(opened).expanduser().resolve()


def refuse_open_blend(output: Path, opened: str | Path | None) -> None:
    """Stop before a headless save replaces the blend Blender was opened with."""
    if not opened:
        return
    if same_file(output, opened):
        path = Path(output).expanduser().resolve()
        raise SystemExit(f"Refusing to overwrite {path}; it is the blend Blender opened")


def new_path(path, source=None):
    path = Path(path).expanduser().resolve()
    if path.exists() or (source and same_file(path, source)):
        raise ValueError(f"Refusing to overwrite {path}; choose a new output")
    return path


def endpoint_error(expected, actual):
    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    return min(
        max(distance(expected[i], actual[i]) for i in range(2)),
        max(distance(expected[i], actual[1 - i]) for i in range(2)),
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def skeleton_signature(rig):
    return digest(
        [
            (
                b.name,
                list(b.head_local),
                list(b.tail_local),
                b.parent.name if b.parent else None,
            )
            for b in rig.data.bones
        ]
    )


def controls(rig):
    return [p for p in rig.pose.bones if not p.name.startswith(INTERNAL)]


def select(bpy, obj):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_select = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def require_rig(bpy, name):
    rig = bpy.data.objects.get(name)
    if (
        not rig
        or rig.type != "ARMATURE"
        or not any(b.name.startswith("DEF-") for b in rig.data.bones)
    ):
        raise ValueError("Specify a generated Rigify armature, not the metarig")
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise ValueError("Mute NLA tracks before capturing or authoring")
    scales = rig.matrix_world.to_scale()
    if (
        min(scales) <= 0
        or max(scales) - min(scales) > 1e-5
        or rig.matrix_world.determinant() <= 0
    ):
        raise ValueError("Nonuniform or reflected rig transforms are unsupported")
    return rig


def bind(bpy, args):
    """Generate on a disposable file; preserve world joints before proxy weighting."""
    import numpy as np
    from blender_rebind import regenerate_rigify
    from blender_rebind_weights import transfer_weights

    mesh, meta = bpy.data.objects.get(args.mesh), bpy.data.objects.get(args.metarig)
    if not mesh or mesh.type != "MESH" or not meta or meta.type != "ARMATURE":
        raise ValueError("Name an existing mesh and fitted metarig")
    if mesh.modifiers or mesh.data.shape_keys or mesh.vertex_groups:
        raise ValueError(
            "bind expects an unbound mesh without modifiers, shape keys or vertex groups"
        )
    if any(b.name.startswith("DEF-") for b in meta.data.bones) or meta.animation_data:
        raise ValueError("Expected an unanimated metarig")
    if getattr(meta.data, "rigify_target_rig", None):
        raise ValueError(
            "Metarig already targets a generated rig; use a fresh fitted input"
        )
    if any(
        max(
            abs(p.matrix_basis[i][j] - (1 if i == j else 0))
            for i in range(4)
            for j in range(4)
        )
        > 1e-6
        for p in meta.pose.bones
    ):
        raise ValueError("Fit the metarig in Edit Mode, not Pose Mode")
    for obj in [mesh, meta]:
        scale = obj.matrix_world.to_scale()
        if (
            min(scale) <= 0
            or max(scale) - min(scale) > 1e-5
            or obj.matrix_world.determinant() <= 0
        ):
            raise ValueError(
                "Apply/resolve nonuniform or reflected transforms before binding"
            )

    def geometry_world():
        return np.array([mesh.matrix_world @ v.co for v in mesh.data.vertices])

    before = geometry_world()
    uv_hash = digest(
        [[list(v.uv) for v in layer.data] for layer in mesh.data.uv_layers]
    )
    joints = {
        b.name: [
            list(meta.matrix_world @ b.head_local),
            list(meta.matrix_world @ b.tail_local),
        ]
        for b in meta.data.bones
    }
    for obj in [mesh, meta]:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        select(bpy, obj)
        bpy.ops.object.transform_apply(
            location=obj == meta, rotation=obj == meta, scale=True
        )
    rig = regenerate_rigify(bpy, meta)
    error = 0.0
    for name, expected in joints.items():
        b = rig.data.bones.get("ORG-" + name)
        if b is None:
            raise ValueError(f"Generated rig missing ORG-{name}")
        error = max(
            error,
            endpoint_error(
                expected,
                [rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local],
            ),
        )
    if error > max(1e-6, max(mesh.dimensions) * 1e-5):
        raise ValueError(
            f"Generated joint alignment error {error}; do not weight this rig"
        )
    proxy_name = "I2L_WEIGHT_PROXY_" + mesh.name
    if bpy.data.objects.get(proxy_name):
        raise ValueError(f"Reserved proxy object already exists: {proxy_name}")
    report = transfer_weights(bpy, mesh, rig, voxel_fraction=args.voxel_fraction)
    select(bpy, mesh)
    bpy.ops.object.vertex_group_normalize_all(
        group_select_mode="BONE_DEFORM", lock_active=False
    )
    deform = {b.name for b in rig.data.bones if b.use_deform}
    groups = {g.index: g.name for g in mesh.vertex_groups}
    used, sums = set(), []
    for vertex in mesh.data.vertices:
        sums.append(sum(g.weight for g in vertex.groups if groups[g.group] in deform))
        used.update(groups[g.group] for g in vertex.groups if g.weight > 0.001)
    unused = sorted(deform - used)
    unexpected = set(unused) - set(args.allow_unused)
    if unexpected or not sums or min(sums) < 0.999 or max(sums) > 1.001:
        raise ValueError(
            f"Weight audit failed: unused={unused}, sum range={min(sums, default=0), max(sums, default=0)}"
        )
    geometry_error = float(np.abs(before - geometry_world()).max())
    if geometry_error > 1e-5 or uv_hash != digest(
        [[list(v.uv) for v in layer.data] for layer in mesh.data.uv_layers]
    ):
        raise ValueError("Binding changed visible geometry or UVs")
    meta.hide_set(True)
    report.update(
        rig=rig.name,
        unused_deform_bones=unused,
        joint_error=error,
        geometry_error=geometry_error,
        weight_sum_range=[min(sums), max(sums)],
        visual_review_required=True,
    )
    return report


def capture(bpy, rig, mapping, tail):
    """Capture an exact control reference plus a rig-derived scale and travel axes."""
    from mathutils import Vector

    required = list(mapping.values()) + tail
    missing = sorted(set(required) - set(rig.pose.bones.keys()))
    if missing:
        raise ValueError(f"Missing controls: {missing}; supply --controls JSON")
    fk = [p.name for p in controls(rig) if "IK_FK" in p and abs(p["IK_FK"]) > 1e-6]
    if fk:
        raise ValueError(f"Capture locomotion references in IK mode (IK_FK=0): {fk}")
    bpy.context.view_layer.update()
    up = (rig.matrix_world.to_3x3().inverted() @ Vector((0, 0, 1))).normalized()
    head = rig.data.bones[mapping["head"]].head_local
    hips = rig.data.bones[mapping["hips"]].head_local
    forward = head - hips
    forward -= up * forward.dot(up)
    if forward.length < 1e-6:
        raise ValueError("Cannot infer horizontal heading from head/hips")
    forward.normalize()
    right = forward.cross(up).normalized()
    floor = sum(rig.data.bones[mapping[k]].head_local.dot(up) for k in FEET) / 4
    height = (
        rig.data.bones[mapping["chest"]].head_local.dot(up) + hips.dot(up)
    ) / 2 - floor
    if height <= 1e-6:
        raise ValueError("Cannot derive support height; check rest skeleton")
    bones = {}
    for p in controls(rig):
        if p.rotation_mode == "AXIS_ANGLE":
            raise ValueError("Axis-angle controls unsupported; use Euler or quaternion")
        bones[p.name] = {
            "basis": [list(row) for row in p.matrix_basis],
            "mode": p.rotation_mode,
            "properties": {
                k: v for k, v in p.items() if isinstance(v, (int, float, bool))
            },
        }
    return {
        "schema_version": 1,
        "skeleton": skeleton_signature(rig),
        "rig": rig.name,
        "rig_linear_transform": [list(row)[:3] for row in list(rig.matrix_world)[:3]],
        "controls": mapping,
        "tail": tail,
        "up": list(up),
        "forward": list(forward),
        "right": list(right),
        "support_height": height,
        "bones": bones,
        "feet": {
            k: [list(row) for row in rig.pose.bones[mapping[k]].matrix] for k in FEET
        },
        "blender_version": bpy.app.version_string,
    }


def activate(rig, action):
    rig.animation_data_create()
    rig.animation_data.action = action
    if action:
        rig.animation_data.action_slot = action.slots[0]


def key(p, frame):
    rotation = (
        "rotation_quaternion" if p.rotation_mode == "QUATERNION" else "rotation_euler"
    )
    for channel in ["location", rotation, "scale"]:
        p.keyframe_insert(channel, frame=frame, group=p.name)
    for name, value in p.items():
        if isinstance(value, (int, float, bool)):
            p.keyframe_insert(f'["{name}"]', frame=frame, group=p.name)


def new_action(bpy, rig, name):
    if name in bpy.data.actions:
        raise ValueError(f"Action {name!r} already exists; choose --name")
    if rig.animation_data and rig.animation_data.action:
        rig.animation_data.action.use_fake_user = True
    result = bpy.data.actions.new(name)
    result.use_fake_user = True
    result.slots.new(id_type="OBJECT", name=rig.name)
    activate(rig, result)
    return result


def linear(action):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"


def animate(bpy, rig, pose, cfg, name, transition=False):
    """Author controls only, then solve IK targets after the coordinated body motion."""
    from mathutils import Matrix, Quaternion, Vector

    if pose.get("schema_version") != 1 or pose.get("skeleton") != skeleton_signature(
        rig
    ):
        raise ValueError(
            "Pose profile is incompatible with this rest skeleton; recapture it"
        )
    if pose.get("rig_linear_transform") != [
        list(row)[:3] for row in list(rig.matrix_world)[:3]
    ]:
        raise ValueError("Rig rotation/scale changed since capture; recapture the pose")
    if set(pose["bones"]) != {p.name for p in controls(rig)}:
        raise ValueError("Control set differs from captured pose")
    for candidate in [name, name + "_Reference"] + (
        [name + "_Transition"] if transition else []
    ):
        if candidate in bpy.data.actions:
            raise ValueError(f"Action already exists: {candidate}")
    mapping, height = pose["controls"], pose["support_height"]
    up, forward, right = (Vector(pose[n]) for n in ["up", "forward", "right"])
    bones = controls(rig)
    bases = {n: Matrix(v["basis"]) for n, v in pose["bones"].items()}

    def restore():
        for p in bones:
            p.rotation_mode = pose["bones"][p.name]["mode"]
            p.matrix_basis = bases[p.name]
            for k, v in pose["bones"][p.name]["properties"].items():
                p[k] = v

    def local_offset(n, z=0.0, pitch=0.0, roll=0.0, yaw=0.0):
        p = rig.pose.bones[n]
        rest = p.bone.matrix_local.to_quaternion()
        q = Quaternion(right, pitch) @ Quaternion(forward, roll) @ Quaternion(up, yaw)
        p.matrix_basis = (
            p.matrix_basis @ (rest.inverted() @ q @ rest).to_matrix().to_4x4()
        )
        p.location += rest.inverted() @ (up * z)

    ref = new_action(bpy, rig, name + "_Reference")
    restore()
    for p in bones:
        key(p, 1)
    result = new_action(bpy, rig, name)
    for frame in range(1, cfg["frames"] + 2):
        restore()
        u = (frame - 1) / cfg["frames"]
        w = 2 * math.pi * u
        local_offset(
            mapping["torso"],
            z=-height * cfg["bob"] * math.cos(2 * w - 1.5),
            roll=math.radians(cfg["body_roll_degrees"]) * math.sin(w - 0.3),
        )
        local_offset(
            mapping["hips"],
            z=height * 0.005 * cfg["hip_motion"] * math.sin(2 * w - 0.4),
            pitch=-0.012 * cfg["hip_motion"] * math.sin(2 * w - 0.4),
        )
        local_offset(
            mapping["chest"],
            z=height * 0.0075 * cfg["chest_motion"] * math.sin(2 * w - 0.7),
            pitch=0.018 * cfg["chest_motion"] * math.sin(2 * w - 0.7),
        )
        local_offset(
            mapping["neck"],
            z=height * 0.00375 * cfg["neck_motion"] * math.sin(2 * w - 1),
            pitch=0.028 * cfg["neck_motion"] * math.sin(2 * w - 1.05),
        )
        local_offset(
            mapping["head"],
            pitch=-0.013 * cfg["head_compensation"] * math.sin(2 * w - 1.05),
        )
        for i, n in enumerate(pose["tail"]):
            local_offset(
                n,
                pitch=math.radians(cfg["tail_pitch_degrees"])
                * math.sin(2 * w - 1 - i * cfg["tail_lag"]),
                yaw=math.radians(cfg["tail_yaw_degrees"])
                * math.sin(w - 0.8 - i * cfg["tail_lag"]),
            )
        for p in bones:
            key(p, frame)
        bpy.context.view_layer.update()
        for limb in FEET:
            x, z, curl = foot(u, limb, cfg)
            target = Matrix(pose["feet"][limb])
            target = Matrix.LocRotScale(
                target.translation + height * (forward * x + up * z),
                Quaternion(right, curl) @ target.to_quaternion(),
                target.to_scale(),
            )
            p = rig.pose.bones[mapping[limb]]
            p.matrix = target
            key(p, frame)
        bpy.context.view_layer.update()
    linear(result)
    result["profile"] = json.dumps(cfg, sort_keys=True)
    report = audit_cycle(bpy, rig, result, cfg["frames"])
    report["visual_review_required"] = True
    report["nominal_speed_rig_units_per_second"] = (
        height * cfg["stride"] / cfg["duty"] * cfg["fps"] / cfg["frames"]
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = cfg["frames"]
    if transition:
        make_transition(bpy, rig, ref, result, mapping, cfg, name + "_Transition")
    bpy.context.scene.render.fps = cfg["fps"]
    bpy.context.scene.frame_set(1)
    return report


def evaluated_matrices(bpy, rig, frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    obj = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {
        p.name: p.matrix.copy() for p in obj.pose.bones if p.name.startswith("DEF-")
    }


def audit_cycle(bpy, rig, action, period):
    activate(rig, action)
    first = evaluated_matrices(bpy, rig, 1)
    last = evaluated_matrices(bpy, rig, period + 1)
    error = max(
        abs(first[n][i][j] - last[n][i][j])
        for n in first
        for i in range(4)
        for j in range(4)
    )
    if not math.isfinite(error) or error > 1e-5:
        raise ValueError(f"Loop closure failed: {error}")
    for frame in range(1, period + 1):
        if any(
            not math.isfinite(v)
            for m in evaluated_matrices(bpy, rig, frame).values()
            for row in m
            for v in row
        ):
            raise ValueError(f"Nonfinite pose at frame {frame}")
    return {"action": action.name, "loop_error": error}


def make_transition(bpy, rig, reference, source, mapping, cfg, name):
    from mathutils import Matrix

    bones = controls(rig)

    def sample(action, frame):
        activate(rig, action)
        evaluated_matrices(bpy, rig, frame)
        return (
            {p.name: p.matrix_basis.copy() for p in bones},
            {k: rig.pose.bones[mapping[k]].matrix.copy() for k in FEET},
        )

    cycles = {f: sample(source, f) for f in range(1, cfg["frames"] + 1)}
    crouch, crouch_feet = sample(reference, 1)
    activate(rig, None)
    for p in bones:
        p.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    neutral_feet = {k: rig.pose.bones[mapping[k]].matrix.copy() for k in FEET}
    result = new_action(bpy, rig, name)
    for frame in range(1, transition_length(cfg) + 1):
        posture, gait, phase = envelope(frame, cfg)
        moving, feet = cycles[int(phase)]
        for p in bones:
            p.matrix_basis = (
                Matrix.Identity(4)
                .lerp(crouch[p.name], posture)
                .lerp(moving[p.name], gait)
            )
            key(p, frame)
        bpy.context.view_layer.update()
        for k in FEET:
            p = rig.pose.bones[mapping[k]]
            p.matrix = neutral_feet[k].lerp(crouch_feet[k], posture).lerp(feet[k], gait)
            key(p, frame)
        bpy.context.view_layer.update()
    linear(result)
    audit_cycle(bpy, rig, result, transition_length(cfg) - 1)
    bpy.context.scene.frame_end = transition_length(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    binding = sub.add_parser("bind", help="Generate and proxy-weight a fitted metarig")
    binding.add_argument("--mesh", required=True)
    binding.add_argument("--metarig", required=True)
    binding.add_argument("--voxel-fraction", type=float, default=0.006)
    binding.add_argument("--allow-unused", action="append", default=[])
    capture_parser = sub.add_parser(
        "capture", help="Save the current reference pose as JSON"
    )
    capture_parser.add_argument(
        "--controls", help="JSON with optional controls mapping and tail list"
    )
    capture_parser.add_argument("--frame", type=int)
    animation = sub.add_parser("animate", help="Build a gait from a captured posture")
    animation.add_argument("--pose", required=True)
    animation.add_argument(
        "--settings", help="JSON object containing documented motion knobs"
    )
    animation.add_argument("--name", default="QuadrupedGait")
    animation.add_argument("--transition", action="store_true")
    for p in [capture_parser, animation]:
        p.add_argument("--rig", required=True)
    for p in [binding, capture_parser, animation]:
        p.add_argument("--out", required=True)
        p.add_argument(
            "--report", help="New JSON audit path; defaults to OUT.report.json"
        )
    args = parser.parse_args(
        argv
        if argv is not None
        else sys.argv[sys.argv.index("--") + 1 :]
        if "--" in sys.argv
        else []
    )
    import bpy

    output = new_path(args.out, bpy.data.filepath)
    report_path = new_path(
        args.report or str(output) + ".report.json", bpy.data.filepath
    )
    if report_path == output:
        raise ValueError("Output and report paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(bpy.data.filepath)
    report = {
        "stage": args.stage,
        "status": "running",
        "blender": bpy.app.version_string,
        "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
        if source.is_file()
        else None,
        "arguments": vars(args),
    }
    try:
        if args.stage == "bind":
            report.update(bind(bpy, args))
        else:
            rig = require_rig(bpy, args.rig)
            if args.stage == "capture":
                if args.frame is not None:
                    bpy.context.scene.frame_set(args.frame)
                mapping = (
                    json.loads(Path(args.controls).read_text()) if args.controls else {}
                )
                if set(mapping) - {"controls", "tail"} or set(
                    mapping.get("controls", {})
                ) - set(DEFAULT_CONTROLS):
                    raise ValueError("Unknown control-map keys")
                pose = capture(
                    bpy,
                    rig,
                    DEFAULT_CONTROLS | mapping.get("controls", {}),
                    mapping.get("tail", DEFAULT_TAIL),
                )
                output.write_text(json.dumps(pose, indent=2))
                report["skeleton"] = pose["skeleton"]
            else:
                cfg = settings(
                    json.loads(Path(args.settings).read_text()) if args.settings else {}
                )
                pose = json.loads(Path(args.pose).read_text())
                report["settings"] = cfg
                report["pose_sha256"] = digest(pose)
                report.update(animate(bpy, rig, pose, cfg, args.name, args.transition))
        if args.stage != "capture":
            bpy.ops.wm.save_as_mainfile(filepath=str(output))
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    print("QUADRUPED_PIPELINE_REPORT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
