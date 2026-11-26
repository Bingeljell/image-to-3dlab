"""Exercise quadruped validation and capture without opening a Blender scene."""

import json
import sys
from types import SimpleNamespace as NS

import blender_quadruped_pipeline as pipeline
import blender_quadruped_walk as walk
import numpy as np
import pytest


class Vector(np.ndarray):
    """Only the vector operations needed by the capture and rest geometry tests."""

    def __new__(cls, coords=(0, 0, 0)):
        return np.asarray(coords, dtype=float).view(cls)

    @property
    def x(self):
        return self[0]

    @property
    def length(self):
        return float(np.linalg.norm(self))

    @property
    def z(self):
        return self[2]

    @property
    def y(self):
        return self[1]

    def normalized(self):
        return self / self.length

    def normalize(self):
        self[:] = self.normalized()

    def cross(self, other):
        return Vector(np.cross(self, other))


class IdentityTransform:
    def __iter__(self):
        return iter(np.eye(4))

    def __matmul__(self, value):
        return value

    def to_3x3(self):
        return self

    def inverted(self):
        return self

    def to_scale(self):
        return (1, 1, 1)

    def determinant(self):
        return 1


class BoneMap(dict):
    def __iter__(self):
        return iter(self.values())


class PoseBone(dict):
    def __init__(self, name):
        super().__init__(IK_FK=0, label="skip")
        self.name = name
        self.rotation_mode = "QUATERNION"
        self.matrix = self.matrix_basis = np.eye(4)


def rest_bone(name, head=(0, 0, 0), tail=(0, 0, 0), parent=None):
    return NS(
        name=name, head_local=Vector(head), tail_local=Vector(tail), parent=parent
    )


@pytest.fixture
def capture_scene(monkeypatch, tmp_path):
    names = set(pipeline.DEFAULT_CONTROLS.values()) | set(pipeline.DEFAULT_TAIL)
    rest = BoneMap({name: rest_bone(name) for name in sorted(names)})
    rest["hips"].head_local = Vector((0, 0, 2))
    rest["head"].head_local = Vector((0, -1, 3))
    rest["chest"].head_local = Vector((0, -1, 2))
    rest["DEF-spine"] = rest_bone("DEF-spine", parent=rest["chest"])
    rig = NS(
        name="rig",
        type="ARMATURE",
        data=NS(bones=rest),
        pose=NS(
            bones=BoneMap(
                {name: PoseBone(name) for name in sorted(names | {"DEF-spine"})}
            )
        ),
        matrix_world=IdentityTransform(),
        animation_data=None,
    )
    source = tmp_path / "input.blend"
    source.write_bytes(b"source")
    frames, updates = [], []
    bpy = NS(
        data=NS(objects={rig.name: rig}, filepath=str(source)),
        context=NS(
            scene=NS(frame_set=frames.append),
            view_layer=NS(update=lambda: updates.append(True)),
        ),
        app=NS(version_string="test Blender"),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    return bpy, rig, source, frames, updates


def test_capture_cli_writes_reference_and_audit(capture_scene, tmp_path):
    _, rig, source, frames, updates = capture_scene
    output = tmp_path / "new" / "pose.json"
    controls = tmp_path / "controls.json"
    controls.write_text(json.dumps({"tail": []}))
    assert (
        pipeline.main(
            [
                "capture",
                "--rig",
                rig.name,
                "--frame",
                "6",
                "--out",
                str(output),
                "--controls",
                str(controls),
            ]
        )
        == 0
    )
    reference = json.loads(output.read_text())
    report = json.loads(output.with_suffix(".json.report.json").read_text())
    assert reference["support_height"] == 2
    assert reference["up"] == [0, 0, 1]
    assert reference["forward"] == [0, -1, 0]
    assert reference["right"] == [-1, 0, 0]
    assert reference["bones"]["head"]["properties"] == {"IK_FK": 0}
    assert "DEF-spine" not in reference["bones"]
    assert set(reference["feet"]) == set(pipeline.FEET)
    assert reference["tail"] == []
    assert report["status"] == "ok"
    assert report["skeleton"] == reference["skeleton"]
    assert report["input"] == str(source)
    assert report["input_sha256"]
    assert frames == [6]
    assert updates == [True]
    assert source.read_bytes() == b"source"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "Missing controls"),
        ("fk", "IK mode"),
        ("heading", "heading"),
        ("height", "support height"),
        ("rotation", "Axis-angle"),
        ("tracks", "Mute NLA"),
        ("scale", "Nonuniform"),
        ("rig", "generated Rigify"),
        ("mapping", "Unknown control-map"),
    ],
)
def test_capture_failures_write_audit_without_reference(
    capture_scene, tmp_path, failure, message
):
    _, rig, _, _, _ = capture_scene
    output = tmp_path / "pose.json"
    report = tmp_path / "logs" / "failure.json"
    args = ["capture", "--rig", rig.name, "--out", str(output), "--report", str(report)]
    if failure == "missing":
        del rig.pose.bones["head"]
    elif failure == "fk":
        rig.pose.bones["head"]["IK_FK"] = 0.5
    elif failure == "heading":
        rig.data.bones["head"].head_local = Vector((0, 0, 3))
    elif failure == "height":
        rig.data.bones["chest"].head_local = Vector((0, -1, -2))
    elif failure == "rotation":
        rig.pose.bones["head"].rotation_mode = "AXIS_ANGLE"
    elif failure == "tracks":
        rig.animation_data = NS(nla_tracks=[NS(mute=False)])
    elif failure == "scale":
        rig.matrix_world.to_scale = lambda: (1, 2, 1)
    elif failure == "rig":
        rig.type = "MESH"
    else:
        mapping = tmp_path / "bad.json"
        mapping.write_text('{"controls": {"missing": "arm"}}')
        args.extend(["--controls", str(mapping)])
    with pytest.raises(ValueError, match=message):
        pipeline.main(args)
    assert not output.exists()
    audit = json.loads(report.read_text())
    assert audit["status"] == "failed"
    assert message in audit["error"]


def test_capture_rejects_shared_output_and_report_path(capture_scene, tmp_path):
    output = str(tmp_path / "pose.json")
    with pytest.raises(ValueError, match="paths must differ"):
        pipeline.main(["capture", "--rig", "rig", "--out", output, "--report", output])


def test_walk_finds_the_only_generated_armature(capture_scene):
    _, rig, _, _, _ = capture_scene
    metarig = NS(type="ARMATURE", data=NS(bones=[rest_bone("spine")]))
    mesh = NS(type="MESH")
    assert walk.find_rig([mesh, metarig, rig]) is rig
    with pytest.raises(RuntimeError, match="no generated"):
        walk.find_rig([mesh, metarig])
    other_rig = NS(name="other", type="ARMATURE", data=rig.data)
    with pytest.raises(RuntimeError, match="several generated"):
        walk.find_rig([rig, other_rig])


def test_walk_reads_local_chain_lengths_and_world_offsets():
    rig = NS(
        data=NS(
            bones={
                "ORG-thigh": rest_bone("ORG-thigh", (0, 0, 5), (0, 1, 3)),
                "ORG-shin": rest_bone("ORG-shin", (0, 1, 3), (0, 2, 1)),
            }
        ),
        matrix_world=IdentityTransform(),
    )
    metarig = NS(
        data=NS(
            bones={
                "thigh": rest_bone("thigh", (0, 0, 5), (0, 0, 3)),
                "shin": rest_bone("shin", (0, 0, 3), (0, 0, 1)),
            }
        )
    )
    assert walk.chain_length(rig, metarig, ["thigh", "shin"]) == 4
    assert walk.chain_length(rig, None, ["thigh", "shin"]) == pytest.approx(
        2 * np.sqrt(5)
    )
    with pytest.raises(RuntimeError, match="cannot measure limb"):
        walk.chain_length(rig, metarig, ["missing"])
    assert walk.reach_geometry(rig, metarig, "thigh", "shin") == {
        "chain_length": 4,
        "vertical_drop": 4,
        "horizontal_at_rest": 2,
    }
    assert walk.reach_geometry(rig, None, "thigh", "shin")[
        "chain_length"
    ] == pytest.approx(2 * np.sqrt(5))


def test_walk_derives_forward_axis_with_missing_control_fallback():
    rig = NS(pose=NS(bones={}), matrix_world=IdentityTransform())
    assert walk.forward_axis(rig) == (1, -1)
    rig.pose.bones.update(
        head=NS(head=Vector((3, 0, 8))), hips=NS(head=Vector((0, 1, 2)))
    )
    assert walk.forward_axis(rig) == (0, 1)
    rig.pose.bones["head"].head = Vector((0, -3, 8))
    assert walk.forward_axis(rig) == (1, -1)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([0, 1, 0], None),
        ([0, 1, 0.1], "Loop closure failed"),
        ([0, float("nan"), 0], "Nonfinite pose at frame 2"),
    ],
)
def test_loop_audit_reads_evaluated_bones(capture_scene, frames, message):
    bpy, rig, _, _, updates = capture_scene
    action = NS(name="cycle", slots=["rig slot"])
    rig.animation_data_create = lambda: setattr(rig, "animation_data", NS())
    depsgraph = object()
    bpy.context.evaluated_depsgraph_get = lambda: depsgraph
    current = NS(index=0)

    def frame_set(frame):
        current.index = frame - 1

    def evaluated(dep):
        assert dep is depsgraph
        transform = np.eye(4)
        transform[0, 3] = frames[current.index]
        deformation = NS(name="DEF-spine", matrix=transform)
        control = NS(name="head", matrix=np.full((4, 4), float("nan")))
        return NS(pose=NS(bones=[control, deformation]))

    bpy.context.scene.frame_set = frame_set
    rig.evaluated_get = evaluated
    if message:
        with pytest.raises(ValueError, match=message):
            pipeline.audit_cycle(bpy, rig, action, 2)
    else:
        assert pipeline.audit_cycle(bpy, rig, action, 2) == {
            "action": "cycle",
            "loop_error": 0,
        }
        assert len(updates) == 4
    assert rig.animation_data.action is action
    assert rig.animation_data.action_slot == "rig slot"


def test_walk_audit_reports_paw_and_body_motion(capture_scene):
    bpy, rig, _, _, _ = capture_scene
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 3
    current = NS(index=0)
    scene.frame_set = lambda frame: setattr(current, "index", frame - 1)
    bpy.context.evaluated_depsgraph_get = lambda: None

    def evaluated(_):
        stride, height = [(0, 0), (1, 0.5), (0, 0)][current.index]
        pose = {key: NS(tail=Vector((stride, 0, height))) for key in walk.PAW.values()}
        pose["torso"] = NS(head=Vector((0, 0, 4 + height)))
        pose[walk.RAKE_CHAIN[0]] = NS(tail=Vector((0, 0, 3)))
        pose[walk.RAKE_CHAIN[1]] = NS(tail=Vector((stride, 0, 2)))
        return NS(matrix_world=rig.matrix_world, pose=NS(bones=pose))

    rig.evaluated_get = evaluated
    audit = walk.audit_cycle(
        bpy, rig, scene, body_height=4, leg_lengths={"F": 2, "B": 4}, axis=0
    )
    assert audit["loop_gap"] == 0
    assert audit["front_rake_excursion_deg"] == pytest.approx(45)
    assert audit["body_bob_pct_height"] == 0.125
    assert audit["limbs"]["FL"] == pytest.approx(
        {
            "stride_pct_leg": 0.5,
            "lift_pct_height": 0.125,
            "planted_fraction": 2 / 3,
        }
    )
    assert audit["limbs"]["BL"]["stride_pct_leg"] == 0.25


def test_head_yaw_uses_weighted_world_centroids(capture_scene):
    _, rig, _, _, _ = capture_scene
    groups = [NS(name="head", index=0), NS(name="neck", index=1)]

    def vertex(coords, group, weight):
        return NS(co=Vector(coords), groups=[NS(group=group, weight=weight)])

    mesh = NS(
        vertex_groups=groups,
        matrix_world=rig.matrix_world,
        data=NS(
            vertices=[
                vertex((2, -2, 5), 0, 1),
                vertex((0, 0, 5), 0, 0.5),
                vertex((100, 100, 0), 0, 0.25),
                vertex((0, 0, 4), 1, 1),
            ]
        ),
    )
    assert walk.measured_head_yaw(rig, mesh, "head", "neck") == pytest.approx(np.pi / 4)
    assert walk.measured_head_yaw(rig, mesh, "unknown", "neck") == 0
    mesh.data.vertices.pop()
    assert walk.measured_head_yaw(rig, mesh, "head", "neck") == 0


def test_walk_motion_can_spend_extra_reach_on_stride():
    args = NS(
        frames=12,
        gait="walk",
        scale=1,
        body_drop=0.5,
        spend_reach=True,
        reach_fill=0.75,
        tail_swing=0.0,
        body_sway=0.0,
    )
    front = {"chain_length": 3, "vertical_drop": 2.5, "horizontal_at_rest": 0}
    hind = {"chain_length": 4, "vertical_drop": 3.5, "horizontal_at_rest": 0}
    frame_count, plan = walk.motion_plan(args, 3, 4, 5, front, hind)
    assert frame_count == 12
    assert plan["sizes"]["front_stride"] == pytest.approx(
        1.5 * np.sqrt((0.95 * 3) ** 2 - 2**2)
    )
    assert plan["sizes"]["hind_stride"] == pytest.approx(
        1.5 * np.sqrt((0.95 * 4) ** 2 - 3**2)
    )
    assert all(frame["lateral"] == 0 for frame in plan["body"])

    args.spend_reach = False
    args.frames = args.tail_swing = args.body_sway = args.body_drop = None
    frame_count, defaults = walk.motion_plan(args, 3, 4, 5, front, hind)
    assert frame_count == walk.GAITS["walk"]["frames"]
    assert defaults["sizes"]["front_stride"] < plan["sizes"]["front_stride"]
    assert any(frame["lateral"] != 0 for frame in defaults["body"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [(None, (0, 0, 0)), ("90, -45, 0", (np.pi / 2, -np.pi / 4, 0))],
)
def test_walk_parse_static_rotation_in_degrees(text, expected):
    assert walk.parse_offset(text) == pytest.approx(expected)


def test_walk_static_rotation_needs_three_axes():
    with pytest.raises(RuntimeError, match="three comma-separated degrees"):
        walk.parse_offset("90,45")


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("schema", "incompatible"),
        ("skeleton", "incompatible"),
        ("transform", "rotation/scale changed"),
        ("controls", "Control set differs"),
        ("action", "Action already exists"),
    ],
)
def test_animating_stops_before_writing_when_pose_is_incompatible(
    capture_scene, fault, message
):
    bpy, rig, _, _, updates = capture_scene
    pose = pipeline.capture(bpy, rig, pipeline.DEFAULT_CONTROLS, pipeline.DEFAULT_TAIL)
    bpy.data.actions = {}
    if fault == "schema":
        pose["schema_version"] = 2
    elif fault == "skeleton":
        pose["skeleton"] = "different rig"
    elif fault == "transform":
        pose["rig_linear_transform"][0][0] = 2
    elif fault == "controls":
        del pose["bones"]["head"]
    else:
        bpy.data.actions["walk_Transition"] = object()
    before = dict(bpy.data.actions)
    with pytest.raises(ValueError, match=message):
        pipeline.animate(bpy, rig, pose, pipeline.settings({}), "walk", transition=True)
    assert bpy.data.actions == before
    assert rig.animation_data is None
    assert updates == [True]


def test_bind_audit_checks_normalized_deform_weights_and_allowed_unused_bones():
    rig = NS(
        data=NS(
            bones=[
                NS(name="deform", use_deform=True),
                NS(name="optional", use_deform=True),
                NS(name="control", use_deform=False),
            ]
        )
    )
    mesh = NS(
        vertex_groups=[NS(index=0, name="deform"), NS(index=1, name="control")],
        data=NS(vertices=[NS(groups=[NS(group=0, weight=1), NS(group=1, weight=0.5)])]),
    )
    assert pipeline.audit_weights(mesh, rig, ["optional"]) == (["optional"], [1, 1])
    with pytest.raises(ValueError, match="unused=\\['optional'\\]"):
        pipeline.audit_weights(mesh, rig, [])
    mesh.data.vertices[0].groups[0].weight = 0.8
    with pytest.raises(ValueError, match=r"sum range=\(0\.8, 0\.8\)"):
        pipeline.audit_weights(mesh, rig, ["optional"])
    mesh.data.vertices.clear()
    with pytest.raises(ValueError, match=r"sum range=\(0, 0\)"):
        pipeline.audit_weights(mesh, rig, ["deform", "optional"])


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing", "Name an existing mesh"),
        ("shape keys", "expects an unbound mesh"),
        ("groups", "expects an unbound mesh"),
        ("generated", "Expected an unanimated metarig"),
        ("action", "Expected an unanimated metarig"),
        ("target", "already targets a generated rig"),
        ("posed", "Fit the metarig in Edit Mode"),
        ("reflected", "reflected transforms"),
    ],
)
def test_bind_rejects_unsafe_fit_without_altering_scene(
    capture_scene, tmp_path, fault, message
):
    bpy, meta, source, _, updates = capture_scene
    mesh = NS(
        type="MESH",
        modifiers=[],
        data=NS(shape_keys=None),
        vertex_groups=[],
        matrix_world=IdentityTransform(),
    )
    bpy.data.objects["mesh"] = mesh
    name = "mesh"
    del meta.data.bones["DEF-spine"]
    if fault == "missing":
        name = "absent"
    elif fault == "shape keys":
        mesh.data.shape_keys = object()
    elif fault == "groups":
        mesh.vertex_groups = [object()]
    elif fault == "generated":
        meta.data.bones["DEF-spine"] = rest_bone("DEF-spine")
    elif fault == "action":
        meta.animation_data = object()
    elif fault == "target":
        meta.data.rigify_target_rig = object()
    elif fault == "posed":
        meta.pose.bones["head"].matrix_basis[0, 3] = 0.5
    else:
        meta.matrix_world.determinant = lambda: -1
    output = tmp_path / "bound.blend"
    with pytest.raises(ValueError, match=message):
        pipeline.main(
            ["bind", "--mesh", name, "--metarig", "rig", "--out", str(output)]
        )
    audit = json.loads(output.with_suffix(".blend.report.json").read_text())
    assert audit["status"] == "failed"
    assert not output.exists()
    assert not updates
    assert source.read_bytes() == b"source"


def test_body_offsets_close_the_loop_and_use_the_supplied_control_names():
    mapping = {
        name: f"control:{name}" for name in ("torso", "hips", "chest", "neck", "head")
    }
    tail = ["tail-a", "tail-b"]
    settings = pipeline.settings({})
    first = dict(pipeline.body_offsets(0, 2, settings, mapping, tail))
    last = dict(pipeline.body_offsets(1, 2, settings, mapping, tail))
    assert list(first) == list(mapping.values()) + tail
    for control, angles in first.items():
        assert angles == pytest.approx(last[control], abs=1e-12)
    assert first["tail-a"]["yaw"] != first["tail-b"]["yaw"]
    taller = dict(pipeline.body_offsets(0, 4, settings, mapping, tail))
    assert taller["control:torso"]["z"] == pytest.approx(
        2 * first["control:torso"]["z"]
    )
    assert taller["control:torso"]["roll"] == first["control:torso"]["roll"]


def test_body_offsets_respect_disabled_motion_and_empty_tail():
    mapping = {name: name for name in ("torso", "hips", "chest", "neck", "head")}
    settings = pipeline.settings(
        {
            "bob": 0,
            "body_roll_degrees": 0,
            "hip_motion": 0,
            "chest_motion": 0,
            "neck_motion": 0,
            "head_compensation": 0,
        }
    )
    offsets = dict(pipeline.body_offsets(0.4, 2, settings, mapping, []))
    assert set(offsets) == set(mapping)
    assert all(
        offset == 0 for control in offsets.values() for offset in control.values()
    )


class RotationControl:
    def __init__(self, angles=(0, 0, 0)):
        self.rotation_euler = angles
        self.rotation_mode = "QUATERNION"

    @property
    def rotation_euler(self):
        return self._angles

    @rotation_euler.setter
    def rotation_euler(self, value):
        self._angles = list(value)


def test_swing_axis_probes_both_directions_and_restores_previous_rotation(
    capture_scene,
):
    bpy, rig, _, _, _ = capture_scene
    control = RotationControl((0.1, 0.2, 0.3))
    tip = NS(tail=Vector())
    rig.pose.bones = {"thigh.L": control, "foot.L": tip}

    def update():
        x, y, z = control.rotation_euler
        tip.tail = Vector((0, x - 3 * y + 2 * z, 0))

    bpy.context.view_layer.update = update
    assert walk.swing_axis(bpy, rig, "thigh.L") == (1, 1)
    assert control.rotation_euler == [0.1, 0.2, 0.3]
    assert control.rotation_mode == "QUATERNION"
    assert tip.tail.y == pytest.approx(0.1)


def test_axis_probe_uses_relative_tip_motion(capture_scene):
    bpy, rig, _, _, _ = capture_scene
    control = RotationControl()
    tip = NS(tail=Vector((0, 0, 3)))
    rig.pose.bones = {"tail": control, "tip": tip}

    def update():
        x, y, z = control.rotation_euler
        tip.tail = Vector((0, 0, 3 + x + 2 * y - 5 * z))

    bpy.context.view_layer.update = update
    assert walk.axis_for(bpy, rig, "tail", "tip", 2) == (2, -1)
    assert control.rotation_euler == [0, 0, 0]
    assert control.rotation_mode == "QUATERNION"
    assert tip.tail.z == 3


@pytest.mark.parametrize("has_mesh", [False, True])
def test_walk_cli_validates_mesh_and_required_controls_before_motion(
    capture_scene, monkeypatch, has_mesh
):
    bpy, rig, _, _, _ = capture_scene
    widget = NS(type="MESH", name="WGT-foot", data=NS(vertices=[object()]))
    meshes = (
        [NS(type="MESH", name="asset", data=NS(vertices=[object()]))]
        if has_mesh
        else []
    )
    bpy.data.objects = [rig, widget, *meshes]
    bpy.context.object = NS(mode="POSE")
    events = []
    bpy.ops = NS(object=NS(mode_set=lambda *, mode: events.append(mode)))
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--gait", "trot"])
    with pytest.raises(
        RuntimeError, match="missing controls" if has_mesh else "no mesh found"
    ):
        walk.main()
    assert events == ["OBJECT"]
    assert rig.animation_data is None


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (None, None),
        ("missing joint", "missing ORG-"),
        ("moved joint", "alignment error"),
        ("reserved obj", "Reserved proxy object"),
        ("moved mesh", "changed visible geometry or UVs"),
        ("edited uv", "changed visible geometry or UVs"),
    ],
)
def test_bind_preserves_world_geometry_joints_and_uvs(
    capture_scene, monkeypatch, fault, message
):
    import blender_rebind_weights as rebind_weights

    bpy, metarig, _, _, _ = capture_scene
    coords = Vector((1, 2, 3))
    uv = NS(uv=(0.2, 0.5))
    data = NS(
        shape_keys=None,
        users=1,
        vertices=[NS(co=coords, groups=[])],
        uv_layers=[NS(data=[uv])],
    )
    mesh = NS(
        name="asset",
        type="MESH",
        modifiers=[],
        data=data,
        vertex_groups=[],
        matrix_world=IdentityTransform(),
        dimensions=(1, 2, 3),
    )
    bpy.data.objects = {mesh.name: mesh, metarig.name: metarig}
    metarig.data.bones = BoneMap({"spine": rest_bone("spine", (0, 0, 0), (0, 1, 0))})
    metarig.pose.bones = BoneMap({"spine": PoseBone("spine")})
    metarig.data.users = 2
    metarig.data.copy = lambda: NS(**{**vars(metarig.data), "users": 1})
    orig_meta_data = metarig.data
    org = rest_bone(
        "ORG-spine", (0, 0, 0), (0.1, 1, 0) if fault == "moved joint" else (0, 1, 0)
    )
    org.use_deform = False
    deform = rest_bone("DEF-spine")
    deform.use_deform = True
    generated_bones = BoneMap({deform.name: deform})
    if fault != "missing joint":
        generated_bones[org.name] = org
    rig = NS(
        type="ARMATURE",
        name="generated",
        data=NS(bones=generated_bones),
        matrix_world=IdentityTransform(),
    )
    active = NS(active=None)
    bpy.context.object = NS(mode="OBJECT")
    bpy.context.view_layer.objects = active
    actions = []

    for obj in (mesh, metarig):
        obj.parent = object()
        obj.select_set = lambda selected, obj=obj: setattr(obj, "selected", selected)
        obj.hide_set = lambda hidden, obj=obj: setattr(obj, "hidden", hidden)
        obj.matrix_world.copy = lambda obj=obj: obj.matrix_world
    if fault == "reserved obj":
        bpy.data.objects["I2L_WEIGHT_PROXY_asset"] = object()

    def transform(*, location, rotation, scale):
        assert active.active in (mesh, metarig)
        assert active.active.parent is None
        assert active.active.selected and scale
        assert location is rotation is (active.active is metarig)
        actions.append(active.active.name)

    def generate():
        assert metarig.data is not orig_meta_data
        assert active.active is metarig
        metarig.data.rigify_target_rig = rig
        actions.append("rigify")

    def transfer(app, obj, armature, *, voxel_fraction):
        assert app is bpy and obj is mesh and armature is rig and voxel_fraction == 0.01
        obj.vertex_groups = [NS(index=0, name="DEF-spine")]
        obj.data.vertices[0].groups = [NS(group=0, weight=1)]
        if fault == "moved mesh":
            obj.data.vertices[0].co = Vector((2, 3, 4))
        elif fault == "edited uv":
            uv.uv = (0.3, 0.5)
        actions.append("transfer")
        return {"mesh": "asset"}

    def normalize(*, group_select_mode, lock_active):
        assert group_select_mode == "BONE_DEFORM" and not lock_active
        assert active.active is mesh
        actions.append("normalize")

    bpy.ops = NS(
        object=NS(
            select_all=lambda *, action: None,
            transform_apply=transform,
            vertex_group_normalize_all=normalize,
        ),
        preferences=NS(addon_enable=lambda *, module: None),
        pose=NS(rigify_generate=generate),
    )
    monkeypatch.setattr(rebind_weights, "transfer_weights", transfer)
    args = NS(mesh="asset", metarig="rig", voxel_fraction=0.01, allow_unused=[])
    if message:
        with pytest.raises(ValueError, match=message):
            pipeline.bind(bpy, args)
        assert not metarig.hidden
        if fault in {"missing joint", "moved joint", "reserved obj"}:
            assert actions == ["asset", "rig", "rigify"]
    else:
        result = pipeline.bind(bpy, args)
        assert result == {
            "mesh": "asset",
            "rig": "generated",
            "unused_deform_bones": [],
            "joint_error": 0,
            "geometry_error": 0,
            "weight_sum_range": [1, 1],
            "visual_review_required": True,
        }
        assert metarig.hidden
        assert actions == ["asset", "rig", "rigify", "transfer", "normalize"]
