"""Exercise the shipped animation workers with in-memory Blender boundaries."""

import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import blender_attack_cycle_worker as attack
import blender_idle_cycle as idle_cli
import blender_idle_cycle_worker as idle
import blender_rigify_walk_cycle_worker as walk
import pytest
from scipy.spatial.transform import Rotation


class Objects(dict):
    def __iter__(self):
        return iter(self.values())


class Matrix:
    def __init__(self, rotation):
        self.rotation = rotation

    @classmethod
    def Rotation(cls, angle, size, axis):
        assert size == 3
        return cls(Rotation.from_euler(axis.lower(), angle))

    def to_3x3(self):
        return self

    def inverted(self):
        return Matrix(self.rotation.inv())

    def __matmul__(self, other):
        return Matrix(self.rotation * other.rotation)

    def to_euler(self, order):
        return vector(self.rotation.as_euler(order.lower()))


def vector(values):
    return SimpleNamespace(**dict(zip("xyz", values, strict=True)))


class Keyframed:
    def __init__(self):
        self.keys = {}
        self.on_keyframe = None

    def keyframe_insert(self, path, *, frame):
        self.on_keyframe()
        value = getattr(self, path)
        self.keys[path, frame] = (value.x, value.y, value.z)


class Bone(Keyframed):
    def __init__(self, *, mode="QUATERNION", rest=None, ik_fk=1.0):
        super().__init__()
        self.rotation_mode = mode
        self.bone = SimpleNamespace(matrix_local=rest or Matrix(Rotation.identity()))
        self.ik_fk = ik_fk
        self.rotation_euler = (1, 1, 1)

    @property
    def rotation_euler(self):
        return self._rotation_euler

    @rotation_euler.setter
    def rotation_euler(self, values):
        self._rotation_euler = vector(values) if isinstance(values, tuple) else values

    def get(self, name, default):
        return self.ik_fk if name == "IK_FK" else default


class Armature(Keyframed):
    def __init__(self, bones, action, *, animation_data=True):
        super().__init__()
        self.name = "rig"
        self.type = "ARMATURE"
        self.pose = SimpleNamespace(bones=Objects(bones))
        self.animation_data = (
            SimpleNamespace(action=None) if animation_data else None
        )
        self.action = action
        self.clear_count = 0
        self.hide_viewport = True

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, values):
        self._location = vector(values)

    def hide_set(self, hidden):
        self.hidden = hidden

    def animation_data_clear(self):
        self.clear_count += 1
        self.animation_data = None


@pytest.fixture
def blender(monkeypatch):
    def setup(bones=None, *, action=None, items=(), animation_data=True):
        action = action or SimpleNamespace(name="Action")
        mesh = SimpleNamespace(name="mesh", type="MESH", hide_render=False)
        marker = SimpleNamespace(name="JOINT_head", type="EMPTY")
        marker.hide_set = lambda hidden: setattr(marker, "hidden", hidden)
        objects = Objects(mesh=mesh, JOINT_head=marker)
        active = SimpleNamespace(active=None)
        frames, modes = [], []
        scene = SimpleNamespace(frame_set=frames.append)

        arm = None
        if bones is not None:
            arm = Armature(bones, action, animation_data=animation_data)
            objects[arm.name] = arm

            def ensure_action():
                if arm.animation_data is None:
                    arm.animation_data = SimpleNamespace(action=None)
                if arm.animation_data.action is None:
                    arm.animation_data.action = action

            # Blender creates an action when the first key is inserted.
            for target in [arm, *arm.pose.bones]:
                target.on_keyframe = ensure_action

        bpy = SimpleNamespace(
            data=SimpleNamespace(objects=objects, actions=list(items)),
            context=SimpleNamespace(
                scene=scene, view_layer=SimpleNamespace(objects=active)
            ),
            ops=SimpleNamespace(
                object=SimpleNamespace(mode_set=lambda *, mode: modes.append(mode))
            ),
        )
        monkeypatch.setitem(sys.modules, "bpy", bpy)
        monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=Matrix))
        return bpy, arm, frames, modes

    return setup


def test_attack_replaces_actions_and_keys_toes_and_undriven_bones(blender, capsys):
    bones = {
        name: Bone() for name in ("frontL_paw", "frontL_toe", "frontR_toe", "head")
    }
    stale = [SimpleNamespace(name=name) for name in ("Attack", "Attack.001")]
    other = SimpleNamespace(name="Idle")
    bpy, arm, frames, modes = blender(bones, items=[*stale, other])
    poses = [{"frontL_paw": 60.0, "tail": 10.0}, {"frontL_paw": -20.0, "tail": 0.0}]

    result = attack.run({"poses": poses, "toe": 0.5, "action_name": "Attack"})

    expected = {
        "frames": 2,
        "action": "Attack",
        "driven_bones": sorted(bones.keys() - {"head"}),
        "missing_bones": ["tail"],
    }
    assert json.loads(capsys.readouterr().out) == result == expected
    assert arm.animation_data.action.name == "Attack"
    assert bpy.data.actions == [*stale, other]
    assert frames == [1, 2, 1]
    assert modes == ["POSE", "OBJECT"]
    assert (bpy.context.scene.frame_start, bpy.context.scene.frame_end) == (1, 2)
    assert bones["frontL_paw"].keys["rotation_euler", 1] == pytest.approx(
        (math.pi / 3, 0, 0)
    )
    assert bones["frontL_toe"].keys["rotation_euler", 2] == pytest.approx(
        (-math.pi / 18, 0, 0)
    )
    for name in ("frontR_toe", "head"):
        assert bones[name].keys == {
            ("rotation_euler", 1): (0, 0, 0),
            ("rotation_euler", 2): (0, 0, 0),
        }
    assert {bone.rotation_mode for bone in bones.values()} == {"XYZ"}


def test_attack_with_no_old_action_or_toes(blender, capsys):
    bpy, _, _, _ = blender({"head": Bone()}, animation_data=False)
    assert attack.run({"poses": [{"head": 0}], "toe": 0, "action_name": "Swing"})[
        "driven_bones"
    ] == ["head"]
    assert bpy.data.actions == []
    assert json.loads(capsys.readouterr().out)["action"] == "Swing"


def test_rigify_warns_about_ik_and_only_keys_sampled_bones(blender, capsys):
    bones = {
        "front_thigh_parent.L": Bone(ik_fk=0.98),
        "thigh_parent.R": Bone(ik_fk=0.99),
        "head": Bone(mode="XYZ"),
        "spine": Bone(),
        "unused": Bone(),
    }
    stale = SimpleNamespace(name="Walk.001")
    other = SimpleNamespace(name="Idle")
    bpy, arm, frames, modes = blender(bones, items=[stale, other])
    poses = [{"head": -45, "missing": 10}, {"spine": 90, "missing": 30}]

    result = walk.run({"poses": poses, "action_name": "Walk"})

    warning, report = capsys.readouterr().out.split("\n", 1)
    assert json.loads(warning)["bones"] == ["front_thigh_parent.L"]
    assert (
        json.loads(report)
        == result
        == {
            "action": "Walk",
            "frames": 2,
            "driven_bones": ["head", "missing", "spine"],
            "missing_bones": ["missing"],
        }
    )
    assert arm.animation_data.action.name == "Walk"
    assert bpy.data.actions == [stale, other]
    assert frames == [1, 2, 1]
    assert modes == ["POSE", "OBJECT"]
    assert (bpy.context.scene.frame_start, bpy.context.scene.frame_end) == (1, 2)
    assert bones["head"].keys["rotation_euler", 1] == pytest.approx(
        (-math.pi / 4, 0, 0)
    )
    assert bones["head"].keys["rotation_euler", 2] == (0, 0, 0)
    assert bones["spine"].keys["rotation_euler", 1] == (0, 0, 0)
    assert bones["spine"].keys["rotation_euler", 2] == pytest.approx(
        (math.pi / 2, 0, 0)
    )
    assert bones["unused"].keys == {}
    assert {bone.rotation_mode for bone in bones.values()} == {"XYZ"}


def test_rigify_can_start_without_animation_data_or_ik_controls(blender, capsys):
    blender({"head": Bone()}, animation_data=False)
    result = walk.run({"poses": [{"head": 12}], "action_name": "Walk"})
    assert json.loads(capsys.readouterr().out) == result
    assert result["missing_bones"] == []


IDLE_ARGS = {
    "frames": 20,
    "breath": 10,
    "head": 0,
    "ear": 0,
    "tail": 5,
    "flick": 10,
    "flick_at": 0,
    "head_yaw": 30,
    "shift": 0.05,
    "ear_turn": 40,
    "ear_turn_at": 0,
    "ear_turn_hold": 0.5,
}


def test_idle_breathes_closes_the_loop_and_swivels_one_ear(blender, capsys):
    names = (
        "spine_01",
        "spine_02",
        "head",
        "jaw",
        "neck",
        "ear_L",
        "ear_R",
        "tail_01",
        "tail_02",
    )
    bones = {name: Bone() for name in names}
    bones["neck"].bone.matrix_local = Matrix.Rotation(math.pi / 2, 3, "X")
    points = [SimpleNamespace(interpolation="LINEAR") for _ in range(2)]
    action = SimpleNamespace(fcurves=[SimpleNamespace(keyframe_points=points)])
    bpy, arm, frames, modes = blender(bones, action=action)

    result = idle.run(IDLE_ARGS)

    assert (
        json.loads(capsys.readouterr().out)
        == result
        == {
            "frames": 20,
            "fcurves": 1,
            "keyframes": 2,
            "missing_bones": [],
        }
    )
    assert arm.clear_count == 1
    assert bpy.context.view_layer.objects.active is arm
    assert arm.animation_data.action is action
    assert frames == [*range(1, 22), 1]
    assert modes == ["POSE", "OBJECT"]
    assert (bpy.context.scene.frame_start, bpy.context.scene.frame_end) == (1, 20)
    assert [p.interpolation for p in points] == ["BEZIER", "BEZIER"]
    assert arm.hidden and arm.hide_render and not arm.hide_viewport
    marker = bpy.data.objects["JOINT_head"]
    assert marker.hidden and marker.hide_render
    assert not bpy.data.objects["mesh"].hide_render
    assert arm.keys["location", 6] == pytest.approx((0, 0, 0.05))
    assert arm.keys["location", 16] == pytest.approx((0, 0, -0.05))
    assert bones["spine_02"].keys["rotation_euler", 6] == pytest.approx(
        (-math.pi / 18, 0, 0)
    )
    assert bones["spine_01"].keys["rotation_euler", 6] == pytest.approx(
        (math.radians(4.5 * math.cos(0.55)), 0, 0)
    )
    assert bones["neck"].keys["rotation_euler", 1] == pytest.approx((0, math.pi / 6, 0))
    assert bones["ear_L"].keys["rotation_euler", 2] == pytest.approx(
        (0.0113800788, 0, 0), abs=1e-9
    )
    assert bones["ear_L"].keys["rotation_euler", 9] == (0, 0, 0)
    # Frames 2, 6, 10 and 12 sample the rise, hold, release and quiet phases of the ear turn.
    assert [
        bones["ear_R"].keys["rotation_euler", f][2] for f in (2, 6, 10, 12)
    ] == pytest.approx([math.radians(a) for a in (40 / 3, 40, 40 / 3, 0)])
    for bone in bones.values():
        assert bone.keys["rotation_euler", 1] == pytest.approx(
            bone.keys["rotation_euler", 21]
        )


def test_idle_uses_layered_curves_and_reports_missing_bones_once(blender):
    point = SimpleNamespace(interpolation="LINEAR")
    bag = SimpleNamespace(fcurves=[SimpleNamespace(keyframe_points=[point])])
    layer = SimpleNamespace(
        strips=[SimpleNamespace(channelbags=[bag]), SimpleNamespace()]
    )
    action = SimpleNamespace(layers=[layer, SimpleNamespace()])
    blender({"head": Bone()}, action=action, animation_data=False)

    summary = idle.run({**IDLE_ARGS, "frames": 2})

    assert summary == {
        "frames": 2,
        "fcurves": 1,
        "keyframes": 1,
        "missing_bones": [
            "ear_L",
            "ear_R",
            "jaw",
            "spine_01",
            "spine_02",
            "tail_01",
            "tail_02",
        ],
    }
    assert point.interpolation == "BEZIER"
    assert idle._fcurves(SimpleNamespace()) == []


@pytest.mark.parametrize(
    ("worker", "message"), [(idle, "no armature"), (walk, "no object named 'rig'")]
)
def test_workers_reject_missing_rig(blender, worker, message):
    blender()
    with pytest.raises(RuntimeError, match=message):
        worker.run({**IDLE_ARGS, "poses": [], "action_name": "Walk"})


def test_idle_cli_bundles_the_worker_and_sends_motion_controls(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "idle",
            "--frames",
            "20",
            "--head-yaw",
            "30",
            "--ear-turn-hold",
            "0.5",
            "--host",
            "studio.local",
            "--port",
            "4321",
        ],
    )

    def send(code, host, port):
        assert (host, port) == ("studio.local", 4321)
        tree = ast.parse(code)
        (bundled_source,) = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
        )
        assert bundled_source == Path(idle.__file__).read_bytes()
        (arguments,) = (
            json.loads(ast.literal_eval(node.args[0]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
        )
        assert arguments["frames"] == 20
        assert arguments["head_yaw"] == 30
        assert arguments["ear_turn_hold"] == 0.5
        assert arguments["breath"] == 2.4
        assert arguments.keys() == IDLE_ARGS.keys()
        return '{"status":"success"}'

    monkeypatch.setattr(idle_cli, "send", send)
    assert idle_cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "success"}


def test_attack_preserves_the_missing_armature_failure(blender):
    blender()
    with pytest.raises(StopIteration):
        attack.run({"poses": [], "toe": 0, "action_name": "Attack"})


WALK_ARGS = {
    "frames": 4,
    "gait": "trot",
    "swing_front": 20,
    "swing_back": 10,
    "bend_front": 30,
    "bend_back": 35,
    "wrist": 25,
    "ankle": 5,
    "base_bend": 8,
    "bob": 0.02,
    "sway": 2,
    "head_yaw": 20,
    "head_pitch": 0,
    "head_level": 5,
    "chest_drop": 12,
    "shoulder_tuck": 4,
    "hind_lead_l": 3,
    "hind_lead_r": 5,
    "body_drop": 0.10,
    "front_crouch": 2,
    "crouch": 10,
    "front_fold_sign": -1,
    "toe": 20,
    "auto_plant": True,
}


@pytest.mark.parametrize(
    ("gait", "front_left_first", "back_right_first"),
    [
        ("trot", 12, 15),
        ("walk", -8, -5),
    ],
)
def test_quadruped_walk_positions_paws_and_seams(
    blender, gait, front_left_first, back_right_first
):
    from blender_walk_cycle_worker import run

    class Coordinate:
        def __init__(self, x, z):
            self.x = x
            self.z = z

        def copy(self):
            return Coordinate(self.x, self.z)

        def __sub__(self, other):
            return SimpleNamespace(
                length_squared=(self.x - other.x) ** 2 + (self.z - other.z) ** 2
            )

    def mesh_data(positions):
        return SimpleNamespace(
            vertices=[SimpleNamespace(co=Coordinate(*point)) for point in positions]
        )

    names = (
        "frontL_upperarm",
        "frontR_upperarm",
        "backL_thigh",
        "backR_thigh",
        "frontL_forearm",
        "frontL_paw",
        "frontL_toe",
        "neck",
        "spine_02",
        "head",
    )
    bones = {name: Bone() for name in names}
    point = SimpleNamespace(interpolation="LINEAR")
    action = SimpleNamespace(fcurves=[SimpleNamespace(keyframe_points=[point])])
    names_by_role = (
        {"neck": "neck_control", "frontL_upperarm": "left_front_control"}
        if gait == "walk"
        else {}
    )
    bones_by_name = {
        names_by_role.get(role, role): bone for role, bone in bones.items()
    }
    bpy, arm, frames, modes = blender(bones_by_name, action=action)
    arm.data = SimpleNamespace(pose_position="REST")
    arm.location = (0, 0, -8)
    marker = bpy.data.objects["JOINT_head"]
    marker.matrix_world = SimpleNamespace(translation=Coordinate(0, 1.05))
    paw_name = "plant_target" if gait == "walk" else "JOINT_frontL_paw"
    paw = SimpleNamespace(name=paw_name, type="EMPTY", matrix_world=marker.matrix_world)
    bpy.data.objects[paw.name] = paw
    bpy.data.objects["PROXY_mesh"] = SimpleNamespace(type="MESH", name="PROXY_mesh")
    updates = []
    bpy.context.view_layer.update = lambda: updates.append(arm.location.z)
    bpy.context.evaluated_depsgraph_get = lambda: "depsgraph"
    mesh = bpy.data.objects["mesh"]
    mesh.data = mesh_data([(10, 0.4), (0, 1.0)])

    def evaluate(depsgraph):
        assert depsgraph == "depsgraph"
        return SimpleNamespace(data=mesh_data([(10, -9), (0, 1.25)]))

    mesh.evaluated_get = evaluate
    arguments = {**WALK_ARGS, "gait": gait}
    if gait == "walk":
        arm.name = "Creature"
        bpy.data.objects["rig"] = Armature({"head": Bone()}, None)
        bpy.data.objects["Creature"] = arm
        bpy.data.objects["decoy_mesh"] = SimpleNamespace(type="MESH", name="decoy_mesh")
        arguments.update(
            bone_names=names_by_role,
            marker_names={"frontL_paw": paw_name},
            rig_name="Creature",
            mesh_name="mesh",
        )
    result = run(arguments)

    assert modes == ["POSE", "OBJECT"]
    assert frames == [1, 2, 3, 4, 5, 1]
    assert updates == [0.0] * 5
    assert bpy.context.view_layer.objects.active is arm
    assert (arm.hide_viewport, arm.hide_render, arm.show_in_front) == (
        False,
        True,
        True,
    )
    assert arm.data.pose_position == "POSE"
    assert marker.hide_render and paw.hide_render
    assert point.interpolation == "BEZIER"
    assert result["frame_range"] == [1, 4]
    assert result["keyframes"] == 1
    assert "ear_L" in result["missing_bones"]
    assert "backL_shin" in result["missing_bones"]
    assert arm.keys["location", 1][2] == pytest.approx(-0.33)
    assert arm.keys["location", 2][2] == pytest.approx(-0.37)
    assert bones["frontL_upperarm"].keys["rotation_euler", 1][0] == pytest.approx(
        math.radians(front_left_first)
    )
    assert bones["backR_thigh"].keys["rotation_euler", 1][0] == pytest.approx(
        math.radians(back_right_first)
    )
    assert bones["frontL_forearm"].keys["rotation_euler", 1][0] == pytest.approx(
        math.radians(-20 if gait == "trot" else -50)
    )
    assert bones["neck"].keys["rotation_euler", 1][2] == pytest.approx(math.radians(20))
    assert bones["spine_02"].keys["rotation_euler", 1][0] == pytest.approx(
        math.radians(12)
    )
    for bone in bones.values():
        assert bone.keys["rotation_euler", 1] == pytest.approx(
            bone.keys["rotation_euler", 5]
        )


@pytest.mark.parametrize(
    ("auto_plant", "has_mesh", "corrections"),
    [(False, True, True), (True, False, False), (True, True, False)],
)
def test_quadruped_walk_can_skip_paw_planting(
    blender, auto_plant, has_mesh, corrections
):
    from blender_walk_cycle_worker import run

    point = SimpleNamespace(interpolation="LINEAR")
    bag = SimpleNamespace(fcurves=[SimpleNamespace(keyframe_points=[point])])
    action = SimpleNamespace(
        layers=[SimpleNamespace(strips=[SimpleNamespace(channelbags=[bag])])]
    )
    bpy, arm, _, _ = blender({"head": Bone()}, animation_data=False, action=action)
    arm.data = SimpleNamespace()
    del bpy.data.objects["JOINT_head"]
    if has_mesh:
        bpy.data.objects["mesh"].data = SimpleNamespace(vertices=[])
    else:
        del bpy.data.objects["mesh"]

    result = run(
        {
            **WALK_ARGS,
            "frames": 2,
            "auto_plant": auto_plant,
            "head_yaw": 0,
            "head_pitch": 15 if corrections else 0,
            "chest_drop": 12 if corrections else 0,
        }
    )

    assert result["fcurves"] == 1
    assert point.interpolation == "BEZIER"
    assert arm.keys["location", 1][2] == pytest.approx(-0.08)


def test_quadruped_walk_rejects_a_missing_armature(blender):
    from blender_walk_cycle_worker import run

    blender()
    with pytest.raises(RuntimeError, match="no armature"):
        run(WALK_ARGS)


def test_quadruped_walk_cli_bundles_gait_and_planting_arguments(monkeypatch, capsys):
    import blender_walk_cycle as cli
    import blender_walk_cycle_worker as worker

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk",
            "--gait",
            "walk",
            "--crouch",
            "12",
            "--front-fold-sign",
            "1",
            "--no-auto-plant",
            "--host",
            "studio.local",
            "--port",
            "4321",
        ],
    )

    def send(code, host, port):
        assert (host, port) == ("studio.local", 4321)
        tree = ast.parse(code)
        (source,) = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
        )
        assert source == Path(worker.__file__).read_bytes()
        (arguments,) = (
            json.loads(ast.literal_eval(node.args[0]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
        )
        assert arguments.keys() == WALK_ARGS.keys()
        assert arguments["gait"] == "walk"
        assert arguments["crouch"] == 12
        assert arguments["front_fold_sign"] == 1
        assert arguments["auto_plant"] is False
        return "sent"

    monkeypatch.setattr(cli, "send", send)
    assert cli.main() == 0
    assert capsys.readouterr().out == "sent\n"


@pytest.mark.parametrize(
    ("worker", "args", "role", "frame", "axis", "angle"),
    [
        (idle, IDLE_ARGS, "ear_R", 6, 2, 40),
        (
            attack,
            {"poses": [{"frontL_paw": 40}], "toe": 0.5, "action_name": "Strike"},
            "frontL_toe",
            1,
            0,
            20,
        ),
        (walk, {"poses": [{"head": 40}], "action_name": "Trot"}, "head", 1, 0, 40),
    ],
)
def test_named_armature_and_remapped_bone_receive_keys(
    blender, worker, args, role, frame, axis, angle
):
    target_bone = Bone()
    bpy, target, _, _ = blender({"custom_bone": target_bone})
    target.name = "Creature"
    decoy = Armature({role: Bone()}, None)
    bpy.data.objects["rig"] = decoy
    bpy.data.objects[target.name] = target

    worker.run({**args, "rig_name": target.name, "bone_names": {role: "custom_bone"}})

    assert bpy.context.view_layer.objects.active is target
    assert target_bone.keys["rotation_euler", frame][axis] == pytest.approx(
        math.radians(angle)
    )
    assert not decoy.pose.bones[role].keys


@pytest.mark.parametrize(
    "module",
    [
        "blender_idle_cycle",
        "blender_attack_cycle",
        "blender_rigify_walk_cycle",
        "blender_walk_cycle",
    ],
)
def test_animation_cli_forwards_rig_and_bone_map(module, tmp_path, monkeypatch):
    import importlib

    cli = importlib.import_module(module)
    names = tmp_path / "bones.json"
    names.write_text('{"head": "head_control"}', encoding="utf-8")
    cli_args = [module, "--rig-name", "Creature", "--bone-map", str(names)]
    if module == "blender_walk_cycle":
        markers = tmp_path / "markers.json"
        markers.write_text('{"frontL_paw": "toe_anchor"}', encoding="utf-8")
        cli_args.extend(["--marker-map", str(markers), "--mesh-name", "Creature Mesh"])
    monkeypatch.setattr(sys, "argv", cli_args)

    def send(code, host, port):
        tree = ast.parse(code)
        (arguments,) = (
            json.loads(ast.literal_eval(node.args[0]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
        )
        assert arguments["rig_name"] == "Creature"
        assert arguments["bone_names"] == {"head": "head_control"}
        if module == "blender_walk_cycle":
            assert arguments["marker_names"] == {"frontL_paw": "toe_anchor"}
            assert arguments["mesh_name"] == "Creature Mesh"
        return "sent"

    monkeypatch.setattr(cli, "send", send)
    assert cli.main() == 0


@pytest.mark.parametrize(
    "body", [None, "invalid", "[]", '{"": "bone"}', '{"head": 1}', '{"head": ""}']
)
def test_animation_names_must_be_readable_nonempty_pairs(tmp_path, body):
    from argparse import ArgumentTypeError

    from animation_cli import read_name_map

    path = tmp_path / "names.json"
    if body is not None:
        path.write_text(body, encoding="utf-8")
    with pytest.raises(ArgumentTypeError, match="name map|Name map"):
        read_name_map(str(path))


@pytest.mark.parametrize("mesh_name", ["unknown", "rig"])
def test_quadruped_walk_checks_explicit_mesh_before_editing_rig(blender, mesh_name):
    from blender_walk_cycle_worker import run

    _, arm, _, modes = blender({"head": Bone()})
    with pytest.raises(RuntimeError, match=f"no mesh named '{mesh_name}'"):
        run({**WALK_ARGS, "mesh_name": mesh_name})
    assert arm.clear_count == 0
    assert modes == []


@pytest.mark.parametrize("worker,name", [(attack, "Attack"), (walk, "Walk")])
@pytest.mark.parametrize("remaining_users", [0, 1, 2])
@pytest.mark.parametrize("old_name", ["requested", "requested.001", "Idle"])
def test_animation_replacement_preserves_other_clips(blender, worker, name, remaining_users, old_name):
    old = SimpleNamespace(name=old_name.replace("requested", name), users=remaining_users)
    unrelated = SimpleNamespace(name=name + ".002", users=1)
    bpy, arm, _, _ = blender({"head": Bone()}, items=[old, unrelated])
    arm.animation_data.action = old
    other_rig = SimpleNamespace(animation_data=SimpleNamespace(action=unrelated))
    bpy.data.objects["OtherRig"] = other_rig
    worker.run({"poses": [{"head": 0}, {"head": 30}], "toe": 0,
                "action_name": name, "rig_name": arm.name})
    assert unrelated in bpy.data.actions
    assert other_rig.animation_data.action is unrelated
    assert (old in bpy.data.actions) == (remaining_users > 0 or old_name == "Idle")
    assert arm.animation_data.action is not old
