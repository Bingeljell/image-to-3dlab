"""Author reference, cycle, and transition actions against in-memory rig data."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import blender_quadruped_pipeline as pipeline
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from tests.test_coverage_blender_quadruped import BoneMap, PoseBone, Vector
from tests.test_coverage_blender_quadruped import capture_scene as reference_scene

capture_scene = reference_scene


class Matrix(np.ndarray):
    def __new__(cls, rows):
        return np.array(rows, dtype=float).view(cls)

    @staticmethod
    def Identity(size):
        return Matrix(np.eye(size))

    @staticmethod
    def LocRotScale(location, rotation, scale):
        transform = Matrix.Identity(4)
        transform[:3, :3] = rotation.as_matrix() * scale
        transform[:3, 3] = location
        return transform

    @property
    def translation(self):
        return Vector(self[:3, 3])

    def to_scale(self):
        return np.linalg.norm(self[:3, :3], axis=0)

    def to_quaternion(self):
        return Quaternion(Rotation.from_matrix(self[:3, :3] / self.to_scale()))

    def to_4x4(self):
        result = Matrix.Identity(4)
        result[:3, :3] = self
        return result

    def lerp(self, other, factor):
        def blend(left, right):
            return left + (right - left) * factor

        rotations = Rotation.concatenate([self.to_quaternion().value, other.to_quaternion().value])
        rotation = Quaternion(Slerp([0, 1], rotations)(factor))
        return self.LocRotScale(blend(self.translation, other.translation), rotation,
                                blend(self.to_scale(), other.to_scale()))


class Quaternion:
    def __init__(self, axis, angle=None):
        self.value = axis if angle is None else Rotation.from_rotvec(np.asarray(axis) * angle)

    def __matmul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.value * other.value)
        return self.value.apply(other)

    def inverted(self):
        return Quaternion(self.value.inv())

    def to_matrix(self):
        return Matrix(self.as_matrix())

    def as_matrix(self):
        return self.value.as_matrix()


class Slots(list):
    def new(self, **spec):
        self.append(spec)
        return spec


class Action(dict):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.slots = Slots()
        self.keys = {}
        self.point = NS(interpolation="BEZIER")
        self.layers = [NS(strips=[NS(channelbags=[NS(fcurves=[NS(keyframe_points=[self.point])])])])]

    def __bool__(self):
        return True


class Actions(dict):
    def new(self, name):
        result = Action(name)
        self[name] = result
        return result


class AnimatedBone(PoseBone):
    def __init__(self, name, animation):
        self.neutral = Matrix.Identity(4)
        self.animation = animation
        self.bone = NS(matrix_local=Matrix.Identity(4))
        super().__init__(name)

    @property
    def matrix_basis(self):
        return self.basis

    @matrix_basis.setter
    def matrix_basis(self, transform):
        self.basis = Matrix(transform)
        self.matrix = self.neutral @ self.basis

    @property
    def location(self):
        return self.basis[:3, 3]

    @location.setter
    def location(self, translation):
        self.basis[:3, 3] = translation
        self.matrix = self.neutral @ self.basis

    def keyframe_insert(self, channel, *, frame, group):
        assert group == self.name
        entries = self.animation.action.keys.setdefault(frame, {})
        channels = entries.setdefault(self.name, NS(channels=set())).channels
        channels.add(channel)
        entries[self.name] = NS(channels=channels, basis=self.matrix_basis.copy(), matrix=self.matrix.copy())


@pytest.fixture
def animation_scene(capture_scene, monkeypatch):
    bpy, rig, source, _, updates = capture_scene
    animation = NS(action=None, action_slot=None, nla_tracks=[])
    rig.animation_data_create = lambda: animation
    rig.animation_data = animation
    old = Action("old")
    animation.action = old
    bpy.data.actions = Actions(old=old)
    bones = rig.pose.bones = BoneMap({bone.name: AnimatedBone(bone.name, animation) for bone in rig.pose.bones})
    for limb, name in pipeline.FEET.items():
        # Preserve four distinct stance origins through capture, gait, and transition.
        bones[name].neutral[:3, 3] = (-1 if limb[1] == "L" else 1, -1 if limb[0] == "F" else 1, 0)
        bones[name].matrix_basis = Matrix.Identity(4)
    bones["torso"].location = Vector((0, 0, -0.25))
    bones["head"].rotation_mode = "XYZ"

    def frame_set(frame):
        scene.frame_current = frame
        for name, entry in animation.action.keys.get(frame, {}).items():
            bones[name].matrix_basis = entry.basis
            bones[name].matrix = entry.matrix

    scene = bpy.context.scene
    scene.frame_set = frame_set
    scene.render = NS(fps=0)
    bpy.context.evaluated_depsgraph_get = lambda: None
    # Observe the transform keyed on the torso through one deform child. This
    # exercises the saved-action playback and closure check without simulating IK.
    rig.evaluated_get = lambda _: NS(pose=NS(bones=[NS(name="DEF-spine", matrix=bones["torso"].matrix)]))
    bpy.ops = NS(wm=NS(save_as_mainfile=lambda *, filepath: Path(filepath).write_bytes(b"scene data")))
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector, Matrix=Matrix, Quaternion=Quaternion))
    return bpy, rig, source, updates, old


@pytest.mark.parametrize("transition", [False, True])
def test_pipeline_authors_and_saves_reference_and_closed_actions(
        animation_scene, tmp_path, monkeypatch, transition):
    bpy, rig, source, _, old = animation_scene
    pose_path = tmp_path / "pose.json"
    assert pipeline.main(["capture", "--rig", rig.name, "--out", str(pose_path)]) == 0
    config = {
        "frames": 8, "fps": 20, "gait": "walk",
        "transition_hold": 2, "transition_blend": 3, "transition_cycles": 1,
    }
    settings = tmp_path / "gait.json"
    settings.write_text(json.dumps(config), encoding="utf-8")
    target = tmp_path / "different" / "gait.blend"
    args = ["animate", "--rig", rig.name, "--pose", str(pose_path), "--settings", str(settings), "--out", str(target)]
    if transition:
        args.append("--transition")
    monkeypatch.setattr(sys, "argv", [pipeline.__file__, "--", *args])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(pipeline.__file__, run_name="__main__")
    assert caught.value.code == 0
    assert source.read_bytes() == b"source"
    assert target.read_bytes() == b"scene data"
    report = json.loads(target.with_suffix(".blend.report.json").read_text())
    assert report["status"] == "ok"
    assert report["settings"]["stride"] == 0.35
    assert report["action"] == "QuadrupedGait"
    assert report["loop_error"] == pytest.approx(0, abs=1e-8)
    assert report["pose_sha256"] == pipeline.digest(json.loads(pose_path.read_text()))
    assert report["visual_review_required"] is True
    assert report["nominal_speed_rig_units_per_second"] == pytest.approx(2 * 0.35 * 20 / (0.65 * 8))
    assert bpy.context.scene.render.fps == 20
    assert old.use_fake_user

    ref = bpy.data.actions["QuadrupedGait_Reference"]
    gait = bpy.data.actions["QuadrupedGait"]
    assert list(ref.keys) == [1]
    assert sorted(gait.keys) == list(range(1, 10))
    assert gait.point.interpolation == "LINEAR"
    assert gait.use_fake_user and ref.use_fake_user
    assert ref.keys[1]["torso"].basis.translation == pytest.approx([0, 0, -0.25])
    assert gait.keys[1]["head"].channels == {"location", "scale", "rotation_euler", '["IK_FK"]'}
    assert "DEF-spine" not in gait.keys[1]
    for name in pipeline.FEET.values():
        foot_path = [gait.keys[f][name].matrix.translation for f in range(1, 10)]
        assert foot_path[0] == pytest.approx(foot_path[-1])
        assert max(point[2] for point in foot_path) > 0
        assert np.ptp([point[1] for point in foot_path]) > 0.25
        assert gait.keys[1][name].channels == {"location", "scale", "rotation_quaternion", '["IK_FK"]'}

    if transition:
        combined = bpy.data.actions["QuadrupedGait_Transition"]
        total = pipeline.transition_length(pipeline.settings(config))
        assert combined.point.interpolation == "LINEAR"
        assert combined.keys[1]["torso"].basis.translation == pytest.approx((0, 0, 0))
        assert combined.keys[total]["torso"].basis.translation == pytest.approx((0, 0, 0))
        assert min(row["torso"].basis.translation[2] for row in combined.keys.values()) < -0.2
        assert bpy.context.scene.frame_end == total
    else:
        assert "QuadrupedGait_Transition" not in bpy.data.actions
        assert bpy.context.scene.frame_end == 8
    assert bpy.context.scene.frame_current == 1


def test_pipeline_action_rejects_duplicate_without_replacing_active_action(animation_scene):
    bpy, rig, _, _, old = animation_scene
    with pytest.raises(ValueError, match="Action 'old' already exists"):
        pipeline.new_action(bpy, rig, "old")
    assert list(bpy.data.actions) == ["old"]
    assert rig.animation_data.action is old
    assert not hasattr(old, "use_fake_user")


def test_pipeline_select_leaves_pose_mode_and_activates_visible_mesh():
    scene = []
    target = NS(hide_set=lambda hidden: scene.append(("hidden", hidden)),
                select_set=lambda selected: scene.append(("target selected", selected)), hide_select=True)
    bpy = NS(context=NS(object=NS(mode="POSE"), view_layer=NS(objects=NS(active=None))),
             ops=NS(object=NS(mode_set=Mock(side_effect=lambda **kw: scene.append(("mode", kw["mode"]))),
                             select_all=Mock(side_effect=lambda **kw: scene.append(("select all", kw["action"]))))))
    pipeline.select(bpy, target)
    assert scene == [("mode", "OBJECT"), ("select all", "DESELECT"), ("hidden", False), ("target selected", True)]
    assert bpy.context.view_layer.objects.active is target
    assert target.hide_select is False
