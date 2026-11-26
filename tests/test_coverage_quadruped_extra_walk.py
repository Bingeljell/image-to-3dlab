"""Author gait keys in a scene adapter without running Blender."""

import json
import runpy
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import blender_quadruped_walk as walk
import numpy as np
import pytest

from tests.test_coverage_blender_quadruped import (
    BoneMap,
    IdentityTransform,
    Vector,
    rest_bone,
)


class SceneVector(Vector):
    @Vector.z.setter
    def z(self, value):
        self[2] = value


class KeyedBone(dict):
    def __init__(self, name, origin=(0, 0, 0)):
        super().__init__()
        self.name = name
        self.bone = NS(matrix_local=IdentityTransform())
        self.head = self.tail = SceneVector(origin)
        self.matrix = Translation(origin)
        self.rotation_euler = (1, 2, 3)
        self.rotation_quaternion = (0, 1, 0, 0)
        self.rotation_mode = "QUATERNION"
        self.keys = {}

    @property
    def rotation_euler(self):
        return self.euler

    @rotation_euler.setter
    def rotation_euler(self, value):
        self.euler = SceneVector(value).copy()

    def keyframe_insert(self, channel, *, frame):
        if channel == "location" and self.name in walk.FOOT_IK.values():
            value = self.matrix.translation
        else:
            value = getattr(self, channel)
        self.keys[frame, channel] = tuple(value)


class Translation:
    def __init__(self, point):
        self.translation = SceneVector(point)

    def copy(self):
        return Translation(self.translation.copy())


@pytest.fixture
def walking_scene(monkeypatch, tmp_path):
    names = (set(walk.FOOT_IK.values()) | set(walk.BODY_CHAIN)
             | set(walk.SHOULDER.values()) | set(walk.TAIL_CHAIN)
             | set(walk.FRONT_FK["FL"]) | {"ORG-front_thigh.L", "DEF-body", "MCH-helper", walk.LEG_PARENTS[0]})
    controls = BoneMap({name: KeyedBone(name) for name in sorted(names)})
    controls["head"].head = SceneVector((0, -1, 2))
    controls[walk.LEG_PARENTS[0]]["IK_FK"] = 0.7512
    rest = BoneMap({"DEF-body": rest_bone("DEF-body")})
    for chain in (walk.FRONT_CHAIN, walk.HIND_CHAIN):
        for step, name in enumerate(chain):
            rest["ORG-" + name] = rest_bone("ORG-" + name, (0, 0, 3 - step), (0, -0.25, 2 - step))
    rig = NS(name="test rig", type="ARMATURE", pose=NS(bones=controls), data=NS(bones=rest),
             matrix_world=IdentityTransform(), animation_data=None)

    def create_animation():
        rig.animation_data = NS()
        return rig.animation_data

    rig.animation_data_create = create_animation
    old = NS(name="QuadWalk")

    class Actions(dict):
        def new(self, name):
            action = NS(name=name, slots=NS(new=lambda **kwargs: kwargs))
            self[name] = action
            return action

        def remove(self, action):
            del self[action.name]

    actions = Actions(QuadWalk=old)
    mesh = NS(name="skin", type="MESH", matrix_world=IdentityTransform(), vertex_groups=[], data=NS(
        vertices=[NS(co=SceneVector((0, 0, 0))), NS(co=SceneVector((0, 0, 4)))],
    ))
    scene = NS(frame_set=Mock())

    def update():
        # Distinct local axes and signs. The probes must infer them from moved tips.
        thigh = controls[walk.FRONT_FK["FL"][0]].rotation_euler
        controls[walk.FRONT_FK["FL"][0].replace("thigh", "foot")].tail = SceneVector((0, -thigh[1], 0))
        shoulder = controls[walk.SHOULDER["FL"]].rotation_euler
        controls["ORG-front_thigh.L"].tail = SceneVector((0, shoulder[2], 0))
        for bone, tip in (("head", "head"), (walk.TAIL_CHAIN[0], walk.TAIL_CHAIN[-1])):
            if bone in controls:
                x, y, _ = controls[bone].rotation_euler
                controls[tip].tail = SceneVector((x, 0, y))

    bpy = NS(
        data=NS(objects=[rig, mesh], actions=actions, filepath=str(tmp_path / "input.blend")),
        context=NS(object=NS(mode="POSE"), scene=scene,
                   view_layer=NS(update=update), evaluated_depsgraph_get=lambda: None),
        ops=NS(object=NS(mode_set=Mock()), wm=NS(save_as_mainfile=Mock())),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=SceneVector))
    return bpy, rig, controls, old


def test_walk_entry_authors_closed_cycle_without_resetting_internal_bones(
        walking_scene, tmp_path, monkeypatch, capsys):
    bpy, rig, bones, old = walking_scene
    target = tmp_path / "output" / "walk.blend"
    monkeypatch.setattr(sys, "argv", [walk.__file__, "--", "--frames", "12", "--head-yaw", "25",
                                      "--neck-offset", "15,0,0", "--body-drop", "0.1",
                                      "--shoulder-share", "0.45", "--skip-audit", "--out", str(target)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(walk.__file__, run_name="__main__")
    assert caught.value.code == 0
    report = json.loads(capsys.readouterr().out.split("I2L_GAIT_REPORT ")[1])
    assert report["frames"] == [1, 13]
    assert report["front_swing_axis"] == ["Y", 1]
    assert report["shoulder_axis"] == ["Z", -1]
    assert report["head_yaw_split_deg"] == {"neck": -10.75, "head": -14.25}
    assert report["body_drop_applied"] == 0.1
    assert report["body_bob"] > 0
    assert report["ik_fk_switched_from"] == {walk.LEG_PARENTS[0]: 0.751}
    assert bones[walk.LEG_PARENTS[0]]["IK_FK"] == 0
    assert tuple(bones["MCH-helper"].rotation_euler) == (1, 2, 3)
    assert bones["MCH-helper"].keys == {}
    assert rig.animation_data.action is bpy.data.actions["QuadWalk"]
    assert rig.animation_data.action is not old
    assert rig.animation_data.action_slot == {"id_type": "OBJECT", "name": rig.name}
    bpy.ops.object.mode_set.assert_called_once_with(mode="OBJECT")
    bpy.ops.wm.save_as_mainfile.assert_called_once_with(filepath=str(target))

    for name in walk.FOOT_IK.values():
        keys = bones[name].keys
        assert keys[1, "location"] == pytest.approx(keys[13, "location"])
        assert max(keys[frame, "location"][2] for frame in range(1, 14)) > 0
        assert all(keys[frame, "rotation_quaternion"] == (1, 0, 0, 0) for frame in range(1, 14))
    for name in (*walk.SHOULDER.values(), *walk.TAIL_CHAIN, "head", "neck"):
        keys = bones[name].keys
        values = np.asarray([keys[frame, "rotation_euler"] for frame in range(1, 14)])
        assert values[0] == pytest.approx(values[-1])
        assert np.max(np.ptp(values, axis=0)) > 0


def test_walk_audit_rejects_motion_that_does_not_reach_deform_bones(walking_scene, capsys):
    bpy, rig, bones, _ = walking_scene
    for name in walk.TAIL_CHAIN:
        del bones[name]
    bpy.context.object = None
    pose = {name: NS(tail=SceneVector()) for name in walk.PAW.values()}
    pose["torso"] = NS(head=SceneVector((0, 0, 3)))
    pose[walk.RAKE_CHAIN[0]] = NS(tail=SceneVector((0, 0, 2)))
    pose[walk.RAKE_CHAIN[1]] = NS(tail=SceneVector((0, 0, 1)))
    rig.evaluated_get = lambda _: NS(matrix_world=rig.matrix_world, pose=NS(bones=pose))
    with pytest.raises(RuntimeError, match="authored cycle failed its audit"):
        walk.main(["--frames", "12", "--name", "NoMotion", "--head-yaw", "auto"])
    report = json.loads(capsys.readouterr().out.split("I2L_GAIT_REPORT ")[1])
    assert report["audit"]["limbs"]["FL"]["stride_pct_leg"] == 0
    assert report["audit"]["limbs"]["FR"]["planted_fraction"] == 1
    assert report["audit"]["problems"]
    assert report["tail_bones"] == []
    bpy.ops.wm.save_as_mainfile.assert_not_called()


def test_walk_preserves_yaw_when_pitch_uses_the_same_axis_and_supports_legacy_actions(
        walking_scene, capsys):
    bpy, rig, bones, _ = walking_scene
    update = bpy.context.view_layer.update

    def update_coupled_axes():
        update()
        for bone, tip in (("head", "head"), (walk.TAIL_CHAIN[0], walk.TAIL_CHAIN[-1])):
            x = bones[bone].rotation_euler.x
            bones[tip].tail = SceneVector((x, 0, x))

    bpy.context.view_layer.update = update_coupled_axes
    legacy = NS(name="Legacy")
    bpy.data.actions.new = lambda _: legacy
    assert walk.main(["--frames", "12", "--name", "Legacy", "--head-yaw", "25", "--skip-audit"]) == 0
    assert rig.animation_data.action is legacy
    assert not hasattr(rig.animation_data, "action_slot")
    report = json.loads(capsys.readouterr().out.split("I2L_GAIT_REPORT ")[1])
    assert report["head_yaw_split_deg"]["head"] == -14.25
    for name in ("head", *walk.TAIL_CHAIN):
        poses = np.asarray([bones[name].keys[frame, "rotation_euler"] for frame in range(1, 14)])
        assert np.ptp(poses[:, 0]) > 0
        assert poses[:, 1:] == pytest.approx(np.zeros((13, 2)))
