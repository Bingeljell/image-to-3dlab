"""Verify the live importer with an explicit Blender library boundary."""

import json
import math
import sys
from contextlib import contextmanager
from types import SimpleNamespace as NS

import blender_import_character as client
import blender_import_character_worker as worker
import pytest

from tests.test_coverage_blender_agent_worker import Vector as NumericVector


def test_scenekit_to_blender_rotates_axes_without_mirroring():
    assert worker.sk_to_blender(3, 4, 5) == (3, -5, 4)


class Vector(NumericVector):
    @NumericVector.z.setter
    def z(self, value):
        self[2] = value


class Object:
    def __init__(self, name, kind):
        self.name, self.type = name, kind
        self.parent = None
        self.hide_render = False
        self.location = (0, 0, 0)
        self.rotation_euler = (0, 0, 0)
        self.animation_data = None
        self.animation_created = 0
        self.inverted_world = object()
        self.matrix_world = NS(inverted=lambda: self.inverted_world)

    def animation_data_create(self):
        self.animation_created += 1
        self.animation_data = NS(action=None)


class Objects(list):
    def keys(self):
        return [obj.name for obj in self]

    def get(self, name):
        return next((obj for obj in self if obj.name == name), None)

    def new(self, name, data):
        assert data is None
        obj = Object(name, "EMPTY")
        self.append(obj)
        return obj

    def remove(self, obj, *, do_unlink):
        assert do_unlink
        super().remove(obj)


@pytest.fixture
def live_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    armature = Object("Avatar Rig", "ARMATURE")
    mesh = Object("Avatar Surface", "MESH")
    prior = Object("unsaved camera", "CAMERA")
    widget = Object("WGT-paw", "MESH")
    stray = Object("extra light", "LIGHT")
    library_obj = {obj.name: obj for obj in (armature, mesh, widget, stray)}
    library_actions = {"Idle Loop": NS(name="Idle Loop")}
    objects = Objects([prior])
    linked, transfers, frames, saves = [], [], [], []
    actions = {}
    path = "/tmp/café's fixture.blend"

    @contextmanager
    def load(actual, *, link):
        assert actual == path and not link
        source = NS(objects=list(library_obj), actions=list(library_actions))
        destination = NS()
        yield source, destination
        transfers.append((destination.objects[:], destination.actions[:]))
        destination.objects = [library_obj[name] for name in destination.objects]
        objects.extend(obj for obj in destination.objects if obj is not None)
        objects.extend([widget, stray])
        actions.update({name: library_actions[name] for name in destination.actions})

    bpy = NS(
        data=NS(objects=objects, actions=actions, libraries=NS(load=load)),
        context=NS(
            collection=NS(objects=NS(link=linked.append)),
            scene=NS(frame_set=frames.append),
        ),
        ops=NS(wm=NS(save_mainfile=lambda: saves.append(True))),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return NS(
        bpy=bpy,
        armature=armature,
        mesh=mesh,
        prior=prior,
        widget=widget,
        stray=stray,
        library_obj=library_obj,
        library_actions=library_actions,
        objects=objects,
        linked=linked,
        transfers=transfers,
        frames=frames,
        saves=saves,
        args=dict(
            path=path,
            prefix="new party member",
            armature_name="Avatar Rig",
            mesh_name="Avatar Surface",
            position=(2, 5, 7),
            facing=(5, 20, 7),
            action_name="Idle Loop",
            frame=6,
            do_save=True,
        ),
    )


def test_character_bundle_preserves_import_result_and_scene_state(live_import, capsys):
    scene = live_import
    bundle = client.build_code(**scene.args)
    bridge = {"untouched": object()}
    exec(compile(bundle, "<Blender append request>", "exec"), bridge)
    data = json.loads(capsys.readouterr().out.removeprefix("IMPORT_RESULT "))
    assert (
        data
        == bridge["result"]
        == {
            "root": "new party member_root",
            "armature": "new party member_rig",
            "mesh": "new party member_mesh",
            "location": [2, -7, 5],
            "removed_strays": ["extra light"],
            "widgets_hidden": 1,
            "saved": True,
        }
    )
    assert scene.transfers == [(["Avatar Rig", "Avatar Surface"], ["Idle Loop"])]
    root = scene.objects.get(data["root"])
    assert root.rotation_euler == pytest.approx((0, 0, math.pi / 2))
    assert scene.linked == [scene.armature, scene.mesh, root]
    for obj in (scene.armature, scene.mesh):
        assert obj.parent is root and obj.matrix_parent_inverse is root.inverted_world
    assert scene.prior in scene.objects and not scene.prior.hide_render
    assert scene.widget in scene.objects and scene.widget.hide_render
    assert scene.stray not in scene.objects
    assert scene.armature.animation_data.action is scene.library_actions["Idle Loop"]
    assert scene.armature.animation_created == 1
    assert scene.frames == [6] and scene.saves == [True]


@pytest.mark.parametrize("facing", [None, (2, 99, 7)])
def test_character_plain_import_skips_pose_save_and_zero_horizontal_rotation(
    live_import, facing
):
    scene = live_import
    args = {
        **scene.args,
        "facing": facing,
        "action_name": None,
        "frame": None,
        "do_save": False,
    }
    payload = worker.import_character(scene.bpy, **args)
    root = scene.objects.get(payload["root"])
    assert root.rotation_euler == (0, 0, 0)
    assert scene.transfers == [(["Avatar Rig", "Avatar Surface"], [])]
    assert scene.armature.animation_data is None
    assert not scene.frames and not scene.saves and not payload["saved"]


def test_character_reuses_existing_animation_data(live_import):
    scene = live_import
    existing = NS(action=NS(name="old action"))
    scene.armature.animation_data = existing
    worker.import_character(scene.bpy, **scene.args)
    assert scene.armature.animation_data is existing
    assert existing.action is scene.library_actions["Idle Loop"]
    assert scene.armature.animation_created == 0


@pytest.mark.parametrize("missing", ["Avatar Rig", "Avatar Surface", "Idle Loop"])
def test_character_missing_dependencies_reports_source_and_never_saves(
    live_import, missing
):
    scene = live_import
    if missing == "Idle Loop":
        del scene.library_actions[missing]
        message = "action 'Idle Loop' was not found"
    else:
        # Blender can return None for an unavailable requested datablock.
        scene.library_obj[missing] = None
        message = (
            "expected an armature named 'Avatar Rig' and a mesh named 'Avatar Surface'"
        )
    with pytest.raises(RuntimeError, match=message) as error:
        worker.import_character(scene.bpy, **scene.args)
    assert repr(scene.args["path"]) in str(error.value)
    assert not scene.frames and not scene.saves
