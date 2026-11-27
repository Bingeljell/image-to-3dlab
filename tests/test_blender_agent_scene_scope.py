"""The live agent must not wipe the user's open Blender file.

The generated bpy snippets are executed here against a small fake bpy that models two
scenes. Real Blender is not needed: what matters is which objects the snippet removes.
`bpy.data.objects` spans every scene, so removing through it reaches objects the agent
was never working on.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_agent():
    spec = importlib.util.spec_from_file_location("blender_agent_scope", SCRIPTS / "blender_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent = _load_agent()


class FakeObject:
    def __init__(self, name, kind="MESH"):
        self.name = name
        self.type = kind


class FakeScene:
    def __init__(self, name):
        self.name = name
        self.objects = []


class FakeObjects:
    """bpy.data.objects: every object in the file, whichever scene holds it."""

    def __init__(self, world):
        self.world = world

    def _all(self):
        return [o for s in self.world.scenes for o in s.objects]

    def __iter__(self):
        return iter(dict.fromkeys(self._all()))

    def keys(self):
        return [o.name for o in self]

    def get(self, name, default=None):
        return next((o for o in self if o.name == name), default)

    def __contains__(self, name):
        return self.get(name) is not None

    def __getitem__(self, name):
        obj = self.get(name)
        if obj is None:
            raise KeyError(name)
        return obj

    def remove(self, obj, do_unlink=True):
        for scene in self.world.scenes:
            if obj in scene.objects:
                scene.objects.remove(obj)


class World:
    def __init__(self):
        self.a = FakeScene("A")
        self.b = FakeScene("B")
        self.scenes = [self.a, self.b]
        self.data = types.SimpleNamespace(objects=FakeObjects(self))
        self.context = types.SimpleNamespace(scene=self.a, view_layer=types.SimpleNamespace(update=lambda: None))
        gltf = lambda filepath: self.a.objects.append(FakeObject("imported_mesh"))
        self.ops = types.SimpleNamespace(import_scene=types.SimpleNamespace(gltf=gltf))

    def module(self):
        mod = types.ModuleType("bpy")
        mod.data, mod.context, mod.ops = self.data, self.context, self.ops
        return mod


@pytest.fixture
def world(monkeypatch):
    w = World()
    monkeypatch.setitem(sys.modules, "bpy", w.module())
    monkeypatch.setitem(sys.modules, "mathutils", types.ModuleType("mathutils"))
    return w


def _run(code: str) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(code, "<agent>", "exec"), {})
    return out.getvalue()


def _names(scene):
    return sorted(o.name for o in scene.objects)


def test_import_default_keeps_the_users_existing_meshes(world):
    """An omitted clear_scene used to delete every mesh in the open file."""
    world.a.objects.append(FakeObject("users_mesh"))
    _run(agent.build_call("import_asset", {"path": "/tmp/a.glb"}))
    assert "users_mesh" in _names(world.a), "default import deleted the user's mesh"
    assert "imported_mesh" in _names(world.a)


def test_asked_for_import_clear_leaves_other_scenes_alone(world):
    world.a.objects.append(FakeObject("a_mesh"))
    world.b.objects.append(FakeObject("b_mesh"))
    _run(agent.build_call("import_asset", {"path": "/tmp/a.glb", "clear_scene": True}))
    assert _names(world.b) == ["b_mesh"], "import clear removed an object from scene B"
    assert _names(world.a) == ["imported_mesh"]


def test_clear_scene_leaves_other_scenes_alone(world):
    world.a.objects.append(FakeObject("a_mesh"))
    world.b.objects.append(FakeObject("b_mesh"))
    _run(agent.build_call("clear_scene", {}))
    assert _names(world.b) == ["b_mesh"], "clear_scene removed an object from scene B"
    assert _names(world.a) == []


def test_delete_object_cannot_reach_another_scene(world):
    world.b.objects.append(FakeObject("b_mesh"))
    _run(agent.build_call("delete_object", {"name": "b_mesh"}))
    assert _names(world.b) == ["b_mesh"], "delete_object removed an object from scene B"
