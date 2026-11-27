"""Wind has to have something rigid to bend around.

The Blender code is a generated string, so this runs it against a small fake bpy,
mathutils and KDTree. No Blender is started.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
VERTICES = 4


class EmptyAnchorTree(Exception):
    """The generated code went on to build an anchor tree with no points in it."""


class FakeKDTree:
    def __init__(self, size):
        if size == 0:
            raise EmptyAnchorTree("built an empty anchor tree instead of refusing")

    def insert(self, point, index):
        pass

    def balance(self):
        pass

    def find(self, point):
        return (None, 0, 0.0)


class Colours:
    """Every vertex the same green: foliage, so every vertex moves."""

    domain = "POINT"

    def __init__(self):
        self.data = self

    def __len__(self):
        return VERTICES

    def foreach_get(self, name, buf):
        buf[:] = np.tile([0.1, 0.9, 0.1, 1.0], VERTICES)


class Vertices:
    def __len__(self):
        return VERTICES

    def foreach_get(self, name, buf):
        buf[:] = np.arange(VERTICES * 3, dtype=np.float32)


class FakeObject:
    type = "MESH"

    def __init__(self):
        self.data = types.SimpleNamespace(
            color_attributes=[Colours()], vertices=Vertices(), loops=[]
        )
        self.name = ""

    def select_set(self, _flag):
        pass


class Objects(list):
    def remove(self, *args, **kwargs):
        pass


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "wind_demo_under_test", SCRIPTS / "blender_wind_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_a_label_mask_with_nothing_rigid_is_refused(monkeypatch, tmp_path):
    """Old bug: when every labelled vertex is foliage or flower, the code built an empty
    anchor tree and carried on. It must stop with an error that says why."""
    module = _load_module()
    env = next(iter(module.ENVIRONMENTS))
    code = module.blender_code(
        tmp_path / "m.glb",
        tmp_path / "l.glb",
        tmp_path / "frames",
        "label",
        env,
        64,
        2,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
    )
    all_objects = Objects()
    scene = types.SimpleNamespace(
        objects=Objects(),
        collection=types.SimpleNamespace(children=[]),
        world=object(),
        name="scene",
    )

    def gltf(filepath):
        obj = FakeObject()
        all_objects.append(obj)
        scene.objects.append(obj)

    bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            objects=all_objects,
            scenes=types.SimpleNamespace(get=lambda n: scene, new=lambda n: scene),
            collections=types.SimpleNamespace(remove=lambda c: None),
        ),
        context=types.SimpleNamespace(
            window=types.SimpleNamespace(scene=scene),
            view_layer=types.SimpleNamespace(
                objects=types.SimpleNamespace(active=None)
            ),
        ),
        ops=types.SimpleNamespace(import_scene=types.SimpleNamespace(gltf=gltf)),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=tuple))
    monkeypatch.setitem(
        sys.modules, "mathutils.kdtree", types.SimpleNamespace(KDTree=FakeKDTree)
    )

    with pytest.raises(Exception) as caught:
        exec(compile(code, "<wind>", "exec"), {"__name__": "wind_code"})
    assert not isinstance(caught.value, EmptyAnchorTree), str(caught.value)
    assert "rigid" in str(caught.value).lower()
