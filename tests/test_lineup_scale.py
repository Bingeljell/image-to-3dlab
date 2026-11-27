"""A multi-part asset in the line-up is scaled once, from its combined height.

The script imports bpy at module level, so these tests install a tiny fake bpy and run
the real import_asset against it. No Blender is started.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_lineup_sweep.py"


class Vec:
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz


class Loc:
    """A location that reads like a Blender vector: .x/.y/.z and indexing."""

    def __init__(self):
        self.x = self.y = self.z = 0.0

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]


class Matrix:
    def __init__(self, obj):
        self.obj = obj

    def __matmul__(self, vec):
        o = self.obj
        return Vec(
            (
                vec.x * o.scale[0] + o.location[0],
                vec.y * o.scale[1] + o.location[1],
                vec.z * o.scale[2] + o.location[2],
            )
        )


class FakeMesh:
    """Unit-scale mesh sitting at `z0..z0+height`."""

    type = "MESH"

    def __init__(self, name, z0, height):
        self.name = name
        self.scale = [1.0, 1.0, 1.0]
        self.location = Loc()
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.dimensions = types.SimpleNamespace(z=height)
        self.bound_box = [
            (x, y, z) for x in (0, 1) for y in (0, 1) for z in (z0, z0 + height)
        ]

    @property
    def matrix_world(self):
        return Matrix(self)


@pytest.fixture
def fake_blender(monkeypatch):
    objects: set = set()
    parts = [FakeMesh("body", 0.0, 2.0), FakeMesh("eye", 1.8, 0.2)]
    bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=objects),
        context=types.SimpleNamespace(
            view_layer=types.SimpleNamespace(update=lambda: None)
        ),
        ops=types.SimpleNamespace(
            import_scene=types.SimpleNamespace(
                gltf=lambda filepath: objects.update(parts)
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=Vec))
    spec = importlib.util.spec_from_file_location("lineup_sweep_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, parts


def test_parts_of_one_asset_share_one_scale(fake_blender):
    """Old bug: each part was scaled by its own height, so a 0.2 tall eye on a 2.0 tall
    body grew to the same height as the body. One scale keeps them in proportion."""
    module, parts = fake_blender
    module.import_asset("a.glb", "fox", 0, 1)
    body, eye = parts
    assert body.scale[2] == pytest.approx(0.5)
    assert eye.scale[2] == pytest.approx(body.scale[2])
