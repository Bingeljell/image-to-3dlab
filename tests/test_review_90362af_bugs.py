"""Bugs from the old review commit 90362af that are still present on this branch.

Each test fails today for the reason in its docstring and passes once that bug is fixed.
No Blender, no network, no weights: tiny synthetic inputs only.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import arm_clearance as ac
from scripts import clean_skin_weights, project_markings

# 3. clean_skin_weights ---------------------------------------------------------------


class _Objects(list):
    def remove(self, obj, do_unlink=True):
        super().remove(obj)


def _fake_bpy(names):
    class Obj:
        type = "MESH"

        def __init__(self, name):
            self.name = name

        def keys(self):
            return []

        def find_armature(self):
            return None

    store = _Objects(Obj(n) for n in names)
    nothing = types.SimpleNamespace(
        read_factory_settings=lambda **k: None, gltf=lambda **k: None
    )
    bpy = types.ModuleType("bpy")
    bpy.ops = types.SimpleNamespace(
        wm=nothing, import_scene=nothing, export_scene=nothing
    )
    bpy.data = types.SimpleNamespace(objects=store)
    return bpy, store


def test_only_the_exact_helper_name_is_removed(tmp_path, monkeypatch):
    """`startswith("Icosphere")` deletes a real body that is named Icosphere_body."""
    bpy, store = _fake_bpy(["Icosphere", "Icosphere_body", "Body"])
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_skin_weights",
            "--in",
            str(tmp_path / "in.glb"),
            "--out",
            str(tmp_path / "out.glb"),
        ],
    )

    clean_skin_weights.main()

    assert sorted(o.name for o in store) == ["Body", "Icosphere_body"]


# 4. project_markings --mirror --------------------------------------------------------


class _Stop(Exception):
    pass


def test_mirror_reflects_through_the_model_middle_not_the_origin(tmp_path, monkeypatch):
    """A model spanning x 2..4 must mirror to x 2..4 (swapped), not to x -4..-2."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_translation((3.0, 0.0, 0.0))
    uv = np.zeros((len(box.vertices), 2))
    box.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=Image.new("RGB", (4, 4), (128, 128, 128))
    )
    mesh_path, source_path = tmp_path / "m.glb", tmp_path / "s.png"
    box.export(mesh_path)
    Image.new("RGBA", (8, 8), (200, 50, 50, 255)).save(source_path)

    seen = []

    def fake_project(mesh, *args, **kwargs):
        seen.append(np.array(mesh.vertices))
        n = len(mesh.vertices)
        zeros, none = np.zeros(n), None
        if len(seen) == 2:
            raise _Stop
        return none, np.ones(n, bool), zeros, zeros, none

    monkeypatch.setattr(project_markings, "project", fake_project)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_markings",
            str(mesh_path),
            str(source_path),
            str(tmp_path / "o.glb"),
            "--mirror",
        ],
    )
    with pytest.raises(_Stop):
        project_markings.main()

    original, mirrored = seen
    assert mirrored[:, 0].min() == pytest.approx(original[:, 0].min())
    assert mirrored[:, 0].max() == pytest.approx(original[:, 0].max())


# 5. arm_clearance --------------------------------------------------------------------


def test_opposite_directions_make_a_finite_half_turn():
    """c = -1 divides by (1 + c) = 0, so the matrix comes back full of inf/nan."""
    a, b = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
    with np.errstate(
        all="raise"
    ):  # a divide-by-zero is the bug, not just its nan result
        turned = ac._rotation_between(a, b)
    assert np.isfinite(turned).all()
    assert turned @ a == pytest.approx(b)
