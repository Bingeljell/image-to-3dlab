"""Test the exact weld used by the o_voxel patch."""

import sys
from types import ModuleType

import numpy as np
import pytest

from scripts import patch_ovoxel_weld_before_simplify as patch


class ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def contiguous(self):
        return self


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = ArrayTensor(vertices)
        self.faces = ArrayTensor(faces)
        self.replacements = []

    def read(self):
        return self.vertices, self.faces

    def init(self, vertices, faces):
        self.replacements.append((vertices.values, faces.values))


@pytest.fixture
def tensor_api(monkeypatch):
    module = ModuleType("torch")
    module.float32 = np.float32
    module.int32 = np.int32
    module.as_tensor = lambda values, dtype: ArrayTensor(np.asarray(values, dtype=dtype))
    monkeypatch.setitem(sys.modules, "torch", module)


def test_weld_remaps_faces_and_drops_collapsed_triangles(tensor_api, capsys):
    mesh = Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[3, 1, 2], [0, 3, 1]],
    )
    patch.weld_before_simplify(mesh)
    (vertices, faces), = mesh.replacements
    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert len(vertices) == 3
    assert faces.shape == (1, 3)
    np.testing.assert_array_equal(vertices[faces[0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert "welded 1 duplicate vertices (2 -> 1 faces)" in capsys.readouterr().out


def test_weld_keeps_distinct_nearby_vertices(tensor_api):
    mesh = Mesh([[0, 0, 0], [1e-5, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    patch.weld_before_simplify(mesh)
    assert mesh.replacements == []


def test_weld_patch_routes_through_shared_implementation_and_reverts(
    tmp_path, monkeypatch, tensor_api, capsys
):
    import runpy
    from pathlib import Path

    target = tmp_path / "postprocess.py"
    original = "import math\n\ndef run(mesh):\n    mesh.simplify(20)\n"
    backup = target.with_suffix(".py.i2l-orig")
    repo = Path(patch.__file__).resolve().parents[1]

    def run_patch(*flags):
        monkeypatch.setattr(
            sys, "argv", [patch.__file__, "--target", str(target), "--repo", str(repo), *flags]
        )
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(patch.__file__, run_name="__main__")
        return exc.value.code

    assert "no backup" in run_patch("--revert")
    assert "target not found" in run_patch()
    target.write_text(original)
    assert run_patch() == 0
    assert backup.read_text() == original
    patched = target.read_text()
    assert run_patch() == 0
    assert "already patched" in capsys.readouterr().out
    assert target.read_text() == patched

    mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], [[3, 1, 2]])
    simplifications = []

    def simplify(limit):
        assert len(mesh.replacements) == 1
        simplifications.append(limit)

    mesh.simplify = simplify
    runpy.run_path(str(target))["run"](mesh)
    assert simplifications == [20]
    assert run_patch("--revert") == 0
    assert target.read_text() == original
    assert run_patch() == 0
    assert backup.read_text() == original
