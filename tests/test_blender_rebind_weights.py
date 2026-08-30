from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "blender_rebind_weights.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_rebind_weights", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Mesh:
    name = "Creature"

    def __init__(self, dimensions):
        self.dimensions = dimensions


def test_voxel_size_is_scale_relative():
    assert load_module().derived_voxel_size(Mesh((2, 4, 3))) == pytest.approx(0.048)


@pytest.mark.parametrize("fraction", [0, 0.0009, 0.051, 1])
def test_voxel_fraction_has_safe_bounds(fraction):
    with pytest.raises(ValueError):
        load_module().derived_voxel_size(Mesh((1, 1, 1)), fraction)


def test_weight_transfer_is_fail_closed_and_uses_a_disposable_proxy():
    source = MODULE.read_text()

    assert "ARMATURE_AUTO" in source
    assert 'data_types_verts = {"VGROUP_WEIGHTS"}' in source
    assert "POLYINTERP_NEAREST" in source
    assert "produced no non-empty weight groups" in source
    assert "bpy.data.objects.remove(proxy, do_unlink=True)" in source
