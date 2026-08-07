"""Tests for aligning the high-poly bake source to the low-poly target.

The bug these prevent: the high-poly PLY comes straight out of the generator while the
low-poly GLB has been through the glTF exporter, so their Y and Z axes are swapped. A
bounding-box fit that only scales and translates cannot repair a rotation — it squashes
one mesh to match the other's dimensions and bakes nonsense.

Measured on the real assets:
    high-poly PLY   size (0.324, 1.000, 0.840)
    hero GLB        size (0.318, 0.841, 1.000)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_bake_normals.py"

# The actual measured sizes, so these tests fail if the real orientation ever changes.
REAL_HIGH = (0.324, 1.000, 0.840)
REAL_LOW = (0.318, 0.841, 1.000)


def _load():
    spec = importlib.util.spec_from_file_location("bake_normals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_real_assets_need_the_gltf_permutation(mod):
    """The regression: these two meshes are 90 degrees apart, and it must be detected."""
    name, spread = mod.best_permutation(REAL_HIGH, REAL_LOW)
    assert name == "gltf_to_blender"
    assert spread < 1.05, "the correct permutation should give near-uniform ratios"


def test_identity_permutation_is_visibly_wrong_for_them(mod):
    """Leaving the axes alone must look bad, or the check has no discriminating power."""
    ratios = mod.axis_size_ratios(REAL_HIGH, REAL_LOW, mod.AXIS_PERMUTATIONS["none"])
    spread = max(ratios) / min(ratios)
    assert spread > 1.3, f"identity should be clearly wrong here, got spread {spread}"


def test_already_aligned_meshes_choose_identity(mod):
    """A high-poly already in the target's space must not be needlessly rotated."""
    size = (0.3, 0.9, 1.0)
    name, spread = mod.best_permutation(size, size)
    assert name == "none"
    assert spread == pytest.approx(1.0)


def test_uniform_scale_difference_is_not_mistaken_for_rotation(mod):
    """A high-poly at half scale is still the same orientation."""
    high = (0.15, 0.45, 0.5)
    low = (0.3, 0.9, 1.0)
    name, spread = mod.best_permutation(high, low)
    assert name == "none"
    assert spread == pytest.approx(1.0)


def test_ratios_are_per_axis_not_uniform(mod):
    """The whole point is per-axis comparison; a single number cannot detect a swap."""
    ratios = mod.axis_size_ratios(REAL_HIGH, REAL_LOW, mod.AXIS_PERMUTATIONS["gltf_to_blender"])
    assert len(ratios) == 3
    for r in ratios:
        assert r == pytest.approx(1.0, abs=0.03)


def test_degenerate_mesh_raises(mod):
    with pytest.raises(ValueError):
        mod.best_permutation((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
