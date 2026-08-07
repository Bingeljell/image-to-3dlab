"""Tests for aligning the high-poly bake source to the low-poly target.

The bug these prevent is subtler than it first looks, and I got it wrong once.

In *file* space the two disagree, which suggests a Y/Z swap:
    high-poly PLY   (0.324, 1.000, 0.840)   raw generator output
    hero GLB        (0.318, 0.841, 1.000)   glTF, which is Y-up

But the alignment happens *inside Blender*, and the glTF importer already applies the
Y-up to Z-up conversion on load. By then both are in the same orientation and no
permutation is wanted. "Correcting" the file-space discrepancy rotates them apart: the
per-axis ratio spread went to 1.42, against 1.04 when left alone.

So these tests use **Blender-space** sizes. The lesson is that the comparison must happen
in the space where the fit is applied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_bake_normals.py"

# Sizes as Blender sees them after import — the space the fit actually runs in.
# The low-poly is fractionally narrower in X because it is decimated.
BLENDER_HIGH = (0.324, 1.000, 0.840)
BLENDER_LOW = (0.313, 1.000, 0.841)

# The same two meshes in raw file space, where the GLB is still Y-up. Kept to prove the
# machinery *would* catch a genuine swap, and to document the trap.
FILE_HIGH = (0.324, 1.000, 0.840)
FILE_LOW = (0.318, 0.841, 1.000)


def _load():
    spec = importlib.util.spec_from_file_location("bake_normals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_in_blender_space_no_permutation_is_needed(mod):
    """The real case: the glTF importer has already aligned them, so identity wins."""
    name, spread = mod.best_permutation(BLENDER_HIGH, BLENDER_LOW)
    assert name == "none"
    assert spread < 1.05, "already-aligned meshes should give near-uniform ratios"


def test_file_space_comparison_is_the_trap(mod):
    """Comparing raw file sizes suggests a swap that does not exist inside Blender.

    This is the mistake that produced a misaligned bake. Pinned so the reasoning stays
    visible rather than being rediscovered.
    """
    name, _ = mod.best_permutation(FILE_HIGH, FILE_LOW)
    assert name == "gltf_to_blender", (
        "file-space sizes really do look swapped — which is exactly why the comparison "
        "must be done in Blender space instead"
    )


def test_a_genuine_swap_is_still_detected(mod):
    """The machinery must retain discriminating power for assets that truly are rotated."""
    high = (0.3, 1.0, 0.5)
    low = (0.3, 0.5, 1.0)
    name, spread = mod.best_permutation(high, low)
    assert name == "gltf_to_blender"
    assert spread == pytest.approx(1.0)


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
    ratios = mod.axis_size_ratios(BLENDER_HIGH, BLENDER_LOW, mod.AXIS_PERMUTATIONS["none"])
    assert len(ratios) == 3
    # 5% rather than 3%: decimating 6.2M triangles to 101k pulls the silhouette in
    # slightly, and X is the thinnest axis so it shows there first (0.966 measured).
    # A real orientation error is ~1.4, nowhere near this band.
    for r in ratios:
        assert r == pytest.approx(1.0, abs=0.05)


def test_degenerate_mesh_raises(mod):
    with pytest.raises(ValueError):
        mod.best_permutation((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
