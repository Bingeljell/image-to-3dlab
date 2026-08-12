"""Tests for the retopology + texture-transfer step.

The bpy-free parts are the ones that quietly ruin a 10-minute bake: an out-of-range face
target that re-fragments the atlas we are trying to fix, and a ray distance that either
misses the original surface entirely (empty atlas) or reaches across the body and samples
the far side.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_retopo_bake.py"


def _load():
    spec = importlib.util.spec_from_file_location("retopo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retopo = _load()


def test_defaults_are_sensible():
    source, dest, faces, size, angle, voxel = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert (source, dest) == ("a.glb", "b.glb")
    assert faces == 20000
    assert size == 2048
    assert angle == pytest.approx(math.radians(89.0))
    assert 0.0005 <= voxel <= 0.05


def test_explicit_arguments_are_honoured():
    _, _, faces, size, angle, voxel = retopo.parse_args(
        ["--", "a.glb", "b.glb", "8000", "4096", "60", "0.003"]
    )
    assert faces == 8000
    assert size == 4096
    assert angle == pytest.approx(math.radians(60.0))
    assert voxel == pytest.approx(0.003)


def test_rejects_a_voxel_size_coarse_enough_to_melt_the_subject():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "2048", "89", "0.5"])


def test_rejects_a_voxel_size_fine_enough_to_exhaust_memory():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "2048", "89", "0.00001"])


def test_rejects_a_face_target_that_would_refragment_the_atlas():
    """The whole point is fewer, larger UV islands; 500k quads defeats it."""
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "500000"])


def test_rejects_a_face_target_too_low_to_hold_a_silhouette():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "50"])


def test_rejects_a_bad_atlas_size():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "3000"])


def test_missing_arguments_exit_with_usage():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "only.glb"])


# --- ray distance --------------------------------------------------------------------


def test_ray_distance_scales_with_the_asset():
    """A fixed distance would miss entirely on a small asset and cross-sample on a big one."""
    small = retopo.ray_distance((0.1, 0.2, 0.15))
    large = retopo.ray_distance((10.0, 20.0, 15.0))
    assert large == pytest.approx(small * 100)


def test_ray_distance_uses_the_largest_dimension():
    assert retopo.ray_distance((1.0, 5.0, 2.0)) == pytest.approx(retopo.ray_distance((5.0, 5.0, 5.0)))


def test_ray_distance_is_a_small_fraction_not_the_whole_body():
    """Reaching across the body would sample the far side's colour onto the near side."""
    assert retopo.ray_distance((1.0, 1.0, 1.0)) < 0.1
