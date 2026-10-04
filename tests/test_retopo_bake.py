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
    source, dest, faces, size, angle, voxel, *_surface = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert (source, dest) == ("a.glb", "b.glb")
    assert faces == 20000
    assert size == 2048
    assert angle == pytest.approx(math.radians(89.0))
    assert 0.0005 <= voxel <= 0.05


def test_explicit_arguments_are_honoured():
    _, _, faces, size, angle, voxel, *_surface = retopo.parse_args(
        ["--", "a.glb", "b.glb", "8000", "4096", "60", "0.003"]
    )
    assert faces == 8000
    assert size == 4096
    assert angle == pytest.approx(math.radians(60.0))
    assert voxel == pytest.approx(0.003)


def test_zero_voxel_size_means_skip_the_remesh():
    """0 is an escape hatch, not an invalid size.

    The voxel pass was adopted because the raw mesh looked hopelessly non-manifold, a
    reading that came from measuring it unwelded — 43.7% against 0.63% welded, same file.
    It costs the creases, so going direct has to be expressible.
    """
    _, _, _faces, _size, _angle, voxel, *_surface = retopo.parse_args(
        ["--", "a.glb", "b.glb", "40000", "2048", "89", "0"]
    )
    assert voxel == 0.0


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
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "128"])


@pytest.mark.parametrize("size", [256, 512])
def test_accepts_the_small_atlases_far_lods_use(size):
    """finish_props.py halves the atlas per LOD, so the far ones are 512 and 256."""
    assert retopo.parse_args(["--", "a.glb", "b.glb", "2000", str(size)])[3] == size


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


def test_quadriflow_reduced_detects_a_silent_refusal():
    """QuadriFlow declines with a Blender *warning*, not an exception.

    When its preconditions are unmet the operator leaves the mesh untouched and the script
    carries on. On the Snag that shipped 1.3M triangles from a request for 20,000. An
    unchanged count is the only reliable signal that it did not run.
    """
    assert retopo.quadriflow_reduced(662_328, 20_112, 20_000) is True
    assert retopo.quadriflow_reduced(662_328, 662_328, 20_000) is False


def test_quadriflow_reduced_rejects_a_result_nowhere_near_the_target():
    # Changed, but still vastly larger than asked for: not a retopology.
    assert retopo.quadriflow_reduced(662_328, 400_000, 20_000) is False


def test_quadriflow_reduced_tolerates_approximation():
    # QuadriFlow approximates the target rather than hitting it, so nearby counts pass.
    assert retopo.quadriflow_reduced(600_000, 24_000, 20_000) is True
    assert retopo.quadriflow_reduced(600_000, 59_000, 20_000) is True


def test_surface_knobs_have_neutral_organic_defaults():
    """Only base colour is baked, so these stand in for a metallic-roughness map.

    Defaults are a neutral organic surface rather than Blender's metallic 0 / roughness
    0.5, which reads as dead plastic under any light.
    """
    *_, metallic, roughness, ior = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert (metallic, roughness, ior) == (0.25, 0.65, 1.45)


def test_surface_knobs_are_tunable_per_asset():
    # Wet bark and dry stone want different answers, so these are arguments, not constants.
    *_, metallic, roughness, ior = retopo.parse_args(
        ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.648", "0.686", "1.4"]
    )
    assert (metallic, roughness, ior) == (0.648, 0.686, 1.4)


@pytest.mark.parametrize("bad", [
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "1.5"],
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.5", "-0.2"],
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.5", "0.5", "9.0"],
])
def test_surface_knobs_reject_impossible_values(bad):
    with pytest.raises(SystemExit):
        retopo.parse_args(bad)

def test_the_decimate_ratio_is_computed_against_triangles_not_quads():
    """The bug this fixes doubled every face target this repo has ever set.

    Blender's COLLAPSE decimation applies its ratio to triangles; the voxel remesh before
    it emits quads. Measured on the Pixal3D fox: 200,632 quads = 401,264 triangles, asked
    for 40,000, got 79,991 — almost exactly twice.
    """
    quads = 200632
    triangles = quads * 2

    ratio = retopo.decimate_ratio(40000, triangles)
    assert triangles * ratio == pytest.approx(40000, rel=1e-6)

    # The old calculation, kept here as the thing that must not come back.
    wrong = min(1.0, 40000 / quads)
    assert triangles * wrong == pytest.approx(80000, rel=1e-6)


def test_the_ratio_never_exceeds_one():
    """Asking for more faces than exist must not inflate the mesh."""
    assert retopo.decimate_ratio(100000, 5000) == 1.0


def test_a_degenerate_mesh_does_not_divide_by_zero():
    assert retopo.decimate_ratio(40000, 0) == 1.0
