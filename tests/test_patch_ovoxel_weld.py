"""Tests for the weld-before-simplify patch.

Runs against the REAL vendored postprocess.py where it is present, because a patch tested
only on a synthetic string proves nothing about the file that ships.
"""

from pathlib import Path

import pytest

from scripts.patch_ovoxel_weld_before_simplify import (
    DEFAULT_TARGET,
    MARKER,
    already_patched,
    apply_patch,
)

REAL = Path(__file__).resolve().parents[1] / DEFAULT_TARGET

SYNTHETIC = '''import numpy as np
import torch


def to_glb(vertices, faces, decimation_target=1000000):
    mesh = _MeshBackend()
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if True:
        mesh.simplify(decimation_target * 3, verbose=False)
        mesh.repair_non_manifold_edges()
        mesh.simplify(decimation_target, verbose=False)
    return mesh
'''


def _call_lines(text):
    """Count call sites only -- `def _i2l_weld(mesh):` contains the call as a substring."""
    return sum(1 for line in text.splitlines() if line.strip() == f"{MARKER}(mesh)")


def test_inserts_a_weld_before_every_simplify():
    out = apply_patch(SYNTHETIC, "/repo")
    assert _call_lines(out) == SYNTHETIC.count("mesh.simplify(")


def test_weld_precedes_the_simplify_it_guards():
    for line in apply_patch(SYNTHETIC, "/repo").splitlines():
        if "mesh.simplify(" in line:
            assert prev.strip() == f"{MARKER}(mesh)"
            assert len(prev) - len(prev.lstrip()) == len(line) - len(line.lstrip()), \
                "indentation must match or the call lands in the wrong block"
        prev = line


def test_helper_is_defined_before_first_use():
    out = apply_patch(SYNTHETIC, "/repo")
    assert out.index(f"def {MARKER}(") < out.index(f"{MARKER}(mesh)\n")


def test_repo_path_is_baked_in():
    assert "'/some/repo'" in apply_patch(SYNTHETIC, "/some/repo")


def test_patch_is_idempotent():
    once = apply_patch(SYNTHETIC, "/repo")
    assert apply_patch(once, "/repo") == once
    assert already_patched(once)


def test_unpatched_source_is_detected():
    assert not already_patched(SYNTHETIC)


def test_missing_anchor_raises_rather_than_silently_doing_nothing():
    with pytest.raises(RuntimeError, match="anchor not found"):
        apply_patch("def to_glb():\n    return None\n", "/repo")


@pytest.mark.skipif(not REAL.is_file(), reason="vendored o_voxel not bootstrapped here")
def test_against_the_real_vendored_file():
    """The anchor must exist in the file that actually ships."""
    source = REAL.read_text()
    out = apply_patch(source, "/repo")
    assert _call_lines(out) >= 2, "expected at least two simplify calls"
    assert apply_patch(out, "/repo") == out
    compile(out, "postprocess.py", "exec")


@pytest.mark.skipif(not REAL.is_file(), reason="vendored o_voxel not bootstrapped here")
def test_the_defect_this_patch_targets_is_still_present():
    """If upstream reorders these, the patch is obsolete and we want to know."""
    source = REAL.read_text()
    repair = source.index("mesh.repair_non_manifold_edges()")
    following_simplify = source.index("mesh.simplify(", repair)
    assert following_simplify > repair, "repair_non_manifold_edges precedes a simplify"
