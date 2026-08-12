"""Tests for outward-facing winding repair.

The subtle one is `needs_inverting`: `fix_normals()` makes winding CONSISTENT but can leave
a mesh uniformly inside-out, which is precisely what happened on Flicker (volume -0.02369
after repair). Consistency alone is not the goal; outward is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fix_winding.py"


def _load():
    spec = importlib.util.spec_from_file_location("fix_winding", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fw = _load()
trimesh = pytest.importorskip("trimesh")


def test_negative_volume_needs_inverting():
    assert fw.needs_inverting(-0.02369)


def test_positive_volume_does_not():
    assert not fw.needs_inverting(0.00199)


def test_zero_volume_is_left_alone():
    """An open sheet has ~zero signed volume; flipping it would be a coin toss."""
    assert not fw.needs_inverting(0.0)


# --- against a real mesh --------------------------------------------------------------


def _box(inside_out: bool):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    if inside_out:
        mesh.invert()
    return mesh


def test_repairs_an_inside_out_mesh():
    """Assert the OUTCOME, not the mechanism.

    On a clean watertight box `fix_normals()` already resolves orientation, so the
    explicit invert never fires. On a real generated mesh it does -- Flicker was still at
    volume -0.02369 after `fix_normals()`. Either route is fine; ending up outward-facing
    is the contract.
    """
    mesh = _box(inside_out=True)
    assert mesh.volume < 0
    fw.repair(mesh)
    assert mesh.volume > 0
    assert mesh.is_winding_consistent


def test_leaves_a_correct_mesh_alone():
    mesh = _box(inside_out=False)
    before = mesh.volume
    report = fw.repair(mesh)
    assert not report["inverted"]
    assert mesh.volume == pytest.approx(before)


def test_repair_preserves_geometry():
    """Only winding may change -- vertex positions and face count must not."""
    mesh = _box(inside_out=True)
    vertices_before = mesh.vertices.copy()
    faces_before = len(mesh.faces)
    fw.repair(mesh)
    assert len(mesh.faces) == faces_before
    assert mesh.vertices.shape == vertices_before.shape


def test_report_records_the_before_state():
    mesh = _box(inside_out=True)
    report = fw.repair(mesh)
    assert report["was_volume"] < 0
    assert report["now_volume"] > 0
    assert report["now_consistent"]


def test_invert_fires_when_fix_normals_leaves_it_inside_out():
    """The Flicker case: consistent winding, still negative volume.

    Simulated with a stub because a watertight primitive cannot reproduce it -- trimesh
    resolves those on its own. This is the branch that matters in production.
    """

    class _Stub:
        def __init__(self):
            self.volume = -1.0
            self.is_winding_consistent = False
            self.inverted = False

        def fix_normals(self):
            self.is_winding_consistent = True   # consistent, but still inside-out

        def invert(self):
            self.inverted = True
            self.volume = -self.volume

    stub = _Stub()
    report = fw.repair(stub)
    assert stub.inverted
    assert report["inverted"]
    assert report["now_volume"] > 0
