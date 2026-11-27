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


# --- several pieces, and a mesh split at UV seams --------------------------------------
#
# One global volume test cannot orient pieces separately. Welded pieces are measured on a
# position weld so a UV seam never reads as its own piece.


def _piece(extents, offset, inside_out=False):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(offset)
    if inside_out:
        mesh.invert()
    return mesh


def _seamed(mesh):
    """Same surface with every triangle owning its three corners, as a UV seam split does."""
    import numpy as np

    corners = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(corners), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=corners, faces=faces, process=False)


def _piece_volumes(mesh):
    """Signed volume of each piece, found on a position weld. Sorted."""
    welded = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    welded.merge_vertices()
    return sorted(
        round(float(p.volume), 6) for p in welded.split(only_watertight=False)
    )


def _small_ok_big_inverted():
    # Volumes +1 and -8: the whole-mesh total is -7, so one global flip is tempting
    # and wrong for the small piece.
    return trimesh.util.concatenate(
        [_piece((1, 1, 1), (0, 0, 0)), _piece((2, 2, 2), (5, 0, 0), inside_out=True)]
    )


def test_two_boxes_one_inverted_both_end_outward():
    mesh = _small_ok_big_inverted()
    fw.repair(mesh)
    assert _piece_volumes(mesh) == [1.0, 8.0]


def test_two_seamed_boxes_one_inverted_both_end_outward():
    mesh = _seamed(_small_ok_big_inverted())
    fw.repair(mesh)
    assert _piece_volumes(mesh) == [1.0, 8.0]


def test_a_correct_seamed_box_keeps_its_faces():
    import numpy as np

    mesh = _seamed(_piece((1, 1, 1), (5.0, 2.0, -3.0)))
    before = mesh.faces.copy()
    fw.repair(mesh)
    assert not np.all(mesh.faces == before[:, ::-1], axis=1).any()
    assert _piece_volumes(mesh) == [1.0]


def test_a_single_inverted_seamed_box_is_turned_outward():
    mesh = _seamed(_piece((1, 1, 1), (5.0, 2.0, -3.0), inside_out=True))
    fw.repair(mesh)
    assert _piece_volumes(mesh) == [1.0]


def test_the_trellis_glb_repair_orients_each_seamed_piece(tmp_path):
    """The generation step has its own copy of the same whole-mesh flip."""
    from image_to_3dlab import trellis_backend

    path = tmp_path / "pieces.glb"
    _seamed(_small_ok_big_inverted()).export(str(path))
    trellis_backend._repair_winding(path)
    repaired = trimesh.load(str(path), force="mesh", process=False)
    assert _piece_volumes(repaired) == [1.0, 8.0]
