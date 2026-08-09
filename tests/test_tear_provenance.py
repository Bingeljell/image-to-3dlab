"""Tests for tear provenance classification.

Built on synthetic meshes whose answer is known by construction, because the whole
value of this script is the visible/unseen split — and a classifier that silently
inverts its sign would still produce a plausible-looking table. Two of the geometry
diagnoses in this repo were wrong in exactly that way, from a metric nobody checked
against a case with a known answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tear_provenance import classify, loop_geometry


def _open_box(hole_face: str = "+x") -> trimesh.Trimesh:
    """A unit box with one face removed, leaving a single square boundary loop."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    centres = box.triangles_center
    axis = {"x": 0, "y": 1, "z": 2}[hole_face[1]]
    sign = 1.0 if hole_face[0] == "+" else -1.0
    keep = ~(np.sign(centres[:, axis]) * sign > 0) | (
        np.abs(centres[:, axis]) < 0.49
    )
    return trimesh.Trimesh(vertices=box.vertices, faces=box.faces[keep], process=False)


def test_open_box_has_one_boundary_loop():
    mesh = _open_box("+x")
    result = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)
    assert result["loops_total"] == 1
    assert result["loops_large"] == 1


def test_a_hole_facing_the_camera_is_visible():
    """The case that matters: evidence existed and the mesh tore anyway."""
    mesh = _open_box("+x")
    result = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)

    tear = result["tears"][0]
    assert tear["facing_camera"] is True
    assert tear["occluded"] is False
    assert tear["was_visible"] is True
    assert result["summary"]["visible_to_input_view"] == 1


def test_the_same_hole_seen_from_behind_is_not_visible():
    """Flip only the camera. A sign error in the facing test fails here, not in a table."""
    mesh = _open_box("+x")
    result = classify(mesh, np.array([-1.0, 0.0, 0.0]), min_perimeter=0.1)

    tear = result["tears"][0]
    assert tear["facing_camera"] is False
    assert tear["was_visible"] is False
    assert result["summary"]["facing_away"] == 1
    assert result["summary"]["visible_to_input_view"] == 0


def test_a_camera_facing_hole_behind_a_wall_is_occluded():
    """Facing the camera is not the same as being seen; a limb in front counts."""
    mesh = _open_box("+x")
    blocker = trimesh.creation.box(extents=(0.1, 2.0, 2.0))
    blocker.apply_translation((2.0, 0.0, 0.0))
    scene = trimesh.util.concatenate([mesh, blocker])

    result = classify(scene, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)
    tear = next(t for t in result["tears"] if t["facing_camera"])

    assert tear["occluded"] is True
    assert tear["was_visible"] is False
    assert result["summary"]["occluded"] == 1


def test_a_loose_speck_elsewhere_cannot_invert_the_verdict():
    """The bug the occlusion test caught first.

    An early version chose the rim normal's sign against the whole mesh's centroid, so
    any second component — and the hero fox had 226 of them — dragged that centroid and
    flipped camera-facing tears to facing-away. The sign must come from local geometry.
    """
    mesh = _open_box("+x")
    alone = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)

    speck = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    speck.apply_translation((8.0, 0.0, 0.0))  # far away, drags the global centroid
    with_speck = classify(
        trimesh.util.concatenate([mesh, speck]),
        np.array([1.0, 0.0, 0.0]),
        min_perimeter=0.1,
    )

    assert alone["tears"][0]["facing_camera"] is True
    assert with_speck["tears"][0]["facing_camera"] is True


def test_small_holes_are_not_counted_as_tears():
    """The threshold must match what fill_holes.py already patched, or we double-count."""
    mesh = _open_box("+x")
    result = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=10.0)
    assert result["loops_total"] == 1
    assert result["loops_large"] == 0
    assert result["tears"] == []


def test_perimeter_is_measured_in_scale_relative_units():
    """A unit box's open face has perimeter 4 against a scale of 1."""
    mesh = _open_box("+x")
    result = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)
    assert result["tears"][0]["perimeter_rel"] == pytest.approx(4.0, rel=1e-6)


def test_scaling_the_mesh_does_not_change_the_verdict():
    """Relative units only: a fox exported at 100x must classify identically."""
    small = _open_box("+x")
    big = small.copy()
    big.apply_scale(100.0)

    view = np.array([1.0, 0.0, 0.0])
    a = classify(small, view, min_perimeter=0.1)
    b = classify(big, view, min_perimeter=0.1)

    assert a["summary"] == b["summary"]
    assert a["tears"][0]["perimeter_rel"] == pytest.approx(
        b["tears"][0]["perimeter_rel"], rel=1e-6
    )


def test_rim_normal_points_out_of_the_hole():
    """loop_geometry averages face normals; an inward average inverts every verdict."""
    mesh = _open_box("+x")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    from fill_holes import boundary_loops

    loops = loop_geometry(vertices, faces, boundary_loops(vertices, faces))
    assert len(loops) == 1
    assert loops[0]["normal"][0] > 0.5, "rim normal should face +x, out of the opening"


def test_centroid_is_reported_in_bounding_box_fractions():
    """Absolute coordinates are unreadable across assets; fractions are comparable."""
    mesh = _open_box("+x")
    result = classify(mesh, np.array([1.0, 0.0, 0.0]), min_perimeter=0.1)
    x, y, z = result["tears"][0]["centroid_rel"]
    assert x == pytest.approx(1.0, abs=1e-6)  # the open face is at the +x extreme
    assert y == pytest.approx(0.5, abs=1e-6)
    assert z == pytest.approx(0.5, abs=1e-6)
