"""Tests for region cropping.

The point of cropping is to keep every original triangle so a cage and a surface can be told
apart. So the properties that matter are: nothing is decimated, nothing is duplicated, and
the selection is by centroid rather than "any vertex inside" - the latter drags in a fringe
of faces hanging off the cut plane, which is exactly the visual noise that would muddy the
judgement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crop_mesh.py"


def _load():
    spec = importlib.util.spec_from_file_location("crop_mesh", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cm = _load()


def test_top_half_of_a_box_keeps_faces_above_the_middle():
    box = trimesh.creation.box(extents=(1, 1, 1))
    keep = cm.face_mask(box.vertices, box.faces, {"y": (0.5, 1.0)})
    centroids = box.vertices[box.faces].mean(axis=1)
    assert keep.any()
    assert (centroids[keep][:, 1] >= box.vertices[:, 1].mean() - 1e-9).all()


def test_selection_is_by_centroid_not_any_vertex():
    """A face straddling the plane must be decided once, by its middle."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 1, 0]])   # centroid y = 1/3
    faces = np.array([[0, 1, 2]])
    # Upper half by centroid excludes it, even though one vertex reaches y=1.
    assert not cm.face_mask(verts, faces, {"y": (0.5, 1.0)}).any()
    assert cm.face_mask(verts, faces, {"y": (0.0, 0.5)}).all()


def test_bounds_compose_across_axes():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    one = cm.face_mask(sphere.vertices, sphere.faces, {"y": (0.5, 1.0)})
    two = cm.face_mask(sphere.vertices, sphere.faces, {"y": (0.5, 1.0), "x": (0.5, 1.0)})
    assert two.sum() < one.sum()
    assert (two & ~one).sum() == 0        # the second is a strict subset


def test_fractions_are_relative_to_the_meshs_own_bounds():
    """Same shape, different scale and origin, same fraction kept."""
    small = trimesh.creation.box(extents=(1, 1, 1))
    big = trimesh.creation.box(extents=(10, 10, 10))
    big.apply_translation([100, 100, 100])
    a = cm.face_mask(small.vertices, small.faces, {"y": (0.5, 1.0)})
    b = cm.face_mask(big.vertices, big.faces, {"y": (0.5, 1.0)})
    assert a.sum() == b.sum()


def test_degenerate_axis_does_not_divide_by_zero():
    """A flat mesh has zero span on one axis; that must not produce nan."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mask = cm.face_mask(verts, faces, {"z": (0.0, 1.0)})
    assert mask.all()


def test_every_named_region_is_a_valid_axis_spec():
    for name, bounds in cm.REGIONS.items():
        for axis, (low, high) in bounds.items():
            assert axis in cm.AXES, f"{name} uses unknown axis {axis}"
            assert 0.0 <= low < high <= 1.0, f"{name} has bad bounds"


def test_crop_preserves_original_triangles(tmp_path):
    """No decimation: kept faces must be identical to their originals.

    This is the whole point - a decimated crop could not answer whether the surface is
    solid, because decimation itself removes faces.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    keep = cm.face_mask(sphere.vertices, sphere.faces, {"y": (0.5, 1.0)})
    cropped = sphere.submesh([keep], append=True)
    assert len(cropped.faces) == int(keep.sum())
    original_areas = np.sort(sphere.area_faces[keep])
    assert np.allclose(np.sort(cropped.area_faces), original_areas)
