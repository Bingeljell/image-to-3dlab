"""Tests for exact vertex welding before simplification."""

import numpy as np
import pytest

from scripts.mesh_weld import duplicate_count, weld_vertices


def test_split_vertex_is_rejoined():
    """The exact case repair_non_manifold_edges creates: one point stored twice."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0.0, 0, 0]])
    faces = np.array([[0, 1, 2], [3, 1, 2]])
    v, f = weld_vertices(verts, faces)
    assert len(v) == 3
    # both faces now reference the same welded vertex, so they are the same triangle
    assert np.array_equal(f[0], f[1])


def test_geometry_is_unchanged_when_nothing_is_duplicated():
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    v, f = weld_vertices(verts, faces)
    np.testing.assert_array_equal(v, verts)
    np.testing.assert_array_equal(f, faces)


def test_degenerate_faces_are_dropped():
    """Welding can fold two corners of a face onto one point; that face has no area."""
    verts = np.array([[0.0, 0, 0], [0.0, 0, 0], [1, 0, 0]])
    faces = np.array([[0, 1, 2]])
    _, f = weld_vertices(verts, faces)
    assert len(f) == 0


def test_faces_still_index_valid_vertices():
    rng = np.random.default_rng(0)
    base = rng.random((20, 3))
    verts = np.vstack([base, base[:5]])          # five exact duplicates
    faces = rng.integers(0, len(verts), (40, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    v, f = weld_vertices(verts, faces)
    assert len(v) == 20
    assert f.max() < len(v) and f.min() >= 0


def test_welding_preserves_vertex_positions():
    """A weld must never move a point -- only merge points already identical."""
    rng = np.random.default_rng(1)
    base = rng.random((12, 3))
    verts = np.vstack([base, base[:3]])
    faces = np.array([[0, 1, 2], [12, 3, 4]])
    v, _ = weld_vertices(verts, faces)
    for row in v:
        assert np.any(np.all(np.isclose(base, row), axis=1))


def test_near_but_not_exact_duplicates_are_kept():
    """Exactness is the point: a tolerance weld would fuse adjacent coils."""
    verts = np.array([[0.0, 0, 0], [1e-9, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 2, 3], [1, 2, 3]])
    v, _ = weld_vertices(verts, faces)
    assert len(v) == 4


def test_duplicate_count():
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert duplicate_count(verts) == 2


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        weld_vertices(np.zeros((4, 2)), np.zeros((1, 3), dtype=int))
    with pytest.raises(ValueError):
        weld_vertices(np.zeros((4, 3)), np.zeros((1, 4), dtype=int))


def test_out_of_range_faces_are_rejected():
    with pytest.raises(ValueError):
        weld_vertices(np.zeros((3, 3)), np.array([[0, 1, 9]]))


def test_empty_faces_are_allowed():
    v, f = weld_vertices(np.zeros((3, 3)), np.zeros((0, 3), dtype=int))
    assert len(f) == 0 and len(v) == 1
