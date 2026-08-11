"""Tests for the tear metric that gates post-processing."""

import numpy as np
import pytest

from scripts.ribbon_metric import boundary_stats, verdict


def _tetrahedron():
    """The smallest closed surface: every edge shared by exactly two faces."""
    return np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def test_closed_mesh_scores_zero():
    pct_edges, pct_faces = boundary_stats(_tetrahedron())
    assert pct_edges == pytest.approx(0.0)
    assert pct_faces == pytest.approx(0.0)


def test_single_triangle_is_entirely_boundary():
    """One free triangle is the degenerate 'ribbon': all three edges open."""
    pct_edges, pct_faces = boundary_stats(np.array([[0, 1, 2]]))
    assert pct_edges == pytest.approx(100.0)
    assert pct_faces == pytest.approx(100.0)


def test_open_mesh_counts_faces_not_just_edges():
    """Two triangles sharing one edge: 4 of 5 edges open, but 100% of faces touch one."""
    pct_edges, pct_faces = boundary_stats(np.array([[0, 1, 2], [1, 2, 3]]))
    assert pct_edges == pytest.approx(80.0)
    assert pct_faces == pytest.approx(100.0)


def test_removing_a_face_from_a_closed_mesh_opens_it():
    pct_edges, pct_faces = boundary_stats(_tetrahedron()[:3])
    assert pct_edges > 0
    assert pct_faces == pytest.approx(100.0)


def test_empty_mesh_is_rejected():
    with pytest.raises(ValueError):
        boundary_stats(np.zeros((0, 3), dtype=int))


def test_verdict_thresholds():
    assert verdict(0.0).startswith("PASS")
    assert verdict(9.9).startswith("PASS")
    assert verdict(10.0).startswith("MARGINAL")
    assert verdict(24.9).startswith("MARGINAL")
    assert verdict(25.0).startswith("FAIL")
    assert verdict(40.9).startswith("FAIL"), "the Snag must not pass"


def test_gate_is_configurable():
    assert verdict(12.0, gate=15.0).startswith("PASS")
