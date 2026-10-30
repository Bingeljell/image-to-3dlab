"""Repair must close real holes without treating a texture seam as one."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from scripts.mesh_health import measure, repair_mesh


def _textured(mesh):
    split = mesh.copy()
    split.unmerge_vertices()
    count = len(split.vertices)
    uv = np.linspace(0.0, 1.0, count * 2, dtype=np.float64).reshape(count, 2)
    split.visual = trimesh.visual.TextureVisuals(uv=uv)
    return split, uv


def test_repair_keeps_a_seamed_closed_mesh_closed_and_keeps_its_uvs():
    split, uv = _textured(trimesh.creation.box())
    repaired = repair_mesh(split)
    assert len(repaired.faces) == 12
    assert np.allclose(np.asarray(repaired.visual.uv), uv)
    assert measure(repaired)["watertight"] is True
    assert measure(repaired)["boundary_edges"] == 0


def test_repair_closes_a_small_hole_without_collapsing_uvs():
    # Subdivision 4 makes one face a small fraction of the sphere, which is the
    # hole size the filler is meant to close.
    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.faces = sphere.faces[1:]
    split, uv = _textured(sphere)
    repaired = repair_mesh(split)
    assert measure(repaired)["watertight"] is True
    assert measure(repaired)["boundary_edges"] == 0
    got = np.asarray(repaired.visual.uv)
    assert np.allclose(got[: len(uv)], uv)
    assert len(repaired.faces) == len(split.faces) + 3
