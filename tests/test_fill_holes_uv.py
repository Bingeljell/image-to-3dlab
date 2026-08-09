"""Hole filling must not cost the texture.

The original `fill_holes.py` welded by position and exported the WELDED mesh, which
collapses the vertices glTF splits at every UV seam. Output had `uv: None` and no
material -- `moss_fox_hero_101k_filled.glb` is 1.9 MB against the hero's 12 MB purely
because of this. A repaired asset that cannot be textured is not a shippable asset,
so these tests pin the property down.
"""

from __future__ import annotations

import numpy as np
import trimesh

from scripts.fill_holes import fill


def _open_tetrahedron(with_uv: bool = True) -> trimesh.Trimesh:
    """A tetrahedron with one face removed: a single triangular hole."""
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.4, 1]], dtype=np.float64
    )
    # Three of the four faces, consistently wound outward.
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if with_uv:
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.25, 0.5]])
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    edges = np.sort(
        np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def test_the_hole_is_actually_closed():
    mesh = _open_tetrahedron()
    assert _boundary_edge_count(mesh) == 3
    filled = fill(mesh, max_perimeter=5.0)
    assert _boundary_edge_count(filled) == 0


def test_uvs_survive_and_stay_aligned_with_vertices():
    mesh = _open_tetrahedron()
    filled = fill(mesh, max_perimeter=5.0)

    assert filled.visual.uv is not None, "UVs were dropped -- the original bug"
    assert len(filled.visual.uv) == len(filled.vertices)


def test_original_vertices_and_uvs_are_untouched():
    """Patches are APPENDED. Existing geometry must survive bit-for-bit, or the
    baked 2048 texture no longer lines up with the surface it was baked for."""
    mesh = _open_tetrahedron()
    filled = fill(mesh, max_perimeter=5.0)

    n = len(mesh.vertices)
    assert np.allclose(filled.vertices[:n], mesh.vertices)
    assert np.allclose(filled.visual.uv[:n], mesh.visual.uv)
    assert np.array_equal(filled.faces[: len(mesh.faces)], mesh.faces)


def test_patch_winding_agrees_with_the_surface():
    """A boundary edge belongs to exactly one face, so the patch must traverse it in
    the OPPOSITE direction. Getting this wrong injects randomly-oriented faces, which
    measurably defeats normal recalculation afterwards."""
    mesh = _open_tetrahedron()
    filled = fill(mesh, max_perimeter=5.0)
    assert filled.is_winding_consistent
    assert filled.volume > 0, "inside-out patch would give negative volume"


def test_large_holes_are_left_alone():
    """A big opening is missing evidence, not an artefact; stretching a membrane
    across the back of a head looks worse than the hole."""
    mesh = _open_tetrahedron()
    filled = fill(mesh, max_perimeter=0.01)
    assert len(filled.faces) == len(mesh.faces), "hole should have been skipped"
    assert _boundary_edge_count(filled) == 3


def test_a_uv_seam_across_the_rim_does_not_split_the_loop():
    """glTF duplicates a vertex at a UV seam. Tracing boundaries on raw indices would
    see two open ends instead of one closed rim and refuse to fill it."""
    mesh = _open_tetrahedron()
    # Duplicate vertex 1 with a different UV, and rewire one face to the copy --
    # exactly what a glTF exporter does at a seam.
    vertices = np.vstack([mesh.vertices, mesh.vertices[1]])
    uv = np.vstack([mesh.visual.uv, [0.9, 0.1]])
    faces = mesh.faces.copy()
    faces[1][1] = 4  # face [0,1,3] now reads [0,4,3]
    seamed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    seamed.visual = trimesh.visual.TextureVisuals(uv=uv)

    filled = fill(seamed, max_perimeter=5.0)
    assert len(filled.faces) > len(seamed.faces), "the seam blocked the fill"
    assert filled.visual.uv is not None
    assert len(filled.visual.uv) == len(filled.vertices)


def test_works_without_uvs():
    """Untextured meshes must still fill -- the tool predates the texture path."""
    mesh = _open_tetrahedron(with_uv=False)
    filled = fill(mesh, max_perimeter=5.0)
    assert _boundary_edge_count(filled) == 0


def test_refilling_adds_nothing():
    mesh = _open_tetrahedron()
    once = fill(mesh, max_perimeter=5.0)
    twice = fill(once, max_perimeter=5.0)
    assert len(twice.faces) == len(once.faces)
