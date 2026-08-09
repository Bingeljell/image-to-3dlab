"""Tests for dropping disconnected junk from a generated mesh.

The mesh under repair carries a 2048 texture, and the existing `fill_holes.py`
silently discards it (welding by position collapses the UV-split vertices glTF
creates at every seam). So the property that matters most here is not the face
count -- it is that UVs survive. That is what `test_uvs_survive` pins down.
"""

from __future__ import annotations

import numpy as np
import trimesh

from scripts.remove_loose_parts import loose_face_mask, prune


def _cube(offset: tuple[float, float, float], scale: float = 1.0) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(scale, scale, scale))
    box.apply_translation(offset)
    return box


def _two_islands() -> trimesh.Trimesh:
    """A big cube (12 faces) plus a distant 4-face speck.

    The face counts must DIFFER, or no threshold can separate them and the test
    proves nothing -- the first version of this fixture used two cubes and passed
    a broken assertion for that reason.
    """
    big = _cube((0.0, 0.0, 0.0), scale=10.0)
    speck = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    speck = trimesh.Trimesh(vertices=speck.vertices[:4], faces=[[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    speck.apply_translation((50.0, 0.0, 0.0))
    return trimesh.util.concatenate([big, speck])


def test_mask_keeps_every_face_when_threshold_is_low():
    mesh = _two_islands()
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces=1)
    assert mask.all()


def test_mask_drops_the_smaller_component():
    mesh = _two_islands()
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces=6)
    assert mask.sum() == 12, "the 12-face cube stays, the 4-face speck goes"


def test_threshold_above_everything_drops_everything():
    mesh = _two_islands()
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces=13)
    assert not mask.any()


def test_mask_is_over_original_face_indices():
    """The mask must index the ORIGINAL faces, not a welded/filtered copy.

    Welding drops degenerate faces, which shifts indices. A mask built against the
    welded array and applied to the original silently deletes the wrong triangles.
    """
    mesh = _two_islands()
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces=1)
    assert mask.shape == (len(mesh.faces),)


def test_largest_component_is_always_kept():
    mesh = _two_islands()
    mask = loose_face_mask(mesh.vertices, mesh.faces, min_faces=6)
    kept = mesh.faces[mask]
    # The big cube's vertices span 10 units; the speck sits out at x=50.
    assert kept.size > 0
    xs = mesh.vertices[kept.ravel()][:, 0]
    assert xs.max() < 40.0, "the distant speck should be gone"


def test_welding_is_by_position_so_uv_seams_do_not_split_components():
    """Two triangles sharing an edge but with different UVs are ONE component.

    This is the trap that produced a wrong diagnosis for several sessions: glTF
    splits a vertex at every UV seam, so counting components without welding by
    position counts UV islands instead of geometry.
    """
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64)
    mask = loose_face_mask(vertices, faces, min_faces=2)
    assert mask.all(), "duplicated seam vertices must not split the surface"


def test_uvs_survive():
    """The whole point: pruning must not cost the texture."""
    mesh = _two_islands()
    uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    pruned = prune(mesh, min_faces=6)

    assert pruned.visual.uv is not None, "UVs were dropped"
    assert len(pruned.visual.uv) == len(pruned.vertices)
    assert len(pruned.faces) < len(mesh.faces)


def test_prune_is_idempotent():
    mesh = _two_islands()
    once = prune(mesh, min_faces=6)
    twice = prune(once, min_faces=6)
    assert len(twice.faces) == len(once.faces)
