"""Tests for building a UV mask from a point in 3D."""

import numpy as np
import pytest
from PIL import Image

from scripts.feature_mask import rasterise_uv, select_faces


@pytest.fixture
def two_triangles():
    """Two coplanar triangles, one at the origin and one far away on +X."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],
         [5.0, 0.0, 0.0], [5.1, 0.0, 0.0], [5.0, 0.1, 0.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return vertices, faces


def test_sphere_selects_only_the_near_face(two_triangles):
    vertices, faces = two_triangles
    selected = select_faces(vertices, faces, np.zeros(3), radius=1.0)
    assert selected.tolist() == [True, False]


def test_radius_zero_is_rejected(two_triangles):
    vertices, faces = two_triangles
    with pytest.raises(ValueError):
        select_faces(vertices, faces, np.zeros(3), radius=0.0)


def test_front_only_drops_back_facing_geometry(two_triangles):
    """A sphere reaching through a shell must not also mask the inside surface."""
    vertices, faces = two_triangles
    normals = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    facing = np.array([0.0, -1.0, 0.0])

    both_near = select_faces(vertices, faces, np.zeros(3), 10.0, normals, facing)
    assert both_near.tolist() == [True, True]

    flipped = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    selected = select_faces(vertices, faces, np.zeros(3), 10.0, flipped, facing)
    assert selected.tolist() == [False, True]


def test_rasterise_marks_the_right_corner_and_flips_v():
    """UV origin is bottom-left, image origin is top-left; v must be flipped."""
    uv = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]])
    faces = np.array([[0, 1, 2]])
    mask = np.asarray(rasterise_uv(uv, faces, np.array([True]), size=64, dilate=0))

    # v=0 is the BOTTOM of the atlas, so the triangle lands in the lower-left.
    assert mask[-2, 2] == 255, "expected the mask at the bottom-left"
    assert mask[2, -2] == 0, "top-right must stay empty"


def test_nothing_selected_gives_an_empty_mask():
    uv = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]])
    faces = np.array([[0, 1, 2]])
    mask = np.asarray(rasterise_uv(uv, faces, np.array([False]), size=32, dilate=2))
    assert mask.max() == 0


def test_dilate_grows_the_mask_without_shrinking_it():
    uv = np.array([[0.3, 0.3], [0.5, 0.3], [0.3, 0.5]])
    faces = np.array([[0, 1, 2]])
    tight = np.asarray(rasterise_uv(uv, faces, np.array([True]), 128, dilate=0))
    grown = np.asarray(rasterise_uv(uv, faces, np.array([True]), 128, dilate=3))

    assert grown.sum() > tight.sum()
    assert np.all(grown[tight > 0] == 255), "dilation must not erase original texels"


def test_mask_is_single_channel_and_binary():
    uv = np.array([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
    faces = np.array([[0, 1, 2]])
    img = rasterise_uv(uv, faces, np.array([True]), 64, dilate=1)
    assert isinstance(img, Image.Image) and img.mode == "L"
    assert set(np.unique(np.asarray(img))).issubset({0, 255})
