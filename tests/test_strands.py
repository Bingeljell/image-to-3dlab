"""Tests for `image_to_3dlab.strands`: hair strands a generator leaves in front of a face.

Pixal3D often builds a strand of hair a little in front of the forehead or the eyes where the front picture shows
bare skin. Pixel Match then paints it with the picture's skin, and from the side it reads as a crack across the
face. The count of such strand triangles tells a clean try from a spoiled one, so Create can keep the cleanest of
a few seeds. These tests cover the parts without a model: the colour rule, the face in the picture, the layers
behind a pixel and the grouping into pieces.
"""

import numpy as np

from image_to_3dlab import strands


def test_skin_colours_are_skin_and_hair_colours_are_not():
    pixels = np.array([[[224, 172, 140], [141, 85, 36], [30, 30, 30], [235, 235, 240], [60, 90, 200]]], np.uint8)
    assert strands.skin_mask(pixels)[0].tolist() == [True, True, False, False, False]


def _figure(face_rgb=(224, 172, 140)):
    """A 200 px tall figure: a dark-haired head with a skin-coloured face, a grey body."""
    image = np.zeros((220, 120, 4), np.uint8)
    image[10:210, 30:90] = (90, 90, 95, 255)          # the body
    image[10:50, 40:80] = (40, 30, 25, 255)           # the hair
    image[18:44, 46:74] = (*face_rgb, 255)            # the face
    image[26:30, 52:56] = (20, 20, 20, 255)           # an eye: a hole the face area fills
    return image


def test_the_face_is_the_skin_in_the_head_with_its_holes_filled():
    face, box = strands.face_area(_figure())
    assert box == (46, 18, 74, 44)
    assert face[27, 53]                                # the eye counts as face
    assert not face[100, 60]                           # the body does not


def test_a_figure_without_skin_in_its_head_has_no_face():
    assert strands.face_area(_figure(face_rgb=(60, 90, 200))) is None


def test_fragments_list_every_layer_a_pixel_centre_lies_in():
    tri_xy = np.array([[[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]])
    tri_depth = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    pix, z, tri = strands.fragments(tri_xy, tri_depth, (4, 4))
    at = pix == 1 * 4 + 1                              # the pixel centred at (1.5, 1.5)
    assert sorted(z[at].tolist()) == [1.0, 2.0]
    assert sorted(tri[at].tolist()) == [0, 1]


def test_a_thin_sheet_with_a_short_gap_in_front_of_the_skin_is_a_strand():
    """Layers along one pixel, metres from the camera: a strand's front and back 5 mm apart, 10 mm of air, the
    skin, then the far side of the hollow head. The skin is found behind the strand."""
    pix = np.array([0, 0, 0, 0, 1, 1])
    z = np.array([1.000, 1.005, 1.015, 1.200, 1.015, 1.200])
    depth, strand = strands.skin_depth(pix, z, 2, scale=1.0)
    assert np.allclose(depth, [1.015, 1.015])
    assert strand.tolist() == [True, False]


def test_the_skin_followed_by_the_hollow_head_is_no_strand():
    """The skin itself is a wall a few millimetres thick, then the deep hollow: no short gap, so it stays the skin."""
    pix = np.array([0, 0, 0])
    z = np.array([1.000, 1.006, 1.150])
    depth, strand = strands.skin_depth(pix, z, 1, scale=1.0)
    assert np.allclose(depth, [1.000])
    assert strand.tolist() == [False]


def test_pieces_join_triangles_that_share_a_vertex():
    faces = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 5]])
    label = strands.pieces(faces)
    assert label[0] == label[1]
    assert label[2] == label[3]
    assert label[0] != label[2]
