"""Run the head replacement with small generated geometry and no rendering."""

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from image_to_3dlab import hero


@pytest.fixture
def geometry():
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    old_head = head.copy()
    old_head.apply_translation([0, 1.5, 0])
    torso = trimesh.creation.box(extents=[1.5, 2, 1])
    full = trimesh.util.concatenate([torso, old_head])
    bones = np.r_[np.zeros(len(torso.vertices), int), np.ones(len(old_head.vertices), int)]
    return full, head, bones


@pytest.mark.parametrize(("cut_at", "grow", "expected_cut"), [
    (None, None, None), (1.2, 4, 1.2), ("neck", None, None),
])
def test_graft_matches_and_replaces_head_above_torso(geometry, cut_at, grow, expected_cut):
    full, head, bones = geometry
    body, replacement, report = hero.graft(
        full, head, full_box=(25, 175, 25, 325), head_box=(0, 100, 0, 100),
        crop=(50, 25, 150, 125), head_width=100, bone=bones, shape=(350, 200),
        grow=grow, liner=0, cut_at=cut_at,
    )
    assert report["head_bones"] == [1]
    assert report["cut_at"] == expected_cut
    assert report["outline_margin_px"] == (grow if grow is not None else 10)
    assert report["line_up_cost"] < 1e-10
    assert report["body_faces"] == len(body.faces) >= 12
    assert report["head_faces"] == len(replacement.faces) > 0
    np.testing.assert_allclose(body.bounds[0], [-.75, -1, -.5])
    assert replacement.bounds[1][1] == pytest.approx(2)
    assert replacement.bounds[0][1] > 0.99


def test_graft_rejects_crop_outside_head(geometry):
    full, head, bones = geometry
    with pytest.raises(ValueError, match="does not land on the body's head"):
        hero.graft(full, head, (25, 175, 25, 325), (0, 100, 0, 100),
                   (450, 25, 550, 125), 100, bones, (350, 200))


def test_picture_with_opaque_alpha_uses_grey_cutout(tmp_path):
    picture = Image.new("RGBA", (10, 12), (hero.LIGHT_GREY,) * 3 + (255,))
    ImageDraw.Draw(picture).rectangle((2, 3, 8, 9), fill=(255, 0, 0, 255))
    path = tmp_path / "opaque.png"
    picture.save(path)
    assert hero.picture_box(hero.load_picture(path)) == (2, 8, 3, 9)


def test_skinned_vertices_refuses_mesh_without_joint_weights():
    primitive = trimesh.creation.box().export(file_type="glb")
    with pytest.raises(ValueError, match="no skinned mesh"):
        hero.skinned_vertices(primitive)


def test_mask_sampling_leaves_outside_points_unselected():
    mask = np.ones((4, 4), dtype=bool)
    x = np.array([-1., 0., 3., 4.])
    y = np.array([1., 0., 3., 2.])
    assert hero.on_mask(x, y, mask).tolist() == [False, True, True, False]


def test_neck_in_adjacent_collar_and_skull_slices_has_no_clear_cut():
    ys = np.repeat([0.0, .4, .6, 1.0], 30)
    xs = np.tile(np.linspace(-1.0, 1.0, 30), 4)
    assert hero.neck_height(ys, xs, slices=5) is None
