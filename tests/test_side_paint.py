"""Tests for where a drawn side picture may paint a figure (`--sides-paint safe`).

The rules read the figure off its front picture's cut-out: a synthetic A-pose figure (head, neck, shoulders, a torso
with its arms hanging apart, two legs) stands in for it, and a square seen from the side shows which heights take the
side picture's paint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import side_paint as sp

IDENTITY = ((0, 1, 2), (1, 1, 1))
SIZE, FOV, DIST = 256, 0.6, 3.0
F = (SIZE / 2) / np.tan(FOV / 2)


def _row(z: float) -> int:
    return int(round(SIZE / 2 - F * z / DIST))


def _col(x: float) -> int:
    return int(round(SIZE / 2 - F * x / DIST))   # the front camera looks down -y: world +x is image left


def _figure_alpha() -> np.ndarray:
    """A 1 m figure from z = -0.5 to 0.5 in the front camera, in an A-pose: the arms hang apart from the torso."""
    alpha = np.zeros((SIZE, SIZE), np.uint8)

    def box(z0, z1, x0, x1):
        alpha[_row(z1):_row(z0) + 1, _col(x1):_col(x0) + 1] = 255
    box(0.35, 0.50, -0.08, 0.08)                         # head
    box(0.30, 0.35, -0.03, 0.03)                         # neck
    box(0.22, 0.30, -0.30, 0.30)                         # shoulders, the arms joined
    box(-0.10, 0.22, -0.13, 0.13)                        # torso
    box(-0.05, 0.22, 0.18, 0.30)                         # arms, apart from the torso
    box(-0.05, 0.22, -0.30, -0.18)
    box(-0.50, -0.10, 0.02, 0.12)                        # legs
    box(-0.50, -0.10, -0.12, -0.02)
    return alpha


def _front_view(alpha=None) -> pp.View:
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).T   # looks down -y, up is +z
    c2w[:3, 3] = (0.0, DIST, 0.0)
    image = np.zeros((SIZE, SIZE, 4), np.uint8)
    image[..., 3] = _figure_alpha() if alpha is None else alpha
    return pp.View(image, c2w, FOV, "view_000.png")


def _side_view(paint: str = "safe") -> pp.View:
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T    # looks down -x
    c2w[:3, 3] = (DIST, 0.0, 0.0)
    image = np.zeros((SIZE, SIZE, 4), np.uint8)
    image[..., 0] = 255
    image[..., 3] = 255
    return pp.View(image, c2w, FOV, "view_270.png", paint)


def _side_square(x: float = 0.15):
    """A square in the plane x = `x` facing +x (the side camera), from z = -0.5 to 0.5; v grows downward."""
    positions = np.array([[x, -0.3, -0.5], [x, 0.3, -0.5], [x, 0.3, 0.5], [x, -0.3, 0.5]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
    return positions, faces, uvs


def _texel(z: float, size: int = 64) -> int:
    return int((0.5 - z) * size)


# --- the figure read off the front picture --------------------------------------------------------

def test_runs_are_the_stretches_of_a_row():
    assert sp.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 3), (4, 5), (7, 8)]
    assert sp.runs(np.zeros(4, bool)) == []


def test_the_body_is_the_run_over_the_middle_or_the_nearest_either_side():
    assert sp.body_runs([(0, 3), (5, 10), (12, 14)], 7) == {1}
    assert sp.body_runs([(0, 3), (4, 6), (8, 10), (12, 14)], 7) == {1, 2}   # between the legs


def test_the_front_picture_gives_the_neck_the_hands_and_the_arms():
    layout = sp.front_layout(_figure_alpha())
    assert _row(0.35) <= layout.neck <= _row(0.30)
    assert layout.hands == _row(-0.05)
    assert layout.arm[_row(0.1), _col(0.24)] and layout.arm[_row(0.1), _col(-0.24)]   # both arms
    assert not layout.arm[_row(0.1), _col(0.0)]          # the torso
    assert not layout.arm[_row(-0.3), _col(0.07)]        # a leg
    assert not layout.arm[_row(0.45), :].any()           # nothing on the head


def test_a_figure_whose_arms_touch_its_body_has_no_arm():
    alpha = _figure_alpha()
    alpha[_row(0.22):_row(-0.05) + 1, _col(0.30):_col(-0.30) + 1] = 255   # arms against the torso
    layout = sp.front_layout(alpha)
    assert layout.hands == -1 and not layout.arm.any()


def test_the_band_runs_from_the_neck_to_below_the_hands():
    layout = sp.front_layout(_figure_alpha())
    rows = np.array([_row(0.45), _row(0.1), _row(-0.10), _row(-0.4)], float)
    keep = sp.band_keep(rows, layout)
    assert keep[0] == pytest.approx(1.0)                 # the head
    assert keep[1] == pytest.approx(0.0)                 # the chest
    assert keep[2] == pytest.approx(0.0)                 # a drawn hand may hang 6 % lower than the model's
    assert keep[3] == pytest.approx(1.0)                 # the legs


def test_the_face_is_kept_from_the_sides_but_not_the_ears():
    layout = sp.front_layout(_figure_alpha())
    rows = np.full(3, float(_row(0.42)))
    x = np.array([0.0, 0.07, 0.0])
    toward = np.array([1.0, 0.0, 1.0])
    keep = sp.face_keep(rows, x, toward, head_x=0.0, size=1.0, layout=layout)
    assert keep[0] == pytest.approx(0.0)                 # the face between the temples, facing the front
    assert keep[1] == pytest.approx(1.0)                 # an ear: facing the side, out past the temples
    below = sp.face_keep(np.array([float(_row(0.1))]), np.zeros(1), np.ones(1), 0.0, 1.0, layout)
    assert below[0] == pytest.approx(1.0)                # the chest is the band's business, not the face's


def test_a_side_pictures_own_face_is_its_front_quarter_above_the_neck():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[10:40, 30:70] = 255                            # a head 40 px deep
    alpha[40:90, 20:80] = 255                            # the body
    mask = sp.profile_face_mask(alpha, top=10, neck=39, front_right=True)
    assert mask[20, 69] == pytest.approx(1.0)            # the front edge: the drawn eye and nose
    assert mask[20, 50] == pytest.approx(0.0)            # halfway back: the ear
    assert not mask[60].any()                            # nothing below the neck
    left = sp.profile_face_mask(alpha, top=10, neck=39, front_right=False)
    assert left[20, 30] == pytest.approx(1.0) and left[20, 69] == pytest.approx(0.0)


# --- painting --------------------------------------------------------------------------------------

def test_a_safe_side_paints_the_head_and_the_legs_but_not_the_body_band():
    positions, faces, uvs = _side_square()
    texture = np.full((64, 64, 3), 128, np.uint8)
    reports: list = []
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [_front_view(), _side_view("safe")],
                                   frame=IDENTITY, settings=pp.Settings(align=False, match_colour=False),
                                   side_reports=reports)
    assert weight[_texel(0.44), 32] > 0.9 and out[_texel(0.44), 32, 0] > 240     # the head's height
    assert weight[_texel(0.10), 32] == pytest.approx(0.0)                       # the chest: front and back only
    assert tuple(out[_texel(0.10), 32]) == (128, 128, 128)
    assert weight[_texel(-0.36), 32] > 0.9                                        # the legs
    assert reports and reports[0]["texels_painted"] < reports[0]["texels_seen"]


def test_an_all_side_paints_everywhere_and_a_none_side_nowhere():
    positions, faces, uvs = _side_square()
    texture = np.full((64, 64, 3), 128, np.uint8)
    settings = pp.Settings(align=False, match_colour=False)
    _, all_w = pp.paint_texture(texture, positions, uvs, faces, [_front_view(), _side_view("all")],
                                frame=IDENTITY, settings=settings)
    _, none_w = pp.paint_texture(texture, positions, uvs, faces, [_front_view(), _side_view("none")],
                                 frame=IDENTITY, settings=settings)
    assert all_w[_texel(0.10), 32] > 0.9
    assert none_w.max() == pytest.approx(0.0)


def test_a_safe_side_without_a_front_picture_paints_nothing():
    positions, faces, uvs = _side_square()
    texture = np.full((64, 64, 3), 128, np.uint8)
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [_side_view("safe")], frame=IDENTITY,
                                 settings=pp.Settings(align=False, match_colour=False))
    assert weight.max() == pytest.approx(0.0)


@pytest.mark.parametrize(("paint", "expected"), [("all", 60), ("safe", 200)])
def test_a_drawn_side_alone_cannot_trigger_outside_fill(paint, expected):
    positions, faces, uvs = _side_square()
    texture = np.full((64, 64, 3), 60, np.uint8)
    texture[:, 32:] = 200
    side = _side_view(paint)
    side.image[..., :3] = 60
    side.image[:, SIZE // 2:, 3] = 0
    out, _ = pp.paint_texture(texture, positions, uvs, faces, [side], frame=IDENTITY,
                              settings=pp.Settings(align=False, match_colour=False))
    assert np.all(out[32, 56] == expected)


def test_the_paint_rule_loads_from_transforms_json(tmp_path):
    Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(tmp_path / "view_000.png")
    Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(tmp_path / "view_270.png")
    (tmp_path / "transforms.json").write_text(json.dumps({"camera_angle_x": 0.349, "frames": [
        {"file_path": "view_000.png", "transform_matrix": np.eye(4).tolist()},
        {"file_path": "view_270.png", "transform_matrix": np.eye(4).tolist(), "paint": "safe"}]}))
    views, _ = pp.load_views(tmp_path)
    assert [v.paint for v in views] == ["all", "safe"]
