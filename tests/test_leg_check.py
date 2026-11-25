"""Tests for image_to_3dlab/leg_check.py: do a four-legged subject's legs stand apart in a picture?"""

from __future__ import annotations

import numpy as np
from PIL import Image

from image_to_3dlab import leg_check


def side_mask(open_rows: int, legs_apart: bool = True) -> np.ndarray:
    """A 100 x 160 side view: a body on four legs, `open_rows` rows free under the belly between them."""
    a = np.zeros((120, 200), dtype=bool)
    a[10:100 - open_rows, 20:180] = True        # head, body and the legs' tops
    if legs_apart:
        a[100 - open_rows:100, 20:50] = True    # the front legs
        a[100 - open_rows:100, 150:180] = True  # the hind legs
    else:
        a[100 - open_rows:100, 20:180] = True   # one block down to the floor
    return a


def front_mask(gap: int) -> np.ndarray:
    """A 90 px tall front view: a body over two legs `gap` px apart at the feet."""
    a = np.zeros((100, 120), dtype=bool)
    a[5:70, 30:90] = True
    left_end = 60 - gap // 2
    a[70:95, 30:left_end] = True
    a[70:95, left_end + gap:90] = True
    return a


def test_a_side_view_with_open_space_under_the_body_passes():
    m = leg_check.measure(side_mask(open_rows=30))
    assert m["open_under_body"] >= 0.3
    assert m["parts"][0.08] == 2
    assert leg_check.problems(m, "side") == []


def test_a_side_view_whose_legs_fuse_into_one_block_fails():
    m = leg_check.measure(side_mask(open_rows=30, legs_apart=False))
    assert m["open_under_body"] < leg_check.MIN_OPEN_SIDE
    found = leg_check.problems(m, "side")
    assert found and "open under the body" in found[0]


def test_a_side_view_with_only_a_sliver_open_fails():
    # the Glutwolf's drawn side of round 2: 6 % of the height open under the body, one part above the feet
    m = leg_check.measure(side_mask(open_rows=5))
    assert leg_check.problems(m, "side")


def test_a_front_view_with_a_gap_between_the_feet_passes():
    m = leg_check.measure(front_mask(gap=12))
    assert m["parts"][0.03] == 2
    assert leg_check.problems(m, "front") == []


def test_a_front_view_whose_feet_touch_fails():
    m = leg_check.measure(front_mask(gap=0))
    found = leg_check.problems(m, "front")
    assert found and "feet" in found[0]


def test_an_empty_picture_reports_no_subject():
    m = leg_check.measure(np.zeros((10, 10), dtype=bool))
    assert m["subject"] is None
    assert leg_check.problems(m, "side") == ["no subject in the picture"]


def test_specks_do_not_count_as_legs():
    a = side_mask(open_rows=30, legs_apart=False)
    a[99, 100] = False            # a one-pixel hole in the block is no gap between legs
    m = leg_check.measure(a)
    assert m["parts"][0.03] == 1


def test_a_cut_out_picture_is_read_from_its_alpha(tmp_path):
    rgba = np.zeros((120, 200, 4), dtype=np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = side_mask(open_rows=30).astype(np.uint8) * 255
    path = tmp_path / "side.png"
    Image.fromarray(rgba, "RGBA").save(path)
    m, found = leg_check.check_picture(path, "side")
    assert found == [] and m["parts"][0.08] == 2


def test_the_better_of_two_side_views_is_the_one_with_more_open_space():
    fused = leg_check.measure(side_mask(open_rows=30, legs_apart=False))
    apart = leg_check.measure(side_mask(open_rows=30))
    assert leg_check.better(apart, fused) and not leg_check.better(fused, apart)
