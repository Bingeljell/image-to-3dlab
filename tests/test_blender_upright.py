import itertools

import numpy as np
import pytest

from scripts.blender_split_props import tilt_matrix
from scripts.blender_upright import upright_points


def test_tilted_box_is_level_and_grounded():
    box = np.array(list(itertools.product([-1, 1], [-0.5, 0.5], [0, 3])))
    tilted = box @ tilt_matrix(22, 8).T + [3, 2, 4]
    result = upright_points(tilted)
    assert np.sort(result[:, 2]) == pytest.approx([0] * 4 + [3] * 4, abs=0.01)
    assert result[:, 2].min() == pytest.approx(0)


def test_disabled_correction_leaves_character_untouched():
    points = np.array([[1., 2., 3.], [4., 5., 6.]])
    assert np.array_equal(upright_points(points, enabled=False), points)
