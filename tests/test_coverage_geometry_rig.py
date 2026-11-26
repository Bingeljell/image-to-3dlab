"""Exercise geometry helpers with small local fixtures."""

import numpy as np
import pytest

from scripts import clean_skin_weights


def test_cleaning_uses_bone_distance_and_preserves_weight_total():
    parents = {"upper": None, "lower": "upper", "end": "lower"}
    segments = {
        "upper": (np.array([0, 0, 0]), np.array([0, 1, 0])),
        "lower": (np.array([0, 1, 0]), np.array([0, 2, 0])),
        "end": (np.array([0, 2, 0]), np.array([0, 3, 0])),
    }
    clean = clean_skin_weights.build_vertex_cleaner(parents, segments, max_hops=0)
    assert clean(np.array([0, .5, 0]), {"end": .4, "upper": .2}) == {
        "upper": pytest.approx(.6)
    }
    close = {"upper": .1, "lower": .9}
    assert clean(np.array([0, 1, 0]), close) is close
