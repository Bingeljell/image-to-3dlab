"""Bugs from the old review commit 90362af that are still present on this branch.

Each test fails today for the reason in its docstring and passes once that bug is fixed.
No Blender, no network, no weights: tiny synthetic inputs only.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import arm_clearance as ac


def test_opposite_directions_make_a_finite_half_turn():
    """c = -1 divides by (1 + c) = 0, so the matrix comes back full of inf/nan."""
    a, b = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
    with np.errstate(
        all="raise"
    ):  # a divide-by-zero is the bug, not just its nan result
        turned = ac._rotation_between(a, b)
    assert np.isfinite(turned).all()
    assert turned @ a == pytest.approx(b)
