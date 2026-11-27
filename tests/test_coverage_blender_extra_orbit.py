"""Check the orbit camera's coordinates and scene settings."""

import blender_orbit_camera as orbit
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("target", "azimuth", "elevation", "radius", "location", "direction"),
    [
        ((1, 2, 3), 0, 0, 5, (1, -8, 2), (0, 1, 0)),
        ((1, 2, 3), 90, 0, 5, (6, -3, 2), (-1, 0, 0)),
        ((1, 2, 3), 0, 90, 5, (1, -3, 7), (0, 0, -1)),
    ],
)
def test_orbit_pose_converts_target_and_aims_at_it(
    target, azimuth, elevation, radius, location, direction
):
    position, toward = orbit.camera_pose(target, azimuth, elevation, radius)
    assert position == pytest.approx(location)
    assert np.array(toward) / np.linalg.norm(toward) == pytest.approx(direction)
