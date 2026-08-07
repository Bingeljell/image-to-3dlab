"""Tests for the orbiting key-light sweep.

The whole point of the sweep is that it loops seamlessly and holds elevation constant —
if elevation drifts, the highlight climbs instead of travelling, and relief stops reading.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_light_orbit.py"


def _load():
    """Import the real module. Its Blender import is guarded by sys.path, not bpy."""
    spec = importlib.util.spec_from_file_location("light_orbit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_sweep_is_a_full_circle(mod):
    frames = 48
    first = mod.key_light_rotation(1, frames, 50.0)
    last = mod.key_light_rotation(frames + 1, frames, 50.0)
    assert first[2] == pytest.approx(0.0)
    assert last[2] == pytest.approx(2 * math.pi)


def test_loop_is_seamless(mod):
    """Frame N+1 must land on the same direction as frame 1, modulo a full turn."""
    frames = 24
    first = mod.key_light_rotation(1, frames, 50.0)
    wrap = mod.key_light_rotation(frames + 1, frames, 50.0)
    assert math.isclose((wrap[2] - first[2]) % (2 * math.pi), 0.0, abs_tol=1e-9)


def test_elevation_is_held_constant(mod):
    """Elevation must not drift, or the highlight climbs instead of travelling."""
    frames = 36
    elevations = {mod.key_light_rotation(f, frames, 47.0)[0] for f in range(1, frames + 1)}
    assert len(elevations) == 1
    assert elevations.pop() == pytest.approx(math.radians(47.0))


def test_azimuth_advances_monotonically(mod):
    frames = 30
    azimuths = [mod.key_light_rotation(f, frames, 50.0)[2] for f in range(1, frames + 1)]
    assert all(b > a for a, b in itertools.pairwise(azimuths))


def test_rejects_zero_frames(mod):
    with pytest.raises(ValueError):
        mod.key_light_rotation(1, 0, 50.0)
