"""Tests for softening a baked normal map.

Baked at full strength the map reads crunchy, so the dose has to be adjustable. glTF's
`normalTexture.scale` would be the natural knob but trimesh drops it on export, so the
blend happens in the image — and must therefore be correct here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "attach_normal_map.py"


def _load():
    spec = importlib.util.spec_from_file_location("attach_normal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _img(rgb, size=(4, 4)):
    return Image.fromarray(np.full((size[1], size[0], 3), rgb, dtype=np.uint8), "RGB")


def test_full_strength_is_the_untouched_bake(mod):
    src = _img((200, 60, 240))
    assert np.array_equal(np.asarray(mod.soften(src, 1.0)), np.asarray(src))


def test_zero_strength_is_perfectly_flat(mod):
    """Every pixel must become the neutral normal, or the surface still deviates."""
    out = np.asarray(mod.soften(_img((200, 60, 240)), 0.0))
    assert (out == np.array(mod.FLAT_NORMAL, dtype=np.uint8)).all()


def test_half_strength_lands_midway(mod):
    out = np.asarray(mod.soften(_img((228, 28, 255)), 0.5)).astype(int)
    expected = [(228 + 128) // 2, (28 + 128) // 2, (255 + 255) // 2]
    assert np.allclose(out[0, 0], expected, atol=1)


def test_an_already_flat_map_is_unchanged_at_any_strength(mod):
    """Flat pixels encode no deviation, so scaling them must be a no-op."""
    flat = _img(mod.FLAT_NORMAL)
    for s in (0.0, 0.3, 1.0):
        assert np.array_equal(np.asarray(mod.soften(flat, s)), np.asarray(flat))


def test_output_stays_in_range(mod):
    """Clipping matters: an over-1 strength must not wrap around to garbage."""
    out = np.asarray(mod.soften(_img((250, 5, 255)), 4.0))
    assert out.min() >= 0 and out.max() <= 255


def test_negative_strength_is_rejected(mod):
    with pytest.raises(ValueError):
        mod.soften(_img((200, 60, 240)), -0.5)
