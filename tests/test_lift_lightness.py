"""Tests for the lightness lift -- the complement of the chroma grade."""

import numpy as np
import pytest
from PIL import Image

from scripts.lift_lightness import lift


def _solid(rgb, size=8):
    return Image.fromarray(np.tile(np.array(rgb, dtype=np.uint8), (size, size, 1)), mode="RGB")


def test_gain_one_is_a_no_op():
    src = _solid((90, 70, 30))
    out = np.asarray(lift(src, 1.0).convert("RGB")).astype(int)
    np.testing.assert_allclose(out, np.asarray(src).astype(int), atol=1)


def test_gain_above_one_brightens():
    src = _solid((90, 70, 30))
    out = np.asarray(lift(src, 1.4).convert("RGB")).astype(float)
    assert out.mean() > np.asarray(src).astype(float).mean()


def test_hue_is_preserved():
    """The point of working in LAB: brightening must not shift the colour."""
    src = _solid((120, 60, 30))
    before = np.asarray(src).astype(float)[0, 0]
    after = np.asarray(lift(src, 1.35).convert("RGB")).astype(float)[0, 0]
    # Ratios between channels stand in for hue; they should barely move.
    np.testing.assert_allclose(after[0] / after[1], before[0] / before[1], rtol=0.08)


def test_alpha_survives():
    rgba = np.dstack([
        np.full((4, 4, 3), 80, dtype=np.uint8),
        np.full((4, 4, 1), 123, dtype=np.uint8),
    ])
    out = np.asarray(lift(Image.fromarray(rgba, mode="RGBA"), 1.3))
    assert out.shape[2] == 4
    assert np.all(out[..., 3] == 123)


def test_lightness_is_clamped_not_wrapped():
    """A big gain on a bright texel must saturate to white, never wrap to black."""
    out = np.asarray(lift(_solid((250, 250, 250)), 4.0).convert("RGB")).astype(float)
    assert out.min() > 240


def test_black_stays_black():
    out = np.asarray(lift(_solid((0, 0, 0)), 2.0).convert("RGB")).astype(float)
    assert out.max() < 2, "multiplying L cannot lift a texel that has no lightness"


def test_saturation_is_preserved_not_washed_out():
    """The bug a LAB-L multiply had: brightening must not desaturate.

    Saturation here is (max-min)/max on the sRGB texel, which is what the eye reads as
    'how colourful is this'. Raising LAB's L with a/b held fixed drops it noticeably.
    """
    src = _solid((120, 60, 30))

    def sat(img):
        a = np.asarray(img.convert("RGB")).astype(float)[0, 0]
        return (a.max() - a.min()) / max(a.max(), 1e-6)

    assert sat(lift(src, 1.35)) == pytest.approx(sat(src), rel=0.02)


def test_negative_gain_is_rejected():
    with pytest.raises(ValueError):
        lift(_solid((10, 10, 10)), -1.0)


def test_gain_zero_goes_black():
    out = np.asarray(lift(_solid((200, 100, 50)), 0.0).convert("RGB"))
    assert out.max() == 0
