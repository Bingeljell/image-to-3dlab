"""Tests for softening painted markings in a conditioning image."""

import numpy as np
import pytest
from PIL import Image

from scripts.soften_markings import band_weight, luminance, soften


def _art(marking=65, eye=26, body=200, size=32):
    """A body with one flat marking stripe and one near-black eye patch."""
    rgb = np.full((size, size, 3), body, dtype=np.uint8)
    rgb[8:12, :] = marking
    rgb[20:24, 20:24] = eye
    rgba = np.dstack([rgb, np.full((size, size, 1), 255, dtype=np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def test_band_weight_is_one_inside_and_zero_outside():
    lum = np.array([[10.0, 95.0, 240.0]])
    w = band_weight(lum, 40.0, 150.0)
    assert w[0, 0] == pytest.approx(0.0), "below low must be protected"
    assert w[0, 1] == pytest.approx(1.0), "mid-band is fully affected"
    assert w[0, 2] == pytest.approx(0.0), "above high is already body"


def test_band_weight_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        band_weight(np.zeros((2, 2)), 150.0, 40.0)


def test_markings_lighten_but_eyes_are_protected():
    """The whole point: soften the paint, keep the anatomy."""
    out = np.asarray(soften(_art(), lighten=0.6, feather=0.0).convert("RGB")).astype(float)
    assert out[9, 5].mean() > 65 + 10, "marking should have lightened"
    assert out[22, 22].mean() == pytest.approx(26, abs=3), "eye must be untouched"


def test_lighten_zero_is_a_no_op():
    src = _art()
    out = np.asarray(soften(src, lighten=0.0, feather=0.0).convert("RGB")).astype(int)
    np.testing.assert_allclose(out, np.asarray(src.convert("RGB")).astype(int), atol=1)


def test_lighten_one_erases_the_marking():
    out = np.asarray(soften(_art(), lighten=1.0, feather=0.0).convert("RGB")).astype(float)
    assert out[9, 5].mean() == pytest.approx(200, abs=6)


def test_body_is_left_alone():
    out = np.asarray(soften(_art(), lighten=0.8, feather=0.0).convert("RGB")).astype(float)
    assert out[2, 2].mean() == pytest.approx(200, abs=2)


def test_hue_is_preserved():
    """Scaling RGB uniformly must not tint the markings."""
    rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
    rgb[4:8, :] = (90, 60, 30)
    src = Image.fromarray(
        np.dstack([rgb, np.full((16, 16, 1), 255, dtype=np.uint8)]), mode="RGBA"
    )
    out = np.asarray(soften(src, 0.5, feather=0.0).convert("RGB")).astype(float)
    before, after = np.array([90.0, 60.0, 30.0]), out[5, 5]
    np.testing.assert_allclose(after[0] / after[1], before[0] / before[1], rtol=0.03)


def test_alpha_survives():
    rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
    alpha = np.full((8, 8, 1), 77, dtype=np.uint8)
    out = np.asarray(soften(Image.fromarray(np.dstack([rgb, alpha]), "RGBA"), 0.5))
    assert np.all(out[..., 3] == 77)


def test_lighten_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        soften(_art(), lighten=1.5)


def test_transparent_background_is_not_treated_as_body():
    """Background pixels must not drag the body-colour reference around."""
    rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
    rgb[:, :8] = 255  # bright but fully transparent
    alpha = np.full((16, 16, 1), 255, dtype=np.uint8)
    alpha[:, :8] = 0
    rgb[10:14, 8:] = 65
    src = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    out = np.asarray(soften(src, 1.0, feather=0.0).convert("RGB")).astype(float)
    assert out[12, 12].mean() == pytest.approx(200, abs=6)


def test_luminance_weights_green_most():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    lum = luminance(rgb)
    assert lum[0, 1] > lum[0, 0] > lum[0, 2]


def test_edge_ramp_is_a_fixed_width_not_a_fraction_of_the_band():
    """Regression: a proportional ramp left Flicker's 58-79 markings half-softened.

    With low=40 the protected ramp must be done by ~52, so a marking at 65 is fully
    affected regardless of how far away `high` sits.
    """
    lum = np.array([[65.0]])
    assert band_weight(lum, 40.0, 150.0)[0, 0] == pytest.approx(1.0)
    assert band_weight(lum, 40.0, 400.0)[0, 0] == pytest.approx(1.0)


def test_edge_must_be_positive():
    with pytest.raises(ValueError):
        band_weight(np.zeros((2, 2)), 40.0, 150.0, edge=0.0)


# --- protect mask -------------------------------------------------------------------
# Softening assumes every dark region is flat paint. Where darkness is shading of real
# geometry -- an ear hollow -- lightening it removes a depth cue and the generator builds
# a membrane that tears. Measured on Flicker: holes at the dead-front view went
# 1.00% -> 2.52% of body area from softening the ears, while every other angle improved.


def _protect_mask(size=32, rows=slice(8, 10), value=255):
    """White over part of the marking stripe: 'leave this alone'."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[rows, :] = value
    return Image.fromarray(mask, mode="L")


def test_protected_region_is_not_softened():
    art = _art()
    out = soften(art, lighten=1.0, protect=_protect_mask())
    after = luminance(np.asarray(out.convert("RGB")).astype(np.float32))
    # rows 8:10 are protected, rows 10:12 are the same stripe left unprotected.
    # The unprotected half does not reach the full body level because the feather
    # blurs the band weight down near the stripe's own edge -- what matters is that
    # it moves a long way while the protected half does not move at all.
    assert after[8:10, :].mean() == pytest.approx(65, abs=1.0)
    assert after[10:12, :].mean() > 120


def test_no_protect_mask_matches_the_old_behaviour():
    """The flag must be inert when absent, or every earlier result is invalidated."""
    art = _art()
    a = np.asarray(soften(art, lighten=0.5).convert("RGB"))
    b = np.asarray(soften(art, lighten=0.5, protect=None).convert("RGB"))
    assert np.array_equal(a, b)


def test_black_protect_mask_is_a_no_op():
    art = _art()
    black = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    a = np.asarray(soften(art, lighten=0.6).convert("RGB"))
    b = np.asarray(soften(art, lighten=0.6, protect=black).convert("RGB"))
    assert np.array_equal(a, b)


def test_grey_protect_mask_scales_the_effect():
    """Half-white must soften roughly half as much, so feathered edges behave."""
    art = _art()
    full = luminance(np.asarray(soften(art, 1.0).convert("RGB")).astype(np.float32))
    half = luminance(
        np.asarray(
            soften(art, 1.0, protect=_protect_mask(value=128)).convert("RGB")
        ).astype(np.float32)
    )
    original = 65.0
    lifted_full = full[8:10, :].mean() - original
    lifted_half = half[8:10, :].mean() - original
    assert 0.4 < lifted_half / lifted_full < 0.6


def test_mismatched_mask_size_is_rejected_with_a_useful_message():
    """A silently-ignored mask would look like the protection simply did not work."""
    art = _art(size=32)
    wrong = Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L")
    with pytest.raises(ValueError, match="must match"):
        soften(art, lighten=0.5, protect=wrong)
