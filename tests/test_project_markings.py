"""Tests for painting source markings back onto a generated texture.

The failure modes here are quiet ones: a vertically flipped V axis mirrors every marking,
which looks plausible on a roughly symmetric creature; a barycentric tolerance that is too
tight leaves hairline seams along every shared triangle edge, which reads as exactly the
"jagged lines" this whole exercise exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.project_markings import (
    barycentric_fill,
    blend_markings,
    combine_strength,
    dark_core_strength,
    dilate_into_gutter,
    marking_ratio,
    marking_strength,
    rasterize,
    sample_image,
    strength_from_pair,
    uv_to_texel,
)

# --- UV convention ------------------------------------------------------------------


def test_uv_origin_maps_to_the_bottom_left_row():
    """glTF V runs upward while image rows run down: v=0 is the LAST row."""
    texel = uv_to_texel(np.array([[0.0, 0.0]]), 100, 100)
    assert texel[0, 0] == pytest.approx(0.0)
    assert texel[0, 1] == pytest.approx(99.0)


def test_uv_top_of_v_maps_to_the_first_row():
    texel = uv_to_texel(np.array([[0.0, 1.0]]), 100, 100)
    assert texel[0, 1] == pytest.approx(0.0)


def test_uv_handles_non_square_atlases():
    texel = uv_to_texel(np.array([[1.0, 1.0]]), 64, 32)
    assert texel[0, 0] == pytest.approx(63.0)
    assert texel[0, 1] == pytest.approx(0.0)


# --- marking strength ---------------------------------------------------------------


def test_body_colour_is_not_a_marking():
    assert marking_strength(np.array([[200, 200, 200]])).item() == pytest.approx(0.0)


def test_flat_marking_is_full_strength():
    assert marking_strength(np.array([[65, 65, 65]])).item() == pytest.approx(1.0)


def test_eyes_are_protected_from_repainting():
    """Below --low is real anatomy the generator got right; repainting double-darkens."""
    assert marking_strength(np.array([[26, 26, 26]])).item() == pytest.approx(0.0)


def test_strength_ramps_rather_than_switching():
    """A hard switch would put a stair-step edge on every marking."""
    ramp = marking_strength(np.array([[v, v, v] for v in (40, 46, 52, 100)]))
    assert ramp[0] < ramp[1] < ramp[2] <= ramp[3]


# --- rasterisation ------------------------------------------------------------------


def _square_uv_mesh():
    """Two triangles covering the whole atlas, with per-vertex values 0..3."""
    uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return uv, faces


def test_barycentric_fill_covers_a_simple_triangle():
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    cols, _rows, weights = barycentric_fill(triangle, 8, 8)
    assert cols.size > 0
    assert weights.shape[1] == 3
    # every returned weight set sums to one
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)


def test_barycentric_fill_rejects_a_degenerate_triangle():
    """Collapsed UVs happen on generated meshes; they must not raise or fill garbage."""
    triangle = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    cols, _, _ = barycentric_fill(triangle, 8, 8)
    assert cols.size == 0


def test_barycentric_fill_clips_to_the_atlas():
    triangle = np.array([[-50.0, -50.0], [200.0, 0.0], [0.0, 200.0]], dtype=np.float32)
    cols, rows, _ = barycentric_fill(triangle, 16, 16)
    assert cols.min() >= 0 and cols.max() < 16
    assert rows.min() >= 0 and rows.max() < 16


def test_rasterize_fills_the_whole_atlas_for_full_coverage_uvs():
    """Two triangles spanning the UV square must leave no unpainted texel.

    A gap here is a seam, and a seam is a visible jagged line on the asset.
    """
    uv, faces = _square_uv_mesh()
    values = np.ones((4, 1), dtype=np.float32)
    _, covered = rasterize(uv_to_texel(uv, 32, 32), faces, values, 32, 32)
    assert covered.all()


def test_rasterize_interpolates_across_the_triangle():
    uv, faces = _square_uv_mesh()
    # value 0 at the left edge, 1 at the right edge
    values = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)
    baked, _ = rasterize(uv_to_texel(uv, 32, 32), faces, values, 32, 32)
    assert baked[16, 0, 0] < 0.1
    assert baked[16, 31, 0] > 0.9
    assert 0.4 < baked[16, 16, 0] < 0.6


def test_rasterize_leaves_uncovered_texels_marked_uncovered():
    uv = np.array([[0.0, 1.0], [0.25, 1.0], [0.25, 0.75]])
    faces = np.array([[0, 1, 2]])
    values = np.ones((3, 1), dtype=np.float32)
    _, covered = rasterize(uv_to_texel(uv, 32, 32), faces, values, 32, 32)
    assert covered.any()
    assert not covered.all()


# --- blending -----------------------------------------------------------------------


def test_zero_strength_leaves_the_albedo_untouched():
    albedo = np.full((4, 4, 3), 200.0)
    colours = np.zeros((4, 4, 3))
    out = blend_markings(albedo, colours, np.zeros((4, 4)))
    assert np.allclose(out, albedo)


def test_full_strength_takes_the_source_colour():
    albedo = np.full((4, 4, 3), 200.0)
    colours = np.full((4, 4, 3), 60.0)
    out = blend_markings(albedo, colours, np.ones((4, 4)))
    assert np.allclose(out, 60.0)


def test_amount_scales_the_blend():
    albedo = np.full((2, 2, 3), 200.0)
    colours = np.full((2, 2, 3), 100.0)
    out = blend_markings(albedo, colours, np.ones((2, 2)), amount=0.5)
    assert np.allclose(out, 150.0)


def test_blend_stays_in_range():
    albedo = np.full((2, 2, 3), 250.0)
    colours = np.full((2, 2, 3), 300.0)
    out = blend_markings(albedo, colours, np.ones((2, 2)), amount=2.0)
    assert out.max() <= 255.0


# --- per-texel sampling -------------------------------------------------------------
# The bug this guards against: sampling the source per VERTEX and interpolating the
# resulting colour across a face blurs every marking edge to the width of a triangle,
# turning knife-edge artwork into grey smudge. Coordinates interpolate; colours must not.


def test_sample_image_reads_the_expected_pixel():
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    got = sample_image(pixels, np.array([0.0]), np.array([0.0]))
    assert np.array_equal(got[0], pixels[0, 0])


def test_sample_image_clamps_out_of_range_coordinates():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[3, 3] = (9, 9, 9)
    got = sample_image(pixels, np.array([5.0]), np.array([5.0]))
    assert np.array_equal(got[0], np.array([9, 9, 9]))


def test_sample_image_preserves_a_hard_edge():
    """A step in the source must stay a step -- this is the whole point."""
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, 50:] = 255
    u = np.array([0.49, 0.51])
    v = np.array([0.5, 0.5])
    got = sample_image(pixels, u, v)
    assert got[0][0] == 0
    assert got[1][0] == 255


def test_sampling_coordinates_beats_sampling_colours_across_a_triangle():
    """Reproduces the actual defect: one triangle spanning a hard edge in the source.

    Interpolating vertex colours yields a gradient across the face; interpolating
    coordinates and sampling per texel keeps the edge crisp.
    """
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, 50:] = 255

    uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # the triangle's vertices sit either side of the source's edge
    vertex_u = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    vertex_v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    coords, _ = rasterize(
        uv_to_texel(uv, 64, 64), faces,
        np.stack([vertex_u, vertex_v], axis=1), 64, 64,
    )
    sampled = sample_image(pixels, coords[..., 0], coords[..., 1])
    middle = sampled[32, :, 0]
    # crisp: essentially only 0s and 255s, no long ramp of intermediate values
    intermediate = ((middle > 20) & (middle < 235)).sum()
    assert intermediate <= 2

    # the naive alternative, for contrast: interpolate the COLOUR between vertices
    vertex_colour = sample_image(pixels, vertex_u, vertex_v).astype(np.float32)
    blurred, _ = rasterize(uv_to_texel(uv, 64, 64), faces, vertex_colour, 64, 64)
    ramp = blurred[32, :, 0]
    assert ((ramp > 20) & (ramp < 235)).sum() > 10


# --- recovering the mask from the conditioning pair ----------------------------------
# Thresholding on darkness cannot tell paint from the artwork's own shading, and painting
# the shading in makes the asset look like crumpled paper. Softening is what removed the
# markings, so inverting it recovers exactly what was removed and nothing else.


def _pair(body=200, marking=65, shadow=120, lighten=1.0, protect_shadow=False):
    """An artwork with a marking AND a shaded region in the same luminance band."""
    original = np.full((8, 8, 3), body, dtype=np.float32)
    original[2, :] = marking
    original[5, :] = shadow          # shading, NOT paint
    erased = original.copy()
    target = float(body)
    for row, is_paint in ((2, True), (5, not protect_shadow)):
        if is_paint:
            erased[row, :] = original[row, :] + (target - original[row, :]) * lighten
    return original, erased


def test_pair_recovers_full_strength_where_softening_erased():
    original, erased = _pair()
    strength = strength_from_pair(original, erased)
    assert strength[2, 0] == pytest.approx(1.0, abs=0.02)


def test_pair_ignores_shading_that_softening_left_alone():
    """The whole point: a shadow in the marking band must NOT be repainted."""
    original, erased = _pair(protect_shadow=True)
    strength = strength_from_pair(original, erased)
    assert strength[5, 0] == pytest.approx(0.0, abs=0.02)
    # and darkness-thresholding would have wrongly claimed it
    assert marking_strength(original[5:6, 0:1]).item() > 0.5


def test_pair_recovers_partial_softening():
    original, erased = _pair(lighten=0.5)
    strength = strength_from_pair(original, erased)
    assert strength[2, 0] == pytest.approx(0.5, abs=0.03)


def test_pair_leaves_untouched_body_at_zero():
    original, erased = _pair()
    strength = strength_from_pair(original, erased)
    assert strength[0, 0] == pytest.approx(0.0, abs=0.02)


def test_pair_is_bounded():
    original, erased = _pair()
    strength = strength_from_pair(original, erased)
    assert strength.min() >= 0.0 and strength.max() <= 1.0


# --- marking cores -------------------------------------------------------------------
# Softening protects everything below --low so it does not lighten eyes and claws. That
# also spares the darkest heart of each marking, so the pair-derived mask alone brings the
# markings back grey with their black centres missing.


def test_dark_core_is_full_strength_below_low():
    assert dark_core_strength(np.array([[20, 20, 20]]), low=40.0).item() == pytest.approx(1.0)


def test_dark_core_is_zero_well_above_low():
    assert dark_core_strength(np.array([[120, 120, 120]]), low=40.0).item() == pytest.approx(0.0)


def test_dark_core_ramps_rather_than_switching():
    ramp = dark_core_strength(np.array([[v, v, v] for v in (60, 50, 44, 30)]), low=40.0)
    assert ramp[0] < ramp[1] < ramp[2] <= ramp[3]


def test_combine_takes_the_union_not_the_sum():
    """Overlapping masks must not push strength past 1 and overshoot the blend."""
    pair = np.array([1.0, 0.0, 0.6])
    cores = np.array([1.0, 1.0, 0.2])
    combined = combine_strength(pair, cores)
    assert np.allclose(combined, [1.0, 1.0, 0.6])


def test_cores_fill_in_what_the_pair_mask_misses():
    """The real case: a marking whose centre is darker than --low."""
    original = np.full((8, 8, 3), 200, dtype=np.float32)
    original[3, :] = 65     # marking edge, inside the softening band
    original[4, :] = 25     # marking core, below --low, softening left it alone
    erased = original.copy()
    erased[3, :] = 200      # only the edge was lightened

    pair = strength_from_pair(original, erased)
    assert pair[3, 0] == pytest.approx(1.0, abs=0.02)
    assert pair[4, 0] == pytest.approx(0.0, abs=0.02)   # the gap

    combined = combine_strength(pair, dark_core_strength(original, low=40.0))
    assert combined[3, 0] == pytest.approx(1.0, abs=0.02)
    assert combined[4, 0] == pytest.approx(1.0, abs=0.02)   # gap filled
    assert combined[0, 0] == pytest.approx(0.0, abs=0.02)   # body untouched


# --- gutter padding ------------------------------------------------------------------
# A UV atlas is islands separated by unused gutter. Paint that stops at an island edge
# tears when the renderer filters across the boundary, which reads as ragged papery seams.


def test_dilate_grows_the_covered_region():
    values = np.zeros((16, 16, 1), dtype=np.float32)
    covered = np.zeros((16, 16), dtype=bool)
    values[8, 8, 0] = 5.0
    covered[8, 8] = True
    _, grown = dilate_into_gutter(values, covered, radius=2)
    assert grown.sum() > covered.sum()
    assert grown[8, 10]


def test_dilate_carries_the_nearest_value_outward():
    values = np.zeros((16, 16, 1), dtype=np.float32)
    covered = np.zeros((16, 16), dtype=bool)
    values[8, 8, 0] = 5.0
    covered[8, 8] = True
    filled, _ = dilate_into_gutter(values, covered, radius=3)
    assert filled[8, 9, 0] == pytest.approx(5.0)
    assert filled[8, 10, 0] == pytest.approx(5.0)


def test_dilate_is_a_no_op_at_radius_zero():
    values = np.ones((8, 8, 1), dtype=np.float32)
    covered = np.zeros((8, 8), dtype=bool)
    covered[4, 4] = True
    out, grown = dilate_into_gutter(values, covered, radius=0)
    assert np.array_equal(out, values)
    assert np.array_equal(grown, covered)


def test_dilate_handles_a_fully_covered_atlas():
    values = np.ones((8, 8, 1), dtype=np.float32)
    covered = np.ones((8, 8), dtype=bool)
    out, grown = dilate_into_gutter(values, covered, radius=4)
    assert grown.all() and np.array_equal(out, values)


def test_dilate_handles_an_empty_atlas():
    """A projection that saw nothing must not crash the bake."""
    values = np.zeros((8, 8, 1), dtype=np.float32)
    covered = np.zeros((8, 8), dtype=bool)
    out, grown = dilate_into_gutter(values, covered, radius=4)
    assert not grown.any() and np.array_equal(out, values)


# --- ratio transfer ------------------------------------------------------------------
# The artwork is a rendered image, so its pixels carry its own lighting. Painting those
# absolute colours bakes a second lighting pass into a texture the renderer lights again.
# The ratio between artwork and erased-artwork cancels the lighting and leaves the paint.


def test_ratio_is_one_where_nothing_was_marked():
    body = np.full((4, 4, 3), 200, dtype=np.float32)
    assert np.allclose(marking_ratio(body, body), 1.0)


def test_ratio_darkens_where_the_marking_is():
    original = np.full((4, 4, 3), 50, dtype=np.float32)
    erased = np.full((4, 4, 3), 200, dtype=np.float32)
    assert np.allclose(marking_ratio(original, erased), 0.25)


def test_ratio_cancels_the_artwork_lighting():
    """A marking over a lit flank and a shaded flank must give the SAME multiplier.

    Absolute-colour transfer fails this: it would carry the flank's brightness with it.
    """
    lit_body, shaded_body = 240.0, 120.0
    marking_factor = 0.3
    original = np.array([[[lit_body * marking_factor] * 3, [shaded_body * marking_factor] * 3]])
    erased = np.array([[[lit_body] * 3, [shaded_body] * 3]])
    ratio = marking_ratio(original, erased)
    assert ratio[0, 0, 0] == pytest.approx(ratio[0, 1, 0], abs=0.01)
    assert ratio[0, 0, 0] == pytest.approx(marking_factor, abs=0.01)


def test_ratio_never_brightens():
    original = np.full((2, 2, 3), 250, dtype=np.float32)
    erased = np.full((2, 2, 3), 100, dtype=np.float32)
    assert marking_ratio(original, erased).max() <= 1.0


def test_ratio_survives_near_black_without_dividing_by_zero():
    original = np.zeros((2, 2, 3), dtype=np.float32)
    erased = np.zeros((2, 2, 3), dtype=np.float32)
    out = marking_ratio(original, erased)
    assert np.isfinite(out).all()
