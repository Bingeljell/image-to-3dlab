"""Tests for the eye-patch painter.

These cover the parts that would otherwise only be checked by looking at a render:
the moment fit that decides where the eye goes, the ring structure of the drawn eye,
the rasteriser that resamples the original atlas into patch space, and the padding
that keeps bilinear filtering from fringing at the patch edge.
"""

from argparse import Namespace

import numpy as np
import pytest

from scripts.paint_eyes import (
    EyeStyle,
    constant_orm,
    dilate,
    eye_basis,
    head_forward,
    local_eye_coordinates,
    locate_eye,
    luminance,
    paint_eye,
    parse_colour,
    patch_coordinates,
    rasterise,
    sample_bilinear,
    shade,
    smoothstep,
    style_from_args,
)


def _patch_with_dark_ellipse(size=128, cx=70.0, cy=54.0, a=30.0, b=16.0, angle=0.35):
    """A pale patch with one dark tilted ellipse, standing in for a painted eye."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ca, sa = np.cos(-angle), np.sin(-angle)
    rx = (ca * (xx - cx) - sa * (yy - cy)) / a
    ry = (sa * (xx - cx) + ca * (yy - cy)) / b
    inside = (rx**2 + ry**2) <= 1.0
    rgb = np.full((size, size, 3), 0.45, np.float32)
    rgb[inside] = 0.03
    return rgb


def test_smoothstep_runs_both_directions():
    assert smoothstep(0.0, 1.0, np.array([-1.0, 0.5, 2.0])).tolist() == [0.0, 0.5, 1.0]
    assert smoothstep(1.0, 0.0, np.array([-1.0, 0.5, 2.0])).tolist() == [1.0, 0.5, 0.0]


def test_locate_eye_recovers_a_known_ellipse():
    fit = locate_eye(_patch_with_dark_ellipse())
    assert fit["cx"] == pytest.approx(70.0, abs=1.5)
    assert fit["cy"] == pytest.approx(54.0, abs=1.5)
    assert fit["angle"] == pytest.approx(0.35, abs=0.05)
    assert fit["a"] == pytest.approx(30.0, rel=0.15), "major axis must track the paint"
    assert fit["b"] == pytest.approx(16.0, rel=0.15)


def test_locate_eye_ignores_a_dark_blob_in_the_corner():
    """A nostril or ear interior at the patch edge must not drag the fit off the eye."""
    rgb = _patch_with_dark_ellipse()
    rgb[0:18, 0:18] = 0.01
    fit = locate_eye(rgb)
    assert fit["cx"] == pytest.approx(70.0, abs=3.0)
    assert fit["cy"] == pytest.approx(54.0, abs=3.0)


def test_locate_eye_rejects_a_patch_with_nothing_dark():
    with pytest.raises(ValueError, match="no dark region"):
        locate_eye(np.full((32, 32, 3), 0.6, np.float32))


RAW = EyeStyle(opening_scale=1.0, min_aspect=1.0)


def test_local_coordinates_put_the_opening_on_the_unit_almond():
    fit = locate_eye(_patch_with_dark_ellipse())
    x, y, aspect = local_eye_coordinates((128, 128), fit, RAW)
    centre = (int(round(fit["cy"])), int(round(fit["cx"])))
    assert abs(x[centre]) < 0.05 and abs(y[centre]) < 0.05
    assert aspect == pytest.approx(30.0 / 16.0, rel=0.15), "a wide eye reports a wide aspect"
    # A point one minor axis away across the tilt must land near |y| = 1, and one
    # major axis along it near |x| = aspect: both axes share the same unit.
    ca, sa = np.cos(fit["angle"]), np.sin(fit["angle"])
    px = int(round(fit["cx"] + ca * fit["a"]))
    py = int(round(fit["cy"] + sa * fit["a"]))
    assert abs(x[py, px]) == pytest.approx(aspect, abs=0.1)
    qx = int(round(fit["cx"] - sa * fit["b"]))
    qy = int(round(fit["cy"] + ca * fit["b"]))
    assert abs(y[qy, qx]) == pytest.approx(1.0, abs=0.08)


def test_opening_scale_grows_the_drawn_eye_past_the_measured_paint():
    """Moments read the dark almond small, so the style scales it back up."""
    fit = locate_eye(_patch_with_dark_ellipse())
    _, tight, _ = local_eye_coordinates((128, 128), fit, RAW)
    _, wide, _ = local_eye_coordinates((128, 128), fit, EyeStyle(opening_scale=1.28))
    assert np.abs(wide).max() == pytest.approx(np.abs(tight).max() / 1.28, rel=1e-5)


def test_min_aspect_keeps_a_round_measurement_almond_shaped():
    fit = locate_eye(_patch_with_dark_ellipse(a=20.0, b=19.0, angle=0.0))
    _, _, aspect = local_eye_coordinates((128, 128), fit, EyeStyle(min_aspect=1.30))
    assert aspect == pytest.approx(1.30)


ASPECT = 1.6


def _painted(style=EyeStyle(), size=201, aspect=ASPECT):
    """Paint one eye on a flat grey face; x and y both span -2..2 opening heights."""
    axis = np.linspace(-2.0, 2.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    base = np.full((size, size, 3), 0.30, np.float32)
    rough = np.full((size, size), 0.85, np.float32)
    rgb, out_rough = paint_eye(x, y, base, rough, style, aspect)
    return x, y, rgb, out_rough


def _at(rgb, u, v=0.0):
    """Sample the painted grid at eye-local (u, v)."""
    size = rgb.shape[0]
    i = int(round((v + 2.0) / 4.0 * (size - 1)))
    j = int(round((u + 2.0) / 4.0 * (size - 1)))
    return rgb[i, j]


def test_paint_eye_builds_pupil_iris_and_sclera_in_that_order():
    style = EyeStyle(highlight_strength=0.0, lid_shadow=0.0)
    _, _, rgb, _ = _painted(style)
    pupil = _at(rgb, 0.0)
    iris = _at(rgb, 0.62)
    # Past the iris but still inside a wide opening: the eye corner, where sclera shows.
    sclera = _at(rgb, 1.25)
    assert luminance(pupil) < luminance(iris), "pupil is the darkest part of the eye"
    assert iris[0] > iris[2], "the iris is amber, so red beats blue"
    assert luminance(sclera) > luminance(iris), "sclera at the eye corner is lighter"


def test_paint_eye_keeps_the_iris_circular_on_a_wide_opening():
    """The iris must be a disc on the face, not stretched out to the eye corners."""
    style = EyeStyle(highlight_strength=0.0, lid_shadow=0.0)
    _, _, rgb, _ = _painted(style, size=401, aspect=2.0)
    edge = style.iris_scale
    assert luminance(_at(rgb, edge * 0.8)) < luminance(_at(rgb, edge * 1.25)), (
        "horizontally the iris must end at its own radius, not at the lid"
    )
    assert luminance(_at(rgb, 0.0, edge * 0.8)) < luminance(_at(rgb, edge * 0.8)) + 0.4


def test_paint_eye_leaves_the_face_outside_the_lid_line_alone():
    style = EyeStyle()
    _, _, rgb, rough = _painted(style)
    corner = rgb[0, 0]
    assert corner == pytest.approx([0.30, 0.30, 0.30], abs=1e-6)
    assert rough[0, 0] == pytest.approx(0.85, abs=1e-6)


def test_paint_eye_makes_the_eye_glossy_and_the_lid_line_dark():
    style = EyeStyle()
    _, _, rgb, rough = _painted(style)
    centre = rgb.shape[0] // 2
    assert rough[centre, centre] == pytest.approx(style.eye_roughness, abs=0.02)
    # Just above the opening the liner darkens the original face colour.
    assert luminance(_at(rgb, 0.0, 1.06)) < 0.30
    assert rough[centre, centre] < rough[0, 0], "the eye is glossier than the fur"


def test_paint_eye_puts_the_highlight_where_the_style_asks():
    style = EyeStyle(lid_shadow=0.0)
    _, _, rgb, _ = _painted(style)
    size = rgb.shape[0]

    def index(u, v):
        return (
            int(round((v + 2.0) / 4.0 * (size - 1))),
            int(round((u + 2.0) / 4.0 * (size - 1))),
        )

    hx, hy = style.highlight_at
    here = rgb[index(hx * style.iris_scale, style.iris_rise + hy * style.iris_scale)]
    mirrored = rgb[index(-hx * style.iris_scale, style.iris_rise - hy * style.iris_scale)]
    assert luminance(here) > luminance(mirrored) + 0.2


def test_paint_eye_is_stable_when_run_over_its_own_output():
    """Re-running must not stack liner on liner or brighten the glint twice."""
    style = EyeStyle()
    axis = np.linspace(-2.0, 2.0, 121, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    base = np.full((121, 121, 3), 0.30, np.float32)
    rough = np.full((121, 121), 0.85, np.float32)
    once_rgb, once_rough = paint_eye(x, y, base, rough, style, ASPECT)
    twice_rgb, twice_rough = paint_eye(x, y, once_rgb, once_rough, style, ASPECT)
    inside = (
        np.abs(x / ASPECT) ** style.opening_power + np.abs(y) ** style.opening_power
    ) < 0.8
    assert np.allclose(once_rgb[inside], twice_rgb[inside], atol=1e-6)
    assert np.allclose(once_rough[inside], twice_rough[inside], atol=1e-6)


def test_rasterise_interpolates_across_a_known_triangle():
    tri_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], np.float32)
    tri_attr = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], np.float32)
    attr, mask = rasterise(tri_uv, tri_attr, 16, 16)
    assert mask[0, 0] and mask[0, 15] and mask[15, 0]
    assert not mask[15, 15], "the far corner is outside the triangle"
    # A texel centre maps back to its own patch coordinate for an identity mapping.
    assert attr[4, 2] == pytest.approx([(2 + 0.5) / 16, (4 + 0.5) / 16], abs=1e-5)


def test_rasterise_covers_a_split_quad_without_a_seam():
    quad = np.array(
        [
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
            [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9]],
        ],
        np.float32,
    )
    attr = np.zeros((2, 3, 1), np.float32)
    _, mask = rasterise(quad, attr, 32, 32)
    interior = mask[6:26, 6:26]
    assert interior.all(), "the shared diagonal must not leave uncovered texels"


def test_dilate_grows_the_island_and_copies_its_colour():
    image = np.zeros((9, 9, 3), np.float32)
    mask = np.zeros((9, 9), bool)
    image[4, 4] = (0.2, 0.4, 0.6)
    mask[4, 4] = True
    out, cover = dilate(image, mask, rounds=2)
    assert cover[2:7, 2:7].all(), "two rounds reach two rings out"
    assert not cover[0, 0]
    assert out[3, 4] == pytest.approx([0.2, 0.4, 0.6], abs=1e-6)


def test_dilate_stops_once_everything_is_covered():
    image = np.ones((4, 4, 1), np.float32)
    out, cover = dilate(image, np.ones((4, 4), bool), rounds=5)
    assert cover.all()
    assert out == pytest.approx(np.ones((4, 4, 1), np.float32))


def test_sample_bilinear_reads_texel_centres_exactly():
    image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    uv = np.array([[(2 + 0.5) / 4, (1 + 0.5) / 4]], np.float32)
    assert sample_bilinear(image, uv)[0, 0] == pytest.approx(image[1, 2, 0])


def test_sample_bilinear_blends_between_neighbours():
    image = np.array([[[0.0], [1.0]], [[0.0], [1.0]]], np.float32)
    uv = np.array([[0.5, 0.25]], np.float32)
    assert sample_bilinear(image, uv)[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_eye_basis_is_orthonormal_and_upright():
    right, up = eye_basis(np.array([0.2, -1.0, 0.05]))
    assert np.dot(right, up) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert up[2] > 0.9, "the eye's up must stay close to world up"


def test_patch_coordinates_centre_maps_to_the_middle():
    centre = np.array([1.0, 2.0, 3.0])
    right, up = eye_basis(np.array([0.0, -1.0, 0.0]))
    at_centre = patch_coordinates(centre[None, :], centre, right, up, 0.03)
    assert at_centre[0] == pytest.approx([0.5, 0.5])
    offset = patch_coordinates((centre + up * 0.03)[None, :], centre, right, up, 0.03)
    assert offset[0] == pytest.approx([0.5, 1.0])


# --- the parts that let this run on a creature it has never seen -------------


def test_parse_colour_reads_hex_with_or_without_a_hash():
    assert parse_colour("#ff8000") == pytest.approx((1.0, 128 / 255, 0.0))
    assert parse_colour("ff8000") == pytest.approx((1.0, 128 / 255, 0.0))


def test_parse_colour_reads_floats_and_bytes():
    assert parse_colour("0.5,0.25,0.125") == pytest.approx((0.5, 0.25, 0.125))
    assert parse_colour("255,128,0") == pytest.approx((1.0, 128 / 255, 0.0))


@pytest.mark.parametrize("bad", ["#abc", "1,2", "not a colour", ""])
def test_parse_colour_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_colour(bad)


def test_shade_darkens_without_leaving_the_range():
    assert shade((0.8, 0.4, 0.2), 0.5) == pytest.approx((0.4, 0.2, 0.1))
    assert shade((0.8, 0.4, 0.2), 4.0) == pytest.approx((1.0, 1.0, 0.8))


def _args(**over):
    base = dict(
        iris_scale=None, pupil_scale=None, iris=None, sclera=None, pupil=None
    )
    base.update(over)
    return Namespace(**base)


def test_one_iris_colour_derives_the_rim_and_the_limbal_ring():
    """A caller gives one colour; three that agree is the tool's job, not theirs."""
    style = style_from_args(_args(iris="#6f9ec4"))
    assert style.iris_inner_rgb == pytest.approx(parse_colour("#6f9ec4"))
    assert luminance(np.array(style.iris_outer_rgb)) < luminance(
        np.array(style.iris_inner_rgb)
    )
    assert luminance(np.array(style.limbal_rgb)) < luminance(
        np.array(style.iris_outer_rgb)
    ), "the limbal ring is the darkest of the three"


def test_style_from_args_leaves_untouched_fields_at_their_defaults():
    style = style_from_args(_args(pupil_scale=0.22))
    assert style.pupil_scale == 0.22
    assert style.sclera_rgb == EyeStyle().sclera_rgb
    assert style.iris_inner_rgb == EyeStyle().iris_inner_rgb


def test_constant_orm_packs_roughness_and_metallic_the_gltf_way():
    """Assets whose material carries plain numbers instead of a packed map."""
    orm = constant_orm((4, 6), 0.7, 0.1)
    assert orm.shape == (4, 6, 3)
    assert orm[..., 1] == pytest.approx(0.7), "green is roughness"
    assert orm[..., 2] == pytest.approx(0.1), "blue is metallic"


def test_head_forward_is_perpendicular_to_the_eye_line_and_points_front():
    centres = np.array([[-0.08, -0.41, 0.63], [0.01, -0.36, 0.62]])
    forward = head_forward(centres)
    eye_line = centres[1] - centres[0]
    assert np.dot(forward, eye_line) == pytest.approx(0.0, abs=1e-9)
    assert forward[1] < 0, "a glTF import faces -Y"
    assert np.linalg.norm(forward) == pytest.approx(1.0)


def test_head_forward_does_not_care_which_eye_is_given_first():
    centres = np.array([[-0.08, -0.41, 0.63], [0.01, -0.36, 0.62]])
    assert head_forward(centres) == pytest.approx(head_forward(centres[::-1]))
