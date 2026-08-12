"""Tests for the source-vs-render comparison.

This harness exists because every other metric here measures the mesh against itself.
If its camera convention drifts, or its crop stops finding the subject, the comparison
silently starts comparing two differently-framed images and the eye gets a wrong answer
-- which is the exact failure mode that cost this project two days.

The camera assertions pin the convention to the views ``blender_render_asset`` already
uses, so the two scripts cannot disagree about where "front" is.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_to_source.py"


def _load():
    """Import the real module, not a re-derived copy of it."""
    spec = importlib.util.spec_from_file_location("compare_to_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load()
Image = pytest.importorskip("PIL.Image")


def _blob(size=(200, 200), box=None, colour=(200, 180, 160, 255)):
    """An RGBA image with an opaque rectangle on a transparent field."""
    if box is None:
        box = (
            int(size[0] * 0.30),
            int(size[1] * 0.20),
            int(size[0] * 0.70),
            int(size[1] * 0.80),
        )
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            image.putpixel((x, y), colour)
    return image


# --- camera convention -------------------------------------------------------------


def test_azimuth_zero_is_in_front_at_negative_x():
    """Matches ``front_xneg`` in blender_render_asset, so 'front' means one thing."""
    x, y, z = compare.orbit_position(0.0, 0.0, 10.0)
    assert x == pytest.approx(-10.0)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_azimuth_ninety_is_the_yneg_profile():
    x, y, z = compare.orbit_position(90.0, 0.0, 10.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-10.0)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_default_azimuth_reproduces_the_existing_three_quarter_view():
    """The old view sat at (-0.9d, -0.45d): twice as far along -X as along -Y."""
    x, y, _ = compare.orbit_position(26.565, 0.0, 1.0)
    assert x / y == pytest.approx(2.0, rel=1e-3)


def test_elevation_ninety_looks_straight_down():
    x, y, z = compare.orbit_position(0.0, 90.0, 7.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(7.0)


def test_distance_is_preserved_at_every_angle():
    for azimuth in (0, 27, 90, 180, -140):
        for elevation in (-20, 0, 8, 45):
            x, y, z = compare.orbit_position(azimuth, elevation, 5.0)
            assert math.sqrt(x * x + y * y + z * z) == pytest.approx(5.0)


def test_target_z_raises_the_camera_without_changing_the_orbit():
    flat = compare.orbit_position(30.0, 0.0, 4.0)
    lifted = compare.orbit_position(30.0, 0.0, 4.0, target_z=1.5)
    assert lifted[0] == pytest.approx(flat[0])
    assert lifted[1] == pytest.approx(flat[1])
    assert lifted[2] == pytest.approx(flat[2] + 1.5)


# --- finding and framing the subject -----------------------------------------------


def test_subject_bbox_finds_the_opaque_region():
    assert compare.subject_bbox(_blob()) == (60, 40, 140, 160)


def test_subject_bbox_ignores_almost_transparent_pixels():
    """Anti-aliased cut-outs carry a faint alpha fringe that must not widen the crop."""
    image = _blob()
    image.putpixel((5, 5), (255, 255, 255, 4))
    assert compare.subject_bbox(image) == (60, 40, 140, 160)


def test_subject_bbox_treats_an_image_without_alpha_as_fully_opaque():
    opaque = Image.new("RGB", (40, 30), (10, 20, 30))
    assert compare.subject_bbox(opaque) == (0, 0, 40, 30)


def test_crop_to_subject_removes_the_empty_field():
    cropped = compare.crop_to_subject(_blob(), pad_frac=0.0)
    assert cropped.size == (80, 120)


def test_crop_to_subject_pads_proportionally_to_the_subject():
    cropped = compare.crop_to_subject(_blob(), pad_frac=0.1)
    assert cropped.size == (80 + 24, 120 + 24)


def test_crop_to_subject_does_not_run_off_the_canvas():
    """A subject touching the edge must clamp, not raise or wrap."""
    flush = _blob(size=(100, 100), box=(0, 0, 100, 100))
    assert compare.crop_to_subject(flush, pad_frac=0.2).size == (100, 100)


def test_scale_to_height_preserves_aspect_ratio():
    scaled = compare.scale_to_height(_blob(size=(200, 100)), 300)
    assert scaled.size == (600, 300)


# --- the panels themselves ----------------------------------------------------------


def test_identical_shapes_overlay_as_agreement_only():
    """A perfect match must show no magenta and no green anywhere."""
    blob = _blob()
    overlay = compare.silhouette_overlay(blob, blob, height=120)
    present = {pixel for pixel in overlay.getdata()}
    assert compare.SOURCE_ONLY not in present
    assert compare.RENDER_ONLY not in present
    assert compare.BOTH in present


def test_a_narrower_render_shows_source_only_colour():
    """Coming out too thin must read as magenta -- the colour meaning 'source had this'."""
    broad = _blob(size=(200, 200), box=(40, 20, 160, 180))
    narrow = _blob(size=(200, 200), box=(85, 20, 115, 180))
    overlay = compare.silhouette_overlay(broad, narrow, height=200)
    assert compare.SOURCE_ONLY in set(overlay.getdata())


def test_a_wider_render_shows_render_only_colour():
    """The converse: a limb that ballooned reads as green."""
    narrow = _blob(size=(200, 200), box=(85, 20, 115, 180))
    broad = _blob(size=(200, 200), box=(40, 20, 160, 180))
    overlay = compare.silhouette_overlay(narrow, broad, height=200)
    assert compare.RENDER_ONLY in set(overlay.getdata())


def test_uniform_scale_is_normalised_away_so_only_proportion_is_judged():
    """Camera distance is arbitrary; a half-size copy is not a defect.

    This is the behaviour that makes magenta mean "wrong shape" rather than "wrong
    size" -- without it every panel would be flooded with colour by framing alone.

    The match is not pixel-perfect and cannot be: one mask is scaled up to the common
    height and the other down, and up-scaling dilates a thresholded edge by about a
    pixel. So the guarantee is that disagreement is confined to a hairline at the
    outline, not that it is empty. On a real 900px panel this is under half a percent
    and invisible; the bar here is deliberately at the small end where it is worst.
    """
    import numpy as np

    big = _blob(size=(400, 400), box=(100, 80, 300, 320))
    small = _blob(size=(200, 200), box=(50, 40, 150, 160))
    overlay = np.array(compare.silhouette_overlay(big, small, height=200))

    def count(colour):
        return int((overlay == np.array(colour)).all(axis=2).sum())

    agreed = count(compare.BOTH)
    disagreed = count(compare.SOURCE_ONLY) + count(compare.RENDER_ONLY)
    assert agreed > 0
    # A one-pixel fringe around a ~150x200 shape is ~700px against ~30000px of body.
    assert disagreed / agreed < 0.05


def test_a_genuine_proportion_error_dwarfs_the_resampling_fringe():
    """Guards the tolerance above: a real defect must be an order of magnitude bigger.

    Without this, loosening the fringe tolerance could quietly swallow the very errors
    the overlay exists to reveal.
    """
    import numpy as np

    broad = _blob(size=(200, 200), box=(40, 20, 160, 180))
    narrow = _blob(size=(200, 200), box=(85, 20, 115, 180))
    overlay = np.array(compare.silhouette_overlay(broad, narrow, height=200))

    def count(colour):
        return int((overlay == np.array(colour)).all(axis=2).sum())

    disagreed = count(compare.SOURCE_ONLY) + count(compare.RENDER_ONLY)
    assert disagreed / count(compare.BOTH) > 0.5


def test_contact_sheet_places_every_panel():
    panels = [(_blob(), "source"), (_blob(), "ours"), (_blob(), "grey")]
    sheet = compare.contact_sheet(panels, height=100, gap=10, label_height=20)
    # three panels 66px wide after cropping to 80x120 and scaling to height 100
    assert sheet.height == 100 + 20 + 20
    assert sheet.width > 3 * 60


def test_contact_sheet_survives_a_panel_with_no_subject():
    """An all-transparent render (bad camera, empty frame) must not crash the sheet."""
    empty = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    sheet = compare.contact_sheet([(_blob(), "source"), (empty, "empty")], height=80)
    assert sheet.width > 0


# --- the generated Blender code -----------------------------------------------------


def test_render_is_transparent_so_the_crop_can_find_the_subject():
    """If this regresses, every crop silently becomes the full frame."""
    code = compare.blender_code(Path("a.glb"), Path("/out"), "x", [27.0], 8.0, "studio", grey=False)
    assert "film_transparent = True" in code


def test_grey_mode_clears_existing_materials():
    code = compare.blender_code(Path("a.glb"), Path("/out"), "x", [27.0], 8.0, "studio", grey=True)
    assert "obj.data.materials.clear()" in code
    assert "grey_mode = True" in code


def test_textured_mode_keeps_the_asset_materials():
    code = compare.blender_code(Path("a.glb"), Path("/out"), "x", [27.0], 8.0, "studio", grey=False)
    assert "grey_mode = False" in code


def test_every_requested_angle_reaches_the_render_loop():
    code = compare.blender_code(Path("a.glb"), Path("/out"), "x", [0.0, 27.0, 90.0], 8.0, "studio", grey=False)
    assert "angles = [0.0, 27.0, 90.0]" in code


def test_generated_code_compiles():
    """Catches the syntax error that would otherwise surface only inside Blender."""
    code = compare.blender_code(Path("a.glb"), Path("/out"), "x", [27.0], 8.0, "studio", grey=True)
    compile(code, "<blender>", "exec")
