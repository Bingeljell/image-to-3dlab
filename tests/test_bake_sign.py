"""Tests for resolving the sign of a baked tangent-space normal map.

The defect these describe cost an afternoon of wrong hypotheses. Baking the Snag's
19.2M-face decode onto its 39k retopo produced rainbow confetti, and the first four
explanations — a shattered bake source, flat shading, ray distance, the Metal GPU — were
all measured and ruled out. A ray-cast probe inside Blender found the rays landing on the
right surface, a median 0.0013 away, and coming back with the *normal reversed* in 48.9%
of cases, upper quartile 164 degrees.

The cause is the decode being non-manifold: winding cannot propagate across it, so no
amount of per-component orientation repair converges. The fix does not need the mesh at
all — a tangent-space normal is a deviation from the surface, so its Z is positive by
construction and a negative one can only be a reversed face.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_bake_normals.py"


def _load():
    spec = importlib.util.spec_from_file_location("blender_bake_normals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load()

FLAT = (128, 128, 255)


def _encode(normal):
    """Encode a unit normal the way a baked PNG stores it."""
    return np.clip((np.asarray(normal, dtype=np.float32) + 1) / 2 * 255, 0, 255).astype(
        np.uint8
    )


def test_a_clean_map_is_left_alone():
    flat = np.tile(np.array(FLAT, dtype=np.uint8), (4, 4, 1))
    resolved, flipped = bn.resolve_tangent_sign(flat)
    assert flipped == 0.0
    assert np.array_equal(resolved, flat)


def test_a_reversed_texel_is_negated_back():
    true_normal = np.array([0.3, -0.4, np.sqrt(1 - 0.09 - 0.16)], dtype=np.float32)
    reversed_pixel = _encode(-true_normal)

    image = np.tile(reversed_pixel, (2, 2, 1))
    resolved, flipped = bn.resolve_tangent_sign(image)

    assert flipped == 1.0
    # Back to the true normal, within the rounding the 8-bit encoding costs.
    assert np.allclose(resolved[0, 0], _encode(true_normal), atol=1)


def test_the_flipped_fraction_reports_source_winding():
    """Half reversed is what a non-manifold source looks like; the number is the signal."""
    good = np.tile(np.array(FLAT, dtype=np.uint8), (2, 4, 1))
    bad = np.tile(_encode([0.0, 0.0, -1.0]), (2, 4, 1))
    image = np.concatenate([good, bad], axis=0)

    _resolved, flipped = bn.resolve_tangent_sign(image)
    assert flipped == pytest.approx(0.5)


def test_every_texel_points_out_of_the_surface_afterwards():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(16, 16, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    image = _encode(normals)

    resolved, _flipped = bn.resolve_tangent_sign(image)
    assert (resolved[..., 2] >= 127).all()


def test_the_input_array_is_not_modified_in_place():
    """The caller may still want the raw bake — `--keep-sign` exists for exactly that."""
    image = np.tile(_encode([0.0, 0.0, -1.0]), (2, 2, 1))
    original = image.copy()

    bn.resolve_tangent_sign(image)
    assert np.array_equal(image, original)


def test_an_untouched_texel_comes_back_exactly():
    """Decoding and re-encoding lands a hair under some values; rounding keeps them."""
    values = np.arange(256, dtype=np.uint8)
    image = np.stack([values, values[::-1], np.full(256, 255, dtype=np.uint8)], axis=-1)
    image = image.reshape(16, 16, 3)

    resolved, flipped = bn.resolve_tangent_sign(image)
    assert flipped == 0.0
    assert np.array_equal(resolved, image)
