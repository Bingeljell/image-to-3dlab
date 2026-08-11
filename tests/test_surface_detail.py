"""Tests for the derived surface maps.

These import the shipping functions directly -- never a re-derived copy -- because a
test that re-implements the maths checks something that is not what runs.
"""

import numpy as np
import pytest
from PIL import Image

from scripts.surface_detail import (
    build_maps,
    luminance,
    normal_from_height,
    pack_metallic_roughness,
    roughness_from_luminance,
)


def test_flat_height_gives_flat_normal():
    """A featureless surface must encode as (128, 128, 255) -- straight out."""
    normal = normal_from_height(np.full((16, 16), 0.5, dtype=np.float32), strength=6.0)
    assert normal.shape == (16, 16, 3)
    assert np.all(normal[..., 2] == 255)
    np.testing.assert_allclose(normal[..., 0], 128, atol=1)
    np.testing.assert_allclose(normal[..., 1], 128, atol=1)


def test_slope_tilts_the_normal_and_sign_is_consistent():
    """A ramp rising to the right tilts X one way; the mirrored ramp tilts it the other."""
    ramp = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (8, 1))
    rising = normal_from_height(ramp, strength=6.0)
    falling = normal_from_height(ramp[:, ::-1].copy(), strength=6.0)

    interior = (slice(None), slice(1, -1))
    assert rising[..., 0][interior].mean() < 120
    assert falling[..., 0][interior].mean() > 136
    # Mirroring the input must mirror the encoded X about the neutral 128.
    assert rising[..., 0][interior].mean() + falling[..., 0][interior].mean() == pytest.approx(256, abs=3)


def test_strength_zero_is_a_no_op():
    ramp = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (4, 1))
    normal = normal_from_height(ramp, strength=0.0)
    assert np.all(normal[..., 2] == 255)
    np.testing.assert_allclose(normal[..., :2], 128, atol=1)


def test_normals_are_unit_length():
    rng = np.random.default_rng(0)
    normal = normal_from_height(rng.random((24, 24)).astype(np.float32), strength=4.0)
    vec = normal.astype(np.float32) / 255.0 * 2.0 - 1.0
    np.testing.assert_allclose(np.linalg.norm(vec, axis=2), 1.0, atol=0.02)


def test_normal_from_height_rejects_non_2d():
    with pytest.raises(ValueError):
        normal_from_height(np.zeros((4, 4, 3), dtype=np.float32))


def test_roughness_is_inverted_and_bounded():
    lum = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    rough = roughness_from_luminance(lum, low=0.55, high=0.95)
    assert rough.min() == pytest.approx(0.55, abs=1e-5)
    assert rough.max() == pytest.approx(0.95, abs=1e-5)
    # Darkest texel roughest, brightest smoothest.
    assert rough.flat[0] > rough.flat[-1]


def test_roughness_handles_a_constant_input():
    """A flat albedo must not divide by zero; it collapses to the rough end."""
    rough = roughness_from_luminance(np.full((4, 4), 0.3, dtype=np.float32))
    assert np.all(np.isfinite(rough))
    assert rough.max() == pytest.approx(0.95, abs=1e-5)


def test_roughness_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        roughness_from_luminance(np.zeros((4, 4), dtype=np.float32), low=0.9, high=0.1)


def test_metallic_roughness_packing_follows_gltf():
    """Roughness in G, metallic in B and it must stay zero, or assets turn to chrome."""
    packed = pack_metallic_roughness(np.full((4, 4), 0.5, dtype=np.float32))
    arr = np.asarray(packed)
    assert packed.mode == "RGB"
    assert np.all(arr[..., 2] == 0), "metallic must be 0"
    np.testing.assert_allclose(arr[..., 1], 127, atol=1)


def test_luminance_weights_green_most():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    lum = luminance(rgb)
    assert lum[0, 1] > lum[0, 0] > lum[0, 2]


def test_build_maps_returns_two_usable_images():
    rng = np.random.default_rng(1)
    albedo = Image.fromarray(rng.integers(0, 200, (32, 32, 3), dtype=np.uint8), mode="RGB")
    normal, rough = build_maps(albedo, 6.0, 4, 0.55, 0.95)

    assert normal.size == albedo.size and rough.size == albedo.size
    assert normal.mode == "RGB" and rough.mode == "RGB"
    assert np.all(np.asarray(rough)[..., 2] == 0)
    # A random albedo must produce actual relief, not a flat map.
    assert np.asarray(normal)[..., 0].std() > 1.0


def test_occlusion_packs_into_the_red_channel():
    rough = np.full((4, 4), 0.5, dtype=np.float32)
    ao = np.full((4, 4), 0.25, dtype=np.float32)
    arr = np.asarray(pack_metallic_roughness(rough, ao))
    np.testing.assert_allclose(arr[..., 0], 64, atol=1)
    np.testing.assert_allclose(arr[..., 1], 127, atol=1)
    assert np.all(arr[..., 2] == 0), "metallic must stay 0"


def test_occlusion_size_mismatch_is_rejected():
    with pytest.raises(ValueError):
        pack_metallic_roughness(
            np.zeros((4, 4), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)
        )


def test_ao_strength_zero_fully_opens_the_surface(tmp_path):
    from scripts.surface_detail import load_occlusion

    path = tmp_path / "ao.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path)
    # A pitch-black bake at strength 0 must leave the surface unoccluded, not black.
    np.testing.assert_allclose(load_occlusion(path, (8, 8), 0.0), 1.0, atol=1e-5)
    np.testing.assert_allclose(load_occlusion(path, (8, 8), 1.0), 0.0, atol=1e-5)
    np.testing.assert_allclose(load_occlusion(path, (8, 8), 0.5), 0.5, atol=1e-5)


def test_gloss_mask_overrides_roughness_only_where_set():
    rough = np.full((4, 4), 0.9, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1, 1] = 1.0
    from scripts.surface_detail import apply_gloss

    out = apply_gloss(rough, mask, 0.12)
    assert out[1, 1] == pytest.approx(0.12)
    assert out[0, 0] == pytest.approx(0.9)


def test_gloss_mask_blends_at_soft_edges():
    from scripts.surface_detail import apply_gloss

    out = apply_gloss(
        np.full((2, 2), 1.0, dtype=np.float32), np.full((2, 2), 0.5, dtype=np.float32), 0.0
    )
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_gloss_mask_rejects_bad_input():
    from scripts.surface_detail import apply_gloss

    with pytest.raises(ValueError):
        apply_gloss(np.zeros((2, 2), dtype=np.float32), np.zeros((3, 3), dtype=np.float32), 0.1)
    with pytest.raises(ValueError):
        apply_gloss(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), 1.5)


def test_gloss_mask_flattens_the_normal_there():
    """An eye's painted iris must not be embossed into bumps."""
    rng = np.random.default_rng(3)
    albedo = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8), mode="RGB")
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    normal, _ = build_maps(albedo, 6.0, 4, 0.55, 0.95, None, mask, 0.12)
    arr = np.asarray(normal)
    inside = arr[12:20, 12:20, :2].astype(float)
    np.testing.assert_allclose(inside, 128, atol=2)
    assert arr[:6, :6, 0].std() > 1.0, "outside the mask relief must survive"
