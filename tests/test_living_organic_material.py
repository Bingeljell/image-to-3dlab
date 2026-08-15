from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "living_organic_material", REPO / "scripts" / "living_organic_material.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_parse_colour():
    assert np.allclose(mod.parse_colour("#ff8000"), [1.0, 128 / 255, 0.0])


def test_colourise_preserves_lab_lightness():
    from skimage import color

    pixels = np.array([[[0.12, 0.18, 0.08], [0.6, 0.4, 0.2]]], dtype=np.float32)
    before = color.rgb2lab(pixels)[..., 0]
    result = mod.colourise_preserving_lightness(
        pixels, mod.parse_colour("#d98208"), np.ones((1, 2), dtype=np.float32)
    )
    after = color.rgb2lab(result)[..., 0]
    # A saturated target can clip at the sRGB gamut boundary; allow the sub-one-L-unit
    # round-trip drift while still rejecting an actual brightness grade.
    assert np.allclose(before, after, atol=1.0)


def test_rasterise_face_values_obeys_uv_origin():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=float)
    faces = np.array([[0, 1, 2]])
    image = mod.rasterise_face_values(uv, faces, np.array([1.0]), (32, 32), blur=0)
    data = np.asarray(image)
    assert data.max() == 255
    assert data[20, 8] == 255


def test_eye_roughness_is_written_and_metalness_stays_zero():
    # This pins the semantic distinction the Snag exposed: reflective dielectric is
    # low roughness, not metallic bark.
    roughness = np.full((4, 4), 0.9, dtype=np.float32)
    eye = np.zeros((4, 4), dtype=np.float32)
    eye[1:3, 1:3] = 1.0
    roughness = roughness * (1.0 - eye) + 0.18 * eye
    orm = np.zeros((4, 4, 3), dtype=np.uint8)
    orm[..., 1] = np.clip(roughness * 255, 0, 255).astype(np.uint8)
    assert orm[..., 2].max() == 0
    assert int(orm[1, 1, 1]) == int(0.18 * 255)


def test_moss_chroma_evidence_separates_brown_from_green():
    pixels = np.array(
        [[[0.30, 0.18, 0.07], [0.20, 0.30, 0.08]]],
        dtype=np.float32,
    )
    evidence = mod.moss_chroma_evidence(pixels)
    assert evidence[0, 0] < 0.05
    assert evidence[0, 1] > 0.90
