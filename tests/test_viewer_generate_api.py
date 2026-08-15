"""Cheap, torch-free tests for the viewer Generate API contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "viewer" / "generate_api.py"
SPEC = importlib.util.spec_from_file_location("viewer_generate_api", MODULE_PATH)
api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(api)


def test_parse_tqdm_carriage_return_line():
    event = api.parse_tqdm_line("Sampling shape SLat:  33%|███▎ | 4/12 [07:59<15:58, 119.80s/it]")
    assert event == {
        "label": "Sampling shape SLat", "pct": 33, "step": 4, "total": 12,
        "elapsed": 479.0, "remain": 958.0, "s_per_it": 119.8,
    }


def test_shape_slat_passes_are_disambiguated(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb", {})
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 0) == "shape_slat_coarse"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 4) == "shape_slat_coarse"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 0) == "shape_slat_fine"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 4) == "shape_slat_fine"


@pytest.mark.parametrize("payload", [
    {"resolution": "2048"},
    {"texture_size": 512},
    {"decimation_target": 0},
    {"allow_rembg": "yes"},
])
def test_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api.validate_settings(payload)


def test_validate_settings_applies_demo_defaults():
    settings = api.validate_settings({})
    assert settings == api.DEFAULT_SETTINGS
    assert settings is not api.DEFAULT_SETTINGS


def test_overall_progress_is_stage_weighted():
    assert api._overall_pct("load", 100) == 14
    assert api._overall_pct("bake", 50) == 93
