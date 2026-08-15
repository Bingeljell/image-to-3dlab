"""Pure contract tests for the Stage-3-only TRELLIS runner."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_stage3.py"
SPEC = importlib.util.spec_from_file_location("trellis_stage3", SCRIPT)
stage3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3)


def _bundle():
    return {
        "shape_slat_feats": torch.zeros((7, 32)),
        "coords": torch.zeros((7, 4), dtype=torch.int32),
        "res": 1024,
        "pipeline_type": "1024_cascade",
        "images": ["source.png"],
    }


def test_valid_bundle_is_accepted():
    stage3.validate_bundle(_bundle())


def test_missing_shape_latent_is_rejected():
    payload = _bundle()
    del payload["shape_slat_feats"]
    with pytest.raises(ValueError, match="shape_slat_feats"):
        stage3.validate_bundle(payload)


def test_row_mismatch_is_rejected():
    payload = _bundle()
    payload["coords"] = torch.zeros((6, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="row mismatch"):
        stage3.validate_bundle(payload)


def test_default_geometry_path_matches_generate_outputs(tmp_path):
    path = tmp_path / "snag_latents.pt"
    assert stage3.default_geometry_decode(path) == tmp_path / "snag_decode.pt"


def test_official_texture_defaults_are_explicit():
    args = argparse.Namespace(
        steps=12,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        rescale_t=3.0,
    )
    assert stage3.sampler_params(args) == {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.6, 0.9),
        "rescale_t": 3.0,
    }


def test_runner_can_target_the_clean_space_port_and_stop_after_sampling():
    source = SCRIPT.read_text()
    assert '"--vendor-root"' in source
    assert 'default="sdpa"' in source
    assert '"metal_flash"' in source
    assert '"--sample-only"' in source
    assert "material decode intentionally skipped" in source
    assert "load_rembg=not has_transparent_alpha" in source
