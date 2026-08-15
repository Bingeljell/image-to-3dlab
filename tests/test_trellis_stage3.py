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


def test_resume_texture_latent_flag_is_available(monkeypatch, tmp_path):
    latent = tmp_path / "cached_tex.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "trellis_stage3.py",
            "shape.pt",
            "material.pt",
            "--texture-seed",
            "7",
            "--resume-texture-latent",
            str(latent),
        ],
    )
    args = stage3.parse_args()
    assert args.resume_texture_latent == latent


def test_subdivision_guides_reproduce_frozen_final_coordinates():
    shape_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 3, 3, 3]], dtype=torch.int32
    )
    final_coords = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [16, 16, 16], [31, 31, 31]],
        dtype=torch.int32,
    )
    guides = stage3.derive_subdivision_guide_tensors(
        final_coords, shape_coords, resolution=64, levels=4
    )
    assert len(guides) == 4
    assert guides[0][0].shape == (3, 8)
    assert guides[-1][1].shape[0] == 3

    parent = guides[0][1]
    for features, coords in guides:
        assert torch.equal(coords, parent)
        row, subindex = features.nonzero(as_tuple=True)
        bits = torch.stack(
            (subindex % 2, subindex // 2 % 2, subindex // 4 % 2), dim=1
        ).to(torch.int32)
        parent = torch.cat(
            (coords[row, :1], coords[row, 1:] * 2 + bits), dim=1
        )
    assert {
        tuple(coord) for coord in parent[:, 1:].tolist()
    } == {tuple(coord) for coord in final_coords.tolist()}


def test_subdivision_guides_reject_unmatched_shape_lattice():
    with pytest.raises(ValueError, match="shape lattice"):
        stage3.derive_subdivision_guide_tensors(
            torch.tensor([[63, 63, 63]], dtype=torch.int32),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            resolution=64,
            levels=4,
        )


def test_expensive_inference_calls_are_no_grad_guarded():
    source = SCRIPT.read_text()
    assert source.count("with torch.no_grad():") >= 2
