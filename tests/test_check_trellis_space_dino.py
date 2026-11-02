"""Static contract for the DINO conditioning capture."""

import os
from pathlib import Path

import check_trellis_space_dino as dino

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_trellis_space_dino.py"


def test_offline_wins_even_when_the_shell_says_zero(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    dino.pin_hub_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_capture_records_preprocessing_and_full_feature_tensor():
    source = SCRIPT.read_text()
    assert "Trellis2ImageTo3DPipeline" in source
    assert "pipeline.preprocess_image(image)" in source
    assert "DinoV3FeatureExtractor" in source
    assert 'image_size=args.resolution' in source
    assert 'choices=("auto", "cpu", "mps")' in source
    assert 'extractor.to(device)' in source
    assert '"device": device' in source
    assert 'torch.save(features' in source
    assert 'processed_sha256' in source
    assert 'feature_sha256' in source
