"""Static contract for the DINO conditioning capture."""

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_trellis_space_dino.py"


def test_capture_records_preprocessing_and_full_feature_tensor():
    source = SCRIPT.read_text()
    assert "Trellis2ImageTo3DPipeline" in source
    assert "pipeline.preprocess_image(image)" in source
    assert "DinoV3FeatureExtractor" in source
    assert 'image_size=args.resolution' in source
    assert 'torch.save(features' in source
    assert 'processed_sha256' in source
    assert 'feature_sha256' in source
