"""Pure tests for the TinyCLIP TRELLIS input advisor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "classify_trellis_input.py"
SPEC = importlib.util.spec_from_file_location("classify_trellis_input", SCRIPT)
advisor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["classify_trellis_input"] = advisor
SPEC.loader.exec_module(advisor)


def test_prepare_image_crops_alpha_and_places_subject_on_neutral_square():
    image = Image.new("RGBA", (6, 4), (0, 0, 0, 0))
    image.putpixel((2, 1), (20, 80, 220, 255))
    image.putpixel((3, 1), (20, 80, 220, 255))

    prepared = advisor.prepare_image(image)

    assert prepared.mode == "RGB"
    assert prepared.size == (2, 2)
    assert prepared.getpixel((0, 0)) == (20, 80, 220)
    assert prepared.getpixel((0, 1)) == advisor.NEUTRAL_BACKGROUND


def test_prepare_image_keeps_opaque_rgb_input():
    image = Image.new("RGB", (5, 3), (10, 20, 30))
    prepared = advisor.prepare_image(image)
    assert prepared.mode == "RGB"
    assert prepared.size == (5, 5)
    assert prepared.getpixel((2, 2)) == (10, 20, 30)


def test_summarize_logits_marks_flat_dimensional_and_uncertain():
    flat = advisor.summarize_logits([5.0, 5.0], [1.0, 1.0])
    dimensional = advisor.summarize_logits([1.0, 1.0], [5.0, 5.0])
    uncertain = advisor.summarize_logits([2.0, 2.0], [2.0, 2.0])

    assert flat["verdict"] == "likely_flat"
    assert flat["flat_risk"] > 0.9
    assert dimensional["verdict"] == "likely_dimensional"
    assert dimensional["flat_risk"] < 0.1
    assert uncertain == {
        "flat_risk": 0.5,
        "dimensional_score": 0.5,
        "verdict": "uncertain",
    }


def test_model_revision_is_pinned():
    assert advisor.MODEL_ID == "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    assert len(advisor.MODEL_REVISION) == 40


def test_classifier_uses_checkpoint_fast_tokenizer():
    source = SCRIPT.read_text()
    assert "use_fast=True" in source


def test_high_risk_threshold_is_conservative():
    assert advisor.HIGH_RISK_THRESHOLD == 0.85
    assert advisor.LOW_RISK_THRESHOLD == 0.35
