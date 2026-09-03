#!/usr/bin/env python3
"""Advisory TinyCLIP check for TRELLIS-hostile flat/vector-style inputs.

This is deliberately a warning signal, not a generation gate. The score measures which
of two prompt groups TinyCLIP considers more similar to the image; it is not a calibrated
probability that TRELLIS will fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

MODEL_ID = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
MODEL_REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
NEUTRAL_BACKGROUND = (240, 240, 240)
HIGH_RISK_THRESHOLD = 0.85
LOW_RISK_THRESHOLD = 0.35

FLAT_PROMPTS = (
    "a flat vector illustration of an object",
    "a simple two dimensional cartoon with flat colors",
    "a piece of clip art without realistic shading",
    "a flat outlined drawing of a character",
)
DIMENSIONAL_PROMPTS = (
    "a softly lit three dimensional render of an object",
    "a photograph of a physical three dimensional object",
    "a detailed digital character render with realistic shading",
    "a three dimensional toy with visible light and shadow",
)


def prepare_image(image: Image.Image) -> Image.Image:
    """Crop a transparent cutout and composite it on a neutral square for TinyCLIP."""
    image.load()
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox is not None:
            rgba = rgba.crop(bbox)
        rgb = Image.new("RGB", rgba.size, NEUTRAL_BACKGROUND)
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")

    side = max(rgb.size)
    square = Image.new("RGB", (side, side), NEUTRAL_BACKGROUND)
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return square


def summarize_logits(flat_logits: list[float], dimensional_logits: list[float]) -> dict:
    """Turn prompt-group logits into an explicitly uncalibrated advisory score."""
    if not flat_logits or not dimensional_logits:
        raise ValueError("both prompt groups require at least one logit")
    flat_mean = sum(flat_logits) / len(flat_logits)
    dimensional_mean = sum(dimensional_logits) / len(dimensional_logits)
    delta = max(-60.0, min(60.0, dimensional_mean - flat_mean))
    flat_risk = 1.0 / (1.0 + math.exp(delta))
    verdict = (
        "likely_flat"
        if flat_risk >= HIGH_RISK_THRESHOLD
        else "likely_dimensional"
        if flat_risk <= LOW_RISK_THRESHOLD
        else "uncertain"
    )
    return {
        "flat_risk": round(flat_risk, 6),
        "dimensional_score": round(1.0 - flat_risk, 6),
        "verdict": verdict,
    }


def classify(image_path: Path, *, local_files_only: bool = False) -> dict:
    """Run the pinned TinyCLIP checkpoint on one local image."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    with Image.open(image_path) as source:
        image = prepare_image(source)
    prompts = [*FLAT_PROMPTS, *DIMENSIONAL_PROMPTS]
    processor = CLIPProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        # This checkpoint ships tokenizer.json rather than vocab.json/merges.txt.
        use_fast=True,
    )
    model = CLIPModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    ).eval()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0].float().cpu().tolist()

    split = len(FLAT_PROMPTS)
    summary = summarize_logits(logits[:split], logits[split:])
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "advisory_only": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_sha256": digest,
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="fail instead of downloading the pinned TinyCLIP checkpoint",
    )
    args = parser.parse_args()
    print(json.dumps(classify(args.image.resolve(), local_files_only=args.local_files_only)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
