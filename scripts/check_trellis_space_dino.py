#!/usr/bin/env python3
"""Capture the exact preprocessed image and DINO conditioning tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor/trellis-space-mac"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "metal_flash")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(root / "cache/flex_gemm_autotune.json"),
    )
    sys.path.insert(0, str(root / "TRELLIS.2"))

    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run this conditioning gate on Apple Silicon")

    image = Image.open(args.image)
    pipeline = Trellis2ImageTo3DPipeline()
    processed = pipeline.preprocess_image(image)
    processed_array = np.asarray(processed)
    processed_digest = hashlib.sha256(processed_array.tobytes()).hexdigest()

    extractor = DinoV3FeatureExtractor(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=args.resolution,
    )
    extractor.to("mps")
    features = extractor([processed]).detach().cpu().float().contiguous()
    feature_digest = hashlib.sha256(features.numpy().tobytes()).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = output.with_suffix(".pt")
    image_path = output.with_suffix(".png")
    json_path = output.with_suffix(".json")
    torch.save(features, tensor_path)
    processed.save(image_path)
    report = {
        "source": str(args.image.resolve()),
        "root": str(root),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "resolution": args.resolution,
        "processed_shape": list(processed_array.shape),
        "processed_sha256": processed_digest,
        "feature_shape": list(features.shape),
        "feature_sha256": feature_digest,
        "feature_mean": features.mean().item(),
        "feature_std": features.std().item(),
        "feature_min": features.min().item(),
        "feature_max": features.max().item(),
        "tensor": str(tensor_path),
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
