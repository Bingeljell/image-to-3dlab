#!/usr/bin/env python3
"""Upscale a picture 4x with Real-ESRGAN, then resize it to a square of SIZE pixels.

    hunyuan_mlx/paint/.venv/bin/python -I scripts/upscale_esrgan.py in.png out.png [--size 1024]

Runs in the Hunyuan3D-MLX paint environment, whose install already carries the Real-ESRGAN
weights (BSD-3-Clause); it never downloads them. The hero route feeds Pixal3D a face crop
this way: a plain resize of a ~45 px face blurs, and Pixal3D turns blurred hair into shards,
while the upscaled crop keeps solid hair and real eyelids. Alpha, if any, is carried over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAINT = REPO / "hunyuan_mlx" / "paint"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    sys.path.insert(0, str(PAINT))
    import numpy as np
    from hy3dpaint_mlx import realesrgan as rg
    from PIL import Image

    weights = rg.pick_rrdbnet_weights(str(PAINT))
    if weights is None:
        print("[upscale] no Real-ESRGAN weights: they come with the Hunyuan3D-MLX paint install "
              "(Setup & Status page). Nothing was downloaded.", file=sys.stderr)
        return 2
    with Image.open(args.source) as opened:
        picture = opened.convert("RGBA")
    model = rg.load_rrdbnet(weights)
    up = np.asarray(rg.upscale(model, np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0))
    result = Image.fromarray(np.clip(up * 255, 0, 255).astype(np.uint8))
    result = result.resize((args.size, args.size), Image.LANCZOS)
    alpha = picture.getchannel("A")
    if alpha.getextrema()[0] < 255:
        result.putalpha(alpha.resize((args.size, args.size), Image.LANCZOS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"[upscale] {args.source.name} {picture.size} -> {args.output.name} {result.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
