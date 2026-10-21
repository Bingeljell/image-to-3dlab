#!/usr/bin/env python3
"""Upscale a picture 4x with Real-ESRGAN, then resize it to a square of SIZE pixels.

    .venv/bin/python scripts/upscale_esrgan.py in.png out.png [--size 1024]

Runs in the project's own environment (image_to_3dlab/upscale.py, PyTorch on a Mac, NVIDIA
or AMD) with the 67 MB Real-ESRGAN weights from Setup or scripts/bootstrap_upscaler.py; it
never downloads them. The hero route feeds Pixal3D a face crop this way: a plain resize of
a ~45 px face blurs, and Pixal3D turns blurred hair into shards, while the upscaled crop
keeps solid hair and real eyelids. Alpha, if any, is carried over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from image_to_3dlab import upscale

    if not upscale.installed():
        print("[upscale] Real-ESRGAN is not installed: get it (67 MB) from Setup & Status or "
              "scripts/bootstrap_upscaler.py. Nothing was downloaded.", file=sys.stderr)
        return 2
    with Image.open(args.source) as opened:
        picture = opened.convert("RGBA")
    up = upscale.upscale(upscale.load(), np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0)
    result = Image.fromarray(np.clip(np.round(up * 255), 0, 255).astype(np.uint8))
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
