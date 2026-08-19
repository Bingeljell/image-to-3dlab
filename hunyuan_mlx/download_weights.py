#!/usr/bin/env python3
"""Download Hunyuan3D-MLX weights from Hugging Face.

Fetches shape-stage weights (2.1, 2.0, 2.0-turbo -- pick one with --model, or omit for
all three) into shape/weights/, and paint-stage weights into paint/weights/. Same source
repos Xiong's own dl_any.py/dl_modelscope.py pull from ModelScope, just via Hugging Face
Hub instead -- ModelScope was a region workaround for the original author, not something
needed here; verified 2026-08-19 that both tencent/Hunyuan3D-2 and tencent/Hunyuan3D-2.1
are directly reachable.

2.1 ships only a .ckpt on HF; this converts it to the .safetensors format the shape
pipeline actually loads, via shape/scripts/convert_v21_ckpt.py.

RealESRGAN super-res weights (paint/weights/realesrgan/rrdbnet.npz) aren't part of the
official Tencent HF repos and aren't fetched by this script -- run (needs a torch venv,
dev-time only): `paint/scripts/convert_realesrgan.py`. It downloads the official
xinntao/Real-ESRGAN release and converts it; see that script's docstring.

Usage:
    shape/.venv/bin/python download_weights.py [--model 2.0] [--skip-paint]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHAPE_WEIGHTS = REPO / "shape" / "weights"
PAINT_WEIGHTS = REPO / "paint" / "weights"

# (hf_repo, hf_subdir, local_group_dir) -- local_group_dir matches the layout
# hunyuan_mlx_xiong_generate.py's SHAPE_MODELS / generate_api.py's
# HUNYUAN_XIONG_SHAPE_MODELS already expect.
SHAPE_HF_SOURCES = {
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "Hunyuan3D-2.1"),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", "Hunyuan3D-2"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo", "Hunyuan3D-2"),
}


def download_shape(model: str) -> None:
    from huggingface_hub import hf_hub_download

    hf_repo, hf_dir, local_group = SHAPE_HF_SOURCES[model]
    local_root = SHAPE_WEIGHTS / local_group
    dest = local_root / hf_dir
    dest.mkdir(parents=True, exist_ok=True)

    hf_hub_download(hf_repo, f"{hf_dir}/config.yaml", local_dir=local_root)

    if model == "2.1":
        # 2.1 only ships a .ckpt on HF -- convert once, same as the existing local copy.
        ckpt = hf_hub_download(hf_repo, f"{hf_dir}/model.fp16.ckpt", local_dir=local_root)
        target = dest / "model.fp16.safetensors"
        if not target.is_file():
            print(f"converting {ckpt} -> {target} ...", flush=True)
            subprocess.run(
                [sys.executable, str(REPO / "shape" / "scripts" / "convert_v21_ckpt.py"),
                 ckpt, str(target)],
                check=True,
            )
    else:
        hf_hub_download(hf_repo, f"{hf_dir}/model.fp16.safetensors", local_dir=local_root)

    print(f"{model}: ready at {dest}")


def download_paint() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "tencent/Hunyuan3D-2.1",
        allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        local_dir=PAINT_WEIGHTS,
    )
    # dinov2-giant ships inside paintpbr-v2-1/dinov2/ -- the paint code (run_paint_pbr.py,
    # test_pbr_parity.py) loads it from there directly, no symlink needed.
    print(f"paint weights: ready at {PAINT_WEIGHTS}")
    print(
        "NOTE: RealESRGAN weights (weights/realesrgan/rrdbnet.npz) are NOT covered by "
        "this script -- run paint/scripts/convert_realesrgan.py separately (needs torch)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(SHAPE_HF_SOURCES), default=None,
                         help="download only this shape model; omit to fetch all three")
    parser.add_argument("--skip-paint", action="store_true")
    args = parser.parse_args()

    for model in ([args.model] if args.model else sorted(SHAPE_HF_SOURCES)):
        download_shape(model)
    if not args.skip_paint:
        download_paint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
