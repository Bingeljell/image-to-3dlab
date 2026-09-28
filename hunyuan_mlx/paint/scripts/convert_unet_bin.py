#!/usr/bin/env python3
"""Convert the Hunyuan3D-2.1 paint PBR UNet checkpoint to safetensors.

tencent/Hunyuan3D-2.1 ships `hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin`
(a torch pickle), but run_paint_pbr.py loads it with `mx.load(...)`, which only reads
safetensors. This converts the checkpoint 1:1 (same keys, shapes, dtypes -- fp16) with
no renaming, so the MLX side stays byte-compatible with what upstream ships.

Needs a torch + safetensors environment (dev-time only, like convert_realesrgan.py).
The source .bin is deleted only after a successful, size-plausible conversion
(same policy as the 2.1 shape checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert(bin_path: Path, out_path: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    if out_path.is_file() and out_path.stat().st_size > 0:
        # Idempotent: validate the existing conversion instead of redoing a 4 GB write.
        st = load_file(str(out_path))
        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        if set(st) == set(sd) and all(
            st[k].shape == sd[k].shape and st[k].dtype == sd[k].dtype for k in sd
        ):
            print(f"{out_path} already converted and matches the .bin; nothing to do")
            return
        print(f"{out_path} exists but does not match the .bin; reconverting")

    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    n = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"converting {len(sd)} tensors ({n / 2**30:.2f} GiB) ...", flush=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(out_path), metadata={"format": "pt"})
    got = out_path.stat().st_size
    # serialized file = payload + header; anything wildly off means a truncated write
    if not (n <= got < n + 8 * 2**20):
        out_path.unlink(missing_ok=True)
        sys.exit(f"converted size {got:,} outside payload+header range for {n:,}; removed")
    print(f"wrote {out_path} ({got:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bin_path", type=Path)
    ap.add_argument("out_path", type=Path)
    args = ap.parse_args()
    convert(args.bin_path, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
