#!/usr/bin/env python3
"""Convert the Hunyuan3D-2.1 paint PBR UNet checkpoint to safetensors.

tencent/Hunyuan3D-2.1 ships `hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin`
(a torch pickle), but run_paint_pbr.py loads it with `mx.load(...)`, which only reads
safetensors. This converts the checkpoint 1:1 (same keys, shapes, dtypes -- fp16) with
no renaming, so the MLX side stays byte-compatible with what upstream ships.

Safety properties:
- The source .bin is NEVER modified or deleted; it stays on disk as the
  conversion authority (re-runs validate against it).
- An existing destination is only accepted if it matches the source on keys,
  shapes, dtypes AND values; anything else (truncated write, partial copy,
  same-shaped-but-wrong tensors) is regenerated.
- Output is written to a unique temp file in the destination directory,
  re-loaded, and value-validated against the source BEFORE an atomic rename
  into place. A failed or interrupted conversion can never leave a
  plausible-looking partial destination.

Needs a torch + safetensors environment (dev-time only, like convert_realesrgan.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _matches(st: dict, sd: dict) -> tuple[bool, str]:
    """(ok, reason): exact key set plus identical shapes, dtypes, and values."""
    import torch
    if set(st) != set(sd):
        return False, "key sets differ"
    for k in sd:
        if st[k].shape != sd[k].shape:
            return False, f"{k}: shape {st[k].shape} != {sd[k].shape}"
        if st[k].dtype != sd[k].dtype:
            return False, f"{k}: dtype {st[k].dtype} != {sd[k].dtype}"
        if not torch.equal(st[k], sd[k]):
            return False, f"{k}: values differ"
    return True, ""


def convert(bin_path: Path, out_path: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    n = sum(t.numel() * t.element_size() for t in sd.values())

    if out_path.is_file():
        try:
            ok, why = _matches(load_file(str(out_path)), sd)
        except Exception as e:  # truncated file, bad header, non-safetensors junk
            ok, why = False, f"unreadable ({e})"
        if ok:
            print(f"{out_path} already converted and matches the .bin; nothing to do")
            return
        print(f"existing {out_path.name} is invalid ({why}); regenerating")

    payload = {k: v.contiguous() for k, v in sd.items()}
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent),
                                    prefix=out_path.name + ".", suffix=".convert-tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        print(f"converting {len(payload)} tensors ({n / 2**30:.2f} GiB) ...", flush=True)
        save_file(payload, str(tmp), metadata={"format": "pt"})
        got = tmp.stat().st_size
        if not (n <= got < n + 8 * 2**20):
            raise ValueError(f"serialized size {got:,} outside payload+header range for {n:,}")
        ok, why = _matches(load_file(str(tmp)), sd)
        if not ok:
            raise ValueError(f"re-loaded tensors do not match the source: {why}")
        os.replace(tmp, out_path)  # atomic: destination appears only when valid
        print(f"wrote {out_path} ({got:,} bytes), value-validated against source")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bin_path", type=Path)
    ap.add_argument("out_path", type=Path)
    args = ap.parse_args()
    try:
        convert(args.bin_path, args.out_path)
    except ImportError as e:
        sys.exit(
            f"missing dependency ({e}); this converter needs torch + safetensors. "
            "Run download_weights.py with a python that has both (e.g. the same "
            "env used for convert_realesrgan.py), not the shape venv alone."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
