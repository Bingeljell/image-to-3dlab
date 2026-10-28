"""Convert the v2-1 .ckpt (torch pickle) -> flat fp16 safetensors with
conditioner./model./vae. prefixes that hy3dmlx.convert.load_models expects.

The released inference ckpt is a dict {'model':sd, 'vae':sd, 'conditioner':sd}.
Run with the oracle venv (needs torch):
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/convert_v21_ckpt.py
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

DEFAULT_CKPT = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt"
DEFAULT_OUT = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.safetensors"


def _load_checkpoint(ckpt_path: Path, *, allow_pickle: bool = False):
    """Load a torch checkpoint, failing closed on unsafe pickle unless asked."""
    import torch

    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_pickle:
            raise RuntimeError(
                f"Failed to load {ckpt_path} with weights_only=True. "
                "Pass --allow-pickle to allow the unsafe pickle loader."
            ) from exc
        print(f"  weights_only=True failed, retrying False: {str(exc)[:80]}", flush=True)
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _flatten_checkpoint(ckpt: dict) -> dict:
    """Flatten {'model': sd, 'vae': sd, 'conditioner': sd} or unwrap state_dict."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "model" not in ckpt:
        ckpt = ckpt["state_dict"]

    flat = {}
    subs = [k for k in ("model", "vae", "conditioner") if k in ckpt and isinstance(ckpt[k], dict)]
    if subs:  # dict-of-substatedicts layout
        print(f"  layout: dict of sub-state-dicts {subs}", flush=True)
        for sub in subs:
            for k, v in ckpt[sub].items():
                flat[f"{sub}.{k}"] = v
    else:  # already-flat layout, possibly with a wrapper prefix
        print("  layout: flat", flush=True)
        flat = {k: v for k, v in ckpt.items()}
    return flat


def _prepare_tensors(flat: dict) -> dict:
    """Return fp16, contiguous tensors, skipping non-tensor values."""
    import torch

    out = {}
    for k, v in flat.items():
        if not torch.is_tensor(v):
            continue
        out[k] = (v.half() if v.is_floating_point() else v).contiguous()
    return out


def convert_checkpoint(
    ckpt_path: Path,
    out_path: Path,
    *,
    allow_pickle: bool = False,
) -> int:
    """Convert ``ckpt_path`` to a flat fp16 safetensors file at ``out_path``.

    Loads with ``weights_only=True``. If that fails, raises ``RuntimeError`` unless
    ``allow_pickle`` is true, in which case it retries once with the unsafe pickle
    loader and prints a warning.
    """
    import safetensors.torch

    print(f">> torch.load (cpu) {ckpt_path} ...", flush=True)
    ckpt = _load_checkpoint(ckpt_path, allow_pickle=allow_pickle)

    flat = _flatten_checkpoint(ckpt)
    out = _prepare_tensors(flat)

    pref = collections.Counter(k.split(".")[0] for k in out)
    print("  top-level prefixes:", dict(pref), flush=True)
    print("  total tensors:", len(out), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(out, str(out_path))
    print(f"WROTE {out_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI surface, separated so tests can inspect it without running torch."""
    parser = argparse.ArgumentParser(
        description="Convert the v2-1 .ckpt to flat fp16 safetensors.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_CKPT,
        help="input .ckpt path",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUT,
        help="output .safetensors path",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="allow the unsafe torch pickle loader if weights_only=True fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return convert_checkpoint(
        Path(args.input),
        Path(args.output),
        allow_pickle=args.allow_pickle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
