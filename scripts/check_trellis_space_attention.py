#!/usr/bin/env python3
"""Cheap MPS integration gate for TRELLIS sparse self/cross attention."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor/trellis-space-mac"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--backend", choices=("sdpa", "metal_flash"), default="sdpa")
    args = parser.parse_args()
    root = args.root.resolve()

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", args.backend)
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(root / "cache/flex_gemm_autotune.json"),
    )
    sys.path.insert(0, str(root / "TRELLIS.2"))

    import torch
    import torch.nn.functional as F
    from trellis2.modules.sparse import VarLenTensor, config
    from trellis2.modules.sparse.attention.full_attn import (
        sparse_scaled_dot_product_attention,
    )

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run this integration gate on Apple Silicon")
    if config.ATTN != args.backend:
        raise RuntimeError(f"wrong sparse attention backend: {config.ATTN}")

    torch.manual_seed(0x2A11CE)
    seqlens = [7, 5, 9]
    layout = VarLenTensor.layout_from_seqlen(seqlens)
    heads = 4
    channels = 128 if args.backend == "sdpa" else 64

    # Packed self-attention: the main DiT path.
    qkv_cpu = torch.randn(sum(seqlens), 3, heads, channels, dtype=torch.float32) * 0.2
    qkv = VarLenTensor(qkv_cpu.to("mps"), layout)
    actual_self = sparse_scaled_dot_product_attention(qkv).feats.cpu()
    expected_self = []
    for segment in qkv_cpu.split(seqlens):
        q, k, v = segment.unbind(dim=1)
        expected = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
        )[0].permute(1, 0, 2)
        expected_self.append(expected)
    expected_self = torch.cat(expected_self)

    # Packed-query/dense-context cross-attention: DINO image conditioning in Stage 3.
    context_len = 11
    q_cpu = torch.randn(sum(seqlens), heads, channels, dtype=torch.float32) * 0.2
    k_cpu = torch.randn(len(seqlens), context_len, heads, channels, dtype=torch.float32) * 0.2
    v_cpu = torch.randn(len(seqlens), context_len, heads, channels, dtype=torch.float32) * 0.2
    actual_cross = sparse_scaled_dot_product_attention(
        VarLenTensor(q_cpu.to("mps"), layout),
        k_cpu.to("mps"),
        v_cpu.to("mps"),
    ).feats.cpu()
    expected_cross = []
    offset = 0
    for batch, length in enumerate(seqlens):
        q = q_cpu[offset:offset + length]
        expected = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k_cpu[batch].permute(1, 0, 2).unsqueeze(0),
            v_cpu[batch].permute(1, 0, 2).unsqueeze(0),
        )[0].permute(1, 0, 2)
        expected_cross.append(expected)
        offset += length
    expected_cross = torch.cat(expected_cross)

    checks = {
        "self": (actual_self, expected_self),
        "cross": (actual_cross, expected_cross),
    }
    for name, (actual, expected) in checks.items():
        maximum = (actual - expected).abs().max().item()
        mean = (actual - expected).abs().mean().item()
        print(f"{name}: max_error={maximum:.6g} mean_error={mean:.6g}")
        if not torch.allclose(actual, expected, atol=5e-4, rtol=1e-4):
            raise RuntimeError(f"{name} Metal attention parity failed")

    print(f"TRELLIS {args.backend} attention integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
