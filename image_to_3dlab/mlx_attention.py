"""Fused attention for TRELLIS.2 on Apple Silicon, via MLX.

Why this exists
---------------
TRELLIS.2-4B is 1536 wide over 12 heads, so its head dimension is 128. PyTorch's MPS
backend has no fused attention kernel, and Pedro's Metal kernel only supports head
dimensions through 64, so the model falls back to unfused SDPA that materialises the whole
score matrix. Measured 2026-09-20 at the real Stage-3 shape (9,801 tokens, 12 heads, head
dim 128), attention is 93.2% of sampling time, and sampling is 68% of a 1024 run.

MLX has a fused kernel that handles head dimension 128. At that shape it is 3.2x faster
than torch MPS SDPA at fp32 and 13.8x at fp16. Even converting tensors through host memory
on every call - which is what this module does, because MLX 0.31.2 exposes no zero-copy
path from an MPS tensor - it stays 2.83x and 9.35x ahead respectively.

The functions here are deliberately plain and importable so they can be unit-tested. The
vendored TRELLIS tree only gets a thin dispatch branch that calls into them, injected by
``scripts/patch_trellis_mlx_attention.py``.

Numerics
--------
MLX and torch agree to roughly 1e-4 at fp32, which is reduction-order difference rather
than error. That is small but not nothing, and this repository has previously been caught
out by changes of that size affecting colour and guidance behaviour. fp32 is therefore the
default; fp16 is opt-in through ``I2L_MLX_ATTN_DTYPE=fp16`` and must be validated on a real
generated asset before being trusted.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

DTYPE_ENV = "I2L_MLX_ATTN_DTYPE"


def mlx_available() -> bool:
    """True when MLX can be imported in this interpreter."""
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def resolve_dtype(name: str | None = None):
    """Map a dtype name to an MLX dtype, defaulting to fp32.

    fp32 is the default on purpose: it makes the first integration a pure speed change
    with no precision change, so it cannot disturb guidance-tuned behaviour.
    """
    import mlx.core as mx

    if name is None:
        name = os.environ.get(DTYPE_ENV, "fp32")
    name = name.lower()
    if name in ("fp32", "float32"):
        return mx.float32
    if name in ("fp16", "float16"):
        return mx.float16
    if name in ("bf16", "bfloat16"):
        return mx.bfloat16
    raise ValueError(
        f"unsupported {DTYPE_ENV}={name!r}; expected one of fp32, fp16, bf16"
    )


def split_offsets(seqlen: Sequence[int]) -> list[tuple[int, int]]:
    """Turn packed sequence lengths into (start, stop) slices.

    Pure and trivial, but it is the part that silently corrupts every downstream token if
    it is wrong, so it is separated out and tested directly.
    """
    offsets = []
    cursor = 0
    for length in seqlen:
        offsets.append((cursor, cursor + int(length)))
        cursor += int(length)
    return offsets


def attention_single(q, k, v, scale: float | None = None, dtype=None):
    """Fused attention for one sequence.

    q, k, v are torch tensors laid out ``[L, H, C]``, matching how the vendored sparse
    attention packs its features. Returns a torch tensor ``[Lq, H, C]`` on the same device
    and dtype as ``q``.
    """
    import mlx.core as mx
    import numpy as np
    import torch

    if dtype is None:
        dtype = resolve_dtype()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    def to_mlx(t):
        # [L, H, C] -> [1, H, L, C]; MLX has no zero-copy path from an MPS tensor, so this
        # goes through host memory. Benchmarked: still far ahead of staying in torch.
        arr = t.permute(1, 0, 2).unsqueeze(0).contiguous()
        return mx.array(arr.detach().to("cpu", torch.float32).numpy()).astype(dtype)

    out = mx.fast.scaled_dot_product_attention(
        to_mlx(q), to_mlx(k), to_mlx(v), scale=float(scale), mask=None
    )
    mx.eval(out)
    host = np.array(out.astype(mx.float32))          # [1, H, Lq, C]
    result = torch.from_numpy(host).squeeze(0).permute(1, 0, 2)   # [Lq, H, C]
    return result.to(device=q.device, dtype=q.dtype).contiguous()


def varlen_attention(
    q,
    k,
    v,
    q_seqlen: Sequence[int],
    kv_seqlen: Sequence[int],
    scale: float | None = None,
    dtype=None,
):
    """Block-diagonal attention over packed variable-length sequences.

    This is the drop-in equivalent of the vendored ``sdpa`` branch: each packed sequence
    attends only to its own keys, which is what FlashAttention's varlen form does. Inputs
    are ``[T, H, C]``; the output is ``[T_q, H, C]``.
    """
    import torch

    if len(q_seqlen) != len(kv_seqlen):
        raise ValueError(
            f"q_seqlen and kv_seqlen must describe the same number of sequences; "
            f"got {len(q_seqlen)} and {len(kv_seqlen)}"
        )
    if dtype is None:
        dtype = resolve_dtype()

    parts = []
    for (q_start, q_stop), (kv_start, kv_stop) in zip(
        split_offsets(q_seqlen), split_offsets(kv_seqlen)
    ):
        parts.append(
            attention_single(
                q[q_start:q_stop],
                k[kv_start:kv_stop],
                v[kv_start:kv_stop],
                scale=scale,
                dtype=dtype,
            )
        )
    if not parts:
        return q.new_zeros((0, *q.shape[1:]))
    return torch.cat(parts, dim=0)


def memory_snapshot() -> dict[str, int]:
    """Metal memory MLX is holding right now, in bytes.

    ``active`` is memory backing live arrays; ``cache`` is memory MLX has freed internally
    but kept reserved for reuse rather than returned to the system. The cache is the part
    that matters here: it stays claimed after sampling finishes, while a later stage in the
    same process may need that headroom.

    Returns an empty dict when MLX is absent, so callers need no guard.
    """
    if not mlx_available():
        return {}
    import mlx.core as mx

    return {
        "active": int(mx.get_active_memory()),
        "cache": int(mx.get_cache_memory()),
        "peak": int(mx.get_peak_memory()),
    }


def release_memory() -> dict[str, int]:
    """Return MLX's reserved Metal memory to the system, reporting what was freed.

    Called at a stage boundary rather than after every attention call: clearing the cache
    mid-sampling would force MLX to re-request buffers on the next step and cost more than
    it saves. The point is to hand memory back before a *different* subsystem needs it.

    Reports rather than just acting, because the interesting question is how much was being
    held, and a silent call answers nothing.
    """
    before = memory_snapshot()
    if not before:
        return {}
    import mlx.core as mx

    mx.clear_cache()
    after = memory_snapshot()
    return {
        "cache_before": before["cache"],
        "cache_after": after["cache"],
        "released": before["cache"] - after["cache"],
        "active": after["active"],
        "peak": before["peak"],
    }
