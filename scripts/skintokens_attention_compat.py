"""Provide the PyTorch attention fallback copied into SkinTokens."""

from __future__ import annotations

from torch.nn.functional import scaled_dot_product_attention


def flash_attn_func(q, k, v, causal=False, softmax_scale=None, **kwargs):
    """Match the vendor call's batch, token, head, channel layout and return pair."""
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    out = scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype),
                                      is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous(), None
