"""Attention and accelerator adapters copied into vendored TRELLIS checkouts."""

from __future__ import annotations

import math
import os
import sys
from itertools import accumulate
from pathlib import Path

import torch
from torch.nn.functional import scaled_dot_product_attention


def unpack_attention(arg_count, dim, inputs):
    """Unpack the one-, two- and three-argument forms supported by TRELLIS."""
    if arg_count == 1:
        return inputs["qkv"].unbind(dim=dim)
    if arg_count == 2:
        k, v = inputs["kv"].unbind(dim=dim)
        return inputs["q"], k, v
    return inputs["q"], inputs["k"], inputs["v"]


def mlx_sparse_attention(arg_count, inputs, q_seqlen, kv_seqlen, source_file):
    """Locate the repository and dispatch TRELLIS's packed inputs to MLX."""
    for parent in Path(source_file).resolve().parents:
        if (parent / "image_to_3dlab" / "mlx_attention.py").is_file():
            break
    else:
        raise RuntimeError(
            "mlx attention backend: could not locate image_to_3dlab starting from "
            f"{source_file}. The vendored checkout must live inside the repository."
        )
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from image_to_3dlab.mlx_attention import varlen_attention

    q, k, v = unpack_attention(arg_count, 1, inputs)
    return varlen_attention(q, k, v, q_seqlen, kv_seqlen)


def chunked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """Bound the Metal attention score buffer by chunking each sequence's queries."""
    chunk = int(os.environ.get("SDPA_Q_CHUNK", "1024"))
    output = []
    q_offset = kv_offset = 0
    for q_length, kv_length in zip(q_seqlen, kv_seqlen, strict=False):
        k_part = k[kv_offset:kv_offset + kv_length].transpose(0, 1).unsqueeze(0)
        v_part = v[kv_offset:kv_offset + kv_length].transpose(0, 1).unsqueeze(0)
        for start in range(0, q_length, chunk):
            stop = min(start + chunk, q_length)
            q_part = q[q_offset + start:q_offset + stop].transpose(0, 1).unsqueeze(0)
            result = scaled_dot_product_attention(q_part, k_part, v_part)
            output.append(result.squeeze(0).transpose(0, 1))
        q_offset += q_length
        kv_offset += kv_length
    return torch.cat(output, dim=0)


def metal_attention(q, k, v, q_seqlen, kv_seqlen, device, kernel=None):
    """Pass packed prefix lengths to the fused Metal kernel for small attention heads."""
    if q.shape[-1] > 64:
        raise RuntimeError(
            f"metal_flash supports fast heads only through 64 values; got {q.shape[-1]}. "
            "Use the sdpa backend for TRELLIS.2-4B."
        )
    if kernel is None:
        import flex_gemm

        kernel = flex_gemm.kernels.metal.sparse_attention_fwd
    q_prefix = torch.tensor(list(accumulate(q_seqlen, initial=0)), dtype=torch.int32, device=device)
    kv_prefix = torch.tensor(list(accumulate(kv_seqlen, initial=0)), dtype=torch.int32, device=device)
    return kernel(q, k, v, q_prefix, kv_prefix, max(q_seqlen), max(kv_seqlen),
                  1.0 / math.sqrt(q.shape[-1]))


def aule_dense(q, k, v, kernel):
    """Convert dense TRELLIS tensors to Aule's head-major layout and back."""
    return kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False).transpose(1, 2)


def aule_packed(q, k, v, q_seqlen, kv_seqlen, kernel):
    """Run Aule once per packed sequence or window, with independent offsets."""
    if isinstance(q_seqlen, torch.Tensor):
        q_seqlen = q_seqlen.tolist()
    if isinstance(kv_seqlen, torch.Tensor):
        kv_seqlen = kv_seqlen.tolist()
    output = []
    q_offset = kv_offset = 0
    for q_length, kv_length in zip(q_seqlen, kv_seqlen, strict=False):
        q_part = q[q_offset:q_offset + q_length].unsqueeze(0)
        k_part = k[kv_offset:kv_offset + kv_length].unsqueeze(0)
        v_part = v[kv_offset:kv_offset + kv_length].unsqueeze(0)
        output.append(aule_dense(q_part, k_part, v_part, kernel)[0])
        q_offset += q_length
        kv_offset += kv_length
    return torch.cat(output, dim=0)


def aule_window_self(qkv, lengths, kernel):
    q, k, v = qkv.unbind(dim=1)
    return aule_packed(q, k, v, lengths, lengths, kernel)


def aule_window_cross(q, kv, q_lengths, kv_lengths, kernel):
    k, v = kv.unbind(dim=1)
    return aule_packed(q, k, v, q_lengths, kv_lengths, kernel)


def accelerator_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")


def model_device(model):
    return next(model.parameters()).device


def dino_layers(model):
    return getattr(model, "model", model).layer


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
