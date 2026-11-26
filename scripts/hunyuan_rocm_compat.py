"""Find Hunyuan scheduler steps and chunk paint queries for ROCm."""

import os

import torch
import torch.nn.functional as F


def nearest_timestep_indices(schedule_timesteps, timestep):
    """Return a one-element tensor with the nearest scheduled step."""
    return (schedule_timesteps - timestep).abs().argmin().reshape(1)


def chunked_attention(query, key, value, attention_mask=None):
    """Attend to all keys for each query, limiting the number of queries per call."""
    chunk = int(os.environ.get("HY3D_SDPA_QUERY_CHUNK", "0") or 0)
    if chunk <= 0 or query.shape[-2] <= chunk:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    parts = []
    for start in range(0, query.shape[-2], chunk):
        # The query dimension is -2 in both the tensor and its mask.
        mask = None if attention_mask is None else attention_mask[..., start:start + chunk, :]
        parts.append(F.scaled_dot_product_attention(
            query[..., start:start + chunk, :], key, value,
            attn_mask=mask, dropout_p=0.0, is_causal=False,
        ))
    return torch.cat(parts, dim=-2)
