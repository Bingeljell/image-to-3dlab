"""Chunk MtlBVH distance queries to bound Metal command-buffer work."""

from __future__ import annotations

import os

import torch


def chunked_unsigned_distance(query, positions, return_uvw, out_id, output_dtype=None):
    """Collect distance, face and optional barycentric output in their original order."""
    chunk_size = int(os.environ.get("MTLBVH_QUERY_CHUNK", "65536"))
    count = positions.shape[0]
    if chunk_size <= 0 or count <= chunk_size:
        return query(positions, return_uvw, out_id)

    device = positions.device
    distances = torch.empty(count, dtype=output_dtype or torch.float32, device=device)
    face_ids = torch.empty(count, dtype=torch.int64, device=device)
    uvws = torch.empty((count, 3), dtype=torch.float32, device=device) if return_uvw else None
    for start in range(0, count, chunk_size):
        result = query(positions[start:start + chunk_size], return_uvw, out_id)
        end = start + result[0].shape[0]
        distances[start:end].copy_(result[0])
        face_ids[start:end].copy_(result[1])
        if return_uvw:
            uvws[start:end].copy_(result[2])
        del result
        if device.type == "mps":
            torch.mps.synchronize()
    return distances, face_ids, uvws
