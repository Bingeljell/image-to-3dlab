"""Sparse coordinate lookup copied into the vendored TRELLIS decoder."""

from __future__ import annotations

import torch


class CPUHashMap:
    """Keep only active voxels and return the CUDA-compatible sentinel for misses."""

    def __init__(self, grid_size, device="cpu"):
        self.device = device
        self._entries = {}

    def insert(self, coords_4d):
        coords = coords_4d[:, 1:4].detach().cpu().tolist()
        self._entries = {tuple(coord): index for index, coord in enumerate(coords)}

    def lookup_3d(self, coords_4d):
        coords = coords_4d[:, 1:4].detach().cpu().tolist()
        values = [self._entries.get(tuple(coord), 0xFFFFFFFF) for coord in coords]
        return torch.tensor(values, dtype=torch.long, device=self.device)
