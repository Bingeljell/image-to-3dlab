#!/usr/bin/env python3
"""Pure helpers for visibility-based face culling: keep only what is seen from outside.

A generated mesh arrives shattered -- the thorn-knot Snag has 664 components and 46,253
boundary edges, most of it loose flaps and fragments produced by sign flips where the
decoder could not resolve two nearby surfaces. Every repair that reasons about *distance*
fails on it: merge-by-distance has nothing to weld to, and voxel remeshing either fuses
neighbouring coils (coarse) or faithfully rebuilds the shards as slabs (fine). Both were
measured and both failed.

Visibility asks a different question. Stand outside, look from every direction, and keep
only the triangles you actually saw. That is:

- **topology-preserving** -- it only deletes faces, never moves a vertex, so creases stay
  exactly as sharp as they were;
- **scale-free** -- no threshold, no voxel size, nothing to tune wrong;
- **incapable of fusing anything** -- it never asks whether two surfaces are close, so
  adjacent coils cannot blend into each other. That is the failure mode of every
  alternative, and this method is immune to it by construction.

**Honest limit:** an outward-facing flap floating just above the true surface *is* seen,
so it survives. Expect this to remove interior junk, enclosed fragments and back-facing
debris -- a large fraction, not all of it. It is the first step of a cleanup, not the
whole cleanup.

The rendering half lives in ``blender_visibility_cull.py``; everything here is pure so it
can be tested without Blender.
"""

from __future__ import annotations

import numpy as np

# 8 bits per channel over three channels addresses 16.7M faces, far beyond any mesh we
# generate. Index 0 is reserved so that "nothing here" and "face 0" cannot be confused.
_CHANNEL = 256
MAX_FACES = _CHANNEL**3 - 1


def fibonacci_directions(count: int) -> np.ndarray:
    """`count` unit vectors spread evenly over the sphere.

    A latitude/longitude grid clumps at the poles and would over-sample whatever happens
    to point up. The Fibonacci spiral gives near-uniform coverage, so every part of the
    surface gets a comparable chance of being seen.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    i = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / count)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def index_to_color(index: np.ndarray | int) -> np.ndarray:
    """Encode 0-based face indices as RGB bytes, offset by one so 0 means background."""
    index = np.asarray(index)
    if np.any(index < 0) or np.any(index >= MAX_FACES):
        raise ValueError(f"face index out of encodable range 0..{MAX_FACES - 1}")
    shifted = index.astype(np.int64) + 1
    return np.stack(
        [
            shifted % _CHANNEL,
            (shifted // _CHANNEL) % _CHANNEL,
            (shifted // (_CHANNEL * _CHANNEL)) % _CHANNEL,
        ],
        axis=-1,
    ).astype(np.uint8)


def color_to_index(rgb: np.ndarray) -> np.ndarray:
    """Decode RGB bytes back to face indices. Background (black) decodes to -1."""
    rgb = np.asarray(rgb).astype(np.int64)
    if rgb.shape[-1] != 3:
        raise ValueError("expected a trailing RGB axis of size 3")
    packed = rgb[..., 0] + rgb[..., 1] * _CHANNEL + rgb[..., 2] * _CHANNEL * _CHANNEL
    return packed - 1


def visible_faces(buffers: list[np.ndarray], face_count: int) -> np.ndarray:
    """Boolean mask of faces observed in at least one face-ID buffer.

    Each buffer is an HxWx3 uint8 image whose pixels encode face indices.
    """
    if face_count < 0:
        raise ValueError("face_count must be non-negative")
    seen = np.zeros(face_count, dtype=bool)
    for buffer in buffers:
        idx = color_to_index(buffer).ravel()
        idx = idx[(idx >= 0) & (idx < face_count)]
        seen[idx] = True
    return seen


def cull_report(seen: np.ndarray) -> str:
    total = int(seen.size)
    kept = int(seen.sum())
    pct = 100.0 * kept / total if total else 0.0
    return f"{kept:,} of {total:,} faces visible ({pct:.1f}%); culling {total - kept:,}"
