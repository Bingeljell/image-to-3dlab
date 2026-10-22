"""Calm a finished mesh's surface where the picture paints: Finish's High detail.

Pixal3D's raw faces carry small ridges and dents (goggle-shelf eyes, lip creases) that the
picture already draws with its own shading. Painted onto that surface, a picture a pixel
off reads as scuffs, tears and stretches at 45-90 degrees. Measured 2026-10-20 on seven
characters: Pixal3D's own clean-up at 50k plus ten rounds of this smoothing, only where the
front camera looks, gave the cleanest faces on every one; smoothing at 20k ate real
features (a gator's teeth) and never beat plain 20k. Bodies keep their armour and folds.

The smoothing is Taubin's: each round shrinks toward the neighbours' average, then grows
back by slightly more, so small bumps go and the shape keeps its size. Vertices split by
UV seams are welded by position first, so a seam never opens.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from image_to_3dlab import photo_paint as pp

LAMBDA, MU = 0.5, -0.53
HIGH_ROUNDS = 10
HIGH_FACES = 50000
# Facing ramp: a surface turned more than this far from the camera gets no smoothing,
# and full smoothing once it faces the camera this squarely (cosines).
FACING_FROM, FACING_FULL = 0.1, 0.4


def weld(positions: np.ndarray) -> np.ndarray:
    """Index of each vertex's welded twin: vertices at the same position share one."""
    grid = np.round(positions / (max(np.ptp(positions), 1e-12) * 1e-6)).astype(np.int64)
    return np.unique(grid, axis=0, return_inverse=True)[1].ravel()


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit normals."""
    normals = np.zeros_like(positions, dtype=np.float64)
    face_n = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]],
                      positions[faces[:, 2]] - positions[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def taubin(positions: np.ndarray, faces: np.ndarray, weights: np.ndarray,
           rounds: int) -> np.ndarray:
    """Smoothed positions; `weights` (0..1 per vertex) scales how far each one may move."""
    if rounds <= 0:
        return positions
    key = weld(positions)
    count = key.max() + 1
    pos = np.zeros((count, 3))
    np.add.at(pos, key, positions)
    pos /= np.bincount(key, minlength=count)[:, None]
    weight = np.zeros((count, 1))
    np.maximum.at(weight, key, np.asarray(weights, dtype=np.float64)[:, None])
    welded = key[faces]
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    a, b = edges[:, 0], edges[:, 1]
    degree = np.bincount(np.concatenate([a, b]), minlength=count)[:, None].astype(np.float64)
    for _ in range(rounds):
        for factor in (LAMBDA, MU):
            average = np.zeros_like(pos)
            np.add.at(average, a, pos[b])
            np.add.at(average, b, pos[a])
            average /= np.maximum(degree, 1.0)
            pos = pos + factor * weight * (average - pos)
    return pos[key]


def facing_weights(positions: np.ndarray, normals: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """0..1 per vertex: how squarely the surface faces `camera` (same coordinates)."""
    to_camera = camera - positions
    to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
    cosine = (normals * to_camera).sum(1)
    return np.clip((cosine - FACING_FROM) / (FACING_FULL - FACING_FROM), 0.0, 1.0)


def front_weights(glb: Path, views_dir: Path) -> np.ndarray:
    """Facing weights for a GLB against the first (front) camera of a Pixel Match views dir."""
    positions, _uvs, faces, _tex = pp.read_glb(glb)
    faces = faces.reshape(-1, 3).astype(np.int64)
    views, mesh_scale = pp.load_views(views_dir)
    key = weld(positions)
    welded = np.zeros((key.max() + 1, 3))
    welded[key] = positions
    normals = vertex_normals(welded, key[faces])[key]
    return facing_weights(pp.to_view_space(positions, mesh_scale=mesh_scale),
                          pp.to_view_space(normals), views[0].position)


def _attribute_slots(doc: dict, index: int) -> tuple[int, int, int]:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    return (view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
            view.get("byteStride", 12), accessor["count"])


def _write_vec3(binary: bytearray, slots: tuple[int, int, int], values: np.ndarray) -> None:
    start, stride, count = slots
    for i in range(count):
        struct.pack_into("<3f", binary, start + i * stride, *values[i])


def calm_glb(source: Path, output: Path, weights: np.ndarray, rounds: int = HIGH_ROUNDS) -> dict:
    """Write `source` with its surface calmed by `weights`; UVs, indices and textures as they were."""
    positions, _uvs, faces, _tex = pp.read_glb(source)
    faces = faces.reshape(-1, 3).astype(np.int64)
    calmed = taubin(positions, faces, weights, rounds)
    doc, binary = pp._split_glb(Path(source).read_bytes())
    binary = bytearray(binary)
    attributes = doc["meshes"][0]["primitives"][0]["attributes"]
    _write_vec3(binary, _attribute_slots(doc, attributes["POSITION"]), calmed)
    if "NORMAL" in attributes:
        key = weld(calmed)
        welded = np.zeros((key.max() + 1, 3))
        welded[key] = calmed
        _write_vec3(binary, _attribute_slots(doc, attributes["NORMAL"]),
                    vertex_normals(welded, key[faces])[key])
    accessor = doc["accessors"][attributes["POSITION"]]
    accessor["min"], accessor["max"] = calmed.min(0).tolist(), calmed.max(0).tolist()
    Path(output).write_bytes(pp._join_glb(doc, bytes(binary)))
    moved = np.linalg.norm(calmed - positions, axis=1)
    return {"rounds": rounds, "vertices": int((np.asarray(weights) > 0.5).sum()),
            "moved_max_pct": round(100 * float(moved.max()) / max(float(np.ptp(positions[:, 1])), 1e-12), 3)}
