"""Packing a Kimodo clip: the retarget must read the same motion back from either format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kimodo_retarget as kr
import pack_kimodo_clip as pk


def _random_rotations(shape, seed=0):
    q = np.random.default_rng(seed).normal(size=(*shape, 4))
    return kr.quaternions_to_matrices(q)


def _angle(a, b):
    r = np.einsum("...ji,...jk->...ik", a, b)
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def test_quaternion_round_trip_is_lossless_in_double():
    m = _random_rotations((50, 77))
    assert _angle(m, kr.quaternions_to_matrices(kr.matrices_to_quaternions(m))).max() < 1e-4  # arccos noise


def test_packed_clip_reads_back_within_a_tenth_of_a_degree(tmp_path):
    t = 20
    g = _random_rotations((t, 77), seed=1).astype(np.float32)
    x = np.random.default_rng(2).normal(size=(t, 77, 3)).astype(np.float32)
    kimodo = tmp_path / "kimodo.npz"
    np.savez(kimodo, global_rot_mats=g, posed_joints=x, foot_contacts=np.zeros((t, 6)))
    packed = tmp_path / "packed.npz"
    assert pk.main([str(kimodo), str(packed)]) == 0
    g1, j1, h1 = kr.load_motion(kimodo)
    g2, j2, h2 = kr.load_motion(packed)
    assert _angle(g1, g2).max() < 0.1
    assert np.allclose(j1, j2, atol=1e-6) and np.allclose(h1, h2, atol=1e-6)
    assert np.allclose(h1, x[:, kr.SOMA_INDEX["Hips"]])
    assert packed.stat().st_size < kimodo.stat().st_size


def test_multi_sample_kimodo_output_uses_the_first_sample(tmp_path):
    g = _random_rotations((2, 5, 77), seed=3)
    x = np.zeros((2, 5, 77, 3))
    x[1] += 9.0
    path = tmp_path / "two.npz"
    np.savez(path, global_rot_mats=g, posed_joints=x)
    _, joints0, _ = kr.load_motion(path)
    assert np.allclose(joints0, 0.0)
