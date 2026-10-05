"""Arm clearance: arms that come inside the body outline swing out, others are left alone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arm_clearance as ac

V = np.array


def _proxy(chest_radius=0.15):
    return ac.Proxy(
        body={"chest": chest_radius, "head": 0.1},
        arms={"Left": {"upper": "uarm", "fore": "farm", "hand": "hand",
                       "radius": {"uarm": 0.04, "farm": 0.035, "hand": 0.03}}},
        side_shoulder={"Left": None, "Right": None},
        margin=0.01,
    )


def _segments(elbow, wrist, tip):
    return {
        "chest": (V([0.0, 0, 1.2]), V([0.0, 0, 1.45])),
        "head": (V([0.0, 0, 1.55]), V([0.0, 0, 1.75])),
        "uarm": (V([0.2, 0, 1.45]), V(elbow)),
        "farm": (V(elbow), V(wrist)),
        "hand": (V(wrist), V(tip)),
    }


def _clearance(seg, proxy):
    """Smallest gap between any arm test point and the body, minus the required gap."""
    arm = proxy.arms["Left"]
    gaps = []
    for p, r in ac._test_points(seg, arm):
        for bone, rb in proxy.body.items():
            d, _ = ac.point_segment(p, *seg[bone])
            gaps.append(d - (rb + r + proxy.margin))
    return min(gaps)


def _swing_segments(seg, proxy, swing):
    pivot = seg["uarm"][0]
    moved = dict(seg)
    for bone in ("uarm", "farm", "hand"):
        moved[bone] = tuple(pivot + swing @ (q - pivot) for q in seg[bone])
    return moved


def test_point_segment_clamps_to_the_ends():
    d, c = ac.point_segment(V([2.0, 1, 0]), V([0.0, 0, 0]), V([1.0, 0, 0]))
    assert np.allclose(c, [1, 0, 0]) and d == pytest.approx(np.sqrt(2))


def test_a_forearm_through_the_chest_is_swung_clear():
    """The knight's punch: the forearm cuts across the chest plate."""
    proxy = _proxy()
    seg = _segments(elbow=[0.25, 0.15, 1.3], wrist=[0.02, 0.1, 1.3], tip=[-0.05, 0.12, 1.3])
    assert _clearance(seg, proxy) < 0
    swing = ac.arm_swing(seg, proxy, "Left")
    assert _clearance(_swing_segments(seg, proxy, swing), proxy) > -1e-3


def test_a_hand_in_the_face_is_swung_clear():
    proxy = _proxy()
    seg = _segments(elbow=[0.3, 0.15, 1.45], wrist=[0.08, 0.12, 1.62], tip=[0.02, 0.1, 1.66])
    assert _clearance(seg, proxy) < 0
    swing = ac.arm_swing(seg, proxy, "Left")
    assert _clearance(_swing_segments(seg, proxy, swing), proxy) > -1e-3


def test_an_arm_already_clear_is_left_alone():
    proxy = _proxy()
    seg = _segments(elbow=[0.45, 0, 1.45], wrist=[0.7, 0, 1.45], tip=[0.8, 0, 1.45])
    assert np.allclose(ac.arm_swing(seg, proxy, "Left"), np.eye(3))


def test_skin_radii_ignore_a_few_stray_vertices():
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 200)
    ring = np.c_[0.1 * np.cos(angle), 0.1 * np.sin(angle), rng.uniform(0, 1, 200)]
    points = np.r_[ring, [[2.0, 0, 0.5]]]  # one stray vertex far out
    radii = ac.skin_radii(points, ["bone"] * len(points), {"bone": (V([0.0, 0, 0]), V([0.0, 0, 1]))},
                          ["bone"], 85)
    assert radii["bone"] == pytest.approx(0.1, abs=1e-6)


def test_build_proxy_finds_body_and_arm_bones_by_role():
    mapping = {"b0": "Hips", "b1": "Chest", "b2": "Head", "b3": "LeftArm", "b4": "LeftForeArm",
               "b5": "LeftHand", "b6": "LeftShoulder"}
    segments = {b: (V([0.0, 0, i]), V([0.0, 0, i + 0.5])) for i, b in enumerate(mapping)}
    points = np.array([[0.05, 0, i + 0.2] for i in range(len(mapping))])
    proxy = ac.build_proxy(mapping, points, list(mapping), segments, height=1.8)
    assert set(proxy.body) == {"b0", "b1", "b2", "b6"}
    assert proxy.arms["Left"]["fore"] == "b4" and "Right" not in proxy.arms
    assert proxy.side_shoulder["Left"] == "b6"
    assert proxy.margin == pytest.approx(0.018)


def test_smoothing_eases_in_and_never_shrinks_a_needed_swing():
    swing = ac._exp(V([0, 0, 0.4]))
    swings = [np.eye(3)] * 5 + [swing] + [np.eye(3)] * 5
    out = ac.smooth_swings(swings, radius=3)
    angles = [np.linalg.norm(ac._log(s)) for s in out]
    assert angles[5] == pytest.approx(0.4)
    assert 0 < angles[3] < angles[4] < 0.4  # eases in
    assert angles[0] == pytest.approx(0)


def test_log_and_exp_round_trip():
    v = V([0.3, -0.2, 0.5])
    assert np.allclose(ac._log(ac._exp(v)), v)
