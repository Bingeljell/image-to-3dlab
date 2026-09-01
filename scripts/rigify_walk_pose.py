"""Pose curves for a quadruped trot on a Rigify-generated rig, as pure functions.

Same trot math as blender_walk_cycle.py (swing = amplitude*cos(theta) per leg, phase-
offset by diagonal pair, plus a fold on the lower joints), retargeted to Rigify's FK
control bones instead of our custom rig's bone names. Two real differences from the
custom rig, both confirmed by testing on the live rig rather than assumed:

* Rigify's legs are IK by default (the `IK_FK` custom property on each `*_parent` bone
  defaults to 0.0). FK rotations on `*_fk` bones are silently ignored by the deform
  bones until that slider is set to 1.0 -- this script assumes that's already done.
* The swing axis is local X here too (confirmed by test-rotating front_thigh_fk.L on
  each axis and checking which one actually moved the bone tip forward/back), so the
  same sign convention as the custom rig's swing math applies directly.

No toe bone -- Rigify's generated quadruped FK chain stops at the foot, so there's no
separate toe-roll control to drive (unlike the custom rig's heel+toe split).
"""

from __future__ import annotations

import math

PHASES = {
    "walk": {"backL": 0.0, "frontL": 0.25, "backR": 0.5, "frontR": 0.75},
    "trot": {"frontL": 0.0, "backR": 0.0, "frontR": 0.5, "backL": 0.5},
}

# (fk_upper, fk_mid, fk_low, swing_amp, mid_fold_amp, low_fold_amp)
CHAIN = {
    "frontL": ("front_thigh_fk.L", "front_shin_fk.L", "front_foot_fk.L"),
    "frontR": ("front_thigh_fk.R", "front_shin_fk.R", "front_foot_fk.R"),
    "backL": ("thigh_fk.L", "shin_fk.L", "foot_fk.L"),
    "backR": ("thigh_fk.R", "shin_fk.R", "foot_fk.R"),
}

SPINE_FK = ["spine_fk.004", "spine_fk.005", "spine_fk.006", "spine_fk.007", "spine_fk.008"]


def sample(
    frames: int,
    gait: str = "trot",
    swing_front: float = 30.0,
    swing_back: float = 30.0,
    bend_front: float = 35.0,
    bend_back: float = 35.0,
    low_fold: float = 30.0,
    base_bend: float = 8.0,
    spine_sway: float = 4.0,
) -> list[dict[str, float]]:
    """One pose per frame, X-rotation degrees for every driven FK bone."""
    if frames < 2:
        raise ValueError("a gait cycle needs at least two frames")
    phase = PHASES[gait]
    out = []
    for frame in range(frames + 1):
        t = frame / frames
        pose: dict[str, float] = {}
        for leg, ph in phase.items():
            upper, mid, low = CHAIN[leg]
            theta = 2.0 * math.pi * (ph - t)
            swing_amp = swing_front if leg.startswith("front") else swing_back
            mid_amp = bend_front if leg.startswith("front") else bend_back
            swing = swing_amp * math.cos(theta)
            lift = max(0.0, math.sin(theta)) ** 0.7
            low_lift = max(0.0, math.sin(theta - 0.5)) ** 0.7
            pose[upper] = swing
            pose[mid] = base_bend + mid_amp * lift
            pose[low] = base_bend * 0.5 + low_fold * low_lift
        # light body undulation, twice per stride, small
        for i, bone in enumerate(SPINE_FK):
            pose[bone] = spine_sway * math.sin(2.0 * math.pi * t * 2 + i * 0.3) * 0.3
        out.append(pose)
    return out
