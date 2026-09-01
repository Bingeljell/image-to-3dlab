"""Pose curves for a ram headbutt/charge on the Rigify-generated rig, as pure functions.

Built fresh for this rig, not adapted from the custom-rig headbutt (ram_headbutt_pose.py)
or the gorilla one (attack_pose.py) -- the bone set and what actually deforms the mesh
are different enough that porting numbers across would just be guessing with extra
steps. Conventions below were each confirmed by test-rotating the live rig, not assumed:

* Legs: local X rotation swings them; positive X on a *_fk bone (front or back) swings
  that segment toward the character's front (confirmed via frontL_toe world position).
* Spine: local X rotation on spine_fk bones; positive pitches that segment DOWN and
  forward, negative arches it UP and back (confirmed on spine_fk.006).
* No head/neck articulation is used here at all -- `head`/`neck` pose bones exist with
  real widgets, but confirmed by direct test that posing them changes zero DEF-spine
  bone rotations. They don't move the mesh. The spine_fk chain is the only thing on this
  rig that actually drives visible body deformation, so the "headbutt" reads as a body
  arc and charge rather than an independent head snap -- an honest limitation of this
  particular Rigify metarig (Basic Quadruped has no head/neck deform bones), not a bug
  in this script.

Same four-phase shape as the custom-rig headbutt: anticipate (crouch/gather) -> rear
(body arches back, front legs reach out) -> strike (drive forward and down, front legs
land) -> recover (settle to rest).
"""

from __future__ import annotations


ANTICIPATE_END = 0.30
REAR_END = 0.46
STRIKE_END = 0.56

# (bone, anticipation, rear, strike, settle) -- degrees, rest is 0.
SPINE_FK = ["spine_fk.004", "spine_fk.005", "spine_fk.006", "spine_fk.007", "spine_fk.008"]
SPINE_CURVE = {
    # Later segments (.007/.008, nearer the neck) get more motion than the base
    # (.004, nearer the hips) -- the arc should build toward the front, not bend evenly.
    "spine_fk.004": (-4.0, -10.0, 8.0, 1.0),
    "spine_fk.005": (-6.0, -16.0, 12.0, 1.5),
    "spine_fk.006": (-8.0, -22.0, 16.0, 2.0),
    "spine_fk.007": (-10.0, -28.0, 20.0, 2.5),
    "spine_fk.008": (-12.0, -32.0, 24.0, 3.0),
}

FRONT_LEGS = {
    "frontL": ("front_thigh_fk.L", "front_shin_fk.L", "front_foot_fk.L"),
    "frontR": ("front_thigh_fk.R", "front_shin_fk.R", "front_foot_fk.R"),
}
BACK_LEGS = {
    "backL": ("thigh_fk.L", "shin_fk.L", "foot_fk.L"),
    "backR": ("thigh_fk.R", "shin_fk.R", "foot_fk.R"),
}

# Front legs are landing gear: reach forward (positive) through anticipate+rear, then
# fold BACK toward zero/negative at strike so the paw doesn't overshoot through the
# floor while the spine is still driving the body down -- same lesson learned the hard
# way on the custom rig, applied here from the start instead of re-discovering it.
FRONT_UPPER = (10.0, 32.0, -8.0, 2.0)
FRONT_MID = (6.0, 18.0, -10.0, 1.0)
FRONT_LOW = (0.0, 8.0, -12.0, 0.0)

# Back legs crouch during anticipation (gathering weight), then drive/extend at strike
# (push-off powering the charge forward).
BACK_UPPER = (-14.0, -20.0, 18.0, -2.0)
BACK_MID = (16.0, 22.0, -14.0, 2.0)
BACK_LOW = (0.0, 6.0, -6.0, 0.0)


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def _blend(t: float, a: float, b: float, c: float, d: float) -> float:
    rest = 0.0
    if t <= 0.0 or t >= 1.0:
        return rest
    if t < ANTICIPATE_END:
        return rest + (a - rest) * ease(t / ANTICIPATE_END)
    if t < REAR_END:
        return a + (b - a) * ease((t - ANTICIPATE_END) / (REAR_END - ANTICIPATE_END))
    if t < STRIKE_END:
        return b + (c - b) * ease_out((t - REAR_END) / (STRIKE_END - REAR_END))
    tail = (t - STRIKE_END) / (1.0 - STRIKE_END)
    if tail < 0.35:
        return c + (d - c) * ease(tail / 0.35)
    return d + (rest - d) * ease((tail - 0.35) / 0.65)


def pose_at(t: float) -> dict[str, float]:
    pose: dict[str, float] = {}
    for bone, curve in SPINE_CURVE.items():
        pose[bone] = _blend(t, *curve)
    for legs, (upper_c, mid_c, low_c) in (
        (FRONT_LEGS, (FRONT_UPPER, FRONT_MID, FRONT_LOW)),
        (BACK_LEGS, (BACK_UPPER, BACK_MID, BACK_LOW)),
    ):
        for name, (upper, mid, low) in legs.items():
            pose[upper] = _blend(t, *upper_c)
            pose[mid] = _blend(t, *mid_c)
            pose[low] = _blend(t, *low_c)
    return pose


def sample(frames: int) -> list[dict[str, float]]:
    if frames < 2:
        raise ValueError("a headbutt needs at least two frames")
    return [pose_at(i / frames) for i in range(frames + 1)]
