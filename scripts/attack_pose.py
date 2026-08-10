"""Pose curves for a quadruped slam attack, as pure functions.

Kept separate from the Blender code deliberately. Animation authored inside a template
string cannot be imported, so it cannot be tested, and a gait that reads plausibly in
code can still be wrong in ways only a test catches -- an anticipation that leans the
same way as the strike, a wind-up faster than the blow, a cycle that does not return to
rest and so pops on loop. `blender_attack_cycle.py` holds only the thin `bpy` layer.

The attack is a rear-and-slam, which is what a heavy quadruped brute does: sit back on
the hindquarters, haul the forelimbs up, then drive them down and forward. Four phases,
in normalised time:

    0.00 - 0.30   anticipation -- crouch back, head drops, weight shifts to the hind legs
    0.30 - 0.46   rear up      -- forelimbs lift, spine pitches back, jaw opens
    0.46 - 0.56   strike       -- the blow. Short, so it reads as fast
    0.56 - 1.00   recover      -- settle back to rest

Angles are degrees. Positive X rotation pitches a bone forward/down in Blender's bone
space for this rig, which is why the strike values are positive and the wind-up negative.
"""

from __future__ import annotations

ANTICIPATE_END = 0.30
REAR_END = 0.46
STRIKE_END = 0.56

# Sign convention, MEASURED on the rig rather than assumed -- pose one bone, read where
# the mesh actually goes. Guessing produced an attack that looked arthritic:
#
#   spine_01 / spine_02   negative -> chest rises, back arches BACK
#   neck / head           negative -> head tilts back and up
#   front*_upperarm       POSITIVE -> paw lifts and tucks under the body
#                         negative -> paw drives down and FORWARD
#
# The forelimbs run opposite to the spine, which is the trap: a first pass used negative
# for the rear-up, so the arms reached down and forward while the body tried to rear.
#
# (bone, anticipation, rear-up, strike, settle) -- degrees, rest is 0.
# Anticipation is a small move in the rear-up direction, so the body gathers before it
# commits; winding up towards the blow reads as a stumble.
CURVE: dict[str, tuple[float, float, float, float]] = {
    # The strike values are deliberately much smaller than the rear-up ones. Driving
    # spine, head and forelimbs all hard forward at once buries the face in the floor --
    # the blow should land in front of the animal, not through it.
    "spine_01": (-9.0, -34.0, 8.0, 2.0),
    "spine_02": (-6.0, -22.0, 6.0, 1.5),
    "neck": (-8.0, -30.0, 8.0, 2.0),
    "head": (-6.0, -34.0, 10.0, 2.0),
    "jaw": (0.0, 26.0, 14.0, 0.0),
    "frontL_upperarm": (12.0, 74.0, -26.0, 4.0),
    "frontR_upperarm": (12.0, 74.0, -26.0, 4.0),
    "frontL_forearm": (8.0, 44.0, -14.0, 2.0),
    "frontR_forearm": (8.0, 44.0, -14.0, 2.0),
    "frontL_paw": (0.0, 20.0, -8.0, 0.0),
    "frontR_paw": (0.0, 20.0, -8.0, 0.0),
    # The hindquarters sink to take the weight, then drive back up through the blow.
    "backL_thigh": (-14.0, -24.0, -6.0, -2.0),
    "backR_thigh": (-14.0, -24.0, -6.0, -2.0),
    "backL_shin": (18.0, 28.0, 8.0, 2.0),
    "backR_shin": (18.0, 28.0, 8.0, 2.0),
}


def ease(t: float) -> float:
    """Smoothstep. Eases both ends so a phase does not start or stop with a jerk."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    """Fast to start, slow to finish -- what a blow landing looks like."""
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def phase(t: float) -> str:
    """Which phase a normalised time falls in."""
    if t < ANTICIPATE_END:
        return "anticipate"
    if t < REAR_END:
        return "rear"
    if t < STRIKE_END:
        return "strike"
    return "recover"


def _blend(t: float, rest: float, a: float, b: float, c: float, d: float) -> float:
    """Value of one bone's curve at normalised time t."""
    if t <= 0.0 or t >= 1.0:
        return rest
    if t < ANTICIPATE_END:
        return rest + (a - rest) * ease(t / ANTICIPATE_END)
    if t < REAR_END:
        return a + (b - a) * ease((t - ANTICIPATE_END) / (REAR_END - ANTICIPATE_END))
    if t < STRIKE_END:
        # ease_out, not ease: the blow must be quickest at its start, then land heavy.
        return b + (c - b) * ease_out((t - REAR_END) / (STRIKE_END - REAR_END))
    tail = (t - STRIKE_END) / (1.0 - STRIKE_END)
    if tail < 0.35:
        return c + (d - c) * ease(tail / 0.35)
    return d + (rest - d) * ease((tail - 0.35) / 0.65)


def attack_pose(t: float) -> dict[str, float]:
    """X rotation in degrees for every driven bone at normalised time t in [0, 1]."""
    return {bone: _blend(t, 0.0, *values) for bone, values in CURVE.items()}


def sample(frames: int) -> list[dict[str, float]]:
    """The whole cycle, one pose per frame. Frame 0 and frame `frames` both sit at rest."""
    if frames < 2:
        raise ValueError("an attack needs at least two frames")
    return [attack_pose(i / frames) for i in range(frames + 1)]
