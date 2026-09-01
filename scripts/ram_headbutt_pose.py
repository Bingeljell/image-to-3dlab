"""Pose curves for a ram headbutt/charge attack, as pure functions.

Same structure as `attack_pose.py` (that file's rear-and-slam is for a gorilla-like
brute, not a ram — confirmed by inspection, not guessed). Kept separate and testable for
the same reason: an anticipation that leans the same way as the strike, or a cycle that
does not return to rest, is a bug a test catches and a render does not always show.

A ram's attack really is rear-then-charge, so the anticipation/rear-up beats are close
to `attack_pose.py`'s — a ram genuinely rears onto its hind legs before it hits. What
changes is the strike: the weapon is the horns, not the forelimbs. The head and neck
drive hard forward at strike, the jaw stays closed (a ram headbutts, it does not bite),
and the front legs land the charge rather than throwing a blow of their own. Four
phases, in normalised time:

    0.00 - 0.30   anticipation -- crouch back, head drops, weight shifts to the hind legs
    0.30 - 0.46   rear up      -- forelimbs lift, spine pitches back, ready to drop
    0.46 - 0.56   strike       -- horns drive forward and down, front legs land the charge
    0.56 - 1.00   recover      -- settle back to rest

Angles are degrees. Sign convention carried over from `attack_pose.py`, measured on this
same rig family:

    spine_01 / spine_02   negative -> chest rises, back arches BACK
    neck / head           negative -> head tilts back and up; POSITIVE drives forward/down
    front*_upperarm       POSITIVE -> paw lifts and tucks under the body
                           negative -> paw drives down and FORWARD
"""

from __future__ import annotations

ANTICIPATE_END = 0.30
REAR_END = 0.46
STRIKE_END = 0.56

# (bone, anticipation, rear-up, strike, settle) -- degrees, rest is 0.
# Head and neck strike values are the biggest in the whole curve on purpose -- the horns
# are the weapon, everything else is just carrying them into position. Kept short of
# attack_pose.py's caution line ("driving spine, head and forelimbs all hard forward at
# once buries the face in the floor") by leaving the front legs' strike values negative
# but modest: they plant to catch the charge's weight, they don't throw their own blow.
CURVE: dict[str, tuple[float, float, float, float]] = {
    "spine_01": (-9.0, -30.0, 20.0, 2.0),
    "spine_02": (-6.0, -20.0, 14.0, 1.5),
    "neck": (-8.0, -28.0, 24.0, 2.0),
    "head": (-6.0, -30.0, 30.0, 2.0),
    # A ram headbutts with a closed mouth -- the gorilla curve's open-jaw roar/bite read
    # is wrong here, so this stays near zero throughout instead of opening at strike.
    "jaw": (0.0, 3.0, 0.0, 0.0),
    # Rear-up is negative (forward/down), same direction as the strike -- the legs reach
    # OUT in front to brace for the charge, not tuck backward under the body. This is the
    # opposite sign from attack_pose.py's gorilla curve, which tucks the forelimbs in
    # before throwing them as the weapon; here the legs are just landing gear, so they
    # extend continuously forward from anticipation through the strike.
    #
    # Anticipation raised (12 -> 24) so the legs visibly lift in step with the spine's
    # arch instead of lagging behind it. Rear-up capped (-70 -> -42) -- past that the legs
    # read as an exaggerated yoga pose, not a rearing animal. At strike the upperarm folds
    # BACK UP (toward positive) while the wrist (paw) folds down hard: the shoulder
    # shortens the leg so the paw lands at ground level instead of overshooting through
    # the floor, while the neck and head keep driving down independently for the headbutt.
    # Strike targets set to match where old frame 24 actually landed (computed from the
    # pre-hold curve, ease_out((24-1)/48 through the old shared timing) -- NOT the fully
    # realized 55/8/-55 fold, which the leg used to only approach, never fully reach,
    # before the hold made "arrive early" mean "arrive early at the far more extreme
    # value nobody had actually approved yet."
    "frontL_upperarm": (24.0, -42.0, 3.8, 4.0),
    "frontR_upperarm": (24.0, -42.0, 3.8, 4.0),
    "frontL_forearm": (14.0, -24.0, -8.9, 2.0),
    "frontR_forearm": (14.0, -24.0, -8.9, 2.0),
    "frontL_paw": (0.0, -11.0, -31.8, 0.0),
    "frontR_paw": (0.0, -11.0, -31.8, 0.0),
    # A bit more hindquarter drive than the gorilla curve -- the charge's forward
    # momentum comes from the hind legs pushing off, not just the body toppling forward.
    "backL_thigh": (-14.0, -22.0, -12.0, -2.0),
    "backR_thigh": (-14.0, -22.0, -12.0, -2.0),
    "backL_shin": (18.0, 26.0, 14.0, 2.0),
    "backR_shin": (18.0, 26.0, 14.0, 2.0),
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


# Per-bone timing overrides -- (rear_end, strike_end, hold_end), any of which can be
# None to fall back to the global default. Two different problems, both about a single
# shared REAR_END/STRIKE_END forcing every bone through the same timing:
#
# neck/head: should stop arching BACK earlier than the rest of the rear-up (by frame 19
# of 49, t=0.375) rather than riding the shared boundary at t=REAR_END (~frame 23) --
# otherwise the head is still winding up backward after the eye has already read "ready."
#
# front legs + spine: should reach their strike (planted) position earlier (by frame 25,
# t=0.5) and then HOLD there rather than continuing to ease toward STRIKE_END (~frame 28)
# -- a leg that keeps drifting after it's already planted reads as floaty, not a stomp.
# The hold_end defaults to STRIKE_END, so recovery still begins on the normal schedule.
#
# spine_01/spine_02 are in this group too, and that's the part that actually matters:
# front*_upperarm is parented to spine_02, so a flat LOCAL hold on the leg alone still
# lets the paw's WORLD position sink if the spine keeps pitching forward underneath it
# -- confirmed by measuring frontL_toe.z, which kept dropping through frame 28 even with
# the leg's own rotation frozen. Freezing the spine at the same frame is what actually
# glues the foot to the ground. neck/head hang off spine_02 too but are NOT ancestors of
# the legs, so they're free to keep driving forward on their own separate timing (see
# their rear_end override above) while the shoulder holds still underneath them.
PHASE_OVERRIDE: dict[str, tuple[float | None, float | None, float | None]] = {
    "neck": (0.375, None, None),
    "head": (0.375, None, None),
    "spine_01": (None, 0.5, STRIKE_END),
    "spine_02": (None, 0.5, STRIKE_END),
    "frontL_upperarm": (None, 0.5, STRIKE_END),
    "frontR_upperarm": (None, 0.5, STRIKE_END),
    "frontL_forearm": (None, 0.5, STRIKE_END),
    "frontR_forearm": (None, 0.5, STRIKE_END),
    "frontL_paw": (None, 0.5, STRIKE_END),
    "frontR_paw": (None, 0.5, STRIKE_END),
}


def _blend(
    t: float, rest: float, a: float, b: float, c: float, d: float,
    rear_end: float | None = None, strike_end: float | None = None,
    hold_end: float | None = None,
) -> float:
    """Value of one bone's curve at normalised time t."""
    rear_end = REAR_END if rear_end is None else rear_end
    strike_end = STRIKE_END if strike_end is None else strike_end
    hold_end = strike_end if hold_end is None else hold_end
    if t <= 0.0 or t >= 1.0:
        return rest
    if t < ANTICIPATE_END:
        return rest + (a - rest) * ease(t / ANTICIPATE_END)
    if t < rear_end:
        return a + (b - a) * ease((t - ANTICIPATE_END) / (rear_end - ANTICIPATE_END))
    if t < strike_end:
        # ease_out, not ease: the blow must be quickest at its start, then land heavy.
        return b + (c - b) * ease_out((t - rear_end) / (strike_end - rear_end))
    if t < hold_end:
        return c
    tail = (t - hold_end) / (1.0 - hold_end)
    if tail < 0.35:
        return c + (d - c) * ease(tail / 0.35)
    return d + (rest - d) * ease((tail - 0.35) / 0.65)


def attack_pose(t: float) -> dict[str, float]:
    """X rotation in degrees for every driven bone at normalised time t in [0, 1]."""
    return {
        bone: _blend(t, 0.0, *values, *PHASE_OVERRIDE.get(bone, (None, None, None)))
        for bone, values in CURVE.items()
    }


def sample(frames: int) -> list[dict[str, float]]:
    """The whole cycle, one pose per frame. Frame 0 and frame `frames` both sit at rest."""
    if frames < 2:
        raise ValueError("a headbutt needs at least two frames")
    return [attack_pose(i / frames) for i in range(frames + 1)]
