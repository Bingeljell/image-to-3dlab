#!/usr/bin/env python3
"""Gait curves for the Rigify basic quadruped metarig, as pure functions.

Every creature in this project is rigged from Rigify's `basic_quadruped` metarig, so a
gait authored against that bone set transfers to all of them -- but only if it is
expressed as *proportions*. The ram and the fox are within 5% in height yet have
inverted limb segments (the ram's shin is its longest hind bone, the fox's metatarsus
is), so copying one creature's baked curves onto another over-reaches its IK targets and
lifts the feet off the ground. Everything here is therefore a multiple of the rig's own
leg length or body height, measured per rig.

The constants come from `RamWalk_Natural_Final`, the hand-authored walk that was accepted
after the generic FK cycle was rejected. That cycle failed for three measurable reasons,
all of which this module exists to avoid:

* it lifted the feet 8.72% of body height (the accepted walk uses 2.96%),
* it strode 3.3x too far,
* and it left `hips`, `torso` and `chest` with *zero* motion, so the body was welded in
  space while the legs swung -- which reads as a stiff march rather than an animal.

The legs are driven as **IK foot targets**, not FK rotations: a planted foot is what lets
the body ride over it. The body chain is driven too, and that is not decoration -- it is
the difference between an animal and a marionette.

No `bpy` here. The driver (`blender_quadruped_walk.py`) maps these world-space offsets
into each bone's local space and keys them.
"""

from __future__ import annotations

import math

# Limb keys: F/B = front/back, L/R = left/right.
LIMBS = ("FL", "FR", "BL", "BR")

# Rigify basic_quadruped generated-rig control names.
FOOT_IK = {"FL": "front_foot_ik.L", "FR": "front_foot_ik.R",
           "BL": "foot_ik.L", "BR": "foot_ik.R"}
BODY_CHAIN = ("hips", "torso", "chest", "neck", "head")

# Footfall phase as a fraction of the cycle, plus duty factor (fraction of the cycle a
# foot spends planted). Walk is the four-beat lateral sequence BL -> FL -> BR -> FR that
# mammals use at low speed; trot is the two-beat diagonal pairing used at intermediate
# speed.
# `bob_scale` and `lift_scale` multiply the walk-derived proportions. They are the whole
# difference between a walk and a scamper on this animal: a fox's foreleg rests at 94.3%
# of full IK extension, so it cannot reach further without locking the elbow. Its faster
# gaits buy ground with vertical action and cadence instead of stride -- which is exactly
# what the reference creature's own clips show, measured against its 1.935 height:
# walk 0.62% bob / 3.5% foot lift / 0.75 planted, run 5.89% / 17% / 0.16 planted.
# `scamper` sits between them, and is the fox's real travel gait: canids trot rather than
# walk to cover ground, in a direct-register pattern where the hind foot lands in the
# print the front foot just left.
GAITS = {
    "walk": {"phases": {"BL": 0.00, "FL": 0.25, "BR": 0.50, "FR": 0.75},
             "duty": 0.65, "bob_cycles": 2, "sway_cycles": 1,
             "bob_scale": 1.0, "lift_scale": 1.0, "frames": 32},
    "trot": {"phases": {"FL": 0.00, "BR": 0.00, "FR": 0.50, "BL": 0.50},
             "duty": 0.50, "bob_cycles": 2, "sway_cycles": 0,
             "bob_scale": 1.8, "lift_scale": 1.15, "frames": 26},
    "scamper": {"phases": {"FL": 0.00, "BR": 0.00, "FR": 0.50, "BL": 0.50},
                "duty": 0.45, "bob_cycles": 2, "sway_cycles": 0,
                "bob_scale": 2.6, "lift_scale": 1.30, "frames": 22},
}

# Proportions measured from RamWalk_Natural_Final.
BOB_PER_HEIGHT = 0.0077
HIND_STRIDE_PER_LEG = 0.2473
HIND_LIFT_PER_LEG = 0.0664
FRONT_STRIDE_PER_LEG = 0.2191
FRONT_LIFT_PER_LEG = 0.0602

# Foot lift is a CONSEQUENCE of stride, not a free number. A foot rises only far enough
# to clear the ground while it swings forward, so the two move together: grow the stride
# and the lift follows. Set independently, the ratio drifts, and once lift approaches
# stride the foot goes up as far as it goes forward -- which is the definition of
# marching on the spot. A cycle built that way hit 1.18 and read as a parade step.
#
# The accepted reference walk lands on 0.268 (hind) and 0.275 (front), independently, on
# an animal of different proportions -- so this is the invariant to hold, not the
# absolute lift height.
HIND_LIFT_PER_STRIDE = 0.268
FRONT_LIFT_PER_STRIDE = 0.275

# Lateral body roll, as a fraction of the vertical bob. A walking quadruped does roll
# over its supporting legs, but the roll is far smaller than the rise and fall. Matching
# the two (sway = bob) puts a hip swing at stride frequency, and with the tail swinging
# at that same frequency the pair reinforce into an unmistakable disco strut -- the first
# cycle built this way was nicknamed the Bee Gees walk and kept as a reference for what
# not to ship.
SWAY_PER_BOB = 0.35


def limb_phase(gait: str, limb: str) -> float:
    if gait not in GAITS:
        raise ValueError(f"unknown gait {gait!r}; expected one of {sorted(GAITS)}")
    if limb not in LIMBS:
        raise ValueError(f"unknown limb {limb!r}; expected one of {LIMBS}")
    return GAITS[gait]["phases"][limb]


def is_planted(u: float, phase: float, duty: float) -> bool:
    """True while this foot is in stance, i.e. carrying weight on the ground."""
    return ((u - phase) % 1.0) < duty


def foot_offset(u: float, phase: float, duty: float, stride: float,
                lift: float) -> tuple[float, float]:
    """(fore_aft, up) offset of one foot from its rest position at cycle time `u`.

    Stance drags the planted foot backwards under a body that is moving forwards, so the
    cycle is authored in place; swing arcs it forward again. `up` is exactly zero for the
    whole of stance, which is what keeps the foot from skating.
    """
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must lie in (0, 1), got {duty}")
    local = (u - phase) % 1.0
    if local < duty:
        progress = local / duty
        return stride * (0.5 - progress), 0.0
    progress = (local - duty) / (1.0 - duty)
    return stride * (progress - 0.5), lift * math.sin(math.pi * progress)


def body_bob(u: float, gait: str, amplitude: float) -> float:
    """Vertical travel of the body. Two oscillations per stride for both gaits."""
    cycles = GAITS[gait]["bob_cycles"]
    return amplitude * 0.5 * math.sin(2.0 * math.pi * cycles * u)


def body_sway(u: float, gait: str, amplitude: float,
              per_bob: float = SWAY_PER_BOB) -> float:
    """Lateral weight shift. A four-beat walk rolls once per stride; a trot does not.

    Deliberately a fraction of the vertical bob: rolling as far as the body rises reads
    as a strut rather than a walk.
    """
    cycles = GAITS[gait]["sway_cycles"]
    if not cycles:
        return 0.0
    return amplitude * per_bob * 0.5 * math.sin(2.0 * math.pi * cycles * u)


def stride_and_lift(front_leg_length: float, hind_leg_length: float) -> dict[str, float]:
    """Per-rig stride and lift, as multiples of that rig's own leg lengths."""
    for name, value in (("front", front_leg_length), ("hind", hind_leg_length)):
        if value <= 0:
            raise ValueError(f"{name} leg length must be positive, got {value}")
    front_stride = front_leg_length * FRONT_STRIDE_PER_LEG
    hind_stride = hind_leg_length * HIND_STRIDE_PER_LEG
    return lift_from_stride(front_stride, hind_stride)


def lift_from_stride(front_stride: float, hind_stride: float,
                     clearance: float = 1.0) -> dict[str, float]:
    """Pair each stride with the lift that clears the ground for it."""
    return {
        "front_stride": front_stride,
        "front_lift": front_stride * FRONT_LIFT_PER_STRIDE * clearance,
        "hind_stride": hind_stride,
        "hind_lift": hind_stride * HIND_LIFT_PER_STRIDE * clearance,
    }


# The forelimb is driven in FK, not IK. On an IK foot target with a static shoulder the
# only way for the paw to reach forward is for the wrist and elbow to bend forward, which
# rakes the whole lower leg ahead of the body -- measured at 15.7 degrees from vertical
# against 8.3 at rest on the moss fox. A real foreleg reaches by swinging from the
# shoulder with the column kept straight, which is what these angles do, and it is why the
# accepted reference walk drives the front limbs in FK while the hind stay on IK.
FRONT_FK = {"FL": ("front_thigh_fk.L", "front_shin_fk.L", "front_foot_fk.L"),
            "FR": ("front_thigh_fk.R", "front_shin_fk.R", "front_foot_fk.R")}
SHOULDER = {"FL": "shoulder.L", "FR": "shoulder.R"}
# Deform-chain bones used to audit the result: the paw that must stay on the ground, and
# the limb column whose lean betrays an ankle-led foreleg.
PAW = {"FL": "ORG-front_toe.L", "FR": "ORG-front_toe.R",
       "BL": "ORG-toe.L", "BR": "ORG-toe.R"}
RAKE_CHAIN = ("ORG-front_thigh.L", "ORG-front_foot.L")

# Gain on the shoulder rotation that absorbs a foreleg's fore-aft reach, so the limb
# reaches from the shoulder rather than by bending the wrist forward. Driving the foot IK
# alone against a static shoulder rakes the lower foreleg ahead of the body: measured on
# the moss fox, the lower leg swung 1.3-15.7 degrees from vertical against 8.3 at rest.
#
# DEFAULTS TO ZERO, because measurement says it does harm. Rotating the shoulder by as
# little as 0.08 rad, with nothing else animated, hyperextends the elbow from 141 to 180
# degrees and collapses the wrist from 174 to 116 -- the ankle-hinged-out-in-front defect,
# caused by the very control meant to prevent it. Moving the foot IK target the full
# stride, by contrast, moves the wrist only 173.7 -> 175.9. Left as an opt-in knob rather
# than deleted, since the idea is sound on a rig with reach headroom to spend.
SHOULDER_SHARE = 0.0
HIND_LIMBS = ("BL", "BR")
FRONT_LIMBS = ("FL", "FR")


def front_fk_angles(u: float, phase: float, duty: float, *, stride: float, lift: float,
                    leg_length: float) -> dict[str, float]:
    """Shoulder-led FK angles (radians) for one foreleg at cycle time `u`.

    `thigh` swings the whole column fore and aft so the paw covers `stride` without the
    wrist leading. `shin` folds only during swing, to lift the paw over the ground, and
    `foot` counter-rotates so the paw stays roughly parallel to it rather than pointing
    wherever the limb happens to swing.
    """
    if leg_length <= 0:
        raise ValueError(f"leg length must be positive, got {leg_length}")
    half_swing = math.asin(min(0.99, (stride * 0.5) / leg_length))
    fold_peak = math.asin(min(0.99, lift / leg_length))
    local = (u - phase) % 1.0
    if local < duty:
        swing = half_swing * (1.0 - 2.0 * (local / duty))
        fold = 0.0
    else:
        progress = (local - duty) / (1.0 - duty)
        swing = half_swing * (2.0 * progress - 1.0)
        fold = fold_peak * math.sin(math.pi * progress)
    return {"thigh": swing, "shin": -fold, "foot": -swing * 0.5 + fold * 0.5}

def shoulder_angle(fore_aft: float, leg_length: float,
                   share: float = SHOULDER_SHARE) -> float:
    """Shoulder rotation (radians) that absorbs `share` of a foreleg's reach."""
    if leg_length <= 0:
        raise ValueError(f"leg length must be positive, got {leg_length}")
    return math.asin(min(0.99, abs(fore_aft) / leg_length)) * share * (
        1.0 if fore_aft >= 0 else -1.0)

# Every bound here is a failure this pipeline actually shipped, kept as a number so it
# cannot come back quietly. The generic Rigify cycle was rejected on the ram and then
# again on the fox for the same reasons, and each was invisible in the report of the run
# that produced it -- the action looked authored, the curves were there, and the result
# was wrong.
GAIT_LIMITS = {
    # Rejected cycle lifted the feet 8.72% of body height; the accepted walk uses 2.96%.
    "foot_lift_pct_height": (0.015, 0.055),
    # Rejected cycle strode 3.3x the accepted walk.
    "stride_pct_leg": (0.15, 0.32),
    # FK forelegs with no ground constraint stayed planted 6 frames of 33 (0.18) and
    # skated; IK feet hold ~0.67.
    "planted_fraction": (0.40, 0.92),
    # The generic cycle left torso, hips and chest at exactly zero: the marionette.
    "body_bob_pct_height": (0.003, 0.02),
    # Foreleg rake is MEASURED AND REPORTED BUT NOT GATED. The quantity is real -- an
    # ankle-led foreleg is the defect a reviewer spotted by eye on the moss fox -- but
    # this metric has not earned a threshold yet: read off the authoring bones it gave
    # 6.2 degrees and off the evaluated depsgraph 10.6 for the identical cycle, and its
    # sign disagrees with the renders about which way the limb leans. Gating on it would
    # fail correct cycles and pass bad ones. Judge rake by eye until the measurement is
    # validated against something visible; the bounds above are all reproducible.
    "front_rake_excursion_deg": None,
    # Frame 1 and the repeated last frame must match or the loop visibly pops.
    "loop_gap": 1e-5,
}


def limits_for(gait: str) -> dict:
    """Audit bounds scaled to the gait being checked.

    The bounds were measured on a walk. A scamper legitimately lifts and bobs far more,
    so checking it against walk numbers would reject a correct cycle -- the failure mode
    where a guard trains you to ignore it.
    """
    settings = GAITS.get(gait, GAITS["walk"])
    limits = dict(GAIT_LIMITS)
    low, high = GAIT_LIMITS["foot_lift_pct_height"]
    limits["foot_lift_pct_height"] = (low, high * settings["lift_scale"])
    low, high = GAIT_LIMITS["body_bob_pct_height"]
    limits["body_bob_pct_height"] = (low, high * settings["bob_scale"])
    floor = min(0.40, settings["duty"] - 0.1)
    limits["planted_fraction"] = (floor, GAIT_LIMITS["planted_fraction"][1])
    return limits


def check_gait(measured: dict, gait: str = "walk") -> list[str]:
    """Audit an authored cycle against the failure modes this pipeline has shipped.

    Returns a list of human-readable problems; empty means the cycle cleared every bound.
    Takes plain numbers so it is testable without Blender, and so the historical failures
    can be replayed against it directly.
    """
    problems: list[str] = []
    bounds = limits_for(gait)
    low, high = bounds["body_bob_pct_height"]
    bob = measured.get("body_bob_pct_height", 0.0)
    if bob < low:
        problems.append(
            f"body is welded in place (bob {bob:.4f} of body height, needs >{low}); "
            f"a cycle with dead hips and chest reads as a marionette, not an animal")
    elif bob > high:
        problems.append(f"body bobs {bob:.4f} of body height, above {high}")

    for limb, stats in sorted(measured.get("limbs", {}).items()):
        lift = stats["lift_pct_height"]
        low, high = bounds["foot_lift_pct_height"]
        if not low <= lift <= high:
            problems.append(
                f"{limb} lifts {lift:.4f} of body height, outside {low}-{high} "
                f"(the rejected cycle high-stepped at 0.0872)")
        stride = stats["stride_pct_leg"]
        low, high = bounds["stride_pct_leg"]
        if not low <= stride <= high:
            problems.append(
                f"{limb} strides {stride:.4f} of its leg length, outside {low}-{high}")
        planted = stats["planted_fraction"]
        low, high = bounds["planted_fraction"]
        if not low <= planted <= high:
            problems.append(
                f"{limb} is planted {planted:.2f} of the cycle, outside {low}-{high} "
                f"(an unplanted foot skates along the ground)")

    rake = measured.get("front_rake_excursion_deg")
    limit = GAIT_LIMITS["front_rake_excursion_deg"]
    if limit is not None and rake is not None and rake > limit:
        problems.append(
            f"foreleg rakes through {rake:.1f} degrees of lean, above {limit} -- the "
            f"ankle is leading the paw instead of the limb swinging from the shoulder")

    gap = measured.get("loop_gap", 0.0)
    if gap > GAIT_LIMITS["loop_gap"]:
        problems.append(f"cycle does not close: first and last frame differ by {gap:.6f}")
    return problems

# A two-bone IK chain that is asked to reach past its own length locks straight and dumps
# the leftover bend into the joint below it. On the moss fox the foreleg sits at 94.3% of
# full extension in its rest pose -- only 0.0158 of headroom each way -- so a stride of
# +/-0.077 hyperextended the elbow to a locked 180 degrees and kinked the wrist from 173.7
# degrees to 113. That reads as an ankle shoved out in front of the leg. The hind leg
# rests at 83.5% with 0.1202 of headroom and was never affected, which is why the defect
# looked like a front-leg-only mystery.
#
# Animals do not walk with a locked limb: the body rides lower than in a braced stand,
# which folds the leg and buys back the reach. So rather than shorten the stride, drop the
# body by exactly as much as the requested stride needs.
MAX_IK_EXTENSION = 0.95


def max_half_stride(chain_length: float, vertical_drop: float, horizontal_at_rest: float,
                    max_extension: float = MAX_IK_EXTENSION) -> float:
    """How far the foot target may travel fore or aft before the chain locks."""
    cap = max_extension * chain_length
    reach = cap * cap - vertical_drop * vertical_drop
    if reach <= 0:
        return 0.0
    return max(0.0, math.sqrt(reach) - horizontal_at_rest)


def body_drop_for_reach(chain_length: float, vertical_drop: float,
                        horizontal_at_rest: float, half_stride: float,
                        max_extension: float = MAX_IK_EXTENSION) -> float:
    """How far to lower the body so `half_stride` stays inside the chain's reach.

    Zero when the limb already has the headroom. Returns the drop needed, which the
    caller applies to the body as a constant offset with the bob riding on top.
    """
    cap = max_extension * chain_length
    needed = horizontal_at_rest + half_stride
    if needed >= cap:
        raise ValueError(
            f"stride of {half_stride:.4f} each way is unreachable: the chain caps at "
            f"{cap:.4f} and the foot already sits {horizontal_at_rest:.4f} out")
    return max(0.0, vertical_drop - math.sqrt(cap * cap - needed * needed))

# Secondary motion. A gait with a rigid head and tail reads as a puppet on rails even
# when the legs are perfect -- the same failure as a welded torso, one level up. These
# are small and phase-lagged on purpose: the tail follows the hips rather than leading
# them, and each segment lags a little more than the one before it so the chain whips
# instead of swinging as a board.
HEAD_CHAIN = ("neck", "head")
TAIL_CHAIN = ("spine.003", "spine.002", "spine.001", "spine")  # base to tip
TAIL_SWING_DEG = 3.0
# Vertical tail motion defaults OFF. Swinging side to side at stride frequency while
# lifting at twice that frequency traces a circle at the tail tip -- two perpendicular
# oscillations at different rates is the definition of one -- and it reads as the tail
# stirring rather than trailing.
TAIL_LIFT_DEG = 0.0
TAIL_SEGMENT_LAG = 0.09
# Rotations compound down a chain: four segments each turning by the full amount put four
# times that at the tip. Tapering keeps the base leading and the tip trailing, which is
# how a tail actually follows a body.
TAIL_TAPER = 0.6
HEAD_NOD_DEG = 2.5
HEAD_YAW_DEG = 2.0



def follow_through(u: float, cycles: float, amplitude: float, lag: float = 0.0) -> float:
    """A phase-lagged oscillation, for motion that trails the body rather than leads it."""
    return amplitude * math.sin(2.0 * math.pi * (cycles * u - lag))


def tail_angles(u: float, gait: str, *, swing_deg: float = TAIL_SWING_DEG,
                lift_deg: float = TAIL_LIFT_DEG,
                segment_lag: float = TAIL_SEGMENT_LAG,
                taper: float = TAIL_TAPER) -> list[dict[str, float]]:
    """Per-segment tail angles (radians), base first, each lagging and softer than the last.

    Swing and lift share one frequency on purpose. Driving them at different rates makes
    the tip describe a circle, which looks like stirring rather than following.
    """
    cycles = GAITS[gait]["sway_cycles"] or 1
    out = []
    for index in range(len(TAIL_CHAIN)):
        lag = segment_lag * index
        falloff = taper ** index
        out.append({
            "swing": math.radians(follow_through(u, cycles, swing_deg * falloff, lag)),
            "lift": math.radians(follow_through(u, cycles, lift_deg * falloff, lag)),
        })
    return out


def head_angles(u: float, gait: str, *, nod_deg: float = HEAD_NOD_DEG,
                yaw_deg: float = HEAD_YAW_DEG) -> dict[str, dict[str, float]]:
    """Neck and head angles (radians). The head counter-nods against the body's bob so
    it stays level, which is what animals do and what makes a walk look purposeful."""
    bob_cycles = GAITS[gait]["bob_cycles"]
    sway_cycles = GAITS[gait]["sway_cycles"] or 1
    return {
        "neck": {"nod": math.radians(follow_through(u, bob_cycles, -nod_deg, 0.05)),
                 "yaw": math.radians(follow_through(u, sway_cycles, yaw_deg * 0.5, 0.1))},
        "head": {"nod": math.radians(follow_through(u, bob_cycles, -nod_deg * 0.6, 0.12)),
                 "yaw": math.radians(follow_through(u, sway_cycles, yaw_deg, 0.18))},
    }

def rest_extent(coordinates) -> float:
    """Longest axis of an undeformed mesh, from raw vertex coordinates.

    Deliberately *not* `object.dimensions`: that reflects the evaluated, posed bounding
    box, so a rig caught mid-stride measures taller than the same rig at rest and the
    whole gait silently scales up with it. Measured live on a fox mid-cycle this read
    1.675 against a true 1.484 -- a 13% error in every stride, lift and bob.
    """
    coordinates = list(coordinates)
    if not coordinates:
        raise ValueError("cannot measure an empty mesh")
    spans = []
    for axis in range(3):
        values = [c[axis] for c in coordinates]
        spans.append(max(values) - min(values))
    return max(spans)


def gait_plan(frames: int, gait: str, *, front_leg_length: float, hind_leg_length: float,
              body_height: float, scale: float = 1.0,
              tail_swing_deg: float = TAIL_SWING_DEG,
              sway_per_bob: float = SWAY_PER_BOB,
              stride_override: dict[str, float] | None = None) -> dict:
    """Every value the driver needs to key, for frames 1..frames+1 inclusive.

    The extra final frame repeats frame 1 so the action loops without a pop.
    """
    if frames < 4:
        raise ValueError(f"a cycle needs at least 4 frames, got {frames}")
    sizes = stride_and_lift(front_leg_length, hind_leg_length)
    # Lowering the body folds the limbs and buys reach; if the stride does not grow to
    # use it, the feet rise further than they travel and the gait reads as marching on
    # the spot rather than covering ground. The reference run strides about 2.4x its foot
    # lift; lifting hard while striding short inverts that ratio.
    if stride_override:
        sizes = lift_from_stride(stride_override["front_stride"],
                                 stride_override["hind_stride"],
                                 GAITS[gait]["lift_scale"])
    else:
        sizes = lift_from_stride(sizes["front_stride"], sizes["hind_stride"],
                                 GAITS[gait]["lift_scale"])
    duty = GAITS[gait]["duty"]
    settings = GAITS[gait]
    bob_amp = body_height * BOB_PER_HEIGHT * scale * settings["bob_scale"]
    plan = {"gait": gait, "frames": frames, "duty": duty,
            "sizes": {k: v * scale for k, v in sizes.items()},
            "feet": {limb: [] for limb in LIMBS}, "body": []}
    for index in range(frames + 1):
        u = (index % frames) / frames
        for limb in LIMBS:
            front = limb.startswith("F")
            fore_aft, up = foot_offset(
                u, limb_phase(gait, limb), duty,
                sizes["front_stride" if front else "hind_stride"] * scale,
                sizes["front_lift" if front else "hind_lift"] * scale)
            plan["feet"][limb].append({"frame": index + 1, "fore_aft": fore_aft, "up": up})
        for limb in FRONT_LIMBS:
            plan.setdefault("front_fk", {limb: [] for limb in FRONT_LIMBS})
            plan["front_fk"][limb].append({
                "frame": index + 1,
                **front_fk_angles(u, limb_phase(gait, limb), duty,
                                  stride=sizes["front_stride"] * scale,
                                  lift=sizes["front_lift"] * scale,
                                  leg_length=front_leg_length)})
        plan.setdefault("tail", []).append(
            {"frame": index + 1,
             "segments": tail_angles(u, gait, swing_deg=tail_swing_deg)})
        plan.setdefault("head", []).append(
            {"frame": index + 1, **head_angles(u, gait)})
        plan["body"].append({"frame": index + 1,
                             "up": body_bob(u, gait, bob_amp),
                             "lateral": body_sway(u, gait, bob_amp, sway_per_bob)})
    return plan
