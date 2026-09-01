"""A ram charges with its horns; the gorilla curve it was forked from throws its arms.

These test the properties that separate the two, plus the per-bone hold that stops the
planted forelimb drifting after it has landed. Each has a plausible-looking wrong
version that a render does not reliably expose: a head that keeps winding up after the
eye has read "ready", a paw that sinks through the floor because the spine kept pitching
underneath it, or a jaw that opens on a headbutt.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ram_headbutt_pose import (  # noqa: E402
    ANTICIPATE_END,
    CURVE,
    PHASE_OVERRIDE,
    REAR_END,
    STRIKE_END,
    attack_pose,
    ease,
    ease_out,
    phase,
    sample,
)

HELD = [bone for bone, (_, _, hold) in PHASE_OVERRIDE.items() if hold is not None]
FREE_AT_STRIKE = ("neck", "head")


def test_cycle_starts_and_ends_at_rest():
    """Anything else pops when the clip is retriggered."""
    for bone, value in attack_pose(0.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not start at rest"
    for bone, value in attack_pose(1.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not end at rest"


def test_the_horns_are_the_weapon():
    """Head and neck must drive furthest forward of anything at strike.

    If a leg out-travels the head in the forward direction, the clip reads as a stomp or
    a swipe -- which is exactly what attack_pose.py is for, and why this file exists.
    """
    forward = {bone: values[2] for bone, values in CURVE.items() if values[2] > 0}
    assert max(forward, key=forward.get) == "head"
    assert forward["head"] > forward["neck"] > forward["spine_01"] > forward["spine_02"]


def test_head_and_neck_drive_forward_while_the_spine_arched_back():
    """The wind-up arches back, the blow drives forward. Sharing a sign is a stumble."""
    for bone in ("spine_01", "spine_02", "neck", "head"):
        _, rear_value, strike_value, _ = CURVE[bone]
        assert rear_value < 0, f"{bone}: rear-up {rear_value} hunches instead of arching"
        assert strike_value > 0, f"{bone}: strike {strike_value} does not drive forward"


def test_the_mouth_stays_shut():
    """A ram headbutts; it does not bite. The gorilla curve opens its jaw at strike."""
    assert CURVE["jaw"][2] == pytest.approx(0.0)
    for t in (i / 200 for i in range(201)):
        assert abs(attack_pose(t)["jaw"]) <= 5.0


def test_the_forelegs_reach_out_rather_than_tucking_under():
    """The distinguishing sign flip from attack_pose.py: here the legs are landing gear.

    Positive lifts and tucks the paw under the body (the gorilla's wind-up). A ram
    extends them forward to catch the charge, so the rear-up value must be negative.
    """
    for bone in ("frontL_upperarm", "frontR_upperarm", "frontL_forearm", "frontR_forearm"):
        assert CURVE[bone][1] < 0, f"{bone} tucks under instead of reaching out"


def test_left_and_right_limbs_stay_in_step():
    """A charge lands on both forelegs at once; a mismatch reads as a limp."""
    pairs = [(b, b.replace("frontL", "frontR").replace("backL", "backR"))
             for b in CURVE if b.startswith(("frontL", "backL"))]
    assert pairs, "no left/right pairs found -- the bone naming changed"
    for t in (0.0, 0.2, 0.35, 0.5, 0.55, 0.7, 1.0):
        pose = attack_pose(t)
        for left, right in pairs:
            assert pose[left] == pytest.approx(pose[right]), f"{left}/{right} at t={t}"


def test_the_plant_holds_instead_of_drifting():
    """Legs and spine freeze between their early strike and the shared STRIKE_END.

    A leg that keeps easing after it is already planted reads as floaty rather than as a
    stomp. The spine is in this group because the forelimbs hang off spine_02 -- a flat
    local hold on the leg alone still lets the paw's world position sink.
    """
    assert set(HELD) >= {"spine_01", "spine_02", "frontL_upperarm", "frontL_paw"}
    planted = attack_pose(0.50)
    for t in (0.52, 0.545, STRIKE_END - 1e-6):
        later = attack_pose(t)
        for bone in HELD:
            assert later[bone] == pytest.approx(planted[bone]), f"{bone} drifts at t={t}"


def test_the_head_keeps_driving_while_the_shoulder_holds():
    """Neck and head are not ancestors of the legs, so they run on their own timing.

    Freezing them along with the shoulder would stop the headbutt mid-swing.
    """
    planted = attack_pose(0.50)
    late = attack_pose(STRIKE_END - 1e-6)
    for bone in FREE_AT_STRIKE:
        assert abs(late[bone] - planted[bone]) > 0.5, f"{bone} froze with the shoulder"


def test_the_head_stops_winding_up_early():
    """Its rear extreme lands on the override at 0.375, not the shared boundary at 0.46."""
    for bone in FREE_AT_STRIKE:
        override_rear = PHASE_OVERRIDE[bone][0]
        assert override_rear is not None and override_rear < REAR_END
        rear_value = CURVE[bone][1]
        values = [attack_pose(i / 400)[bone] for i in range(401)]
        peak_at = min(range(len(values)), key=lambda i: abs(values[i] - rear_value)) / 400
        assert peak_at == pytest.approx(override_rear, abs=0.01)


def test_the_blow_is_the_fastest_part_of_the_cycle():
    poses = sample(200)
    bone = "head"

    def peak(lo: float, hi: float) -> float:
        # Attribute each step to the phase of its MIDPOINT; keying on the start hands the
        # step spanning a boundary to the wrong phase.
        idx = [i for i in range(len(poses) - 1) if lo <= (i + 0.5) / 200 < hi]
        return max(abs(poses[i + 1][bone] - poses[i][bone]) for i in idx)

    strike = peak(PHASE_OVERRIDE[bone][0], STRIKE_END)
    windup = peak(ANTICIPATE_END, PHASE_OVERRIDE[bone][0])
    settle = peak(STRIKE_END, 1.0)
    assert strike > windup, f"strike {strike:.3f} is slower than the wind-up {windup:.3f}"
    assert strike > settle, f"strike {strike:.3f} is slower than the recovery {settle:.3f}"


def test_pose_is_continuous_across_every_phase_boundary():
    """Including the per-bone overrides -- a jump at a boundary is a visible snap."""
    boundaries = {ANTICIPATE_END, REAR_END, STRIKE_END, 0.375, 0.5}
    for boundary in sorted(boundaries):
        before = attack_pose(boundary - 1e-7)
        after = attack_pose(boundary + 1e-7)
        for bone in CURVE:
            assert before[bone] == pytest.approx(after[bone], abs=0.05), (
                f"{bone} snaps at t={boundary}"
            )


def test_no_bone_exceeds_a_sane_limit():
    """Guards a typo turning 42 degrees into 420."""
    for t in (i / 200 for i in range(201)):
        for bone, value in attack_pose(t).items():
            assert abs(value) <= 90.0, f"{bone} reaches {value:.1f} at t={t:.2f}"


def test_phase_boundaries_are_ordered():
    assert 0.0 < ANTICIPATE_END < REAR_END < STRIKE_END < 1.0
    assert phase(0.0) == "anticipate"
    assert phase(ANTICIPATE_END) == "rear"
    assert phase(REAR_END) == "strike"
    assert phase(STRIKE_END) == "recover"


def test_easing_helpers():
    assert ease(0.0) == 0.0 and ease(1.0) == 1.0
    assert ease(0.5) == pytest.approx(0.5)
    assert ease_out(0.0) == 0.0 and ease_out(1.0) == 1.0
    # ease_out must be ahead of linear early on -- that is what makes the blow snap
    assert ease_out(0.25) > 0.25


def test_sample_returns_one_pose_per_frame_plus_the_loop_point():
    assert len(sample(40)) == 41
    with pytest.raises(ValueError):
        sample(1)
