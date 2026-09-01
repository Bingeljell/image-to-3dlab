"""The same headbutt as the custom rig, but this metarig can only bend its spine.

Basic Quadruped generates no head or neck DEFORM bones, so the charge has to read as a
body arc. That limitation is pinned here deliberately: the `head` and `neck` pose bones
exist with real widgets and are tempting to drive, but doing so moves nothing and would
waste a Blender session before anyone noticed. The rest is the shape of the attack --
arch back, drive forward, front legs fold rather than punch through the floor.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rigify_headbutt_pose import (  # noqa: E402
    ANTICIPATE_END,
    BACK_LEGS,
    BACK_LOW,
    BACK_MID,
    BACK_UPPER,
    FRONT_LEGS,
    FRONT_LOW,
    FRONT_MID,
    FRONT_UPPER,
    REAR_END,
    SPINE_CURVE,
    SPINE_FK,
    STRIKE_END,
    ease,
    ease_out,
    pose_at,
    sample,
)


def test_cycle_starts_and_ends_at_rest():
    for bone, value in pose_at(0.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not start at rest"
    for bone, value in pose_at(1.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not end at rest"


def test_no_head_or_neck_bone_is_driven():
    """Confirmed on the rig: posing head/neck changes zero DEF-spine rotations.

    They deform nothing on this metarig, so driving them is dead motion. If someone
    'fixes' the flat-headed look by adding them back, this catches it.
    """
    driven = set(pose_at(0.5))
    assert not any("head" in b or "neck" in b for b in driven), sorted(driven)


def test_every_driven_bone_is_a_named_rigify_control():
    """Rigify silently ignores a pose written to a bone it did not generate."""
    expected = set(SPINE_FK) | {b for chain in FRONT_LEGS.values() for b in chain} \
        | {b for chain in BACK_LEGS.values() for b in chain}
    assert set(pose_at(0.5)) == expected
    assert set(SPINE_CURVE) == set(SPINE_FK)


def test_the_arc_builds_toward_the_front():
    """A spine that bends evenly reads as a folding ruler, not a charging animal.

    Segments nearer the neck must travel further than those at the hips, in every phase.
    """
    for slot, name in enumerate(("anticipate", "rear", "strike", "settle")):
        magnitudes = [abs(SPINE_CURVE[bone][slot]) for bone in SPINE_FK]
        assert magnitudes == sorted(magnitudes), f"{name} does not build forward"
        assert magnitudes[-1] > magnitudes[0], f"{name} is flat across the spine"


def test_the_spine_arches_back_then_drives_forward():
    """Positive pitches a segment down and forward; negative arches it up and back.
    A rear-up and a strike sharing a sign means the body never winds up."""
    for bone in SPINE_FK:
        _, rear_value, strike_value, _ = SPINE_CURVE[bone]
        assert rear_value < 0, f"{bone}: rear {rear_value} hunches instead of arching"
        assert strike_value > 0, f"{bone}: strike {strike_value} does not drive forward"


def test_the_front_legs_reach_out_then_fold_back():
    """They are landing gear. Reaching forward through the wind-up and STAYING there
    drives the paw through the floor while the spine is still pitching down."""
    for curve, name in ((FRONT_UPPER, "upper"), (FRONT_MID, "mid"), (FRONT_LOW, "low")):
        _, rear_value, strike_value, _ = curve
        assert rear_value > 0, f"front {name}: rear {rear_value} does not reach forward"
        assert strike_value < 0, f"front {name}: strike {strike_value} does not fold back"


def test_the_back_legs_gather_then_push_off():
    """The charge's momentum comes from the hind legs extending, not the body toppling."""
    anticipate, rear, strike, _ = BACK_UPPER
    assert anticipate < 0 and rear < 0, "the hindquarters do not crouch to gather"
    assert strike > 0, "the hindquarters never drive the charge forward"
    assert BACK_MID[0] > 0 and BACK_MID[2] < 0, "the hock does not extend at strike"
    assert BACK_LOW[2] < 0


def test_left_and_right_legs_stay_in_step():
    for t in (0.0, 0.2, 0.35, 0.46, 0.5, 0.7, 1.0):
        pose = pose_at(t)
        for legs in (FRONT_LEGS, BACK_LEGS):
            (_, left), (_, right) = sorted(legs.items())
            for lb, rb in zip(left, right):
                assert pose[lb] == pytest.approx(pose[rb]), f"{lb}/{rb} at t={t}"


def test_the_front_and_back_legs_oppose_each_other_at_strike():
    """Front folds back while back extends forward -- that opposition IS the charge."""
    pose = pose_at(STRIKE_END - 1e-6)
    assert pose["front_thigh_fk.L"] < 0 < pose["thigh_fk.L"]


def test_the_strike_is_the_fastest_part_of_the_cycle():
    poses = sample(200)
    bone = SPINE_FK[-1]

    def peak(lo: float, hi: float) -> float:
        # Each step is attributed to the phase of its midpoint; keying on the start hands
        # the step spanning a boundary to the wrong phase.
        idx = [i for i in range(len(poses) - 1) if lo <= (i + 0.5) / 200 < hi]
        return max(abs(poses[i + 1][bone] - poses[i][bone]) for i in idx)

    strike = peak(REAR_END, STRIKE_END)
    assert strike > peak(ANTICIPATE_END, REAR_END), "the wind-up outruns the blow"
    assert strike > peak(STRIKE_END, 1.0), "the recovery outruns the blow"


def test_pose_is_continuous_across_phase_boundaries():
    for boundary in (ANTICIPATE_END, REAR_END, STRIKE_END):
        before = pose_at(boundary - 1e-7)
        after = pose_at(boundary + 1e-7)
        for bone in before:
            assert before[bone] == pytest.approx(after[bone], abs=0.05), (
                f"{bone} snaps at t={boundary}"
            )


def test_no_bone_exceeds_a_sane_limit():
    for t in (i / 200 for i in range(201)):
        for bone, value in pose_at(t).items():
            assert abs(value) <= 90.0, f"{bone} reaches {value:.1f} at t={t:.2f}"


def test_phase_boundaries_are_ordered():
    assert 0.0 < ANTICIPATE_END < REAR_END < STRIKE_END < 1.0


def test_easing_helpers():
    assert ease(0.0) == 0.0 and ease(1.0) == 1.0
    assert ease(0.5) == pytest.approx(0.5)
    assert ease_out(0.0) == 0.0 and ease_out(1.0) == 1.0
    assert ease_out(0.25) > 0.25


def test_sample_returns_one_pose_per_frame_plus_the_loop_point():
    assert len(sample(40)) == 41
    with pytest.raises(ValueError):
        sample(1)
