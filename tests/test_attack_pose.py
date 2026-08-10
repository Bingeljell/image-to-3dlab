"""The attack must read as an attack, not merely animate.

These test properties a reader cannot check by eye in a table of numbers: that the
anticipation leans the opposite way to the blow, that the blow is the fastest thing in
the cycle, that the cycle returns to rest so it does not pop on loop, and that the two
forelimbs stay in step. Each of these has a plausible-looking wrong version.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from attack_pose import CURVE, REAR_END, attack_pose, ease, ease_out, phase, sample


def test_cycle_starts_and_ends_at_rest():
    """Anything else pops when the clip loops."""
    for bone, value in attack_pose(0.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not start at rest"
    for bone, value in attack_pose(1.0).items():
        assert value == pytest.approx(0.0), f"{bone} does not end at rest"


def test_anticipation_opposes_the_strike():
    """A wind-up that leans the same way as the blow reads as a stumble, not a threat."""
    for bone in ("frontL_upperarm", "spine_01", "head"):
        anticipate, _, strike, _ = CURVE[bone]
        assert anticipate * strike < 0 or abs(anticipate) < 1e-9, (
            f"{bone}: anticipation {anticipate} does not oppose strike {strike}"
        )


def test_the_blow_is_the_fastest_part_of_the_cycle():
    poses = sample(48)
    bone = "frontL_upperarm"

    def peak(lo: float, hi: float) -> float:
        # Attribute each step to the phase of its MIDPOINT. Keying on the start instead
        # hands the step that spans a boundary to the wrong phase -- and the strike's
        # opening step is the largest in the cycle, so the wind-up window would steal it
        # and the test would fail on a correct animation.
        idx = [i for i in range(len(poses) - 1) if lo <= (i + 0.5) / 48 < hi]
        return max(abs(poses[i + 1][bone] - poses[i][bone]) for i in idx)

    strike = peak(0.46, 0.56)
    windup = peak(0.30, 0.46)
    settle = peak(0.56, 1.0)
    assert strike > windup, f"strike {strike:.2f} is slower than the wind-up {windup:.2f}"
    assert strike > settle, f"strike {strike:.2f} is slower than the recovery {settle:.2f}"


def test_forelimbs_stay_in_step():
    """A two-footed slam: left and right must match exactly, or it reads as a limp."""
    for t in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
        pose = attack_pose(t)
        for left, right in (("frontL_upperarm", "frontR_upperarm"),
                            ("frontL_forearm", "frontR_forearm"),
                            ("backL_thigh", "backR_thigh")):
            assert pose[left] == pytest.approx(pose[right])


def test_forelimbs_travel_further_than_hind_limbs():
    """It is a front-limb slam; if the back legs move most, it is a kick."""
    poses = sample(60)
    front = max(abs(p["frontL_upperarm"]) for p in poses)
    back = max(abs(p["backL_thigh"]) for p in poses)
    assert front > back * 1.5, f"front {front:.1f} vs back {back:.1f}"


def test_the_forelimb_lifts_before_it_strikes():
    """Peak lift must land in the rear-up phase, the low point in the strike phase."""
    poses = sample(100)
    values = [p["frontL_upperarm"] for p in poses]
    lift_at = min(range(len(values)), key=lambda i: values[i]) / 100
    blow_at = max(range(len(values)), key=lambda i: values[i]) / 100
    # The peak lift sits exactly on the rear/strike boundary, which phase() attributes
    # to "strike"; what matters is that the lift completes before the blow travels.
    assert lift_at <= REAR_END + 0.01, f"peak lift at t={lift_at:.2f} is too late"
    assert phase(blow_at) in {"strike", "recover"}, f"blow at t={blow_at:.2f}"
    assert lift_at < blow_at


def test_no_bone_exceeds_a_sane_limit():
    """Guards a typo turning 46 degrees into 460."""
    for t in (i / 200 for i in range(201)):
        for bone, value in attack_pose(t).items():
            assert abs(value) <= 90.0, f"{bone} reaches {value:.1f} at t={t:.2f}"


def test_pose_is_continuous_across_phase_boundaries():
    """A jump at a boundary is a visible snap."""
    for boundary in (0.30, 0.46, 0.56):
        before = attack_pose(boundary - 1e-7)
        after = attack_pose(boundary + 1e-7)
        for bone in CURVE:
            assert before[bone] == pytest.approx(after[bone], abs=0.05), (
                f"{bone} snaps at t={boundary}"
            )


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
