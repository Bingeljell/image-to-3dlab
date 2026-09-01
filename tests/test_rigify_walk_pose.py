"""A trot on the Rigify FK rig: diagonal pairs move together, and the cycle loops.

The expensive failure this guards is not a wrong angle -- it is a wrong bone NAME.
Rigify silently ignores a pose written to a bone that does not exist, so a typo here
costs a full Blender session before anyone notices the leg never moved. The gait
properties are the other half: a trot whose diagonals drift apart reads as a limp, and
a cycle that does not close pops on every loop.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rigify_walk_pose import CHAIN, PHASES, SPINE_FK, sample  # noqa: E402

DIAGONALS = (("frontL", "backR"), ("frontR", "backL"))


def test_every_driven_bone_is_a_named_rigify_control():
    """A pose written to a bone Rigify did not generate is silently dropped."""
    expected = {b for chain in CHAIN.values() for b in chain} | set(SPINE_FK)
    assert set(sample(8)[0]) == expected
    for chain in CHAIN.values():
        for bone in chain:
            assert bone.endswith((".L", ".R")) and "_fk" in bone, bone
    for bone in SPINE_FK:
        assert bone.startswith("spine_fk."), bone


def test_the_chain_stops_at_the_foot():
    """Rigify's quadruped FK chain has no toe control, unlike our custom rig's
    heel+toe split -- driving a 'toe' bone here would be a no-op."""
    for chain in CHAIN.values():
        assert len(chain) == 3
    assert not any("toe" in b for chain in CHAIN.values() for b in chain)


@pytest.mark.parametrize("gait", sorted(PHASES))
def test_the_cycle_closes(gait):
    """First and last frame must match exactly or the gait pops every stride."""
    poses = sample(24, gait=gait)
    for bone, value in poses[0].items():
        assert value == pytest.approx(poses[-1][bone]), f"{bone} does not close"


def test_a_trot_moves_its_diagonals_together():
    """The definition of a trot: front-left with back-right, front-right with back-left."""
    poses = sample(24, gait="trot", swing_front=30.0, swing_back=30.0,
                   bend_front=35.0, bend_back=35.0)
    for pose in poses:
        for front, back in DIAGONALS:
            assert pose[CHAIN[front][0]] == pytest.approx(pose[CHAIN[back][0]]), (
                f"{front} and {back} are out of step"
            )


def test_a_trot_moves_its_two_diagonals_in_opposition():
    """Half a stride apart. In phase, the animal hops rather than trots."""
    frames = 24
    poses = sample(frames, gait="trot")
    for offset in (0, 3, 7, 11):
        now = poses[offset][CHAIN["frontL"][0]]
        half_later = poses[offset + frames // 2][CHAIN["frontR"][0]]
        assert now == pytest.approx(half_later), f"diagonals not opposed at frame {offset}"


def test_a_walk_puts_all_four_legs_on_different_beats():
    """Four evenly spaced phases -- the property that makes it a walk, not a trot."""
    assert sorted(PHASES["walk"].values()) == [0.0, 0.25, 0.5, 0.75]
    assert sorted(PHASES["trot"].values()) == [0.0, 0.0, 0.5, 0.5]
    # Compared over the whole stride, not one frame: at t=0 the two quarter-phase legs
    # happen to share a swing value (they are passing through zero in opposite
    # directions), so a single-frame check would wrongly report only three beats.
    poses = sample(24, gait="walk")
    curves = {leg: tuple(round(p[CHAIN[leg][0]], 6) for p in poses) for leg in PHASES["walk"]}
    assert len(set(curves.values())) == 4, "two legs share a beat -- that is not a walk"


def test_the_swing_reaches_its_amplitude_and_no_further():
    poses = sample(120, swing_front=30.0, swing_back=18.0)
    for leg, chain in CHAIN.items():
        want = 30.0 if leg.startswith("front") else 18.0
        reached = max(abs(p[chain[0]]) for p in poses)
        assert reached == pytest.approx(want, abs=0.05), f"{leg} swings {reached:.2f}"


def test_the_lower_joints_never_fold_backwards():
    """The fold is a lift, added to a base bend; a negative value hyperextends the knee."""
    base_bend = 8.0
    poses = sample(120, base_bend=base_bend)
    for chain in CHAIN.values():
        assert min(p[chain[1]] for p in poses) >= base_bend - 1e-9
        assert min(p[chain[2]] for p in poses) >= base_bend * 0.5 - 1e-9


def test_the_spine_sway_stays_a_light_undulation():
    """It is body follow-through, not a second gait -- it must not swamp the legs."""
    sway = 4.0
    poses = sample(120, spine_sway=sway)
    reached = max(abs(p[bone]) for p in poses for bone in SPINE_FK)
    assert reached <= sway * 0.3 + 1e-9
    assert reached > 0.0


def test_the_spine_undulates_twice_per_stride():
    poses = sample(240, spine_sway=4.0)
    values = [p[SPINE_FK[0]] for p in poses]
    crossings = sum(
        1 for i in range(len(values) - 1)
        if values[i] <= 0.0 < values[i + 1] or values[i] >= 0.0 > values[i + 1]
    )
    assert crossings == 4, f"{crossings} zero crossings -- not two cycles per stride"


def test_zero_swing_leaves_the_legs_still():
    poses = sample(24, swing_front=0.0, swing_back=0.0)
    for chain in CHAIN.values():
        assert all(p[chain[0]] == pytest.approx(0.0) for p in poses)


def test_sample_returns_one_pose_per_frame_plus_the_loop_point():
    assert len(sample(40)) == 41
    with pytest.raises(ValueError):
        sample(1)


def test_an_unknown_gait_is_rejected_rather_than_silently_walking():
    with pytest.raises(KeyError):
        sample(24, gait="gallop")


def test_the_swing_is_a_clean_cosine():
    """Pins the phase convention: frontL's swing peaks at t=0 for the trot."""
    frames = 96
    poses = sample(frames, gait="trot", swing_front=30.0)
    for frame in (0, 12, 24, 48):
        t = frame / frames
        want = 30.0 * math.cos(2.0 * math.pi * (0.0 - t))
        assert poses[frame][CHAIN["frontL"][0]] == pytest.approx(want)
