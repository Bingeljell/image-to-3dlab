"""Skin-weight clean-up: weights from bones far away in the tree are dropped, near ones kept."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clean_skin_weights as cw

# hips -> spine -> chest -> arm -> hand -> finger1 -> finger2 ; chest -> neck ;
# hips -> leg -> foot -> toe ; hand -> thumb
PARENTS = {
    "hips": None, "spine": "hips", "chest": "spine", "arm": "chest", "hand": "arm",
    "finger1": "hand", "finger2": "finger1", "thumb": "hand", "neck": "chest",
    "leg": "hips", "foot": "leg", "toe": "foot",
}


def test_tree_distance_counts_hops_both_ways():
    hops = cw.tree_hops(PARENTS)
    assert hops["finger2"]["finger2"] == 0
    assert hops["finger2"]["thumb"] == 3
    assert hops["hand"]["chest"] == 2
    assert hops["finger2"]["toe"] == 9


def test_a_fingertip_weighted_to_the_toe_loses_the_toe():
    """The kung-fu hero: right fingertips carried weight from the right toe, so raising
    the hand stretched them into spikes."""
    hops = cw.tree_hops(PARENTS)
    cleaned = cw.prune_weights({"finger2": 0.6, "finger1": 0.2, "toe": 0.2}, hops, "finger2")
    assert set(cleaned) == {"finger2", "finger1"}
    assert sum(cleaned.values()) == pytest.approx(1.0)
    assert cleaned["finger2"] == pytest.approx(0.75)


def test_neighbouring_blends_are_kept_untouched():
    hops = cw.tree_hops(PARENTS)
    for weights in ({"finger2": 0.5, "thumb": 0.5}, {"arm": 0.6, "chest": 0.4},
                    {"spine": 0.5, "leg": 0.5}):
        assert cw.prune_weights(weights, hops, max(weights, key=weights.get)) == weights


def test_empty_and_single_weights_pass_through():
    hops = cw.tree_hops(PARENTS)
    assert cw.prune_weights({}, hops, "toe") == {}
    assert cw.prune_weights({"toe": 1.0}, hops, "toe") == {"toe": 1.0}


def test_a_fingertip_owned_only_by_the_toe_goes_to_the_bone_it_sits_on():
    """Left after the first fix: points weighted wholly to the toe kept it as their main
    bone. Where a point sits decides; if nothing near is left, the nearest bone takes it."""
    hops = cw.tree_hops(PARENTS)
    assert cw.prune_weights({"toe": 1.0}, hops, "finger2") == {"finger2": 1.0}
    cleaned = cw.prune_weights({"toe": 0.7, "finger1": 0.3}, hops, "finger2")
    assert cleaned == pytest.approx({"finger1": 1.0})


def test_nearest_bone_measures_to_the_segment_not_the_head():
    import numpy as np
    segments = {"arm": (np.array([0.0, 0, 0]), np.array([1.0, 0, 0])),
                "toe": (np.array([0.9, -1.0, 0]), np.array([1.0, -1.0, 0]))}
    assert cw.nearest_bone(np.array([0.95, -0.3, 0]), segments) == "arm"
    assert cw.nearest_bone(np.array([0.95, -0.8, 0]), segments) == "toe"


def test_a_hand_hanging_beside_the_thigh_keeps_its_hand_weight():
    """Arms-down characters: a hand point sits nearer the thigh than the hand bone, so the
    thigh became its anchor and the hand weight (9 hops away) was dropped. The rigger had
    it right; the clean-up glued hands to thighs. A bone right beside the point stays."""
    hops = cw.tree_hops(PARENTS)
    weights = {"hand": 0.9, "leg": 0.1}
    assert cw.prune_weights(weights, hops, "leg", close={"hand", "leg"}) == weights
    assert cw.prune_weights({"hand": 1.0}, hops, "leg", close={"hand", "leg"}) == {"hand": 1.0}


def test_close_bones_are_within_slack_of_the_nearest_not_across_the_body():
    import numpy as np
    segments = {"hand": (np.array([0.30, 0.0, 0]), np.array([0.30, -0.10, 0])),
                "leg": (np.array([0.20, 0.0, 0]), np.array([0.20, -0.40, 0])),
                "toe": (np.array([0.20, -0.90, 0]), np.array([0.20, -0.95, 0]))}
    point = np.array([0.26, -0.05, 0])  # between hand and thigh, a touch nearer the thigh
    assert cw.close_bones(point, segments, slack=0.05) == {"hand", "leg"}
    assert "toe" not in cw.close_bones(np.array([0.30, -0.12, 0]), segments, slack=0.05)


def test_the_toe_is_still_dropped_from_a_fingertip_when_it_is_far_away():
    import numpy as np
    hops = cw.tree_hops(PARENTS)
    segments = {"finger2": (np.array([0.5, 0.0, 0]), np.array([0.55, 0.0, 0])),
                "finger1": (np.array([0.45, 0.0, 0]), np.array([0.5, 0.0, 0])),
                "toe": (np.array([0.1, -0.9, 0]), np.array([0.1, -0.95, 0]))}
    point = np.array([0.54, 0.0, 0])
    close = cw.close_bones(point, segments, slack=0.05)
    cleaned = cw.prune_weights({"finger2": 0.6, "finger1": 0.2, "toe": 0.2}, hops, "finger2", close=close)
    assert set(cleaned) == {"finger2", "finger1"}


# --- Ear weight tied to an arm: the old fixes e23cbde and b8abb4c, still missing on main. ---

def test_an_ear_above_the_head_tied_to_a_hand_goes_to_the_head():
    """A fingertip parked inside an ear sits above the head joint. A real hand sits
    below it. Without the retie, a walk drags the ear out with the arm."""
    import numpy as np
    up = np.array([0.0, 0.0, 1.0])
    head_at = np.array([0.0, 0.0, 0.4])
    hands = {"finger", "hand"}
    ear = cw.retie_head_weight(
        {"finger": 0.7, "hand": 0.3}, np.array([0.3, 0.0, 0.72]), "head", head_at, hands, up)
    assert ear == {"head": pytest.approx(1.0)}
    held = {"finger": 1.0}
    hand = cw.retie_head_weight(held, np.array([0.3, 0.0, 0.37]), "head", head_at, hands, up)
    assert hand is held
    neck = {"neck": 1.0}
    same = cw.retie_head_weight(neck, np.array([0.3, 0.0, 0.72]), "head", head_at, hands, up)
    assert same is neck


def test_an_ear_tied_to_the_forearm_goes_to_the_head_but_the_shoulder_stays():
    assert cw.arm_below_shoulder("RightForeArm")
    assert cw.arm_below_shoulder("LeftArm")
    assert cw.arm_below_shoulder("LeftHandMiddle4")
    assert not cw.arm_below_shoulder("LeftShoulder")
    assert not cw.arm_below_shoulder("Head")


def test_a_body_that_is_not_two_legged_is_not_retied():
    import numpy as np
    parents = {"root": None, "a": "root", "b": "root", "c": "root"}
    heads = {name: np.zeros(3) for name in parents}
    assert cw.head_and_hands(parents, heads, np.array([0.0, 0.0, 1.0])) == (None, set())
