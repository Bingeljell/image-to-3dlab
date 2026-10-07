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
