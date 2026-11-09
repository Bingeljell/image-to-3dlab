"""Mirroring a Kimodo clip: strafe left becomes strafe right, and back again unchanged."""
from __future__ import annotations

import numpy as np

from scripts import mirror_kimodo_clip as mirror
from scripts.kimodo_retarget import SOMA77, SOMA_INDEX


def _clip():
    """A small clip with a different rotation on each side, in the packed layout."""
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, len(SOMA77), 1))
    quats[:, SOMA_INDEX["LeftArm"]] = [0.9, 0.1, 0.2, 0.3]
    quats[:, SOMA_INDEX["RightArm"]] = [0.7, -0.2, 0.4, 0.1]
    quats[:, SOMA_INDEX["LeftLeg"]] = [0.8, 0.3, -0.1, 0.2]
    rest = np.zeros((len(SOMA77), 3), np.float32)
    rest[SOMA_INDEX["LeftArm"]] = [0.3, 0.1, 0.2]
    rest[SOMA_INDEX["RightArm"]] = [-0.3, 0.1, 0.2]
    hips = np.array([[0.4, 0.5, 0.9], [-0.4, 0.6, 0.9]], np.float32)
    return quats, rest, hips


def test_a_left_joint_lands_on_its_right_counterpart_mirrored():
    quats, rest, hips = _clip()
    turned, turned_rest, _ = mirror.mirror(quats, rest, hips)
    left = SOMA_INDEX["LeftArm"]
    right = SOMA_INDEX["RightArm"]
    # LeftArm's turn, mirrored, is now the right arm's; the left arm gets the mirrored right one.
    assert np.allclose(turned[0, right], [0.9, 0.1, -0.2, -0.3])
    assert np.allclose(turned[0, left], [0.7, -0.2, -0.4, -0.1])
    assert np.allclose(turned_rest[right], [-0.3, 0.1, 0.2])
    assert np.allclose(turned_rest[left], [0.3, 0.1, 0.2])


def test_the_hips_travel_the_other_way_and_keep_their_height():
    *_, turned_hips = mirror.mirror(*_clip())
    assert np.allclose(turned_hips, [[-0.4, 0.5, 0.9], [0.4, 0.6, 0.9]])


def test_mirroring_twice_gives_the_clip_back():
    quats, rest, hips = _clip()
    back = mirror.mirror(*mirror.mirror(quats, rest, hips))
    assert np.allclose(back[0], quats, atol=1e-6)
    assert np.allclose(back[1], rest, atol=1e-6)
    assert np.allclose(back[2], hips, atol=1e-6)


def test_the_shipped_strafe_right_is_the_mirror_of_the_left():
    """The clip in the repo, not a fixture: a strafe right that is not the left one mirrored
    would walk into the character's own body."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "image_to_3dlab" / "motions"
    left = np.load(root / "strafe_left_b.npz")
    right = np.load(root / "strafe_right.npz")
    turned = mirror.mirror(left["quats"], left["joints0"], left["hips"])
    assert np.allclose(right["quats"].astype(np.float32), turned[0], atol=2e-3)
    assert np.allclose(right["hips"], turned[2], atol=1e-5)
    assert float(right["fps"]) == float(left["fps"])
    # And it is not simply a copy: each side's turn is the other side's, mirrored.
    left_arm, right_arm = SOMA_INDEX["LeftArm"], SOMA_INDEX["RightArm"]
    assert np.allclose(right["quats"][:, left_arm].astype(np.float32),
                       left["quats"][:, right_arm].astype(np.float32) * [1, 1, -1, -1], atol=2e-3)
