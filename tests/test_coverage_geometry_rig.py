"""Exercise geometry helpers with small local fixtures."""

import numpy as np
import pytest

from scripts import clean_skin_weights, kimodo_retarget


def test_cleaning_uses_bone_distance_and_preserves_weight_total():
    parents = {"upper": None, "lower": "upper", "end": "lower"}
    segments = {
        "upper": (np.array([0, 0, 0]), np.array([0, 1, 0])),
        "lower": (np.array([0, 1, 0]), np.array([0, 2, 0])),
        "end": (np.array([0, 2, 0]), np.array([0, 3, 0])),
    }
    clean = clean_skin_weights.build_vertex_cleaner(parents, segments, max_hops=0)
    assert clean(np.array([0, .5, 0]), {"end": .4, "upper": .2}) == {
        "upper": pytest.approx(.6)
    }
    close = {"upper": .1, "lower": .9}
    assert clean(np.array([0, 1, 0]), close) is close


def test_sole_weights_use_four_strongest_foot_influences():
    pairs = [
        ("toe1", .3), ("toe2", .4), ("toe3", .8), ("toe4", .5), ("toe5", .7),
    ]
    assert kimodo_retarget.sole_weights(pairs) == pytest.approx(
        {"toe3": .8 / 2.4, "toe5": .7 / 2.4, "toe4": .5 / 2.4, "toe2": .4 / 2.4}
    )
    assert kimodo_retarget.sole_weights([("toe1", .19)]) == {}
    assert kimodo_retarget.sole_weights([]) == {}
    assert kimodo_retarget.sole_weights([("toe1", .20)]) == {"toe1": 1.0}


def test_retarget_parser_accepts_paths_and_animation_options(tmp_path):
    rig, motion, output = (tmp_path / name for name in ("rig.glb", "walk.npz", "walk.glb"))
    args = kimodo_retarget.build_parser().parse_args([
        "--rig", str(rig), "--motion", str(motion), "--out", str(output),
        "--loop", "--keep-whole", "--speed", "1.5", "--in-place",
    ])
    assert (args.rig, args.motion, args.out) == (rig, motion, output)
    assert args.loop and args.keep_whole and args.in_place
    assert args.speed == 1.5
    assert not args.no_align
    assert not args.no_ground
