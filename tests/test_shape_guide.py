"""Tests for `scripts/shape_guide.py`: the layout maths of the shape-guide kit, without Blender.

The kit builds a detailed vehicle from a few numbers (#259). Where a hand-built model placed every window by hand, the
kit spreads them by a rule; the bus it started from is the yardstick.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import shape_guide as sg


def test_a_side_without_doors_comes_out_like_the_hand_built_bus():
    # the bus's left side, placed by hand: seven panes between 0.10 m pillars from 5.12 back to -4.48 (the first
    # 1.20 m, the others 1.30 m); the rule spreads them evenly and lands within 10 cm of every hand-placed edge
    panes = sg.spread_windows(-4.48, 5.12, pitch=1.40, pillar=0.10)
    by_hand = [(3.92, 5.12), (2.52, 3.82), (1.12, 2.42), (-0.28, 1.02), (-1.68, -0.38), (-3.08, -1.78), (-4.48, -3.18)]
    assert len(panes) == len(by_hand)
    for (a, b), (c, d) in zip(panes, by_hand):
        assert abs(a - c) < 0.10 and abs(b - d) < 0.10


def test_windows_keep_a_pillar_clear_of_every_door():
    doors = [(3.92, 5.22), (-0.10, 1.25), (-4.62, -3.32)]
    panes = sg.spread_windows(-5.40, 5.12, pitch=1.40, pillar=0.20, keep_out=doors)
    for y0, y1 in panes:
        for d0, d1 in doors:
            assert y1 <= d0 - 0.2 + 1e-9 or y0 >= d1 + 0.2 - 1e-9


def test_the_panes_of_one_stretch_are_equal_and_run_front_to_back():
    panes = sg.spread_windows(-3.0, 3.0, pitch=1.40, pillar=0.20)
    lengths = {round(y1 - y0, 6) for y0, y1 in panes}
    assert len(lengths) == 1
    assert [y0 for y0, _ in panes] == sorted((y0 for y0, _ in panes), reverse=True)
    assert panes[0][1] == pytest.approx(3.0) and panes[-1][0] == pytest.approx(-3.0)


def test_a_stretch_too_short_for_a_window_stays_closed():
    # between two doors 0.6 m apart there is no room for a pane and two pillars
    panes = sg.spread_windows(-5.0, 5.0, pitch=1.40, pillar=0.20, keep_out=[(-2.0, -0.3), (0.3, 2.0)])
    assert all(not (-0.3 < (y0 + y1) / 2 < 0.3) for y0, y1 in panes)


def test_the_pitch_must_leave_room_for_glass():
    with pytest.raises(ValueError, match="pitch"):
        sg.spread_windows(-3.0, 3.0, pitch=0.20, pillar=0.20)
