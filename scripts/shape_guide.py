#!/usr/bin/env python3
"""The layout maths of the shape-guide kit: a detailed vehicle from a few numbers (#259).

`blender_shape_guide.py` builds the vehicle in Blender; everything here is plain maths, tested
without Blender (`tests/test_shape_guide.py`). The kit started from a hand-built 12 m bus whose
every window was placed by hand; here the same layout comes from rules, so a new vehicle needs
only its numbers.
"""

from __future__ import annotations

# A pane shorter than this is not worth a window: the stretch stays closed panelling.
MIN_PANE = 0.40


def spread_windows(start: float, end: float, pitch: float, pillar: float,
                   keep_out: list[tuple[float, float]] | tuple = ()) -> list[tuple[float, float]]:
    """Side windows along a vehicle's length, as (rear, front) pairs from the front backwards.

    The band runs from `start` (rear) to `end` (front). Each stretch of it that no door
    (`keep_out`, (rear, front) pairs) covers, less a `pillar` on each side of a door, is split
    into equal panes about `pitch` long including one pillar, with a `pillar` between panes.
    A stretch too short for a pane of `MIN_PANE` stays closed.
    """
    if pitch <= pillar:
        raise ValueError(f"the pitch ({pitch} m) must be longer than the pillar ({pillar} m)")
    stretches = [(start, end)]
    for d0, d1 in sorted(keep_out):
        cut = []
        for s0, s1 in stretches:
            if d1 + pillar <= s0 or d0 - pillar >= s1:
                cut.append((s0, s1))
                continue
            if d0 - pillar > s0:
                cut.append((s0, d0 - pillar))
            if d1 + pillar < s1:
                cut.append((d1 + pillar, s1))
        stretches = cut
    panes = []
    for s0, s1 in sorted(stretches, reverse=True):
        length = s1 - s0
        count = max(1, round((length + pillar) / pitch))
        pane = (length - (count - 1) * pillar) / count
        if pane < MIN_PANE:
            continue
        for k in range(count):
            front = s1 - k * (pane + pillar)
            panes.append((front - pane, front))
    return panes
