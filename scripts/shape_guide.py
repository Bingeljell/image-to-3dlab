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


def seat_rows(start: float, end: float, pitch: float, depth: float,
              keep_out: list[tuple[float, float]] | tuple = (),
              housings: list[tuple[float, float]] | tuple = (),
              margin: float = 0.05) -> list[tuple[float, bool]]:
    """Seats along one wall of a bus, as (rear edge, on a wheel housing) from the front backwards.

    One seat sits centred on top of each wheel housing (as the hand-built bus has them); the
    others stand on the floor at `pitch`, starting at the front end of every stretch that no door
    (`keep_out`) or housing covers, with `margin` to each, and all of them inside start..end.
    """
    rows = [((h0 + h1) / 2 - depth / 2, True) for h0, h1 in housings
            if start <= (h0 + h1) / 2 - depth / 2 and (h0 + h1) / 2 + depth / 2 <= end]
    stretches = [(start, end)]
    for d0, d1 in sorted(list(keep_out) + list(housings)):
        cut = []
        for s0, s1 in stretches:
            if d1 + margin <= s0 or d0 - margin >= s1:
                cut.append((s0, s1))
                continue
            if d0 - margin > s0:
                cut.append((s0, d0 - margin))
            if d1 + margin < s1:
                cut.append((d1 + margin, s1))
        stretches = cut
    for s0, s1 in stretches:
        y0 = s1 - depth
        while y0 >= s0 - 1e-9:
            rows.append((y0, False))
            y0 -= pitch
    return sorted(rows, key=lambda row: -row[0])


def door_poles(doors: list[tuple[float, float]], x: float, front_limit: float) -> list[tuple[float, float]]:
    """Handrail poles at both edges of every door, left out where an edge lies ahead of the
    driver's partition (`front_limit`): there the dashboard stands."""
    return [(x, y) for y0, y1 in doors for y in (y0, y1) if y < front_limit]
