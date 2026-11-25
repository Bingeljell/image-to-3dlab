"""Do a four-legged subject's legs stand apart in a picture?

A generator builds what the outline shows: where the legs touch in a picture, it fuses them into one
lump or doubles them. On ArcherSoul's pets (2026-11-16) every drawn side view that built clean legs
left 21-36 % of the subject's height open under the body between the front and the hind legs; the
one that built a fused lump (a puppy drawn from three quarters behind, its legs bunched) left 6 %,
and above its feet the legs were one part. Front and back views of the same pets showed two feet
apart with gaps of 8-25 % of the width.

Only for four-legged subjects: a person seen from the side shows one leg in front of the other, so
the side check would always fail there. The measures are shares of the subject's own size, so the
picture's resolution and the subject's size in the frame do not matter.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

LEVELS = (0.03, 0.08, 0.15)   # heights above the lowest pixel, as shares of the subject's height
MIN_OPEN_SIDE = 0.12          # side view: at least this share of the height free under the body
MIN_FEET_GAP = 0.02           # front or back view: at least this share of the width between two feet (fronts: 3-25 %)
MIN_PART = 0.015              # a part narrower than this share of the width is a speck, not a leg
ALPHA_CUT = 128


def _runs(row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """(start, end) of every run of covered pixels at least `min_len` long."""
    edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_len]


def _merged(runs: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    """Runs whose gap is narrower than `min_gap` (a speck of background) joined into one part."""
    out: list[tuple[int, int]] = []
    for s, e in runs:
        if out and s - out[-1][1] < min_gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def measure(covered: np.ndarray) -> dict:
    """Parts and gaps across the outline near the floor, and the open space under the body.

    `covered` is a 2-D boolean mask of the subject (True where it is). Returns the subject's size,
    per level of `LEVELS` the number of separate parts and the widest gap between them (share of the
    width), and `open_under_body`: the tallest see-through column between the outermost feet, from
    the floor up (share of the height).
    """
    ys, xs = np.nonzero(covered)
    if not len(ys):
        return {"subject": None, "parts": {}, "gaps": {}, "open_under_body": 0.0}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = covered[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    min_len = max(2, round(MIN_PART * w))
    parts, gaps = {}, {}
    for level in LEVELS:
        row = sub[h - 1 - int(level * h)]
        runs = _merged(_runs(row, min_len), min_len)
        parts[level] = len(runs)
        gaps[level] = round(max(((b[0] - a[1]) / w for a, b in pairwise(runs)), default=0.0), 3)
    floor = np.flatnonzero(sub[h - 1 - int(LEVELS[0] * h)])
    open_share = 0.0
    if len(floor):
        span = sub[:, floor.min():floor.max() + 1][::-1]          # rows from the floor up
        hit = np.where(span.any(axis=0), span.argmax(axis=0), h)  # first covered row in each column
        open_share = float(hit.max()) / h
    return {"subject": [int(w), int(h)], "parts": parts, "gaps": gaps, "open_under_body": round(open_share, 3)}


def problems(m: dict, view: str) -> list[str]:
    """What is wrong with the legs in one view ("side", "front" or "back"); empty when nothing is."""
    if m["subject"] is None:
        return ["no subject in the picture"]
    found = []
    if view == "side":
        if m["open_under_body"] < MIN_OPEN_SIDE:
            found.append(f"only {m['open_under_body']:.0%} of the height is open under the body between the "
                         f"front and the hind legs (clean builds had 21-36 %, at least {MIN_OPEN_SIDE:.0%} needed)")
        if m["parts"][LEVELS[1]] < 2:
            found.append(f"the legs are one part {LEVELS[1]:.0%} above the feet, so the generator will fuse them")
    else:
        if m["parts"][LEVELS[0]] < 2 or m["gaps"][LEVELS[0]] < MIN_FEET_GAP:
            found.append(f"the feet touch in the {view} view (gap {m['gaps'][LEVELS[0]]:.0%} of the width, "
                         f"at least {MIN_FEET_GAP:.0%} needed)")
    return found


def better(a: dict, b: dict) -> bool:
    """Is measurement `a` a better side view than `b` (more open space under the body)?"""
    return (a["open_under_body"], a["parts"].get(LEVELS[1], 0)) > (b["open_under_body"], b["parts"].get(LEVELS[1], 0))


def covered_of(picture: Image.Image) -> np.ndarray:
    """The subject's mask of a cut-out picture (its alpha above half)."""
    return np.asarray(picture.convert("RGBA"))[:, :, 3] > ALPHA_CUT


def check_picture(path: Path, view: str, cut_out=None) -> tuple[dict, list[str]]:
    """Measure one picture file; a picture without a cut-out is cut out first by `cut_out` (RGB -> RGBA)."""
    with Image.open(path) as opened:
        picture = opened.convert("RGBA") if opened.mode == "RGBA" else None
        if picture is None or picture.getchannel("A").getextrema() == (255, 255):
            if cut_out is None:
                from image_to_3dlab import matte

                def cut_out(im):
                    return matte.cut_out(im)[0]
            picture = cut_out(opened.convert("RGB"))
    m = measure(covered_of(picture))
    return m, problems(m, view)
