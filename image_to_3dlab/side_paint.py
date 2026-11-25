"""Where a drawn side picture may paint a figure, for a four-view build painted with `--sides-paint safe`.

A side picture drawn by the picture engine (`scripts/make_side_views.py`) shapes a figure well (whole boots, the right
depth), but it is drawn, not photographed: its hand never hangs exactly where the model's hand is, and its profile's
eye never sits exactly where the model's eye is. Pixel Match paints each picture onto everything it sees, so a drawn
hand landed on a coat beside the holsters, and a drawn profile eye landed on the brow as a second eye above the real
one. Safe sides keep their paint where they are right and only there:

- below the hands (legs, boots) and on the arms and hands themselves, which only the sides see well (a fingerless
  glove's back);
- not on the body between the neck and a little below the fingertips: the front and the back paint it;
- above the neck the ears and the side hair, but not the face (what faces the front between the temples), and nothing
  from the front quarter of the side picture's own head outline, where a drawn profile has its eye, brow and mouth.

Everything is read from the front picture, so no rig is needed: in an A-pose the arms hang apart from the body, and
each row of the front picture's cut-out shows them as runs of their own beside the body's run. A figure whose hands
touch its body keeps those hands front-and-back painted, as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BAND_MARGIN = 0.06   # a drawn hand may hang this share of the figure's height lower than the model's
FADE = 0.02          # the band's edges and the neck fade over this share of the height
HAIR_SKIP = 0.10     # pieces starting in the top share of the figure are hair or a hat brim, never arms
ARM_TOP = 0.45       # an arm starts at the shoulder, above this share of the height from the top; lower pieces beside
                     # the body are coat tails, feet or a weapon's tip
ARM_MIN_AREA = 0.002  # and covers at least this share of the figure's cut-out (a strand or a buckle does not)
NECK_DROP = 0.5      # up from the armpits, the neck is the first row narrower than this share of the widest one below it
FACE_HALF_WIDTH = 0.034   # the face lies between the temples, this share of the height either side of the head's middle
FACE_FACING = (0.35, 0.65)  # and faces the front: no side paint from the first value on, fully off from the second
PROFILE_FACE = 0.25  # the front share of the head's outline in a side picture, where a drawn profile has its face
PROFILE_FADE = 0.05


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True in a row of booleans, as (start, end) with the end excluded."""
    padded = np.concatenate([[False], np.asarray(row, bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def body_runs(row_runs: list[tuple[int, int]], centre: int) -> set[int]:
    """Which runs of a row are the body: the one over the figure's middle column, or, between the legs, the nearest one
    either side of it. Every other run (further out) is an arm, a hand or what it holds."""
    for i, (a, b) in enumerate(row_runs):
        if a <= centre < b:
            return {i}
    left = [i for i, (a, b) in enumerate(row_runs) if b <= centre]
    right = [i for i, (a, b) in enumerate(row_runs) if a > centre]
    return {i for i in (left[-1:] + right[:1])}


@dataclass(frozen=True)
class FrontLayout:
    """A figure's layout in its front picture (rows from the top)."""

    top: int          # the first row of the cut-out
    bottom: int       # the last row
    centre: int       # the head's middle column
    neck: int         # the neck's row: above it is the head
    hands: int        # the lowest row of an arm or hand, -1 when no arm hangs apart from the body
    arm: np.ndarray   # H x W booleans: the arms' and hands' pixels

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def front_layout(alpha: np.ndarray) -> FrontLayout:
    """Read the neck, the arms and the fingertips off the front picture's cut-out (`alpha`, H x W, 0..255)."""
    mask = np.asarray(alpha) > 127
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        raise ValueError("the front picture's cut-out is empty")
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    head = mask[top:top + max(1, round(0.08 * height))]
    centre = int(np.median(np.nonzero(head)[1]))
    beside = np.zeros_like(mask)
    widths = np.zeros(mask.shape[0], int)
    for r in range(top, bottom + 1):
        row_runs = runs(mask[r])
        body = body_runs(row_runs, centre)
        widths[r] = sum(b - a for i, (a, b) in enumerate(row_runs) if i in body)
        for i, (a, b) in enumerate(row_runs):
            if i not in body:
                beside[r, a:b] = True
    # the arms: on each side the largest piece beside the body that starts at the shoulders
    from scipy import ndimage
    labels, count = ndimage.label(beside, structure=np.ones((3, 3)))
    arm = np.zeros_like(mask)
    armpit, hands = None, -1
    if count:
        index = np.arange(1, count + 1)
        area = ndimage.sum(beside, labels, index)
        boxes = ndimage.find_objects(labels)
        centres = ndimage.center_of_mass(beside, labels, index)
        for side in (-1, 1):
            best = None
            for k, box, a, (_, cx) in zip(index, boxes, area, centres):
                starts = (box[0].start - top) / height
                if (cx - centre) * side <= 0 or a < ARM_MIN_AREA * mask.sum():
                    continue
                if not HAIR_SKIP <= starts <= ARM_TOP:
                    continue
                if best is None or a > best[1]:
                    best = (k, a, box)
            if best is not None:
                arm |= labels == best[0]
                armpit = best[2][0].start if armpit is None else min(armpit, best[2][0].start)
                hands = max(hands, best[2][0].stop - 1)
    start = armpit if armpit is not None else top + height // 2
    neck, widest = top, 0
    for r in range(start, top - 1, -1):
        widest = max(widest, widths[r])
        if widths[r] < NECK_DROP * widest:
            neck = r
            break
    return FrontLayout(top, bottom, centre, neck, hands, arm)


def band_keep(rows: np.ndarray, layout: FrontLayout) -> np.ndarray:
    """1 where a drawn side may paint by height (above the neck, below the band), 0 inside the body band, a soft edge
    between. `rows` are each texel's row in the front picture."""
    rows = np.asarray(rows, float)
    fade = max(FADE * layout.height, 1.0)
    low = layout.hands + BAND_MARGIN * layout.height if layout.hands >= 0 else float(layout.bottom)
    above = np.clip((layout.neck - rows) / fade, 0.0, 1.0)
    below = np.clip((rows - low) / fade, 0.0, 1.0)
    keep = np.maximum(above, below)
    return keep * keep * (3 - 2 * keep)


def arm_keep(cols: np.ndarray, rows: np.ndarray, layout: FrontLayout) -> np.ndarray:
    """1 where a texel lies on an arm or hand as the front picture shows them, else 0."""
    h, w = layout.arm.shape
    ix = np.floor(np.asarray(cols, float)).astype(np.int64)
    iy = np.floor(np.asarray(rows, float)).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(len(ix))
    out[inside] = layout.arm[iy[inside], ix[inside]]
    return out


def face_keep(rows: np.ndarray, x: np.ndarray, toward_front: np.ndarray, head_x: float, size: float,
              layout: FrontLayout) -> np.ndarray:
    """0 on the face (above the neck, facing the front, between the temples), 1 elsewhere. `x` is each texel's sideways
    position and `size` the figure's height, both in the views' units; `toward_front` its normal's share toward the
    front camera."""
    fade = max(FADE * layout.height, 1.0)
    head = np.clip((layout.neck - np.asarray(rows, float)) / fade, 0.0, 1.0)
    lo, hi = FACE_FACING
    facing = np.clip((np.asarray(toward_front, float) - lo) / (hi - lo), 0.0, 1.0)
    inner = np.clip((FACE_HALF_WIDTH - np.abs(np.asarray(x, float) - head_x) / size) / 0.008, 0.0, 1.0)
    return 1.0 - head * facing * inner


def profile_face_mask(alpha: np.ndarray, top: int, neck: int, front_right: bool,
                      share: float = PROFILE_FACE) -> np.ndarray:
    """1 where a side picture shows the front `share` of the head's outline between its `top` and `neck` rows (the
    drawn profile's eye, brow, nose and mouth; an ear sits about halfway back), with a soft edge; 0 elsewhere.
    `front_right`: the face looks to the right of the picture. Each row is measured from its own front edge, the
    head's depth as the rows' median width."""
    mask = np.asarray(alpha) > 127
    height, width = mask.shape
    out = np.zeros((height, width))
    rows = [(r, *np.flatnonzero(mask[r])[[0, -1]]) for r in range(max(top, 0), min(neck, height - 1) + 1)
            if mask[r].any()]
    if not rows:
        return out
    depth = float(np.median([hi - lo for _, lo, hi in rows]))
    band = share * depth
    fade = max(PROFILE_FADE * depth, 1.0)
    col = np.arange(width)
    for r, lo, hi in rows:
        dist = (hi - col) if front_right else (col - lo)
        out[r] = np.clip((band + fade - dist) / fade, 0.0, 1.0) * mask[r]
    return out
