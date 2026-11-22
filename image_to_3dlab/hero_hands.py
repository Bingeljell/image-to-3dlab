"""Hero hands: a hand built on its own, with real fingers, grafted onto a figure at the wrist, as pure functions.

Pixal3D spends its detail on its whole box, so a hand that is a few dozen pixels of a full-body picture comes out
as a mitten. The same hand cropped, upscaled, redrawn flat (the fingers apart, seen from the back of the hand) and
built on its own fills the box and keeps five fingers. This module holds the maths that finds the hands, counts
the digits of the drawings and of the builds, and puts the new hand on the figure; `scripts/hero_hands.py` runs
the steps.

The head graft (`hero.py`) taught what to avoid at the seam (issue #229): two skins left overlapping where they
do not meet, and a cut that follows a saw-tooth of whole faces. Here both models are cut straight through by a
plane (triangles split on it), at the wrist, below a cuff, and the two rims are stitched into one surface: no
overlap, no gap. The rigged figure keeps its skeleton and every body weight; the new hand takes the old hand's
weights, so it moves with the hand bone and, where the rig has them, the finger bones.
"""
from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
WELD_DIGITS = 6           # positions are welded on a grid of 1e-6 of the model's size
SIDES = ("right", "left")
FORWARD = np.array([0.0, 0.0, 1.0])   # Pixal3D models face +Z (glTF's front)
DRAWN_DOWN = np.array([0.0, -1.0, 0.0])   # a drawn hand hangs straight down, and its build keeps the drawing's frame
PROFILE_STEPS = 64        # sections along the limb when looking for the wrist
ARM_SHARE = 0.25          # a figure's limb is looked at from this share of the forearm above the rig's wrist joint
SEAM_RISE = 1.15          # the wrist's section is at least this much narrower than the widest on both sides of it
                          # (the palm, and a cuff or the forearm); a gap between fingers is shallower (1.05-1.14 on
                          # the Revolverheld's mittens), its wrists 1.24 and 1.27, a built hand's 1.31
SEAM_SHIFT = 0.04         # the cut sits this share of the hand's length below the narrowest section, toward the
                          # fingers, so it runs through the glove or the wrist and never grazes a cuff's rim
REACH_SHARE = 1.6         # the old hand is removed only within this many hand lengths of the seam: an infinite
                          # plane would also cut a coat tail hanging beyond it
TIP_PERCENTILE = 99.5     # a hand's length: this percentile of its points' distances from the seam's centre
MAX_INFLUENCES = 4
FOREARM_SHARE = 0.5       # the crop takes the forearm's points this share of the hand's length above the wrist,
                          # so the drawing keeps the cuff it has to match
HAND_CROP_PAD = 1.25      # crop side vs the hand's own size
SCALE_LIMITS = (0.7, 1.4)   # a new wrist this much narrower or wider than the old one needs a human's glance
TURN_LIMIT_DEG = 45.0       # the figure's own palm (the old hand's thinnest direction) this far from the placed one
                            # says the pose is not the usual A-pose with the palms to the thighs
THUMB_HINT = 0.03           # the thumb's cleft reaches at least this share of the hand's length nearer the wrist than
                            # the deepest cleft between the fingers ...
THUMB_CLEFT = 0.035         # ... and a cleft this deep at the row's other end is a second thumb, not a splayed little
                            # finger: on 32 drawings checked by a second reviewer, thumbs lay 0.04-0.12 deeper, little
                            # fingers at most 0.02, second thumbs and doubtful digits 0.04-0.14 (AssetFurnace Support D248)


# --- Mesh pieces ----------------------------------------------------------------------------------------------

def weld_groups(vertices: np.ndarray) -> np.ndarray:
    """A group id per vertex, equal for vertices at the same position (glTF splits vertices along UV seams)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if not len(vertices):
        return np.zeros(0, dtype=np.int64)
    extent = float(np.ptp(vertices, axis=0).max()) or 1.0
    keys = np.round(vertices / extent * 10 ** WELD_DIGITS).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    return group.reshape(-1)


def weld(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(welded points, faces on them, group of each input vertex): one point per position."""
    group = weld_groups(vertices)
    points = np.zeros((group.max() + 1 if len(group) else 0, 3))
    points[group] = np.asarray(vertices, dtype=np.float64)
    return points, group[np.asarray(faces, dtype=np.int64)], group


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def plane_basis(normal: np.ndarray) -> np.ndarray:
    """Two unit vectors spanning the plane with this normal (rows), any turn."""
    n = unit(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(n, helper))
    return np.stack([u, np.cross(n, u)])


def chain_loops(edges: np.ndarray) -> list[np.ndarray]:
    """Closed loops from directed edges (a -> b), as arrays of the ids in walking order.

    Every id should have one edge out and one in. Where a cut grazes a vertex shared by two loops, an id has two
    out; the walk then takes the first unused one, so each edge lands in exactly one loop. Open chains (a mesh
    that was not closed there) come back too, so nothing is lost silently.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    outgoing: dict[int, list[int]] = {}
    for index, (a, _) in enumerate(edges):
        outgoing.setdefault(int(a), []).append(index)
    used = np.zeros(len(edges), dtype=bool)
    incoming = {int(b) for _, b in edges}
    # Open chains first, from their free ends, so a closed walk never starts in their middle.
    starts = [i for i, (a, _) in enumerate(edges) if int(a) not in incoming] + list(range(len(edges)))
    loops = []
    for first in starts:
        if used[first]:
            continue
        loop, edge = [int(edges[first, 0])], first
        while edge is not None and not used[edge]:
            used[edge] = True
            nxt = int(edges[edge, 1])
            if nxt == loop[0]:
                break
            loop.append(nxt)
            edge = next((e for e in outgoing.get(nxt, []) if not used[e]), None)
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def _signed(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = (np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)) @ unit(normal)
    d[np.abs(d) < 1e-12] = -1e-12            # a vertex exactly on the plane counts as kept
    return d


def _cut_edges(faces: np.ndarray, out: np.ndarray, select: np.ndarray):
    """Per corner k, the faces whose one odd corner is k: ("in", a, b, c) keeps only a, ("out", a, b, c) loses only a."""
    outs = out[faces].sum(axis=1)
    for k in range(3):
        a, b, c = faces[:, k], faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]
        one_in = select & (outs == 2) & ~out[a]
        one_out = select & (outs == 1) & out[a]
        yield "in", a[one_in], b[one_in], c[one_in]
        yield "out", a[one_out], b[one_out], c[one_out]


def crossings(points: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray,
              select: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(the points where a plane crosses a welded mesh's edges, the segments between them as pairs of their ids,
    each running as the kept faces' boundary). SELECT (bool per face) limits the cut to some faces."""
    faces = np.asarray(faces, dtype=np.int64)
    d = _signed(points, origin, normal)
    chosen = np.ones(len(faces), bool) if select is None else np.asarray(select, dtype=bool)
    segments = []
    for kind, a, b, c in _cut_edges(faces, d > 0, chosen):
        if len(a):
            ab, ca = np.sort(np.stack([a, b], 1), 1), np.sort(np.stack([c, a], 1), 1)
            # the outline runs as the kept faces' boundary: P(a,b) -> P(c,a) when only a is kept, else reversed
            segments.append(np.stack([ab, ca], 1) if kind == "in" else np.stack([ca, ab], 1))
    if not segments:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64)
    seg = np.concatenate(segments)                       # (s, 2 ends, 2 vertex ids of the crossed edge)
    keys, inverse = np.unique(seg.reshape(-1, 2), axis=0, return_inverse=True)
    t = d[keys[:, 0]] / (d[keys[:, 0]] - d[keys[:, 1]])
    crossing = points[keys[:, 0]] + (points[keys[:, 1]] - points[keys[:, 0]]) * t[:, None]
    return crossing, inverse.reshape(-1, 2)


def section_loops(points: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray,
                  select: np.ndarray | None = None) -> list[np.ndarray]:
    """The closed outlines where a plane cuts a welded mesh (`weld`), as arrays of 3D points.

    SELECT (bool per face) limits the cut to some faces: an arm, not the coat beside it.
    """
    crossing, seg = crossings(points, faces, origin, normal, select)
    return [crossing[loop] for loop in chain_loops(seg) if len(loop) >= 3]


def clip(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray,
         cuttable: np.ndarray | None = None, attributes: tuple[np.ndarray, ...] = ()):
    """Cut a mesh straight through by a plane and keep the side NORMAL points away from.

    Triangles that cross the plane are split on it (a saw-tooth of whole faces left the old neck's edge jagged in
    issue #229). New vertices sit on the crossed edges, one per edge as the input numbers it, so a UV seam stays a
    seam; every per-vertex ATTRIBUTE (UVs, normals, skin weights as one row per vertex) is interpolated along the
    edge. Faces where CUTTABLE is False are kept whole whatever side they are on: the cut is for one limb, not for
    the coat beside it.

    Returns (vertices, faces, attributes, rim): RIM holds the cut's outline as edges between new vertices,
    oriented as boundary edges of the kept faces.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    d = _signed(vertices, origin, normal)
    out = d > 0
    chosen = np.ones(len(faces), bool) if cuttable is None else np.asarray(cuttable, dtype=bool)
    outs = out[faces].sum(axis=1)
    keep = ~chosen | (outs == 0)
    pending = list(_cut_edges(faces, out, chosen))
    crossed = [np.stack(pair, 1) for _, a, b, c in pending for pair in ((a, b), (c, a))]
    crossed = np.concatenate(crossed) if crossed else np.zeros((0, 2), np.int64)
    unique, inverse = np.unique(np.sort(crossed, axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    t = d[unique[:, 0]] / (d[unique[:, 0]] - d[unique[:, 1]]) if len(unique) else np.zeros(0)
    base = len(vertices)
    mixed = vertices[unique[:, 0]] + (vertices[unique[:, 1]] - vertices[unique[:, 0]]) * t[:, None]
    out_attributes = []
    for attribute in attributes:
        values = np.asarray(attribute, dtype=np.float64)
        flat = values.reshape(len(values), -1)
        between = flat[unique[:, 0]] + (flat[unique[:, 1]] - flat[unique[:, 0]]) * t[:, None]
        out_attributes.append(np.concatenate([flat, between]).reshape((-1, *values.shape[1:])))
    new_faces, rims, cursor = [faces[keep]], [], 0
    for kind, a, b, c in pending:
        m = len(a)
        p_ab, p_ca = base + inverse[cursor:cursor + m], base + inverse[cursor + m:cursor + 2 * m]
        cursor += 2 * m
        if kind == "in":                        # a kept: the triangle (a, P(a,b), P(c,a))
            new_faces.append(np.stack([a, p_ab, p_ca], 1))
            rims.append(np.stack([p_ab, p_ca], 1))
        else:                                   # a removed: the quad (P(a,b), b, c, P(c,a))
            new_faces += [np.stack([p_ab, b, c], 1), np.stack([p_ab, c, p_ca], 1)]
            rims.append(np.stack([p_ca, p_ab], 1))
    return (np.concatenate([vertices, mixed]), np.concatenate(new_faces), tuple(out_attributes),
            np.concatenate(rims) if rims else np.zeros((0, 2), np.int64))


def compact(vertices: np.ndarray, faces: np.ndarray, attributes: tuple[np.ndarray, ...] = ()):
    """Drop vertices no face uses. Returns (vertices, faces, attributes, new index of each old vertex or -1)."""
    used = np.unique(np.asarray(faces, dtype=np.int64))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (np.asarray(vertices)[used], remap[faces], tuple(np.asarray(a)[used] for a in attributes), remap)


def components(faces: np.ndarray, group: np.ndarray) -> np.ndarray:
    """A component id per face: faces sharing a (welded) vertex belong together."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(group)[np.asarray(faces, dtype=np.int64)]
    if not len(f):
        return np.zeros(0, dtype=np.int64)
    m, count = len(f), int(f.max()) + 1
    # one graph of faces and vertices, each face joined to its three corners
    rows = np.repeat(np.arange(m), 3)
    graph = coo_matrix((np.ones(3 * m), (rows, m + f.reshape(-1))), shape=(m + count, m + count))
    _, label = connected_components(graph, directed=False)
    return label[:m]


def loop_area(points: np.ndarray, normal: np.ndarray, signed: bool = False) -> float:
    """The area a closed outline encloses, seen along NORMAL (counter-clockwise positive when SIGNED)."""
    uv = np.asarray(points, dtype=np.float64) @ plane_basis(normal).T
    x, y = uv[:, 0], uv[:, 1]
    area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
    return area if signed else abs(area)


def contains(points: np.ndarray, normal: np.ndarray, point: np.ndarray) -> bool:
    """Whether POINT, seen along NORMAL, lies inside the closed outline (even-odd rule)."""
    basis = plane_basis(normal)
    uv = np.asarray(points, dtype=np.float64) @ basis.T
    px, py = np.asarray(point, dtype=np.float64) @ basis.T
    x0, y0 = uv[:, 0], uv[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.count_nonzero(crosses & (px < at)) % 2)


def limb_loop(loops: list[np.ndarray], normal: np.ndarray, centre: np.ndarray) -> np.ndarray | None:
    """Of a section's outlines, the one round the limb: the largest that holds CENTRE, else the one nearest it."""
    if not loops:
        return None
    holding = [loop for loop in loops if contains(loop, normal, centre)]
    if holding:
        return max(holding, key=lambda loop: loop_area(loop, normal))
    return min(loops, key=lambda loop: float(np.linalg.norm(loop.mean(axis=0) - centre)))


# --- The wrist ------------------------------------------------------------------------------------------------

PIECE_SHARE = 0.05   # an outline under this share of a section's largest is a sliver, not a piece (a cuff's rim
                     # left specks of 1e-6 beside the Revolverheld's left wrist, of 1.2e-3)
SECTION_PIXELS = 200  # a limb's sections are filled on a grid this many pixels across its widest: a wrist ~100 across
SOLID_SHARE = 0.5     # a section's piece filling less of its outline's hull is a shell round a hollow: solid sections
                      # of the Revolverheld's, the Zauberer's and the Waechter's builds fill 0.97-1.0 of it, the
                      # Waechter's knuckle plate (open toward the palm) and cuff lip 0.12-0.18
HOLLOW_LIMIT = 0.05   # a built hand with more of its sections below the wrist hollow has a hole for a palm: the
                      # Waechter's first gauntlet 7 of 38, six good gloved and bare builds none


def filled(segments: np.ndarray, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """(inside, corner): flat SEGMENTS (s, 2 ends, 2) drawn as lines on a grid of PIXEL and filled from outside;
    INSIDE is every cell no path from outside reaches without crossing a line (the lines too), CORNER the plane
    position of cell (0, 0)."""
    import cv2

    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    corner = seg.reshape(-1, 2).min(axis=0) - 2 * pixel
    cells = np.round((seg - corner) / pixel).astype(np.int32)
    canvas = np.zeros((int(cells[..., 1].max()) + 3, int(cells[..., 0].max()) + 3), np.uint8)
    cv2.polylines(canvas, cells.reshape(-1, 2, 1, 2), False, 1)       # 8-connected lines ...
    seen = np.zeros((canvas.shape[0] + 2, canvas.shape[1] + 2), np.uint8)
    cv2.floodFill(canvas, seen, (0, 0), 2, flags=4)                    # ... that a 4-connected fill cannot cross
    return canvas != 2, corner


def mask_pieces(inside: np.ndarray, least: float = PIECE_SHARE, solid: float = SOLID_SHARE) -> dict:
    """A filled section's pieces, slivers under LEAST of the largest dropped: {"labels" (0 outside or not a piece,
    piece k is k + 1), "areas", "centres" (in cells), "hollow" (per piece)}.

    Armour leaves two kinds of shell a fill from outside misreads (the Waechter's gauntlet build, AssetFurnace
    Support D241). A cuff's lip hanging round the wrist, open on one side, is a thin piece whose outline wraps
    another piece: it is not a piece of its own and adds nothing. A knuckle plate built as a shell open toward a
    hollow palm fills less than SOLID of its outline's hull: it is hollow and counts as the hull it encloses, or
    the open shell reads as a wrist a tenth as wide as the hand."""
    import cv2

    n, labels, stats, centres = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=4)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    big = np.flatnonzero(areas >= least * areas.max(initial=0.0))
    hulls, hull_areas = {}, {}
    for k in big:
        x, y, w, h = stats[k + 1, :4]
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == k + 1)
        hulls[k] = cv2.convexHull(np.stack([xs + x, ys + y], 1).astype(np.int32))
        hull_areas[k] = max(float(cv2.contourArea(hulls[k])), areas[k])
    wraps = {k for k in big for j in big if j != k and hull_areas[k] > hull_areas[j]
             and cv2.pointPolygonTest(hulls[k], (float(centres[j + 1, 0]), float(centres[j + 1, 1])), False) >= 0}
    kept = [k for k in big if k not in wraps]
    keep = np.zeros(n, dtype=np.int64)
    keep[1:][kept] = np.arange(1, len(kept) + 1)
    hollow = np.array([areas[k] < solid * hull_areas[k] for k in kept], dtype=bool)
    sized = np.array([hull_areas[k] if h else areas[k] for k, h in zip(kept, hollow)])
    return {"labels": keep[labels], "areas": sized.reshape(-1), "centres": centres[1:][kept].reshape(-1, 2),
            "hollow": hollow}


def pieces(loops: list[np.ndarray], normal: np.ndarray, least: float = PIECE_SHARE) -> int:
    """How many separate pieces a section's outlines make: an outline inside another (a double-walled build's
    inner skin, a hollow sleeve's lining) is part of that one, and slivers do not count. A wrist is one piece,
    a row of fingers several."""
    loops = [np.asarray(loop, dtype=np.float64) for loop in loops if len(loop) >= 3]
    if not loops:
        return 0
    basis = plane_basis(normal)
    segments = np.concatenate([np.stack([loop, np.roll(loop, -1, axis=0)], 1) @ basis.T for loop in loops])
    pixel = max(float(np.ptp(segments.reshape(-1, 2), axis=0).max()), 1e-12) / SECTION_PIXELS
    return len(mask_pieces(filled(segments, pixel)[0], least)["areas"])


def section_fill(points: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray,
                 select: np.ndarray | None = None, pixel: float | None = None) -> dict:
    """A section through a welded mesh as the solid it outlines: {"area" (NaN where the plane misses), "pieces",
    "centre" (of the piece holding ORIGIN, else the largest; None where it misses), "piece_area" (that piece's),
    "hollow" (whether a piece is a shell round a hollow, `mask_pieces`)}.

    Filled from outside, so its area is what the outer wall encloses. A Pixal3D build has an inner skin about a
    millimetre in, and where the two walls touch, one walk ran round both and their areas cancelled; elsewhere the
    inner skin's area was added. Measured as outlines, the Waechter's gauntlet build read 0.009 where it enclosed
    about 0.05, and the wrist was taken there, on the knuckle plate (AssetFurnace Support D241)."""
    normal = unit(normal)
    crossing, seg = crossings(points, faces, origin, normal, select)
    if not len(seg):
        return {"area": float("nan"), "pieces": 0, "centre": None}
    basis = plane_basis(normal)
    flat = crossing @ basis.T
    if pixel is None:
        pixel = max(float(np.ptp(flat, axis=0).max()), 1e-12) / SECTION_PIXELS
    inside, corner = filled(flat[seg], pixel)
    found = mask_pieces(inside)
    labels, areas, centres = found["labels"], found["areas"], found["centres"]
    if not len(areas):
        return {"area": float("nan"), "pieces": 0, "centre": None}
    here = np.round((np.asarray(origin, dtype=np.float64) @ basis.T - corner) / pixel).astype(int)
    holder = (labels[here[1], here[0]] if 0 <= here[1] < labels.shape[0] and 0 <= here[0] < labels.shape[1]
              else 0)
    piece = holder - 1 if holder > 0 else int(np.argmax(areas))
    middle = corner + centres[piece] * pixel
    centre = middle @ basis + (np.asarray(origin, dtype=np.float64) @ normal) * normal
    return {"area": float(areas.sum()) * pixel ** 2, "pieces": len(areas), "centre": centre,
            "piece_area": float(areas[piece]) * pixel ** 2, "hollow": bool(found["hollow"].any())}


def section_pixel(points: np.ndarray, axis: np.ndarray, faces: np.ndarray | None = None,
                  select: np.ndarray | None = None) -> float:
    """One grid size for every section of a limb: its widest extent across AXIS over SECTION_PIXELS."""
    p = np.asarray(points, dtype=np.float64)
    if select is not None and faces is not None:
        p = p[np.unique(np.asarray(faces)[np.asarray(select, dtype=bool)])]
    flat = p @ plane_basis(axis).T
    return max(float(np.ptp(flat, axis=0).max()) if len(flat) else 0.0, 1e-12) / SECTION_PIXELS


def sections_along(points: np.ndarray, faces: np.ndarray, start: np.ndarray, axis: np.ndarray, ts: np.ndarray,
                   select: np.ndarray | None = None) -> list[dict]:
    """The limb's sections at START + t AXIS for every t (`section_fill`), on one grid."""
    axis = unit(axis)
    pixel = section_pixel(points, axis, faces, select)
    return [section_fill(points, faces, np.asarray(start, dtype=np.float64) + t * axis, axis, select, pixel)
            for t in ts]


def profile(points: np.ndarray, faces: np.ndarray, start: np.ndarray, axis: np.ndarray, ts: np.ndarray,
            select: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(area, pieces) of the limb's sections at START + t AXIS for every t: the solid all its outlines enclose
    together (past the knuckles the fingers are five; `section_fill`), NaN where the plane misses it, and how
    many separate pieces they make.

    All outlines, not the one round the axis: between the fingers the axis holds none, the nearest finger
    stood in for the hand, and the walk took a knuckle for the wrist."""
    sections = sections_along(points, faces, start, axis, ts, select)
    return (np.array([s["area"] for s in sections], dtype=np.float64),
            np.array([s["pieces"] for s in sections], dtype=np.int64))


def running_median(values: np.ndarray, width: int = 3) -> np.ndarray:
    """VALUES with each one the median of its WIDTH neighbours, NaN kept: one odd section is not a wrist."""
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return a
    half = width // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(a, half, mode="edge"), width)
    out = np.full(len(a), np.nan)
    finite = ~np.isnan(windows)
    for i in np.flatnonzero(~np.isnan(a)):
        out[i] = np.median(windows[i][finite[i]])
    return out


def seam_index(areas: np.ndarray, counts: np.ndarray | None = None, rise: float = SEAM_RISE) -> int | None:
    """The wrist in a profile of section areas along an arm, from the arm (first) to the fingertips (last): the
    valley nearest the fingertips that is one piece and RISE deep.

    A wrist is narrower than the palm on one side and a cuff or the forearm on the other: a valley whose widest
    sections on both sides are at least RISE times wider. It is one piece (COUNTS, from `profile`), which a gap
    between two fingers is not. Of the valleys left, the wrist is the first above the hand: a drawn hand with a
    long sleeve has a second valley above its cuff, wider in area than the wrist and so deeper by area alone (the
    Revolverheld's seed-4 hand: its cuff came along onto the figure as a second cuff). Not simply the narrowest
    section: a fingertip is narrower still. Smoothed over three sections first; wider, the median hid the
    Revolverheld's left wrist, narrow for one section below its cuff. None when no valley qualifies (a mitten as
    wide as its sleeve).
    """
    a = running_median(areas)
    one = np.ones(len(a), bool) if counts is None else np.asarray(counts) == 1
    valid = np.flatnonzero(~np.isnan(a))
    best = None
    for k in range(1, len(valid) - 1):
        i = valid[k]
        if a[i] <= 0 or not one[i] or a[i] > a[valid[k - 1]] or a[i] > a[valid[k + 1]]:
            continue
        if min(a[valid[:k]].max(), a[valid[k + 1:]].max()) >= rise * a[i]:
            best = int(i)
    return best


def limb_seam(points: np.ndarray, faces: np.ndarray, start: np.ndarray, axis: np.ndarray, length: float,
              select: np.ndarray | None = None, steps: int = PROFILE_STEPS, shift: float = SEAM_SHIFT) -> dict:
    """Where to cut a limb whose sections run from START (up the arm) along AXIS for LENGTH (past the fingertips):
    {"cut": the plane's point on the axis, "centre": the section's middle there, "found": False when no wrist
    shows in the profile (the cut then sits halfway and the caller should say so), "areas", "ts", "hollow": the
    share of the sections past the cut that hold a shell round a hollow (`mask_pieces`)}."""
    ts = np.linspace(0.0, length, steps)
    sections = sections_along(points, faces, start, axis, ts, select)
    areas = np.array([s["area"] for s in sections], dtype=np.float64)
    counts = np.array([s["pieces"] for s in sections], dtype=np.int64)
    low = seam_index(areas, counts)
    found = low is not None
    t_low = ts[low] if found else 0.5 * length
    valid = np.flatnonzero(~np.isnan(areas))
    t_tip = ts[valid[-1]] if len(valid) else length
    t_cut = t_low + shift * max(t_tip - t_low, 0.0)
    here = np.asarray(start, dtype=np.float64) + t_cut * unit(axis)
    section = section_fill(points, faces, here, axis, select, section_pixel(points, axis, faces, select))
    found_piece = section["centre"] is not None
    past = [s for t, s in zip(ts, sections) if t > t_cut and s["pieces"]]
    return {"cut": here, "centre": section["centre"] if found_piece else here, "found": found,
            "area": section["piece_area"] if found_piece else float("nan"), "areas": areas, "ts": ts,
            "hollow": sum(s["hollow"] for s in past) / max(len(past), 1)}


def reach_of(points: np.ndarray, centre: np.ndarray) -> float:
    """A hand's length: how far its points beyond the seam reach from the seam's centre (a high percentile, so a
    stray splinter does not count)."""
    return float(np.percentile(np.linalg.norm(np.asarray(points) - centre, axis=1), TIP_PERCENTILE))


def principal_axes(points: np.ndarray) -> np.ndarray:
    """The points' main directions as rows, longest first."""
    p = np.asarray(points, dtype=np.float64)
    p = p - p.mean(axis=0)
    _, vectors = np.linalg.eigh(p.T @ p / max(len(p), 1))
    return vectors.T[::-1]


def hand_frame(beyond: np.ndarray, seam_centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(finger direction, palm normal up to sign) of a hand's points BEYOND its seam: the fingers point from the
    seam's centre to the hand's middle; the palm normal is the hand's thinnest direction, square to the fingers."""
    fingers = unit(np.asarray(beyond, dtype=np.float64).mean(axis=0) - seam_centre)
    thin = principal_axes(beyond)[2]
    thin = thin - (thin @ fingers) * fingers
    return fingers, unit(thin)


def figure_back(fingers: np.ndarray, side: str, forward: np.ndarray = FORWARD) -> np.ndarray:
    """The way the back of the figure's hand faces: the thumb forward (a hanging A-pose hand with the palm to the
    thigh, or a T-pose hand palm down), so the back looks out and up. For a right hand thumb = back x fingers."""
    thumb = forward - (forward @ fingers) * fingers
    thumb = unit(thumb)
    back = np.cross(fingers, thumb)
    return unit(back if side == "right" else -back)


def rotation_between(a_from: np.ndarray, b_from: np.ndarray, a_to: np.ndarray, b_to: np.ndarray) -> np.ndarray:
    """The rotation taking direction A_FROM to A_TO and, square to it, B_FROM to B_TO."""
    def frame(a, b):
        a = unit(a)
        b = unit(b - (b @ a) * a)
        return np.stack([a, b, np.cross(a, b)], 1)

    return frame(a_to, b_to) @ frame(a_from, b_from).T


def placement(seam_from: np.ndarray, fingers_from: np.ndarray, back_from: np.ndarray, length_from: float,
              seam_to: np.ndarray, fingers_to: np.ndarray, back_to: np.ndarray, length_to: float) -> np.ndarray:
    """The 4x4 transform putting a built hand on the figure's: its seam's centre on the figure's, its fingers
    along the old hand's, the back of the hand where the figure's faces, scaled to the old hand's length."""
    turn = rotation_between(fingers_from, back_from, fingers_to, back_to)
    k = length_to / length_from
    tf = np.eye(4)
    tf[:3, :3] = k * turn
    tf[:3, 3] = np.asarray(seam_to, dtype=np.float64) - k * turn @ np.asarray(seam_from, dtype=np.float64)
    return tf


def moved(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """POINTS moved by the 4x4 transform TF."""
    return np.asarray(points, dtype=np.float64) @ tf[:3, :3].T + tf[:3, 3]


def mirror_x(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None):
    """A right hand made left: mirrored across x = 0, the faces turned so they still face out."""
    v = np.asarray(vertices, dtype=np.float64) * [-1.0, 1.0, 1.0]
    n = None if normals is None else np.asarray(normals, dtype=np.float64) * [-1.0, 1.0, 1.0]
    return v, np.asarray(faces)[:, ::-1].copy(), n


# --- The seam -------------------------------------------------------------------------------------------------

def rim_loop(vertices: np.ndarray, rim: np.ndarray, normal: np.ndarray, centre: np.ndarray):
    """(vertex ids in walking order, welded positions) of the cut's outline round the limb.

    RIM is `clip`'s edge list; a UV seam that crosses the cut leaves two vertices at one place, so the outline is
    walked on welded positions and each place keeps one of its vertices."""
    ids = np.unique(rim)
    group = weld_groups(np.asarray(vertices)[ids])
    first = np.zeros(group.max() + 1, dtype=np.int64)
    first[group[::-1]] = ids[::-1]                       # one vertex per place (the lowest id)
    lookup = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    lookup[ids] = group
    loops = chain_loops(lookup[rim])
    points = [np.asarray(vertices)[first[loop]] for loop in loops]
    best = limb_loop(points, normal, centre)
    if best is None:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    index = next(i for i, p in enumerate(points) if p is best)
    return first[loops[index]], best


def bridge(a_points: np.ndarray, b_points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Triangles (as pairs of (loop, index): 0 for A, 1 for B) stitching two rims into one surface.

    A is the body's rim and B the new hand's, each as its own faces' boundary runs; seen along NORMAL they turn
    opposite ways. B is walked backwards so both turn alike, both start at the point nearest the same direction
    from their centres, and the strip advances on whichever rim is further behind in its own length, so a rim
    with more points still meets the other evenly. Each triangle runs its rim edge against the edge's own face,
    so the stitched surface faces out like both halves.
    """
    a = np.asarray(a_points, dtype=np.float64)
    b = np.asarray(b_points, dtype=np.float64)[::-1]
    basis = plane_basis(normal)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    angle_a = np.arctan2(*((a - ca) @ basis.T).T[::-1])
    angle_b = np.arctan2(*((b - cb) @ basis.T).T[::-1])
    sa, sb = int(np.argmin(np.abs(angle_a))), int(np.argmin(np.abs(angle_b)))
    a_order, b_order = np.roll(np.arange(len(a)), -sa), np.roll(np.arange(len(b)), -sb)
    b_ids = (len(b) - 1 - b_order)                     # back to B's own numbering

    def progress(points, order):
        steps = np.linalg.norm(np.diff(points[np.r_[order, order[0]]], axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)]) / max(steps.sum(), 1e-12)

    ua, ub = progress(a, a_order), progress(b, b_order)
    i = j = 0
    triangles = []
    while i < len(a) or j < len(b):
        ai, ai1 = a_order[i % len(a)], a_order[(i + 1) % len(a)]
        bj, bj1 = b_ids[j % len(b)], b_ids[(j + 1) % len(b)]
        if j >= len(b) or (i < len(a) and ua[i + 1] <= ub[j + 1]):
            triangles.append(((0, ai), (1, bj), (0, ai1)))
            i += 1
        else:
            triangles.append(((1, bj), (1, bj1), (0, ai)))
            j += 1
    return np.array(triangles, dtype=np.int64)


def turn_of(loop: np.ndarray, normal: np.ndarray) -> float:
    """+1 when the rim turns counter-clockwise seen along NORMAL, else -1."""
    return 1.0 if loop_area(loop, normal, signed=True) > 0 else -1.0


def rim_pieces(vertices: np.ndarray, rim: np.ndarray) -> list[np.ndarray]:
    """Every outline a cut left, as vertex ids (welded: one per place)."""
    if not len(rim):
        return []
    group = weld_groups(np.asarray(vertices))
    first = np.zeros(group.max() + 1, dtype=np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    return [first[loop] for loop in chain_loops(group[rim])]


def outer_wall(vertices: np.ndarray, faces: np.ndarray, rim: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Which faces belong to the piece with the largest rim, after a cut.

    A Pixal3D build is double-walled: a second skin about a millimetre inside the first, joined to it only at an
    open edge (a sleeve's top). Cut below that edge, the two skins come apart, and the outer one is the piece
    whose rim encloses the most. Floating crumbs the build had go too."""
    group = weld_groups(vertices)
    label = components(faces, group)
    pieces = rim_pieces(vertices, rim)
    if not pieces:
        return np.ones(len(faces), bool)
    widest = max(pieces, key=lambda ids: loop_area(np.asarray(vertices)[ids], normal))
    corner = group[widest[0]]
    owner = label[np.flatnonzero((group[np.asarray(faces)] == corner).any(axis=1))[0]]
    return label == owner


ATTRIBUTES = ("normals", "uvs", "dense", "marks")   # marks: which arm a vertex belongs to, carried through cuts


def cut_part(part: dict, origin: np.ndarray, normal: np.ndarray, cuttable: np.ndarray | None = None):
    """(PART cut by the plane, keeping the side NORMAL points away from; the cut's rim edges in its numbering).
    A part is a dict of positions and faces with any of normals, uvs and dense skin weights."""
    present = [k for k in ATTRIBUTES if part.get(k) is not None]
    vertices, faces, attributes, rim = clip(part["positions"], part["faces"], origin, normal, cuttable,
                                            tuple(part[k] for k in present))
    vertices, faces, attributes, remap = compact(vertices, faces, attributes)
    out = {**part, "positions": vertices, "faces": faces, **dict(zip(present, attributes))}
    rim = remap[rim]
    return out, rim[(rim >= 0).all(axis=1)]


def keep_faces(part: dict, keep: np.ndarray, rim: np.ndarray | None = None):
    """PART with only the KEEP faces (and the vertices they use); RIM renumbered along."""
    present = [k for k in ATTRIBUTES if part.get(k) is not None]
    vertices, faces, attributes, remap = compact(part["positions"], np.asarray(part["faces"])[keep],
                                                 tuple(part[k] for k in present))
    out = {**part, "positions": vertices, "faces": faces, **dict(zip(present, attributes))}
    if rim is None:
        return out, None
    rim = remap[rim]
    return out, rim[(rim >= 0).all(axis=1)]


def blend_wrist(points: np.ndarray, cut: np.ndarray, axis: np.ndarray, centre_from: np.ndarray,
                centre_to: np.ndarray, ratio: float, span: float) -> np.ndarray:
    """POINTS of a placed hand eased into the figure's wrist: at the cut the hand's section is scaled by RATIO and
    its middle moved from CENTRE_FROM to CENTRE_TO; the change fades out over SPAN along AXIS toward the fingers.

    The hand is sized by its length, and a drawn hand's wrist need not be as thick as the figure's by the same
    share (the Revolverheld's was about a tenth thinner). Without this the stitch would be a step."""
    axis = unit(axis)
    p = np.asarray(points, dtype=np.float64)
    s = (p - cut) @ axis
    fade = np.clip(1.0 - s / max(span, 1e-12), 0.0, 1.0)[:, None]
    shift = np.asarray(centre_to, dtype=np.float64) - centre_from
    shift = shift - (shift @ axis) * axis
    line = centre_from + np.outer(s, axis)                 # the hand's middle line at each point's height
    radial = p - line
    radial -= np.outer(radial @ axis, axis)
    return p + fade * (radial * (ratio - 1.0) + shift)


GAP_SHARE = 0.02     # the stitch is this share of the hand's length long: a short band, not a flat ring
BLEND_SHARE = 0.25   # the wrist eases into the figure's over this share of the hand's length


def figure_wrist(positions: np.ndarray, faces: np.ndarray, bone: np.ndarray, forearm: int, hand: list[int],
                 elbow: np.ndarray, wrist: np.ndarray, side: str, forward: np.ndarray = FORWARD) -> dict:
    """Where the figure's arm is cut and how a new hand must sit there.

    BONE is every vertex's main bone; FOREARM and HAND (the hand bone and every bone below it) name the arm;
    ELBOW and WRIST are the rig's joints. The arm is cut square to the forearm, just below its narrowest section
    between the palm and a cuff (or the forearm). The new hand's fingers take the old hand's direction (from the
    seam's centre to the old hand's middle); the back of the hand faces out (`figure_back`); the old hand's own
    thinnest direction is kept as "palm" to check that against."""
    points, wfaces, group = weld(positions, faces)
    bone_w = np.zeros(len(points), dtype=np.int64)
    bone_w[group] = np.asarray(bone)
    arm = np.isin(bone_w, [forearm, *hand])
    select = arm[wfaces].any(axis=1)
    axis = unit(np.asarray(wrist, dtype=np.float64) - elbow)
    start = np.asarray(wrist, dtype=np.float64) - ARM_SHARE * (np.asarray(wrist) - elbow)
    on_hand = points[np.isin(bone_w, hand)]
    if not len(on_hand):
        raise ValueError(f"no vertex of the figure belongs to the {side} hand's bones")
    seam = limb_seam(points, wfaces, start, axis, 1.02 * float(((on_hand - start) @ axis).max()), select)
    beyond = arm & ((points - seam["cut"]) @ axis > 0)
    length = reach_of(points[beyond], seam["centre"])
    beyond &= np.linalg.norm(points - seam["centre"], axis=1) < REACH_SHARE * length
    fingers, palm = hand_frame(points[beyond], seam["centre"])
    back = figure_back(fingers, side, forward)
    return {"side": side, "cut": seam["cut"], "axis": axis, "centre": seam["centre"], "found": seam["found"],
            "area": seam["area"], "fingers": fingers, "back": back, "palm": palm, "length": length,
            "reach": REACH_SHARE * length, "arm": [int(forearm), *map(int, hand)],
            "turn_check_deg": float(np.degrees(np.arccos(min(1.0, abs(float(palm @ back))))))}


def hand_wrist(positions: np.ndarray, faces: np.ndarray, toward_camera: np.ndarray = FORWARD) -> dict:
    """A built hand's own wrist and frame. The drawing hangs the hand fingers down with the back of the hand to the
    camera (Pixal3D's +Z), so its long axis is turned to point down and its thinnest direction to the camera.
    "hollow" is the share of its sections below the wrist that are a shell round an empty inside: over
    HOLLOW_LIMIT, the build is a plate with a hole where its palm should be (the Waechter's first gauntlet)."""
    points, wfaces, _ = weld(positions, faces)
    axis = principal_axes(points)[0]
    axis = -axis if axis[1] > 0 else axis
    middle = points.mean(axis=0)
    along = (points - middle) @ axis
    seam = limb_seam(points, wfaces, middle + along.min() * axis, axis, float(np.ptp(along)))
    beyond = (points - seam["cut"]) @ axis > 0
    fingers, palm = hand_frame(points[beyond], seam["centre"])
    back = palm if palm @ toward_camera >= 0 else -palm
    return {"cut": seam["cut"], "axis": axis, "centre": seam["centre"], "found": seam["found"],
            "area": seam["area"], "fingers": fingers, "back": back, "hollow": round(seam["hollow"], 3),
            "length": reach_of(points[beyond], seam["centre"])}


def arm_faces(positions: np.ndarray, faces: np.ndarray, bone: np.ndarray, plan: dict) -> np.ndarray:
    """The faces of PLAN's arm near its seam: the only ones its cut may touch."""
    on_arm = np.isin(np.asarray(bone), plan["arm"])[np.asarray(faces)].any(axis=1)
    middle = np.asarray(positions)[np.asarray(faces)].mean(axis=1)
    return on_arm & (np.linalg.norm(middle - plan["centre"], axis=1) < plan["reach"])


def hand_onto(hand: dict, tf: np.ndarray, plan: dict, gap: float):
    """(the built HAND cut at the figure's seam moved GAP toward the fingers, put in place by TF; its rim).

    The cut is the figure's plane carried back into the hand's own space, so after the move the two rims lie in
    parallel planes and the stitch between them is an even band. Only the outer skin is kept (`outer_wall`)."""
    inverse = np.linalg.inv(tf)
    origin = moved(inverse, plan["cut"] + gap * plan["axis"])
    normal = unit(inverse[:3, :3] @ -plan["axis"])
    part, rim = cut_part(hand, origin, normal)
    part, rim = keep_faces(part, outer_wall(part["positions"], part["faces"], rim, normal), rim)
    turn = tf[:3, :3] / np.cbrt(np.linalg.det(tf[:3, :3]))
    out = {**part, "positions": moved(tf, part["positions"])}
    if part.get("normals") is not None:
        out["normals"] = part["normals"] @ turn.T
    return out, rim


def stitch(body: dict, body_rim: np.ndarray, hand: dict, hand_rim: np.ndarray, plan: dict,
           blend: float = BLEND_SHARE) -> tuple[dict, dict]:
    """The placed HAND eased into the BODY's cut wrist and stitched to it; the stitch belongs to the hand's part.

    Returns (hand part with the band added, report). Every band triangle has corners of its own, at the rim points
    it joins (an engine or Blender welds them on import), with the body's or the hand's skin weights as the corner
    lies, and all three painted from one point of the hand's rim: neighbouring rim points can lie in different
    islands of the hand's texture, and a colour blended between two islands striped the first Revolverheld's
    stitch with the fingers' skin."""
    axis = plan["axis"]
    body_ids, body_loop = rim_loop(body["positions"], body_rim, axis, plan["centre"])
    hand_ids, hand_loop = rim_loop(hand["positions"], hand_rim, axis, plan["centre"])
    if not len(body_ids) or not len(hand_ids):
        raise ValueError(f"the {plan['side']} wrist's cut left no rim to stitch")
    ratio = float(np.sqrt(loop_area(body_loop, axis) / max(loop_area(hand_loop, axis), 1e-18)))
    hand = {**hand, "positions": blend_wrist(hand["positions"], hand_loop.mean(axis=0), axis,
                                             hand_loop.mean(axis=0), body_loop.mean(axis=0), ratio,
                                             blend * plan["length"])}
    hand_loop = hand["positions"][hand_ids]
    alike = turn_of(body_loop, axis) == turn_of(hand_loop, axis)
    triangles = bridge(body_loop, hand_loop, axis)
    on_body = triangles[:, :, 0] == 0
    index = triangles[:, :, 1]
    from_body = body_ids[np.where(on_body, index, 0)]
    from_hand = hand_ids[np.where(on_body, 0, index)]
    anchor = from_hand[np.arange(len(triangles)), np.argmax(~on_body, axis=1)]   # each has a hand corner

    def corners(key):
        values = np.where(on_body[..., None], body[key][from_body], hand[key][from_hand])
        return values.reshape(-1, values.shape[-1])

    added = {"positions": corners("positions")}
    if hand.get("uvs") is not None:
        added["uvs"] = np.repeat(hand["uvs"][anchor], 3, axis=0)
    if hand.get("normals") is not None:
        added["normals"] = (corners("normals") if body.get("normals") is not None
                            else np.repeat(hand["normals"][anchor], 3, axis=0))
    if hand.get("dense") is not None:
        added["dense"] = corners("dense")
    faces = len(hand["positions"]) + np.arange(3 * len(triangles)).reshape(-1, 3)
    out = dict(hand)
    for key, values in added.items():
        out[key] = np.concatenate([hand[key], values])
    out["faces"] = np.concatenate([hand["faces"], faces])
    band = np.linalg.norm(body_loop.mean(axis=0) - hand_loop.mean(axis=0))
    report = {"wrist_ratio": round(ratio, 3), "rim_points": [len(body_ids), len(hand_ids)],
              "stitch_faces": int(len(faces)), "band_length": float(band), "rims_turn_alike": bool(alike)}
    return out, report


def merge_parts(parts: list[dict]) -> dict:
    """One part from several sharing a material (both hands)."""
    out = {"material": parts[0].get("material"), "faces": [], "positions": []}
    offset = 0
    for part in parts:
        out["faces"].append(np.asarray(part["faces"]) + offset)
        out["positions"].append(part["positions"])
        offset += len(part["positions"])
    for key in ("faces", "positions"):
        out[key] = np.concatenate(out[key])
    for key in ATTRIBUTES:
        if all(part.get(key) is not None for part in parts):
            out[key] = np.concatenate([part[key] for part in parts])
    return out


def side_faces(part: dict, side_of_face: np.ndarray, side: int) -> dict:
    """The faces of one hand of a merged part, as a part of their own (for its digit count)."""
    return keep_faces(part, np.asarray(side_of_face) == side)[0]


def arm_stub(part: dict, plan: dict, share: float = FOREARM_SHARE) -> dict:
    """The figure's arm just above PLAN's cut: SHARE of the hand's length of it, as far round the axis as twice
    the wrist's radius (a cuff, not the coat beside it).

    A grafted hand is counted with this stub on, framed as its drawing was (a sleeve on top): counted from its
    cut alone, the outline's wrist point sat so near the fingers that a little finger read as a second thumb."""
    p = np.asarray(part["positions"], dtype=np.float64)
    middle = p[np.asarray(part["faces"])].mean(axis=1)
    along = (middle - plan["cut"]) @ plan["axis"]
    off = middle - plan["centre"] - np.outer((middle - plan["centre"]) @ plan["axis"], plan["axis"])
    radius = 2.0 * np.sqrt(max(plan["area"], 0.0) / np.pi)
    keep = (along <= 0) & (along > -share * plan["length"]) & (np.linalg.norm(off, axis=1) < radius)
    return keep_faces({"positions": p, "faces": part["faces"]}, keep)[0]


def graft_model(body: dict, bone: np.ndarray, plans: dict[str, dict], hands: dict[str, dict],
                gap_share: float = GAP_SHARE) -> tuple[dict, dict, dict]:
    """BODY with each side's old hand cut off at PLANS' seams and HANDS put on and stitched there.

    BODY is a part (positions, faces, and any of normals, uvs, dense skin weights); BONE its vertices' main
    bones. HANDS[side] holds the built hand's "part" (mirrored already for a left hand built as a right) and its
    "tf" (`placement`). Every model of one figure that shares the figure's frame takes the same plans: the rigged
    mesh, and the finer models its paint and bumps are baked from. With dense weights, the new hand borrows
    the old hand's (`borrowed_weights`). Returns (body part, both hands as one part, report per side)."""
    sides = list(plans)
    bone = np.asarray(bone)
    work = {**body, "marks": np.stack([np.isin(bone, plans[s]["arm"]) for s in sides], 1).astype(float)}
    old = {}
    for side in sides:
        plan = plans[side]
        near = np.isin(bone, plan["arm"]) & (np.linalg.norm(body["positions"] - plan["centre"], axis=1)
                                            < plan["reach"])
        old[side] = (body["positions"][near], body["dense"][near]) if body.get("dense") is not None else None
    placed_parts, report, side_of_face = [], {}, []
    for index, side in enumerate(sides):
        plan = plans[side]
        faces = np.asarray(work["faces"])
        middle = work["positions"][faces].mean(axis=1)
        cuttable = (work["marks"][faces, index] > 0).any(axis=1) & (
            np.linalg.norm(middle - plan["centre"], axis=1) < plan["reach"])
        work, body_rim = cut_part(work, plan["cut"], plan["axis"], cuttable)
        pieces = rim_pieces(work["positions"], body_rim)
        areas = sorted((loop_area(work["positions"][ids], plan["axis"]) for ids in pieces), reverse=True)
        placed, hand_rim = hand_onto(hands[side]["part"], hands[side]["tf"], plan, gap_share * plan["length"])
        if old[side] is not None:
            placed["dense"] = borrowed_weights(placed["positions"], *old[side])
        placed, stitched = stitch(work, body_rim, placed, hand_rim, plan)
        placed_parts.append(placed)
        side_of_face.append(np.full(len(placed["faces"]), index))
        report[side] = {**stitched, "rim_pieces": len(pieces),
                        "rim_rest_share": round(float(sum(areas[1:]) / max(sum(areas), 1e-18)), 4)}
    body_out = {k: v for k, v in work.items() if k != "marks"}
    hands_out = merge_parts(placed_parts)
    hands_out["side_of_face"] = np.concatenate(side_of_face)
    return body_out, hands_out, report


# --- Skin weights ---------------------------------------------------------------------------------------------

def dense_weights(joints: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    """One row of COUNT bone weights per vertex, from glTF's four (joint, weight) pairs."""
    joints = np.asarray(joints, dtype=np.int64)
    out = np.zeros((len(joints), count))
    np.add.at(out, (np.repeat(np.arange(len(joints)), joints.shape[1]), joints.reshape(-1)),
              np.asarray(weights, dtype=np.float64).reshape(-1))
    return out


def top_weights(dense: np.ndarray, most: int = MAX_INFLUENCES) -> tuple[np.ndarray, np.ndarray]:
    """(joints, weights) with the MOST largest weights of each row, summing to one again."""
    dense = np.asarray(dense, dtype=np.float64)
    order = np.argsort(-dense, axis=1, kind="stable")[:, :most]
    weights = np.take_along_axis(dense, order, axis=1)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    weights[total[:, 0] <= 0, 0] = 1.0
    return np.where(weights > 0, order, 0), weights


def borrowed_weights(new_points: np.ndarray, old_points: np.ndarray, old_dense: np.ndarray) -> np.ndarray:
    """Each new point takes the weights of the nearest old one: the new hand moves as the old hand did (the hand
    bone, and the finger bones where the rig has them), and its wrist blends into the forearm as the old one did."""
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(np.asarray(old_points, dtype=np.float64)).query(np.asarray(new_points), workers=-1)
    return np.asarray(old_dense)[nearest]


# --- Rig ------------------------------------------------------------------------------------------------------

def arm_bones(parents: dict[str, str | None], heads: dict[str, np.ndarray]) -> tuple[dict[str, dict], np.ndarray]:
    """({side: {"forearm", "hand", "below"}}, the figure's frame) from a rig's skeleton, found by the tree's shape
    with the mapper the preset moves use (`scripts/kimodo_retarget.py`), never by name. SIDE is the figure's own:
    its right hand. The frame's columns are its left, up and forward."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from kimodo_retarget import map_skintokens_to_soma

    mapping, frame = map_skintokens_to_soma(parents, heads, up=np.array([0.0, 1.0, 0.0]))
    children: dict[str, list[str]] = {}
    for bone, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, []).append(bone)

    def first(role: str) -> str | None:
        named = [b for b, r in mapping.items() if r == role and mapping.get(parents.get(b)) != role]
        return named[0] if named else None

    arms = {}
    for side in SIDES:
        prefix = side.capitalize()
        hand, forearm = first(f"{prefix}Hand"), first(f"{prefix}ForeArm")
        if hand is None or forearm is None:
            continue
        below, stack = [], list(children.get(hand, []))
        while stack:
            bone = stack.pop()
            below.append(bone)
            stack.extend(children.get(bone, []))
        arms[side] = {"forearm": forearm, "hand": hand, "below": sorted(below)}
    return arms, frame


# --- Picture side ---------------------------------------------------------------------------------------------

HAND_PROMPT = (
    "The same {side} hand as in the reference image, a stylized 3D game character's hand with smooth simple "
    "skin like in the reference, not a photo. Now seen straight from the back of the hand, the fingers "
    "pointing down and slightly apart. A {side} hand seen from its back with the fingers down has its thumb on "
    "the {thumb} side of the picture: one short thumb sticking out to the {thumb} at the height of the knuckles. "
    "Then from {thumb} to {other} the index finger, the middle finger (the longest), the ring finger and the "
    "little finger (the shortest, on the far {other}). Four fingers and one thumb, nothing on the {other} side but "
    "the little finger. {look} Nothing else in the picture. Flat even lighting, plain light grey background.")
DEFAULT_LOOK = "The same glove or bare skin, the same sleeve and cuff at the top, in the same colours as in the reference."


def hand_prompt(side: str, look: str | None = None) -> str:
    """The picture editor's prompt that redraws a cropped hand flat: naming the thumb's side and the fingers in
    order is what made it draw five (a prompt without them drew 0 good hands of 4, this one 2 of 8)."""
    thumb, other = ("right", "left") if side == "right" else ("left", "right")
    return HAND_PROMPT.format(side=side, thumb=thumb, other=other, look=(look or DEFAULT_LOOK).strip())


def digit_count(mask: np.ndarray, top_band: float = 0.25, prominence: float = 0.06,
                thumb_hint: float = THUMB_HINT, thumb_cleft: float = THUMB_CLEFT,
                thumb_side: str | None = None) -> dict:
    """The digits of a hand seen flat with the fingers down, counted on its outline.

    The outline is walked round; its distance from the wrist (the middle of the outline's top band, where the
    sleeve enters) peaks at every fingertip. Peaks clearly higher than the dips beside them are digits; the top
    TOP_BAND of the height (the sleeve's corners) holds none. The dips between neighbouring digits are the clefts,
    their depth measured along the hand. The thumb is told by its cleft, not its length: lay a hand flat on a table
    and the gap between thumb and index finger runs deeper toward the wrist than the gaps between the fingers. So the
    thumb is the row's end whose cleft lies deeper than the fingers' clefts (by THUMB_HINT of the hand's length) and
    deeper than the other end's; a cleft at the other end that is just as deep (THUMB_CLEFT) is a second thumb, a
    shallower one a splayed little finger. The length rule this replaces took good drawings' little fingers for
    second thumbs. With THUMB_SIDE ('left' or 'right' in the picture, where the picture editor was told to put it) a
    thumb on the other side fails the count. A normal hand: five digits, one thumb.
    """
    import cv2
    from scipy.signal import find_peaks

    mask = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return {"digits": 0, "thumbs": 0, "ok": False, "tips": []}
    mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=len)[:, 0, :].astype(float)
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    band = ys < y0 + 0.08 * (y1 - y0)
    wrist = np.array([xs[band].mean(), y0])
    reach = np.linalg.norm(outline - wrist, axis=1)
    scale = reach.max()
    signal = np.where(outline[:, 1] > y0 + top_band * (y1 - y0), reach, 0.0)
    k = len(signal)
    peaks, _ = find_peaks(np.concatenate([signal] * 3), prominence=prominence * scale, distance=max(3, k // 80))
    peaks = sorted({p % k for p in peaks if k <= p < 2 * k})              # the outline is closed
    tips = sorted((float(outline[p, 0]), float(outline[p, 1]), float(reach[p] / scale)) for p in peaks)
    result = {"digits": len(tips), "thumbs": 0, "ok": False, "thumb_at": None, "clefts": [],
              "tips": [(round(x), round(y), round(q, 2)) for x, y, q in tips], "wrist": wrist.round().tolist()}
    if len(peaks) < 3:
        return result
    # each stretch of the outline between neighbouring tips dips to a cleft; the one stretch that runs through the
    # sleeve band (where the signal is 0) is the hand's outer edge, so the row of digits starts and ends beside it
    # a cleft's depth is measured along the hand (wrist to the fingertips' middle), not as a distance from the wrist:
    # the thumb's cleft sits at the side of the palm, where the sideways offset hides how far up it reaches
    axis = np.mean(outline[peaks], axis=0) - wrist
    axis /= np.linalg.norm(axis)
    along = (outline - wrist) @ axis
    along_signal = np.where(signal > 0, along / along[peaks].max(), 0.0)
    lows = []
    for a, b in zip(peaks, peaks[1:] + peaks[:1]):
        stretch = along_signal[a:b + 1] if b > a else np.concatenate([along_signal[a:], along_signal[:b + 1]])
        lows.append(float(max(stretch.min(), 0.0)))
    outer = [i for i, low in enumerate(lows) if low == 0.0]
    if len(outer) != 1:
        return result
    j, n = outer[0], len(peaks)
    row = [peaks[(j + 1 + i) % n] for i in range(n)]                      # the digits in order along the hand
    clefts = [lows[(j + 1 + i) % n] for i in range(n - 1)]                # clefts[i] lies between row[i] and row[i+1]
    # measured against the deepest cleft between the fingers: two fingers that touch leave a shallow one, which a
    # median of so few would follow (a gauntlet's build read a second thumb that way)
    middle = clefts[1:-1] or clefts
    usual = float(min(middle))
    deep = [(usual - clefts[0], row[0], row[-1]), (usual - clefts[-1], row[-1], row[0])]
    deep.sort(key=lambda d: d[0], reverse=True)
    (deepest, thumb, other), (second, _, _) = deep
    if deepest >= thumb_hint:
        result["thumbs"] = 2 if second >= thumb_cleft else 1
    result["clefts"] = [round(c, 2) for c in clefts]
    if result["thumbs"] == 1:
        result["thumb_at"] = "right" if outline[thumb, 0] > outline[other, 0] else "left"
    result["ok"] = (len(tips) == 5 and result["thumbs"] == 1
                    and (thumb_side is None or result["thumb_at"] == thumb_side))
    return result


def views_count(counts: list[tuple[int, int]]) -> dict:
    """A built hand's count from its outlines seen all round: a hand seen edge-on shows fewer digits (they stack),
    never more, so its count is the most frequent one among the views that show the most digits."""
    if not counts:
        return {"digits": 0, "thumbs": 0, "ok": False, "per_view": []}
    top = max(d for d, _ in counts)
    digits, thumbs = Counter(c for c in counts if c[0] >= top).most_common(1)[0][0]
    return {"digits": int(digits), "thumbs": int(thumbs), "ok": digits == 5 and thumbs == 1,
            "per_view": [int(d) for d, _ in counts]}


SMALL_SPAN = 3        # triangles up to this many pixels across are tested pixel by pixel, all at once


def silhouette(points2d: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """The filled outline of a mesh projected to 2D (pixel units, y down) as a SIZE x SIZE mask: every pixel whose
    centre lies in a triangle.

    Exact at the pixel centres, so a gap between two fingers two pixels wide stays open: a closing that filled the
    pinholes left by marking only the corners of a raw build's million tiny triangles also sealed those gaps, and
    a five-fingered build counted four. Small triangles are tested together; larger ones are filled one by one."""
    import cv2

    p = np.asarray(points2d, dtype=np.float64)
    tri = p[np.asarray(faces, dtype=np.int64)]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    small = ((hi - lo) <= SMALL_SPAN).all(axis=1)
    mask = np.zeros((size, size), np.uint8)
    t = tri[small]
    first = np.ceil(lo[small])
    area = ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
            - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    sign = np.where(area < 0, -1.0, 1.0)
    for dx in range(SMALL_SPAN + 1):
        for dy in range(SMALL_SPAN + 1):
            c = first + [dx, dy]
            inside = (c <= hi[small]).all(axis=1) & (area != 0)
            for k in range(3):
                a, b = t[:, k], t[:, (k + 1) % 3]
                edge = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                inside &= edge * sign >= 0
            x, y = c[inside, 0].astype(int), c[inside, 1].astype(int)
            ok = (x >= 0) & (y >= 0) & (x < size) & (y < size)
            mask[y[ok], x[ok]] = 1
    for corners in np.round(tri[~small] * 16).astype(np.int32):        # 4 bits of sub-pixel precision
        cv2.fillConvexPoly(mask, corners, 1, lineType=cv2.LINE_8, shift=4)
    return mask


def count_built(vertices: np.ndarray, faces: np.ndarray, axis: np.ndarray, views: int = 12,
                size: int = 800) -> dict:
    """The digits of a built hand: its outline seen from VIEWS directions round AXIS (wrist to fingertips), the
    wrist at the top of each picture, each counted by `digit_count`, combined by `views_count`.

    For a build, AXIS is the drawing's own down (DRAWN_DOWN), and for a grafted hand that direction carried onto
    the figure: round the build's longest direction instead (it leans with the cuff), two five-fingered builds of
    round 22 counted four. Round the drawing's down, all six builds checked by eye came out right."""
    v = np.asarray(vertices, dtype=np.float64)
    down = unit(axis)
    across = plane_basis(down)
    centre = (v.min(axis=0) + v.max(axis=0)) / 2
    radius = float(np.linalg.norm(v - centre, axis=1).max()) * 1.05
    counts = []
    for k in range(views):
        angle = 2 * np.pi * k / views
        side = np.cos(angle) * across[0] + np.sin(angle) * across[1]       # picture's x, square to the view
        rel = v - centre
        px = (rel @ side / radius + 1) * size / 2
        py = (rel @ down / radius + 1) * size / 2                          # the wrist at the top
        result = digit_count(silhouette(np.stack([px, py], 1), faces, size))
        counts.append((result["digits"], result["thumbs"]))
    return views_count(counts)


def hand_points(positions: np.ndarray, bone: np.ndarray, plan: dict) -> np.ndarray:
    """The points a hand's crop must hold: the old hand past the seam and the arm up to FOREARM_SHARE of the hand's
    length above it, so the drawing sees the cuff it has to match."""
    p = np.asarray(positions, dtype=np.float64)
    along = (p - plan["cut"]) @ plan["axis"]
    near = np.linalg.norm(p - plan["centre"], axis=1) < plan["reach"]
    return p[np.isin(np.asarray(bone), plan["arm"]) & near & (along > -FOREARM_SHARE * plan["length"])]


def crop_box(px: np.ndarray, py: np.ndarray, width: int, height: int, pad: float = HAND_CROP_PAD):
    """A square crop (x0, y0, x1, y1) round a hand's pixels, padded and kept in the picture."""
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    side = min(max(x1 - x0, y1 - y0) * pad, width, height)
    left = int(round(np.clip((x0 + x1) / 2 - side / 2, 0, width - side)))
    top = int(round(np.clip((y0 + y1) / 2 - side / 2, 0, height - side)))
    side = int(round(side))
    return left, top, left + side, top + side


# --- GLB in and out -------------------------------------------------------------------------------------------

def read_part(data: bytes) -> tuple[dict, bytes, dict]:
    """(document, binary, part) of a GLB holding one mesh with one primitive: positions, faces, and any of
    normals, uvs and (when skinned) the skin weights as dense rows; its material as (document, binary, index).

    Node transforms are refused: every model of a figure has to share one frame for a plan to fit them all."""
    from image_to_3dlab.glb_turn import parse_glb
    from image_to_3dlab.hero import _accessor

    doc, binary = parse_glb(data)
    nodes = [n for n in doc.get("nodes", []) if "mesh" in n]
    primitives = [p for mesh in doc.get("meshes", []) for p in mesh.get("primitives", [])]
    if len(nodes) != 1 or len(primitives) != 1:
        raise ValueError("expected one mesh with one part (a model with several parts, like a Hero head's "
                         "body and head, is not supported: put the hands on before the head)")
    if any(key in nodes[0] for key in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("the model's mesh is moved by its node; expected a model in the figure's own frame")
    primitive = primitives[0]
    attributes = primitive["attributes"]
    part = {"positions": _accessor(doc, binary, attributes["POSITION"]).astype(np.float64),
            "faces": _accessor(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)}
    for key, name in (("normals", "NORMAL"), ("uvs", "TEXCOORD_0")):
        if name in attributes:
            part[key] = _accessor(doc, binary, attributes[name]).astype(np.float64)
    if "JOINTS_0" in attributes and doc.get("skins"):
        weights = _accessor(doc, binary, attributes["WEIGHTS_0"]).astype(np.float64)
        component = doc["accessors"][attributes["WEIGHTS_0"]]["componentType"]
        if component != 5126:                     # normalised bytes or shorts
            weights /= {5121: 255.0, 5123: 65535.0}[component]
        part["dense"] = dense_weights(_accessor(doc, binary, attributes["JOINTS_0"]), weights,
                                      len(doc["skins"][0]["joints"]))
    if "material" in primitive:
        part["material"] = (doc, binary, primitive["material"])
    return doc, binary, part


def finish_part(part: dict, smooth: bool = False) -> dict:
    """A part ready to write: normals of unit length (recomputed smooth for SMOOTH), dense skin weights cut to
    four a vertex, the cut's bookkeeping dropped."""
    from image_to_3dlab.pixal3d_native import smooth_normals

    out = {k: v for k, v in part.items() if k not in ("marks", "side_of_face", "dense")}
    if smooth or part.get("normals") is None:
        out["normals"] = smooth_normals(part["positions"], part["faces"])
    else:
        n = np.asarray(part["normals"], dtype=np.float64)
        out["normals"] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if part.get("dense") is not None:
        out["joints"], out["weights"] = top_weights(part["dense"])
    return out


# --- GLB out --------------------------------------------------------------------------------------------------

class GlbWriter:
    """A GLB put together from new geometry and pieces of other GLBs (a skeleton, a material and its pictures),
    each copied with only what it uses."""

    def __init__(self, asset: dict | None = None):
        self.doc: dict = {"asset": asset or {"version": "2.0"}, "buffers": [{"byteLength": 0}], "bufferViews": [],
                          "accessors": []}
        self.blob = bytearray()
        self._copied: dict[tuple[int, str, int], int] = {}

    def view(self, data: bytes, target: int | None = None) -> int:
        self.blob += b"\0" * (-len(self.blob) % 4)
        entry = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target:
            entry["target"] = target
        self.blob += data
        self.doc["bufferViews"].append(entry)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, values: np.ndarray, component: int, kind: str, target: int | None = None,
                 bounds: bool = False, normalized: bool = False) -> int:
        dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[component]
        values = np.ascontiguousarray(values, dtype=dtype)
        entry = {"bufferView": self.view(values.tobytes(), target), "componentType": component,
                 "count": int(len(values) if kind != "SCALAR" else values.size), "type": kind}
        if normalized:
            entry["normalized"] = True
        if bounds:
            entry["min"] = values.min(axis=0).astype(float).tolist()
            entry["max"] = values.max(axis=0).astype(float).tolist()
        self.doc["accessors"].append(entry)
        return len(self.doc["accessors"]) - 1

    def copy_accessor(self, doc: dict, binary: bytes, index: int) -> int:
        key = (id(doc), "accessor", index)
        if key not in self._copied:
            accessor = dict(doc["accessors"][index])
            view = doc["bufferViews"][accessor["bufferView"]]
            start = view.get("byteOffset", 0)
            entry = {k: v for k, v in view.items() if k not in ("buffer", "byteOffset", "byteLength")}
            accessor["bufferView"] = self.view(binary[start:start + view["byteLength"]])
            self.doc["bufferViews"][-1].update(entry)
            self.doc["accessors"].append(accessor)
            self._copied[key] = len(self.doc["accessors"]) - 1
        return self._copied[key]

    def copy_material(self, doc: dict, binary: bytes, index: int) -> int:
        """Material INDEX of another GLB, with its textures, pictures and samplers."""
        from image_to_3dlab.glb_parts import (
            _renumber_textures,
            _texture_refs,
        )

        key = (id(doc), "material", index)
        if key in self._copied:
            return self._copied[key]
        material = copy.deepcopy(doc["materials"][index])
        found: set[int] = set()
        _texture_refs(material, found)
        texture_map = {}
        for old in sorted(t for t in found if t < len(doc.get("textures", []))):
            texture = copy.deepcopy(doc["textures"][old])
            for holder in [texture, *texture.get("extensions", {}).values()]:
                if "source" in holder:
                    holder["source"] = self._copy_image(doc, binary, holder["source"])
            if "sampler" in texture:
                texture["sampler"] = self._copy_sampler(doc, texture["sampler"])
            self.doc.setdefault("textures", []).append(texture)
            texture_map[old] = len(self.doc["textures"]) - 1
            self._uses(texture, doc)
        material = _renumber_textures(material, texture_map)
        self._uses(material, doc)
        self.doc.setdefault("materials", []).append(material)
        self._copied[key] = len(self.doc["materials"]) - 1
        return self._copied[key]

    def _uses(self, node, doc: dict) -> None:
        """Declare every extension NODE carries, as required where its source GLB required it."""
        if isinstance(node, dict):
            for name, value in node.items():
                if name == "extensions" and isinstance(value, dict):
                    for extension in value:
                        for key in ("extensionsUsed", "extensionsRequired"):
                            if key == "extensionsUsed" or extension in doc.get(key, []):
                                listed = self.doc.setdefault(key, [])
                                if extension not in listed:
                                    listed.append(extension)
                self._uses(value, doc)
        elif isinstance(node, list):
            for item in node:
                self._uses(item, doc)

    def _copy_image(self, doc: dict, binary: bytes, index: int) -> int:
        key = (id(doc), "image", index)
        if key not in self._copied:
            image = dict(doc["images"][index])
            if "bufferView" in image:
                view = doc["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                image["bufferView"] = self.view(binary[start:start + view["byteLength"]])
            self.doc.setdefault("images", []).append(image)
            self._copied[key] = len(self.doc["images"]) - 1
        return self._copied[key]

    def _copy_sampler(self, doc: dict, index: int) -> int:
        key = (id(doc), "sampler", index)
        if key not in self._copied:
            self.doc.setdefault("samplers", []).append(dict(doc["samplers"][index]))
            self._copied[key] = len(self.doc["samplers"]) - 1
        return self._copied[key]

    def primitive(self, part: dict) -> dict:
        """A primitive from a part: positions and faces, and any of normals, uvs, joints + weights, material."""
        positions = np.asarray(part["positions"], dtype=np.float32)
        attributes = {"POSITION": self.accessor(positions, 5126, "VEC3", 34962, bounds=True)}
        if part.get("normals") is not None:
            attributes["NORMAL"] = self.accessor(part["normals"], 5126, "VEC3", 34962)
        if part.get("uvs") is not None:
            attributes["TEXCOORD_0"] = self.accessor(part["uvs"], 5126, "VEC2", 34962)
        if part.get("joints") is not None:
            joints = np.asarray(part["joints"])
            attributes["JOINTS_0"] = self.accessor(joints, 5121 if joints.max(initial=0) < 256 else 5123, "VEC4",
                                                   34962)
            attributes["WEIGHTS_0"] = self.accessor(part["weights"], 5126, "VEC4", 34962)
        faces = np.asarray(part["faces"]).reshape(-1)
        out = {"attributes": attributes, "mode": 4,
               "indices": self.accessor(faces, 5125 if faces.max(initial=0) >= 65535 else 5123, "SCALAR", 34963)}
        if part.get("material") is not None:
            out["material"] = part["material"]
        return out

    def build(self) -> bytes:
        from image_to_3dlab.glb_turn import build_glb

        self.doc["buffers"][0]["byteLength"] = len(self.blob)
        return build_glb(self.doc, bytes(self.blob))


def regraft_glb(template: bytes, parts: list[dict]) -> bytes:
    """TEMPLATE (a GLB with one mesh, rigged or not) with that mesh's primitives replaced by PARTS.

    Nodes, scenes, the skin (its bind matrices) and any moves are copied; each part names its material as
    (document, binary, index) of the GLB it comes from. A part with joints and weights is skinned to the
    template's skeleton."""
    from image_to_3dlab.glb_turn import parse_glb

    doc, binary = parse_glb(template)
    meshes = [i for i, node in enumerate(doc.get("nodes", [])) if "mesh" in node]
    if len({doc["nodes"][i]["mesh"] for i in meshes}) != 1:
        raise ValueError("expected one mesh in the model")
    out = GlbWriter(copy.deepcopy(doc.get("asset", {"version": "2.0"})))
    for key in ("scene", "scenes", "nodes"):
        if key in doc:
            out.doc[key] = copy.deepcopy(doc[key])
    for skin in doc.get("skins", []):
        skin = copy.deepcopy(skin)
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = out.copy_accessor(doc, binary, skin["inverseBindMatrices"])
        out.doc.setdefault("skins", []).append(skin)
    for animation in doc.get("animations", []):
        animation = copy.deepcopy(animation)
        for sampler in animation.get("samplers", []):
            sampler["input"] = out.copy_accessor(doc, binary, sampler["input"])
            sampler["output"] = out.copy_accessor(doc, binary, sampler["output"])
        out.doc.setdefault("animations", []).append(animation)
    primitives = []
    for part in parts:
        part = dict(part)
        if part.get("material") is not None:
            part["material"] = out.copy_material(*part["material"])
        primitives.append(out.primitive(part))
    mesh = copy.deepcopy(doc["meshes"][doc["nodes"][meshes[0]]["mesh"]])
    mesh["primitives"] = primitives
    out.doc["meshes"] = [mesh]
    for node in out.doc["nodes"]:
        if "mesh" in node:
            node["mesh"] = 0
    return out.build()
