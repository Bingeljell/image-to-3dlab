"""Hair strands a generator leaves in front of a face, counted per model: the cleaner of two tries has fewer.

Pixal3D often builds a strand of hair hanging a little in front of the forehead or the eyes where the front picture
shows bare skin. Pixel Match then paints the strand with the picture's skin, and from every side but the front it
reads as a crack across the face. A new seed of the same picture often has fewer such strands, so counting them
lets a run keep the cleanest of a few tries (`scripts/strand_score.py`).

A triangle counts as a strand when all of these hold:

* seen from the front camera it lies in front of the face's skin at its pixel, where the skin is the first layer
  that is not a thin sheet with a short air gap behind it (a strand: a sheet up to 15 mm thick, then 1-40 mm of
  air, then another surface; the skin itself is followed by the hollow head, much deeper);
* its pixel lies in the face of the front picture (the largest skin-coloured region of the head, closed and with
  its holes filled, so eyes and brows count) and a strand sheet was found there or close by;
* the generator painted it unlike the picture there and not as skin (Pixal3D paints its strands as hair and its
  nose, lips and cheeks as skin, so the face itself never counts);
* it belongs to a connected piece of at least `min_piece` such triangles that holds at least `min_seeds` clear
  ones and starts in the upper part of the face (strands hang from the hair; a mask's rim or a collar in front of
  the cheeks starts lower and is no strand).

Distances are for a 1.85 m figure and scale with the model's size. Tuned on two figures whose strands were seen
in renders from eight sides (ArcherSoul's Purpur-Samurai and Assassine): the count followed every visual verdict
(the Assassine's tries 573 / 362 / 0, the Samurai's 1,940 / 1,483 / 854).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import view_align

REF_HEIGHT = 1.85
THIN, GAP_MIN, GAP_MAX, IN_FRONT, DEDUPE = 0.015, 0.001, 0.040, 0.002, 0.0003
SS = 3          # supersampling of the picture's pixel grid
LAYERS = 10


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Skin-coloured pixels (a YCrCb rule wide enough for painted light and dark skin)."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    return (cr > 133) & (cr < 180) & (cb > 77) & (cb < 135) & (y > 60)


def face_area(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The face in a cut-out RGBA picture and its box (x0, y0, x1, y1), or None when its head shows no skin.

    The head is the top fifth of the figure; the face is its largest skin-coloured region, closed and with its
    holes (eyes, brows) filled.
    """
    alpha = image[..., 3] > 128
    rows = np.flatnonzero(alpha.any(axis=1))
    if not len(rows):
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    head_h = max(1, (bottom - top) // 5)
    head = np.zeros_like(alpha)
    head[top:top + head_h] = alpha[top:top + head_h]
    skin = skin_mask(image[..., :3]) & head
    label, count = ndimage.label(skin)
    if count == 0:
        return None
    sizes = ndimage.sum(skin, label, range(1, count + 1))
    face = label == (1 + int(np.argmax(sizes)))
    r = max(2, head_h // 40)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    face = ndimage.binary_closing(face, structure=xx ** 2 + yy ** 2 <= r * r, iterations=2)
    face = ndimage.binary_fill_holes(face)
    ys, xs = np.nonzero(face)
    return face, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def fragments(tri_xy: np.ndarray, tri_depth: np.ndarray, size: tuple[int, int], chunk: int = 4096):
    """Every (pixel, depth, triangle) where a pixel centre lies inside a triangle: all layers, no depth test.

    `tri_xy` is (n, 3, 2) in pixels, `tri_depth` (n, 3); pixels are numbered row by row in a grid of `size`
    (height, width).
    """
    height, width = size
    out_p, out_z, out_t = [np.zeros(0, np.int64)], [np.zeros(0)], [np.zeros(0, np.int64)]
    for start in range(0, len(tri_xy), chunk):
        xy = tri_xy[start:start + chunk]
        dz = tri_depth[start:start + chunk]
        lo = np.clip(np.floor(xy.min(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        hi = np.clip(np.ceil(xy.max(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        extent = hi - lo + 1
        counts = extent[:, 0] * extent[:, 1]
        counts[(xy.max(axis=1) < 0).any(axis=1) | (xy.min(axis=1)[:, 0] > width)
               | (xy.min(axis=1)[:, 1] > height)] = 0
        if counts.sum() == 0:
            continue
        tri = np.repeat(np.arange(len(xy)), counts)
        offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        px = lo[tri, 0] + offset % extent[tri, 0]
        py = lo[tri, 1] + offset // extent[tri, 0]
        cx, cy = px + 0.5, py + 0.5
        a, b, c = xy[tri, 0], xy[tri, 1], xy[tri, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        keep = np.abs(den) > 1e-12
        den = np.where(keep, den, 1.0)
        w0 = ((b[:, 1] - c[:, 1]) * (cx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (cy - c[:, 1])) / den
        w1 = ((c[:, 1] - a[:, 1]) * (cx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (cy - c[:, 1])) / den
        w2 = 1.0 - w0 - w1
        inside = keep & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        tri = tri[inside]
        out_p.append(py[inside] * width + px[inside])
        out_z.append(w0[inside] * dz[tri, 0] + w1[inside] * dz[tri, 1] + w2[inside] * dz[tri, 2])
        out_t.append(tri + start)
    return np.concatenate(out_p), np.concatenate(out_z), np.concatenate(out_t)


def skin_depth(pix: np.ndarray, z: np.ndarray, n_pix: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the skin's depth (inf where nothing) and whether a strand sheet lies in front of it.

    Along a pixel the layers are walked from the camera: a thin sheet (front and back closer than THIN) followed by
    a short air gap (GAP_MIN to GAP_MAX) is a strand, and the walk goes on behind it; the first layer that is not
    such a sheet is the skin. Lengths are for a 1.85 m figure, times `scale`.
    """
    order = np.lexsort((z, pix))
    pix, z = pix[order], z[order]
    keep = np.ones(len(pix), bool)
    keep[1:] = ~((pix[1:] == pix[:-1]) & (z[1:] - z[:-1] < DEDUPE * scale))    # one hit on a shared edge
    pix, z = pix[keep], z[keep]
    first = np.ones(len(pix), bool)
    first[1:] = pix[1:] != pix[:-1]
    start = np.flatnonzero(first)
    rank = np.arange(len(pix)) - np.repeat(start, np.diff(np.r_[start, len(pix)]))
    layers = np.full((n_pix, LAYERS), np.inf)
    sel = rank < LAYERS
    layers[pix[sel], rank[sel]] = z[sel]
    idx = np.zeros(n_pix, np.int64)
    active = np.isfinite(layers[:, 0])
    rows = np.arange(n_pix)
    for _ in range(LAYERS // 2 - 1):
        d0 = layers[rows, np.minimum(idx, LAYERS - 1)]
        d1 = layers[rows, np.minimum(idx + 1, LAYERS - 1)]
        d2 = layers[rows, np.minimum(idx + 2, LAYERS - 1)]
        sheet = (active & (idx + 2 < LAYERS) & np.isfinite(d2) & (d1 - d0 < THIN * scale)
                 & (d2 - d1 > GAP_MIN * scale) & (d2 - d1 < GAP_MAX * scale))
        if not sheet.any():
            break
        idx = np.where(sheet, idx + 2, idx)
        active = sheet
    return layers[rows, np.minimum(idx, LAYERS - 1)], idx > 0


def pieces(faces: np.ndarray) -> np.ndarray:
    """A piece number per triangle: triangles that share a vertex are in the same piece."""
    parent = np.arange(len(faces))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first: dict[int, int] = {}
    for i, corners in enumerate(faces):
        for v in corners:
            j = first.setdefault(int(v), i)
            if j != i:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(len(faces))], dtype=np.int64)


@dataclass
class StrandReport:
    """What `count_strands` found: the strand triangles' indices and the numbers behind them."""
    triangles: np.ndarray
    faces: int
    face_box: tuple[int, int, int, int] | None
    floating: int = 0
    evidence: int = 0
    pieces: int = 0
    align_iou: tuple[float, float] | None = None

    @property
    def count(self) -> int:
        return int(len(self.triangles))


def count_strands(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray, texture: np.ndarray,
                  views: list, mesh_scale: float, *, colour_gap: float = 30.0, min_piece: int = 40,
                  min_seeds: int = 5, top_share: float = 0.4) -> StrandReport:
    """The strand triangles of a Pixal3D model (as `photo_paint.read_glb` gives it) with its stored views."""
    front = next(v for v in views if "000" in v.name or v.name == "view")
    front, rep = view_align.align_view(front, positions, uvs, faces, texture, pp.PIXAL3D_FRAME, mesh_scale)
    image = front.image
    found = face_area(image)
    empty = np.zeros(0, np.int64)
    if found is None:
        return StrandReport(empty, len(faces), None, align_iou=(rep.iou_before, rep.iou_after))
    face, (fx0, fy0, fx1, fy1) = found
    vp = pp.to_view_space(positions, pp.PIXAL3D_FRAME, mesh_scale)
    s = float(max(np.ptp(positions, axis=0))) / REF_HEIGHT
    x, y, depth = pp.project(vp, front)

    m = 6
    tx, ty = x[faces], y[faces]
    near = ((tx.max(1) >= fx0 - m) & (tx.min(1) <= fx1 + m) & (ty.max(1) >= fy0 - m) & (ty.min(1) <= fy1 + m))
    cand = np.flatnonzero(near)
    width, height = (fx1 - fx0 + 2 * m) * SS, (fy1 - fy0 + 2 * m) * SS
    gx, gy = (x - (fx0 - m)) * SS, (y - (fy0 - m)) * SS
    tri_xy = np.stack([gx[faces[cand]], gy[faces[cand]]], axis=2)
    pix, z, _tri = fragments(tri_xy, depth[faces[cand]], (height, width))
    skin, strand_px = skin_depth(pix, z, width * height, s)
    skin, strand_px = skin.reshape(height, width), strand_px.reshape(height, width)

    face_ss = np.zeros((height, width), bool)
    sub = face[max(0, fy0 - m):fy1 + m, max(0, fx0 - m):fx1 + m]
    face_ss[:sub.shape[0] * SS, :sub.shape[1] * SS] = np.kron(sub, np.ones((SS, SS), bool))[:height, :width]
    strand_near = ndimage.binary_dilation(strand_px, iterations=3 * SS)

    centre = vp[faces[cand]].mean(axis=1)
    cx, cy, cz = pp.project(centre, front)
    px = np.clip(((cx - (fx0 - m)) * SS).astype(int), 0, width - 1)
    py = np.clip(((cy - (fy0 - m)) * SS).astype(int), 0, height - 1)
    uv = uvs[faces[cand]].mean(axis=1)
    th, tw = texture.shape[:2]
    own = texture[np.clip((uv[:, 1] * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1), :3]
    pic = image[np.clip(cy.astype(int), 0, image.shape[0] - 1), np.clip(cx.astype(int), 0, image.shape[1] - 1), :3]
    gap = np.linalg.norm(own.astype(float) - pic.astype(float), axis=1)
    in_front = cz < skin[py, px] - IN_FRONT * s
    drop = in_front & face_ss[py, px] & ~skin_mask(own[None])[0]
    seed = drop & strand_near[py, px] & (gap > colour_gap)
    report = StrandReport(empty, len(faces), (fx0, fy0, fx1, fy1), int(drop.sum()), int(seed.sum()),
                          align_iou=(rep.iou_before, rep.iou_after))
    sel = np.flatnonzero(drop)
    if not len(sel):
        return report
    label = pieces(faces[cand[sel]])
    ids, inv, counts = np.unique(label, return_inverse=True, return_counts=True)
    seeds = np.bincount(inv, weights=seed[sel].astype(float), minlength=len(ids))
    top = np.full(len(ids), np.inf)
    np.minimum.at(top, inv, cy[sel])
    go = (counts >= min_piece) & (seeds >= min_seeds) & (top <= fy0 + top_share * (fy1 - fy0))
    report.pieces = int(go.sum())
    report.triangles = cand[sel[go[inv]]]
    return report
