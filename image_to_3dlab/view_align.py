"""Nudge a picture onto the model before Pixel Match paints from it.

The 3D model never matches its picture exactly: the camera fit is a pixel or two off and
about 1% in size, and the model builds some features a little away from where they are
drawn (brows, a smile, a fringe). Painting straight from the picture then puts brows on
the forehead and ink lines beside the bumps they belong to. Two steps, both cautious:

- **Outline fit.** Scale and shift the picture (within a few pixels and ~3%) so its cut-out
  sits on the model's outline as seen by the same camera. Skipped unless the two outlines
  already mostly agree, so a picture with no real cut-out is left alone.
- **Feature matching.** Compare textured patches of the picture with the model's own paint
  seen from the same camera (that paint was made on this geometry, so it lines up with
  it), and move the picture's colours by the small offsets found. Only confident matches
  count: a patch must have detail, match well, and match clearly better than nearby
  positions (a repeating pattern does not). Offsets are spread smoothly, fade to zero away
  from any match, and are capped. Plain dense optical flow was tried and returned junk on
  flat cloth and skin (2026-10-12); do not swap it in.

Sizes are given for a 768-pixel picture and scale with the picture's width.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve

from image_to_3dlab import photo_paint as pp

REFERENCE_WIDTH = 768
MAX_SHIFT_PX = 6          # outline fit and feature moves, at REFERENCE_WIDTH
SCALES = tuple(round(1 + 0.005 * k, 3) for k in range(-6, 7))  # 0.97 .. 1.03
# The outline fit only runs when the outlines already overlap this well: below it the
# picture has no usable cut-out (or is a different shape) and moving it would be a guess.
MIN_OUTLINE_IOU = 0.8


@dataclass(frozen=True)
class OutlineFit:
    scale: float
    dx: int
    dy: int
    iou_before: float
    iou_after: float


@dataclass(frozen=True)
class AlignReport:
    scale: float
    dx: int
    dy: int
    iou_before: float
    iou_after: float
    matches: int


def render_own_paint(positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
                     texture: np.ndarray, view: pp.View, frame, mesh_scale: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The model's own paint as `view`'s camera sees it: (seen mask, H x W x 3 uint8)."""
    height, width = view.image.shape[:2]
    view_pos = pp.to_view_space(positions, frame, mesh_scale)
    vx, vy, vz = pp.project(view_pos, view)
    face_of, bary = pp.rasterize(np.stack([vx, vy], axis=1)[faces], vz[faces], (height, width))
    seen = face_of >= 0
    uv = (bary[seen][:, :, None] * uvs[faces[face_of[seen]]]).sum(axis=1)
    th, tw = texture.shape[:2]
    tx = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
    ty = np.clip((uv[:, 1] * th).astype(np.int64), 0, th - 1)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[seen] = texture[ty, tx, :3]
    return seen, image


def fit_outline(model: np.ndarray, picture: np.ndarray, max_shift: int = MAX_SHIFT_PX,
                scales=SCALES) -> OutlineFit:
    """Scale (about the model's centre) and shift taking the model's outline onto the
    picture's: picture(s * (p - c) + c + d) lines up with model(p). Exhaustive over
    `scales`; every whole-pixel shift within `max_shift` at once by correlation."""
    model, picture = model.astype(bool), picture.astype(bool)
    if not model.any() or not picture.any():
        return OutlineFit(1.0, 0, 0, 0.0, 0.0)
    height, width = model.shape
    ys, xs = np.nonzero(model)
    centre = np.array([ys.mean(), xs.mean()])
    pic_area = picture.sum()
    flipped = picture.astype(np.float64)

    def iou_grid(scale: float) -> np.ndarray:
        scaled = affine_transform(model.astype(np.float64), [1 / scale, 1 / scale],
                                  offset=centre - centre / scale, order=0) > 0.5
        overlap = fftconvolve(flipped, scaled[::-1, ::-1].astype(np.float64), mode="full")
        window = overlap[height - 1 - max_shift: height + max_shift,
                         width - 1 - max_shift: width + max_shift]
        window = np.rint(window)
        return window / np.maximum(scaled.sum() + pic_area - window, 1)

    base = iou_grid(1.0)
    before = float(base[max_shift, max_shift])
    best = OutlineFit(1.0, 0, 0, before, before)
    if before < MIN_OUTLINE_IOU:
        return best
    # Nearest to "no change" first, so ties keep the picture where it is.
    for scale in sorted(scales, key=lambda s: (abs(s - 1), s)):
        grid = base if scale == 1.0 else iou_grid(scale)
        dy, dx = np.unravel_index(np.argmax(grid), grid.shape)
        if grid[dy, dx] > best.iou_after + 1e-9:
            best = OutlineFit(float(scale), int(dx - max_shift), int(dy - max_shift), before,
                              float(grid[dy, dx]))
    return best


def _normalised_grey(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if not mask.any():
        return np.full(grey.shape, 128, dtype=np.uint8)
    mean, spread = grey[mask].mean(), grey[mask].std() + 1e-6
    grey = np.where(mask, (grey - mean) / spread * 40 + 128, 128)
    return np.clip(grey, 0, 255).astype(np.uint8)


def match_features(own: np.ndarray, picture: np.ndarray, mask: np.ndarray, max_px: float,
                   min_score: float = 0.85, min_lead: float = 0.15
                   ) -> tuple[np.ndarray, int]:
    """Displacement field (H x W x 2, x then y) with own(p) ~ picture(p + field), and the
    number of confident matches it came from. Zero where nothing confident is near."""
    height, width = mask.shape
    k = width / REFERENCE_WIDTH
    step, half, sigma = max(2, round(8 * k)), max(3, round(12 * k)), max(2.0, 24 * k)
    reach = int(np.ceil(max_px))
    a, b = _normalised_grey(own, mask), _normalised_grey(picture, mask)
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), 1.0),
                                                   cv2.CV_32F)), (0, 0), half / 2)
    field = np.zeros((height, width, 2), dtype=np.float32)
    if not mask.any():
        return field, 0
    floor = max(float(np.percentile(detail[mask], 70)), 1e-3)
    total = np.zeros((height, width, 2), dtype=np.float32)
    hits = np.zeros((height, width), dtype=np.float32)
    matches = 0
    for y in range(half + reach, height - half - reach, step):
        for x in range(half + reach, width - half - reach, step):
            if not mask[y, x] or detail[y, x] < floor:
                continue
            patch = a[y - half:y + half + 1, x - half:x + half + 1]
            search = b[y - half - reach:y + half + reach + 1, x - half - reach:x + half + reach + 1]
            scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (px, py) = cv2.minMaxLoc(scores)
            # Clearly better than most nearby positions, or it is a repeating pattern.
            runner_up = np.sort(scores.ravel())[-int(scores.size * 0.15)]
            if not np.isfinite(peak) or peak < min_score or peak - runner_up < min_lead:
                continue
            dx, dy = px - reach, py - reach
            if abs(dx) >= reach or abs(dy) >= reach:  # at the edge of the search: unsure
                continue
            total[y, x] = (dx, dy)
            hits[y, x] = 1
            matches += 1
    if not matches:
        return field, 0
    total = cv2.GaussianBlur(total, (0, 0), sigma)
    hits = cv2.GaussianBlur(hits, (0, 0), sigma)
    confidence = np.clip(hits / (hits.max() * 0.05 + 1e-9), 0, 1)
    field = total / np.maximum(hits, 1e-9)[..., None] * confidence[..., None]
    length = np.linalg.norm(field, axis=2, keepdims=True)
    field *= np.minimum(1.0, max_px / np.maximum(length, 1e-6))
    field[~mask] = 0
    return field.astype(np.float32), matches


def align_view(view: pp.View, positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
               texture: np.ndarray, frame, mesh_scale: float) -> tuple[pp.View, AlignReport]:
    """`view` with its picture nudged onto the model (same camera, new pixels)."""
    image = view.image
    height, width = image.shape[:2]
    max_px = MAX_SHIFT_PX * width / REFERENCE_WIDTH
    seen, own = render_own_paint(positions, uvs, faces, texture, view, frame, mesh_scale)
    fit = fit_outline(seen, image[..., 3] > 127, max_shift=max(1, round(max_px)))
    (cy, cx), s = np.argwhere(seen).mean(axis=0) if seen.any() else (0.0, 0.0), fit.scale
    inverse = np.array([[s, 0, cx * (1 - s) + fit.dx], [0, s, cy * (1 - s) + fit.dy]])
    moved = cv2.warpAffine(image, inverse, (width, height),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    both = seen & (moved[..., 3] > 127)
    field, matches = match_features(own, moved[..., :3], both, max_px)
    if matches:
        gx, gy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        shifted = cv2.remap(moved, gx + field[..., 0], gy + field[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        shifted[..., 3] = moved[..., 3]  # colours move onto features; the cut-out stays fitted
        moved = shifted
    report = AlignReport(fit.scale, fit.dx, fit.dy, fit.iou_before, fit.iou_after, matches)
    return pp.View(moved, view.cam_to_world, view.fov_x, view.name, view.paint), report
