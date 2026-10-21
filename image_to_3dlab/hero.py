"""Hero route: a full-body model with a high-detail head grafted on, as pure functions.

Pixal3D spends its detail on its whole box, so a face that is a few dozen pixels of a
full-body picture comes out soft (goggle eyes, smeared paint). The same face cropped,
upscaled and built on its own fills the box and keeps real eyelids. This module holds the
maths that puts that head back on the body; `scripts/hero_route.py` runs the steps.

The head is found from a throwaway rig of the body: the topmost bone region, plus any bone
whose middle sits inside the head's height (the neck when it carries hair or a collar).
That needs no names and no hand-drawn box, and matched the hand-picked heads on both
characters it was tried on (warrior: one bone; elf: head + neck).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

LIGHT_GREY = 236          # the studio's plain background, for pictures without a cut-out
GREY_TOLERANCE = 25
CROP_PAD = 1.3            # crop side vs the head's own size, so hair and ears fit
CROP_DROP = 0.05          # shift the crop down a little: chin and neck matter more than air
HEAD_FLOOR_PERCENTILE = 5
OUTLINE_GROW_PX = 4       # the least margin round the real head's outline
OUTLINE_GROW_SHARE = 0.1  # else a tenth of the head's width
LINER_SHRINK = 0.04       # the body's own head, kept as a liner under the new hair, shrinks this much

_COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def picture_box(image: Image.Image) -> tuple[int, int, int, int]:
    """The figure's box (x0, x1, y0, y1). A cut-out (alpha) wins; else anything far from
    the studio's light grey, which misreads darker or graded backgrounds as figure."""
    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if alpha.min() < 128:
            ys, xs = np.nonzero(alpha > 127)
            return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    rgb = np.asarray(image.convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(rgb - LIGHT_GREY).max(-1) > GREY_TOLERANCE)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def frame(vertices: np.ndarray, box: tuple[int, int, int, int]):
    """(mesh centre, world units per pixel, box centre). Pixal3D fits every model to the
    same box, so a model's scale against its picture comes from the two heights."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    x0, x1, y0, y1 = box
    return (lo + hi) / 2, (hi[1] - lo[1]) / (y1 - y0), np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def project(points: np.ndarray, fr) -> tuple[np.ndarray, np.ndarray]:
    """Front-view picture pixels (x, y) of world points; picture y runs down."""
    centre, per_px, box_centre = fr
    return (box_centre[0] + (points[:, 0] - centre[0]) / per_px,
            box_centre[1] - (points[:, 1] - centre[1]) / per_px)


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype, width = _COMPONENTS[accessor["componentType"]], _WIDTH[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride", item)
    rows = np.lib.stride_tricks.as_strided(
        np.frombuffer(binary, dtype=np.uint8, offset=start),
        shape=(accessor["count"], stride), strides=(stride, 1))
    return rows[:, :item].copy().view(dtype).reshape(-1, width)


def skinned_vertices(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Positions and dominant joint of every vertex of every skinned primitive in a GLB."""
    length = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20:20 + length])
    binary = glb[20 + length + 8:]
    positions, joints = [], []
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "JOINTS_0" not in attributes:
                continue
            positions.append(_accessor(document, binary, attributes["POSITION"]))
            index = _accessor(document, binary, attributes["JOINTS_0"])
            weight = _accessor(document, binary, attributes["WEIGHTS_0"])
            joints.append(index[np.arange(len(index)), weight.argmax(axis=1)])
    if not positions:
        raise ValueError("the rig has no skinned mesh")
    return np.concatenate(positions), np.concatenate(joints).astype(int)


def boxed(points: np.ndarray) -> np.ndarray:
    """Points centred and scaled to unit height, so two exports of one model line up."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (points - (lo + hi) / 2) / (hi[1] - lo[1])


def regions(vertices: np.ndarray, rig_positions: np.ndarray, rig_joints: np.ndarray) -> np.ndarray:
    """The bone of every vertex: the nearest rigged vertex once both are boxed to one height
    (the rigger may re-export the mesh at another scale or orientation)."""
    _, nearest = cKDTree(boxed(rig_positions)).query(boxed(vertices), workers=-1)
    return rig_joints[nearest]


def head_bones(bone: np.ndarray, heights: np.ndarray) -> list[int]:
    """The topmost bone region, plus every bone whose median height sits inside it."""
    bones = np.unique(bone)
    middle = {int(b): float(np.median(heights[bone == b])) for b in bones}
    top = max(middle, key=middle.get)
    floor = np.percentile(heights[bone == top], HEAD_FLOOR_PERCENTILE)
    return sorted(b for b, m in middle.items() if m >= floor)


def crop_box(px: np.ndarray, py: np.ndarray, width: int, height: int,
             pad: float = CROP_PAD, drop: float = CROP_DROP) -> tuple[int, int, int, int]:
    """A square crop (x0, y0, x1, y1) round the head's pixels, padded and kept in the picture."""
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    side = min(max(x1 - x0, y1 - y0) * pad, width, height)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + side * drop
    left = int(round(np.clip(cx - side / 2, 0, width - side)))
    top = int(round(np.clip(cy - side / 2, 0, height - side)))
    side = int(round(side))
    return left, top, left + side, top + side


def crop_to_full(full_frame, head_frame, crop: tuple[int, int, int, int], head_width: int) -> np.ndarray:
    """Transform taking the head model (framed by its crop picture) into the full model's frame:
    a head point at crop pixel p sits at full pixel crop0 + p * (crop side / head picture side)."""
    c_f, s_f, b_f = full_frame
    c_h, s_h, b_h = head_frame
    r = (crop[2] - crop[0]) / head_width
    k = s_f * r / s_h
    shift = c_f - c_h * k
    shift[0] += (crop[0] + b_h[0] * r - b_f[0]) * s_f
    shift[1] -= (crop[1] + b_h[1] * r - b_f[1]) * s_f
    tf = np.diag([k, k, k, 1.0])
    tf[:3, 3] = shift
    return tf


def place_on(real: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Transform putting NEW on REAL: the same centre and the same height (no other scale guess)."""
    lo_r, hi_r = real.min(axis=0), real.max(axis=0)
    lo_n, hi_n = new.min(axis=0), new.max(axis=0)
    k = (hi_r[1] - lo_r[1]) / (hi_n[1] - lo_n[1])
    tf = np.diag([k, k, k, 1.0])
    tf[:3, 3] = (lo_r + hi_r) / 2 - k * (lo_n + hi_n) / 2
    return tf


def silhouette(faces: np.ndarray, px: np.ndarray, py: np.ndarray, shape: tuple[int, int],
               grow: int = 0) -> np.ndarray:
    """Front-view outline of FACES as a filled mask. Whole triangles, not corner points:
    corners alone leave gaps on a low-poly mesh, and anything over a gap was dropped as a hole."""
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for a, b, c in faces:
        draw.polygon([(px[a], py[a]), (px[b], py[b]), (px[c], py[c])], fill=255)
    mask = ndimage.binary_fill_holes(np.asarray(canvas) > 0)
    return ndimage.binary_dilation(mask, iterations=grow) if grow else mask


def on_mask(px: np.ndarray, py: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Which points land on MASK pixels."""
    x, y = np.round(px).astype(int), np.round(py).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    hit = np.zeros(len(x), bool)
    hit[ok] = mask[y[ok], x[ok]]
    return hit


def liner_split(vertices: np.ndarray, real: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices with the old head shrunk, which vertices go). The old head's back half stays
    as a liner under the new hair, shrunk about its centre by SHRINK.

    Thin hair layers thin out further when cut down and have nothing beneath, so gaps showed
    air. The old head is solid and painted from the same picture (its back from the back
    picture), so a gap now shows hair. Only the back half: the new face sits a little behind
    the old one, and the old face pushed through it. Models face +Z. SHRINK 0 drops it all.
    """
    if not shrink:
        return vertices, real.copy()
    moved = vertices.copy()
    inner = np.flatnonzero(real)
    centre = vertices[inner].mean(axis=0)
    moved[inner] = centre + (vertices[inner] - centre) * (1 - shrink)
    return moved, real & (vertices[:, 2] > centre[2])


def graft(full, head, full_box, head_box, crop, head_width, bone, shape, grow=None, liner=LINER_SHRINK):
    """Swap FULL's head region for HEAD (both trimesh meshes; HEAD is moved in place).

    1. The real head: FULL's head bones. 2. HEAD placed by the crop box, then on the real
    head at the same centre and height. 3. Fine line-up, position only. 4. FULL loses the
    front half of its head region (the back half stays, shrunk by LINER, as a liner under
    the new hair; LINER=0 drops it all); HEAD keeps what lands inside the real head's
    outline, grown a little. Returns (body, head part, report).
    """
    import trimesh

    picked = head_bones(bone, full.vertices[:, 1])
    real = np.isin(bone, picked)
    full_frame = frame(full.vertices, full_box)
    fpx, fpy = project(full.vertices, full_frame)
    outline = silhouette(full.faces[real[full.faces].all(axis=1)], fpx, fpy, shape)

    head.apply_transform(crop_to_full(full_frame, frame(head.vertices, head_box), crop, head_width))
    new = on_mask(*project(head.vertices, full_frame), outline)
    if not new.any():
        raise ValueError("the head model does not land on the body's head: check the crop")
    place = place_on(full.vertices[real], head.vertices[new])
    head.apply_transform(place)

    rng = np.random.default_rng(0)
    new = on_mask(*project(head.vertices, full_frame), outline)
    src, dst = head.vertices[new], full.vertices[real]
    src = src[rng.choice(len(src), min(30000, len(src)), replace=False)]
    dst = dst[rng.choice(len(dst), min(80000, len(dst)), replace=False)]
    fine, _, cost = trimesh.registration.icp(src, dst, max_iterations=60, scale=False)
    head.apply_transform(fine)

    if grow is None:
        # A margin in proportion to the head: the new head's hair often stands wider than the
        # body's own head, and a few fixed pixels trimmed it into holes at the sides.
        cols = np.flatnonzero(outline.any(axis=0))
        grow = max(OUTLINE_GROW_PX, int(round(OUTLINE_GROW_SHARE * (cols.max() - cols.min()))))
    keep = on_mask(*project(head.vertices, full_frame), ndimage.binary_dilation(outline, iterations=grow))
    body_mesh = full.copy()
    body_mesh.vertices, gone = liner_split(full.vertices, real, liner)
    body = body_mesh.submesh([np.flatnonzero(~gone[full.faces].any(axis=1))], append=True)
    part = head.submesh([np.flatnonzero(keep[head.faces].all(axis=1))], append=True)
    report = {"head_bones": picked, "outline_margin_px": int(grow), "liner_shrink": liner,
              "scale_to_real_head": round(float(place[0, 0]), 4),
              "line_up_cost": float(cost), "body_faces": int(len(body.faces)),
              "head_faces": int(len(part.faces))}
    return body, part, report


MATERIAL_TOLERANCE = 0.3   # a change this big is worth a human's glance
SPREAD_LIMITS = (0.8, 1.25)  # how far matching may stretch or squeeze a channel's contrast


def sample_texture(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    """Texture values (0..1, RGB) at glTF UVs (v runs up the image)."""
    a = np.asarray(image.convert("RGB")).astype(float) / 255
    h, w = a.shape[:2]
    u, v = np.clip(uv[:, 0], 0, 1), np.clip(uv[:, 1], 0, 1)   # atlases sit in 0..1; 1.0 must not wrap to 0
    x = np.round(u * (w - 1)).astype(int)
    y = np.round((1 - v) * (h - 1)).astype(int)
    return a[y, x]


def paired_points(head_vertices: np.ndarray, body_vertices: np.ndarray, reach: float):
    """(head index, body index) pairs of points that sit on the same spot, within REACH.

    The two heads are lined up, so the nearest body point is the same part of the character:
    hair against hair, skin against skin. Comparing whole regions instead mixed things up
    (the mage's body head region took in some of her bright hood, and her black hair was
    'corrected' to brown and her face to white).
    """
    distance, nearest = cKDTree(body_vertices).query(head_vertices, workers=-1)
    close = np.flatnonzero(distance < reach)
    return close, nearest[close]


def match_to_body(head_texture: Image.Image, head_seen: np.ndarray, body_seen: np.ndarray,
                  channels: tuple[int, ...] = (0, 1, 2)) -> tuple[Image.Image, dict]:
    """HEAD_TEXTURE with each channel's level and contrast moved to the body's.

    The head and body are built in separate runs and come out in different states: one
    woman's head crop was called all metal (1.0) while her body said cloth (0.0), and its
    paint sat darker and more saturated. A hero must look like one piece, so the head always
    follows the body, which is most of the character. HEAD_SEEN and BODY_SEEN are the two
    textures sampled at paired points on the same head (0..1 RGB rows, row i of each is the
    same spot); a channel with no spread in the head (all metal) simply lands on the body's
    level. Contrast changes are kept within SPREAD_LIMITS.
    """
    a = np.asarray(head_texture.convert("RGB")).astype(float) / 255
    report = {}
    for c in channels:
        mh, sh = float(head_seen[:, c].mean()), float(head_seen[:, c].std())
        mb = float(body_seen[:, c].mean())
        if sh > 1e-3:
            # A straight-line fit over the paired points (body = stretch * head + shift):
            # matching spreads instead let a few bright points blow light skin out to white.
            slope = float(np.polyfit(head_seen[:, c], body_seen[:, c], 1)[0])
            stretch = float(np.clip(slope, *SPREAD_LIMITS))
        else:
            stretch = 0.0
        a[..., c] = (a[..., c] - mh) * stretch + mb
        report["rgb"[c]] = {"head": round(mh, 3), "body": round(mb, 3), "stretch": round(stretch, 2)}
    out = Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    report["big_change"] = any(abs(v["head"] - v["body"]) > MATERIAL_TOLERANCE for v in report.values())
    return out, report


def load_picture(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGBA") if opened.mode in ("RGBA", "LA", "P") else opened.convert("RGB")
