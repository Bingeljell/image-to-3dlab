#!/usr/bin/env python3
"""Repaint a generated head's eyes into a dedicated high-resolution patch.

**The problem this exists for.** Image-to-3D backends do not model eyes. They paint
them, into the same atlas that carries the whole body, at whatever resolution the
atlas happens to allocate -- and eyes are small, so they get almost nothing. On the
moss fox (TRELLIS, 2026-09-19) a clay render of the head shows a smooth muzzle with a
faint mound where each eye belongs, and the 1024x1024 atlas spends about 1,500 texels
on each eye: roughly a 25x25 patch once the lids are subtracted. At that size there is
no iris edge, no pupil and no highlight, just a smear a couple of dozen pixels wide,
and the two eyes usually do not even match, because the generator hallucinated each
side independently. Every backend in this repository has the same failure mode; the
fox is only the asset it was first measured on.

**Why a patch instead of repainting the atlas.** Growing the shared atlas to fix a
1% region costs 4x or 16x the texture for no gain anywhere else, and TRELLIS atlases
are shattered into hundreds of tiny islands, so an eye is not one contiguous area you
can paint by hand. Instead each eye gets its own small material and its own planar UV
projection: the eye faces are re-unwrapped along the gaze axis into half of one patch
image, which makes the mapping between a texel and a point on the face analytic. At a
1024x512 patch each eye gets ~300 texels across instead of ~25, and the shared atlas
is left untouched.

**How the surrounding fur survives.** The patch is not painted from scratch. The eye
faces are rasterised into patch space first, interpolating their *original* atlas UVs,
so the patch starts as a resampled copy of what was already there. Only the eye itself
is painted over the top, and it fades into that copy, so the boundary between the patch
material and the atlas material shows the same colour on both sides.

**Placement is measured, not guessed.** ``locate_eye`` finds the dark almond in the
resampled patch by image moments -- centre, tilt and extent -- so the new eye lands
exactly where the generator put the old one, which is what keeps it consistent with
the eyelid shading painted around it. That is also what makes this work on a creature
it has never seen: the only per-asset input is a rough point inside each eye.

Run it against a live Blender (the same execute_code socket every other ``blender_*``
helper in this repo uses). The two centres are approximate -- anywhere inside the eye
is close enough, the moment fit does the rest -- and the left of the head comes first::

    python scripts/paint_eyes.py --object geometry_0 \\
        --eye -0.0777 -0.408 0.6282 --eye 0.0114 -0.3569 0.6229

Everything about the eye itself is a knob, because creatures differ::

    # a pale blue eye with a slit pupil, on a character called "snag"
    python scripts/paint_eyes.py --object snag --material SnagEyes \\
        --eye ... --eye ... --iris "#6f9ec4" --pupil-scale 0.22

``--revert`` undoes the whole edit, ``--export`` writes a GLB without the working UV
layer, and both need only ``--object``.

**What it assumes about the asset**: one triangulated mesh whose first material feeds
base colour from an image texture. A metallic/roughness texture is used if the
material has one and synthesised from the material's constant values if it does not.
Faces are found by projecting along the gaze, so it suits a head with normal eye
placement; a creature with eyes on stalks or on top of its skull wants ``--forward``.
"""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass, replace

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Where the untouched atlas UVs are kept so the painter can be re-run while tuning.
BACKUP_UV_LAYER = "UVMap_pre_eyes"


# --------------------------------------------------------------------------- style


@dataclass(frozen=True)
class EyeStyle:
    """Everything about how one eye is drawn, in eye-local units.

    Lengths are fractions of the measured opening unless noted. Colours are sRGB
    0..1, matching Blender's byte-image pixel buffer (which is *not* linearised).
    """

    # Below 2 the superellipse pinches its corners, which is what makes an eye an
    # almond with a canthus at each end rather than a rounded rectangle.
    opening_power: float = 1.65
    opening_scale: float = 1.08  # grows the measured almond; moments read small
    min_aspect: float = 1.25  # an eye is wider than it is tall, however it was painted
    feather: float = 0.10  # softness of the opening edge, in opening units
    iris_scale: float = 1.10  # iris radius / opening half-height
    pupil_scale: float = 0.36  # pupil radius / iris radius
    iris_rise: float = 0.06  # iris centre lifted, in opening half-heights
    limbal_width: float = 0.13  # dark rim, fraction of iris radius
    liner_width: float = 0.17  # dark lid line across the opening edge, in opening units
    liner_strength: float = 0.75  # a lid, not an inked cartoon outline
    lid_shadow: float = 0.34  # how much the upper lid darkens the eye
    highlight_scale: float = 0.26  # highlight radius / iris radius
    highlight_at: tuple[float, float] = (-0.36, 0.38)  # in iris radii
    highlight_strength: float = 0.80
    pupil_rgb: tuple[float, float, float] = (0.050, 0.040, 0.038)
    iris_inner_rgb: tuple[float, float, float] = (0.86, 0.61, 0.19)
    iris_outer_rgb: tuple[float, float, float] = (0.48, 0.27, 0.06)
    limbal_rgb: tuple[float, float, float] = (0.09, 0.060, 0.040)
    # Warm, not grey: a cool sclera on a green face reads as a plastic googly eye,
    # and the reference art barely shows any of it anyway.
    sclera_rgb: tuple[float, float, float] = (0.68, 0.61, 0.47)
    liner_rgb: tuple[float, float, float] = (0.065, 0.062, 0.045)
    streak_count: int = 16
    streak_strength: float = 0.06
    eye_roughness: float = 0.16  # wet, so it catches a specular highlight
    liner_roughness: float = 0.55


# ------------------------------------------------------------------- pure painting


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite fade from 0 at ``edge0`` to 1 at ``edge1``; works in either direction."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance of an ...x3 array of 0..1 colours."""
    return rgb.astype(np.float32) @ _LUMA


def locate_eye(rgb: np.ndarray, threshold: float = 0.16) -> dict:
    """Measure the dark almond in a resampled eye patch.

    Returns centre ``(cx, cy)`` in pixel coordinates, the tilt ``angle`` in radians
    (positive = counter-clockwise in image axes, y down), and the half-axes ``a``
    (along the tilt) and ``b`` (across it) in pixels.

    Image moments rather than a bounding box: the dark region has ragged edges and a
    tail into the eyelid crease, and moments weight that tail by how dark it is
    instead of letting one stray texel set the size.
    """
    lum = luminance(rgb)
    weight = np.clip((threshold - lum) / threshold, 0.0, 1.0).astype(np.float64)
    # Keep the search near the middle: the patch corners often catch a nostril or the
    # dark inside of an ear, and either would drag the centroid off the eye.
    h, w = weight.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    radius = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0)
    weight *= smoothstep(min(h, w) * 0.48, min(h, w) * 0.30, radius)

    total = weight.sum()
    if total <= 0:
        raise ValueError("no dark region found in the eye patch")
    cx = float((weight * xx).sum() / total)
    cy = float((weight * yy).sum() / total)
    dx = xx - cx
    dy = yy - cy
    mxx = float((weight * dx * dx).sum() / total)
    myy = float((weight * dy * dy).sum() / total)
    mxy = float((weight * dx * dy).sum() / total)
    angle = 0.5 * float(np.arctan2(2.0 * mxy, mxx - myy))
    common = np.sqrt(max((mxx - myy) ** 2 + 4.0 * mxy**2, 0.0))
    major = np.sqrt(max(0.5 * (mxx + myy + common), 1e-12))
    minor = np.sqrt(max(0.5 * (mxx + myy - common), 1e-12))
    # A uniform ellipse has second moment a/2 along each axis; the dark paint is not
    # uniform, so 2.0 is the empirical factor that put the drawn opening on top of it.
    return {"cx": cx, "cy": cy, "angle": angle, "a": 2.0 * major, "b": 2.0 * minor}


def paint_eye(
    x: np.ndarray,
    y: np.ndarray,
    base_rgb: np.ndarray,
    base_roughness: np.ndarray,
    style: EyeStyle = EyeStyle(),  # noqa: B008 - frozen, so one shared default is safe
    aspect: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one eye over ``base_rgb``.

    ``x`` and ``y`` are eye-local coordinates in units of the opening's half-*height*,
    +y up, so the opening spans -1..1 vertically and -``aspect``..``aspect`` across.
    Keeping both axes in the same unit is what makes the iris a circle on a wide
    almond opening instead of an ellipse stretched to the eye corners.

    Returns the new colour and roughness, both the shape of the inputs.
    """
    p = style.opening_power
    t = np.abs(x / aspect) ** p + np.abs(y) ** p
    inside = smoothstep(1.0 + style.feather, 1.0 - style.feather, t)
    # A ring centred on the opening edge: rises through the last of the eye and
    # falls away into the fur, so it reads as an eyelid rather than a drawn circle.
    liner = (
        style.liner_strength
        * smoothstep(1.0 - style.liner_width * 0.5, 1.0, t)
        * smoothstep(1.0 + style.liner_width, 1.0 + style.liner_width * 0.35, t)
    )

    iris_r = style.iris_scale
    r = np.hypot(x, y - style.iris_rise) / iris_r

    # Sclera first, then the iris disc, then the pupil, each laid over the last.
    rgb = np.broadcast_to(np.asarray(style.sclera_rgb, np.float32), base_rgb.shape).copy()

    theta = np.arctan2(y - style.iris_rise, x)
    # Fibres belong to the outer iris; running them into the pupil makes a sunburst.
    streaks = 1.0 + style.streak_strength * np.sin(style.streak_count * theta) * smoothstep(
        style.pupil_scale, 1.0, r
    )
    grade = np.clip(r, 0.0, 1.0)[..., None]
    iris = (
        np.asarray(style.iris_inner_rgb, np.float32) * (1.0 - grade)
        + np.asarray(style.iris_outer_rgb, np.float32) * grade
    ) * streaks[..., None]
    # Radial streaks must not blow past white or the iris gets a neon rim.
    iris = np.clip(iris, 0.0, 1.0)
    limbal_start = 1.0 - style.limbal_width
    to_limbal = smoothstep(limbal_start, 1.0, r)[..., None]
    iris = iris * (1.0 - to_limbal) + np.asarray(style.limbal_rgb, np.float32) * to_limbal

    iris_cover = smoothstep(1.02, 0.98, r)[..., None]
    rgb = rgb * (1.0 - iris_cover) + iris * iris_cover

    pupil_cover = smoothstep(style.pupil_scale * 1.06, style.pupil_scale * 0.94, r)[..., None]
    rgb = rgb * (1.0 - pupil_cover) + np.asarray(style.pupil_rgb, np.float32) * pupil_cover

    # The upper lid casts a shadow over the top of the eye; without it a bright
    # sclera at the top edge reads as a bulging ping-pong ball.
    shade = 1.0 - style.lid_shadow * smoothstep(0.1, 1.0, y)
    rgb = rgb * shade[..., None]

    hx, hy = style.highlight_at
    hr = np.hypot(x - hx * iris_r, y - style.iris_rise - hy * iris_r) / (
        style.highlight_scale * iris_r
    )
    glint = style.highlight_strength * smoothstep(1.0, 0.25, hr)[..., None]
    rgb = rgb * (1.0 - glint) + glint

    # The lid line goes on last. Drawn underneath, the eye's own feathered edge eats
    # most of it and the eye ends up with no outline at all.
    liner_rgb = np.asarray(style.liner_rgb, np.float32)
    out = base_rgb * (1.0 - inside[..., None]) + rgb * inside[..., None]
    out = out * (1.0 - liner[..., None]) + liner_rgb * liner[..., None]

    rough = base_roughness * (1.0 - inside) + style.eye_roughness * inside
    rough = rough * (1.0 - liner) + style.liner_roughness * liner
    return np.clip(out, 0.0, 1.0), np.clip(rough, 0.0, 1.0)


# ------------------------------------------------------------- pure rasterisation


def rasterise(
    tri_uv: np.ndarray, tri_attr: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise triangles given in 0..1 patch space, interpolating attributes.

    ``tri_uv`` is (T, 3, 2) with u to the right and v up (texel row 0 is v=0, which is
    how Blender's pixel buffer is ordered). ``tri_attr`` is (T, 3, K). Returns the
    (height, width, K) attribute image and a boolean coverage mask.
    """
    attr = np.zeros((height, width, tri_attr.shape[2]), np.float32)
    mask = np.zeros((height, width), bool)
    px = tri_uv[:, :, 0] * width - 0.5
    py = tri_uv[:, :, 1] * height - 0.5
    for i in range(tri_uv.shape[0]):
        ax, bx, cx = px[i]
        ay, by, cy = py[i]
        x0 = max(int(np.floor(min(ax, bx, cx))), 0)
        x1 = min(int(np.ceil(max(ax, bx, cx))), width - 1)
        y0 = max(int(np.floor(min(ay, by, cy))), 0)
        y1 = min(int(np.ceil(max(ay, by, cy))), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        gy, gx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        l2 = 1.0 - l0 - l1
        # A half-texel of slop so neighbouring triangles do not leave a gap between
        # them; overlap is harmless because both write the same interpolated value.
        eps = -0.02
        hit = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not hit.any():
            continue
        values = (
            l0[..., None] * tri_attr[i, 0]
            + l1[..., None] * tri_attr[i, 1]
            + l2[..., None] * tri_attr[i, 2]
        )
        sub_attr = attr[y0 : y1 + 1, x0 : x1 + 1]
        sub_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        sub_attr[hit] = values[hit]
        sub_mask[hit] = True
    return attr, mask


def dilate(image: np.ndarray, mask: np.ndarray, rounds: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Grow ``image`` outward into unmasked texels, one ring per round.

    Bilinear filtering reaches outside a UV island, so an un-padded island fringes
    black along every seam. Each round copies the average of the covered neighbours.
    """
    out = image.copy()
    cover = mask.copy()
    for _ in range(rounds):
        filled = cover.astype(np.float32)
        acc = np.zeros_like(out)
        cnt = np.zeros_like(filled)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            acc += np.roll(np.roll(out * filled[..., None], dy, axis=0), dx, axis=1)
            cnt += np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
        grow = (~cover) & (cnt > 0)
        if not grow.any():
            break
        out[grow] = acc[grow] / cnt[grow][..., None]
        cover |= grow
    return out, cover


def parse_colour(text: str) -> tuple[float, float, float]:
    """A colour from ``#rrggbb``, ``rrggbb`` or ``r,g,b`` floats, as sRGB 0..1.

    Creatures differ, and the iris is the one thing a caller always wants to change,
    so it has to be reachable from the command line without editing the style class.
    """
    value = text.strip().lstrip("#")
    if "," in value:
        parts = [float(v) for v in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected three components in {text!r}")
        if max(parts) > 1.0:  # 0..255 is the other way people write this
            parts = [v / 255.0 for v in parts]
        return tuple(min(max(v, 0.0), 1.0) for v in parts)
    if len(value) != 6:
        raise ValueError(f"expected #rrggbb or r,g,b, got {text!r}")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def shade(rgb: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    """Scale a colour's brightness, for deriving an iris rim from its main tone."""
    return tuple(min(max(c * factor, 0.0), 1.0) for c in rgb)


def constant_orm(shape: tuple[int, int], roughness: float, metallic: float) -> np.ndarray:
    """A flat glTF metallic/roughness image: G is roughness, B is metallic.

    Not every backend packs a metallic/roughness map -- some materials carry plain
    numbers instead. Synthesising the map the painter expects is what lets the same
    code path serve both, rather than the tool refusing an asset for having a simpler
    material than the one it was written against.
    """
    orm = np.zeros((*shape, 3), np.float32)
    orm[..., 1] = roughness
    orm[..., 2] = metallic
    return orm


def sample_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample an (H, W, K) image at (..., 2) UV coordinates, v=0 at row 0, wrapping."""
    h, w = image.shape[:2]
    fx = uv[..., 0] * w - 0.5
    fy = uv[..., 1] * h - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x1, y1 = x0 + 1, y0 + 1
    x0 %= w
    x1 %= w
    y0 %= h
    y1 %= h
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bot = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bot * ty


def patch_coordinates(
    positions: np.ndarray, centre: np.ndarray, right: np.ndarray, up: np.ndarray, extent: float
) -> np.ndarray:
    """Project world positions into 0..1 patch space for one eye."""
    delta = positions - centre
    s = (delta @ right) / (2.0 * extent) + 0.5
    t = (delta @ up) / (2.0 * extent) + 0.5
    return np.stack([s, t], axis=-1)


def eye_basis(gaze: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Right and up vectors for an eye looking along ``gaze``."""
    gaze = np.asarray(gaze, np.float64)
    gaze = gaze / np.linalg.norm(gaze)
    right = np.cross(gaze, np.asarray(world_up, np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, gaze)
    return right, up / np.linalg.norm(up)


def local_eye_coordinates(
    shape: tuple[int, int], fit: dict, style: EyeStyle = EyeStyle()  # noqa: B008 - frozen
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-texel eye coordinates, plus the opening's width-to-height ratio.

    ``shape`` is (height, width) of one eye's half of the patch and ``fit`` the output
    of :func:`locate_eye`. Both axes are divided by the *minor* half-axis, so a unit
    step means the same distance on the face whichever way it points. Patch rows run
    bottom-up (row 0 is v=0), so the returned +y points up on the face.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - fit["cx"]
    dy = yy - fit["cy"]
    ca, sa = np.cos(-fit["angle"]), np.sin(-fit["angle"])
    aspect = max(fit["a"] / fit["b"], style.min_aspect)
    scale = fit["b"] * style.opening_scale
    rx = (ca * dx - sa * dy) / scale
    ry = (sa * dx + ca * dy) / scale
    return rx, ry, aspect


# ------------------------------------------------------------------- Blender side


def blender_payload(
    object_name: str,
    eyes: list[tuple[float, float, float]],
    extent: float,
    patch: int,
    splay: float,
    style: EyeStyle,
    repo: str,
    material_name: str = "EyePatch",
    forward: tuple[float, float, float] | None = None,
) -> str:
    """The snippet handed to the live Blender; it imports this module for the maths."""
    # Blender keeps the module between calls, so reload it or every style tweak
    # would silently run the first version that was ever imported.
    return (
        "import sys, importlib\n"
        f"sys.path.insert(0, {repo!r})\n"
        "import scripts.paint_eyes as _pe\n"
        "importlib.reload(_pe)\n"
        "from scripts.paint_eyes import apply_in_blender, EyeStyle\n"
        f"result = apply_in_blender({object_name!r}, {eyes!r}, {extent!r}, {patch!r},"
        f" {splay!r}, EyeStyle(**{style.__dict__!r}), {material_name!r}, {forward!r})\n"
        "import json; print('RES ' + json.dumps(result))\n"
    )


def head_forward(centres: np.ndarray) -> np.ndarray:
    """Which way the head faces, from the line between the two eyes.

    The interocular axis fixes the facing direction to a sign: the head looks along one
    of the two horizontal perpendiculars. Blender's glTF importer lands a generated
    character facing -Y, so that is the branch taken -- pass ``forward`` explicitly for
    an asset that was rotated, or for a creature whose eyes are not on the front.
    """
    right = centres[1] - centres[0]
    right = right / np.linalg.norm(right)
    forward = np.array([right[1], -right[0], 0.0])
    forward /= np.linalg.norm(forward)
    return -forward if forward[1] > 0 else forward


def apply_in_blender(
    object_name: str,
    eyes: list[tuple[float, float, float]],
    extent: float = 0.045,
    patch: int = 640,
    splay: float = 0.35,
    style: EyeStyle | None = None,
    material_name: str = "EyePatch",
    forward: tuple[float, float, float] | None = None,
) -> dict:
    """Build the eye patch inside a running Blender and bind it to the eye faces."""
    import bpy

    style = style or EyeStyle()
    obj = bpy.data.objects[object_name]
    mesh = obj.data
    if any(len(p.loop_indices) != 3 for p in mesh.polygons[:64]):
        raise ValueError("paint_eyes expects a triangulated mesh")

    loops = len(mesh.loops)
    verts = len(mesh.vertices)
    # Painting rewrites the eye faces' UVs, so the run is not repeatable unless the
    # atlas coordinates are kept somewhere. Stash them on the first run and always
    # resample from the stash: tuning the style then costs one more call, not a
    # reimport of the GLB.
    source_layer = mesh.uv_layers.get(BACKUP_UV_LAYER)
    if source_layer is None:
        source_layer = mesh.uv_layers.new(name=BACKUP_UV_LAYER, do_init=True)
        source_layer.data.foreach_set(
            "uv", _layer_values(mesh.uv_layers.active, loops)
        )
    mesh.uv_layers.active = mesh.uv_layers[0]
    uv = _layer_values(source_layer, loops).reshape(loops, 2)
    loop_vert = np.empty(loops, np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    co = np.empty(verts * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(verts, 3)
    matrix = np.array(obj.matrix_world)
    world = co @ matrix[:3, :3].T + matrix[:3, 3]

    tris = np.arange(loops).reshape(-1, 3)
    tri_pos = world[loop_vert[tris]]
    tri_uv = uv[tris]
    face_normal = np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0])
    lengths = np.linalg.norm(face_normal, axis=1, keepdims=True)
    face_normal = face_normal / np.where(lengths == 0, 1.0, lengths)

    centres = np.asarray(eyes, np.float64)
    head_right = centres[1] - centres[0]
    head_right /= np.linalg.norm(head_right)
    if forward is None:
        forward = head_forward(centres)
    else:
        forward = np.asarray(forward, np.float64)
        forward /= np.linalg.norm(forward)

    source = _read_image(_named_image(obj, "Base Color"))
    orm_image_in = _named_image(obj, "Separate Color", required=False)
    orm = (
        _read_image(orm_image_in)
        if orm_image_in is not None
        else constant_orm(source.shape[:2], *_constant_surface(obj))
    )

    width = patch * len(centres)
    albedo = np.zeros((patch, width, 3), np.float32)
    rough_metal = np.zeros((patch, width, 3), np.float32)
    report = {"eyes": []}
    selection: list[tuple[np.ndarray, np.ndarray]] = []

    for index, centre in enumerate(centres):
        lateral = head_right if index == 1 else -head_right
        gaze = forward + lateral * splay
        gaze /= np.linalg.norm(gaze)
        right, up = eye_basis(gaze)

        delta = tri_pos - centre
        depth = delta @ gaze
        sx = delta @ right
        sy = delta @ up
        near = (
            (np.abs(sx) <= extent * 0.98).all(1)
            & (np.abs(sy) <= extent * 0.98).all(1)
            & (np.abs(depth) <= extent * 0.8).all(1)
            & (face_normal @ gaze > 0.15)
        )
        chosen = np.nonzero(near)[0]
        if chosen.size == 0:
            raise ValueError(f"eye {index} selected no faces; check the centre")

        local = patch_coordinates(tri_pos[chosen], centre, right, up, extent)
        base_uv, covered = rasterise(local, tri_uv[chosen], patch, patch)
        base_rgb = sample_bilinear(source, base_uv)
        base_orm = sample_bilinear(orm, base_uv)
        base_rgb, _ = dilate(base_rgb, covered, rounds=8)
        base_orm, covered = dilate(base_orm, covered, rounds=8)

        fit = locate_eye(base_rgb)
        # Both eyes share one basis, so one highlight offset puts both glints on the
        # same side of the face -- which is what a single light source does.
        x, y, aspect = local_eye_coordinates((patch, patch), fit, style)
        painted, rough = paint_eye(x, y, base_rgb, base_orm[..., 1], style, aspect)

        column = slice(index * patch, (index + 1) * patch)
        eyeball = smoothstep(
            1.0 + style.feather,
            1.0 - style.feather,
            np.abs(x / aspect) ** style.opening_power + np.abs(y) ** style.opening_power,
        )
        albedo[:, column] = painted
        rough_metal[:, column, 0] = base_orm[..., 0]
        rough_metal[:, column, 1] = rough
        # An eye is never metal; whatever the generator packed there only dulls it.
        rough_metal[:, column, 2] = base_orm[..., 2] * (1.0 - eyeball)

        offset = np.array([index / len(centres), 0.0])
        scale = np.array([1.0 / len(centres), 1.0])
        selection.append((chosen, local * scale + offset))
        report["eyes"].append(
            {
                "faces": int(chosen.size),
                "gaze": [round(float(v), 4) for v in gaze],
                "fit": {k: round(float(v), 3) for k, v in fit.items()},
                "aspect": round(float(aspect), 3),
            }
        )

    eye_image = _make_image(f"{material_name}_albedo", albedo, "sRGB")
    orm_image = _make_image(f"{material_name}_orm", rough_metal, "Non-Color")
    material = _eye_material(material_name, eye_image, orm_image)
    slot = _material_slot(mesh, material)

    uv_data = mesh.uv_layers.active.data
    for chosen, coords in selection:
        for row, face in enumerate(chosen):
            polygon = mesh.polygons[int(face)]
            polygon.material_index = slot
            for corner, loop in enumerate(polygon.loop_indices):
                uv_data[loop].uv = tuple(coords[row, corner])
    mesh.update()
    report["material_slot"] = slot
    report["patch"] = [width, patch]
    return report


def revert_in_blender(object_name: str, material: str = "EyePatch") -> dict:
    """Put the mesh back on its original atlas UVs and drop the patch material.

    The backup UV layer makes the paint a reversible edit, which matters because the
    only way to judge an eye is to look at it next to the one it replaced.
    """
    import bpy

    obj = bpy.data.objects[object_name]
    mesh = obj.data
    backup = mesh.uv_layers.get(BACKUP_UV_LAYER)
    if backup is None:
        return {"reverted": False, "reason": "no backup UV layer; nothing was painted"}
    loops = len(mesh.loops)
    mesh.uv_layers[0].data.foreach_set("uv", _layer_values(backup, loops))
    mesh.uv_layers.remove(backup)
    mesh.uv_layers.active = mesh.uv_layers[0]
    slots = [i for i, m in enumerate(mesh.materials) if m is not None and m.name == material]
    faces = 0
    if slots:
        indices = np.zeros(len(mesh.polygons), np.int32)
        mesh.polygons.foreach_get("material_index", indices)
        faces = int((indices == slots[0]).sum())
        indices[indices == slots[0]] = 0
        mesh.polygons.foreach_set("material_index", indices)
        mesh.materials.pop(index=slots[0])
    mesh.update()
    return {"reverted": True, "faces": faces}


def export_in_blender(object_name: str, path: str) -> dict:
    """Write the painted object out as a GLB, without the UV backup riding along.

    The backup layer is a working file, not part of the asset: left in place the
    exporter ships it as TEXCOORD_1 on every vertex, which costs megabytes and means
    nothing to a renderer. It is removed for the write and put back afterwards, so
    the live scene can still be reverted or re-tuned.
    """
    import bpy

    obj = bpy.data.objects[object_name]
    mesh = obj.data
    loops = len(mesh.loops)
    backup = mesh.uv_layers.get(BACKUP_UV_LAYER)
    stashed = _layer_values(backup, loops) if backup else None
    if backup is not None:
        mesh.uv_layers.remove(backup)
        mesh.uv_layers.active = mesh.uv_layers[0]

    selected = [o for o in bpy.context.scene.objects if o.select_get()]
    active = bpy.context.view_layer.objects.active
    try:
        for other in bpy.context.scene.objects:
            other.select_set(False)
        obj.select_set(True)
        if obj.parent is not None:
            obj.parent.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.export_scene.gltf(
            filepath=path, export_format="GLB", use_selection=True, export_materials="EXPORT"
        )
    finally:
        for other in bpy.context.scene.objects:
            other.select_set(other in selected)
        bpy.context.view_layer.objects.active = active
        if stashed is not None:
            restored = mesh.uv_layers.new(name=BACKUP_UV_LAYER, do_init=False)
            restored.data.foreach_set("uv", stashed)
            mesh.uv_layers.active = mesh.uv_layers[0]
    import os

    return {"path": path, "bytes": os.path.getsize(path)}


def _layer_values(layer, loops: int) -> np.ndarray:
    """Flat copy of a UV layer's coordinates."""
    values = np.empty(loops * 2, np.float32)
    layer.data.foreach_get("uv", values)
    return values


def _named_image(obj, target: str, required: bool = True):
    """The image texture feeding a named socket or node of the first material."""
    tree = obj.data.materials[0].node_tree
    for link in tree.links:
        if link.from_node.type != "TEX_IMAGE" or link.from_node.image is None:
            continue
        if target in (link.to_socket.name, link.to_node.name):
            return link.from_node.image
    if required:
        raise ValueError(f"no image texture feeds {target!r} on {obj.data.materials[0].name!r}")
    return None


def _constant_surface(obj) -> tuple[float, float]:
    """A material's flat roughness and metallic, for assets with no packed map."""
    for node in obj.data.materials[0].node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return (
                float(node.inputs["Roughness"].default_value),
                float(node.inputs["Metallic"].default_value),
            )
    return 0.8, 0.0


def _read_image(image) -> np.ndarray:
    w, h = image.size
    buf = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)[..., :3].copy()


def _make_image(name: str, rgb: np.ndarray, colorspace: str):
    import bpy

    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    h, w = rgb.shape[:2]
    image = bpy.data.images.new(name, w, h, alpha=False)
    image.colorspace_settings.name = colorspace
    alpha = np.ones((h, w, 1), np.float32)
    image.pixels.foreach_set(np.concatenate([rgb, alpha], axis=2).ravel())
    image.pack()
    return image


def _eye_material(name: str, albedo, orm):
    import bpy

    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (280, 0)
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.location = (-200, 140)
    tex.image = albedo
    orm_tex = tree.nodes.new("ShaderNodeTexImage")
    orm_tex.location = (-400, -220)
    orm_tex.image = orm
    split = tree.nodes.new("ShaderNodeSeparateColor")
    split.location = (-120, -220)
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(orm_tex.outputs["Color"], split.inputs["Color"])
    tree.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return material


def _material_slot(mesh, material) -> int:
    for index, existing in enumerate(mesh.materials):
        if existing is not None and existing.name == material.name:
            return index
    mesh.materials.append(material)
    return len(mesh.materials) - 1


# -------------------------------------------------------------------------- driver


def send(code: str, host: str = "localhost", port: int = 9876, timeout: int = 300) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(timeout)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
    return b"".join(chunks).decode("utf-8", errors="replace")


def style_from_args(args) -> EyeStyle:
    """Fold the command line's overrides into the default style."""
    style = EyeStyle()
    if args.iris_scale is not None:
        style = replace(style, iris_scale=args.iris_scale)
    if args.pupil_scale is not None:
        style = replace(style, pupil_scale=args.pupil_scale)
    if args.iris:
        iris = parse_colour(args.iris)
        # One colour in, a whole iris out: the outer tone and the limbal ring are
        # darker versions of it, so a caller never has to supply three that agree.
        style = replace(
            style,
            iris_inner_rgb=iris,
            iris_outer_rgb=shade(iris, 0.55),
            limbal_rgb=shade(iris, 0.11),
        )
    if args.sclera:
        style = replace(style, sclera_rgb=parse_colour(args.sclera))
    if args.pupil:
        style = replace(style, pupil_rgb=parse_colour(args.pupil))
    return style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", default="geometry_0", help="mesh object to paint (the glTF import default)"
    )
    parser.add_argument(
        "--material",
        default="EyePatch",
        help="name for the eye material and its images; give each character its own",
    )
    parser.add_argument(
        "--eye",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help="world-space point inside one eye; pass twice, left of the head first",
    )
    parser.add_argument("--extent", type=float, default=0.045, help="patch half-width (metres)")
    parser.add_argument("--patch", type=int, default=640, help="texels per eye")
    parser.add_argument("--splay", type=float, default=0.35, help="outward gaze splay")
    parser.add_argument("--iris-scale", type=float, default=None, help="iris size, 1.0 fills the lid")
    parser.add_argument("--pupil-scale", type=float, default=None, help="pupil size within the iris")
    parser.add_argument("--iris", help="iris colour as #rrggbb or r,g,b; the rim is derived from it")
    parser.add_argument("--sclera", help="sclera colour as #rrggbb or r,g,b")
    parser.add_argument("--pupil", help="pupil colour as #rrggbb or r,g,b")
    parser.add_argument(
        "--forward",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="direction the head faces; inferred from the eye line when omitted",
    )
    parser.add_argument("--export", metavar="GLB", help="write the result to a GLB and exit")
    parser.add_argument(
        "--revert",
        action="store_true",
        help="undo a previous run: restore the atlas UVs and drop the patch material",
    )
    parser.add_argument("--repo", default=".", help="repository root, for Blender's sys.path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    import os

    if args.export or args.revert:
        repo = os.path.abspath(args.repo)
        call = (
            f"_pe.export_in_blender({args.object!r}, {os.path.abspath(args.export)!r})"
            if args.export
            else f"_pe.revert_in_blender({args.object!r})"
        )
        print(
            send(
                "import sys, importlib\n"
                f"sys.path.insert(0, {repo!r})\n"
                "import scripts.paint_eyes as _pe\n"
                "importlib.reload(_pe)\n"
                f"import json; print('RES ' + json.dumps({call}))\n",
                args.host,
                args.port,
            )
        )
        return

    if not args.eye or len(args.eye) != 2:
        parser.error("--eye must be given exactly twice")
    try:
        style = style_from_args(args)
    except ValueError as problem:
        parser.error(str(problem))

    code = blender_payload(
        args.object,
        [tuple(e) for e in args.eye],
        args.extent,
        args.patch,
        args.splay,
        style,
        os.path.abspath(args.repo),
        args.material,
        tuple(args.forward) if args.forward else None,
    )
    print(send(code, args.host, args.port))


if __name__ == "__main__":
    main()
