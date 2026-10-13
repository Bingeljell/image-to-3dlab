"""Paint a model's base colour with the real pixels of the photos it was made from.

Every repaint in this repo (Hunyuan's, and the generators' own texture stages) *redraws*
the surface. Anything small and exact in the source (a logo, a number, a face) comes back
as a lookalike, because the model is copying a few dozen blurry pixels it cannot read. The
photo already holds the right answer for every surface it can see, so this copies it there
and leaves the generated paint only where no photo looks.

The inputs are a list of views (image + camera), not one photo. A single-image run is a
list of one; turnaround sheets or generated side views slot in later without changes here.

Pure numpy + Pillow: no Blender, no GPU. Each step is a plain function so the tests can
exercise the real code:

- `to_view_space` maps GLB coordinates into the frame the cameras were written in.
- `project` puts 3D points into a view's pixels, using the same formula as pixal3d.cpp
  (`src/proj_grid.cpp`, restated in its `tools/silhouette_iou.py`).
- `rasterize` draws triangles into a pixel grid, nearest-first: once in UV space (which
  texel is which surface point) and once per camera (which surface each pixel sees).
- `view_weights` decides how much to trust each texel's photo pixel: seen, facing the
  camera, away from the silhouette edge, inside the photo's matte.
- `paint_texture` blends the views over the existing texture; `replace_base_colour`
  swaps the result into the GLB, leaving every other byte of it alone.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

# GLB axis order and signs for Pixal3D's camera frame. Found by the 48-way search in
# pixal3d.cpp's silhouette_iou.py and identical on every asset tried (a robot and a small
# creature, 2026-09-28), including after Finish's retopology, which keeps coordinates.
PIXAL3D_FRAME: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 2, 1), (-1, 1, 1))


@dataclass(frozen=True)
class View:
    """One photo and the camera that saw it (NeRF / transforms.json conventions)."""

    image: np.ndarray          # H x W x 4, uint8 RGBA; alpha is the photo's matte
    cam_to_world: np.ndarray   # 4 x 4; the camera looks down its own -Z
    fov_x: float               # horizontal field of view, radians
    name: str = "view"

    @property
    def focal_px(self) -> float:
        return (self.image.shape[1] / 2.0) / np.tan(self.fov_x / 2.0)

    @property
    def position(self) -> np.ndarray:
        return self.cam_to_world[:3, 3]


def load_views(directory: Path) -> tuple[list[View], float]:
    """Read a transforms.json directory (as pixal3d.cpp stages one) into views.

    Returns the views and the file's mesh_scale.
    """
    meta = json.loads((directory / "transforms.json").read_text())
    views = []
    for frame in meta["frames"]:
        image = np.asarray(Image.open(directory / frame["file_path"]).convert("RGBA"))
        fov = float(frame.get("camera_angle_x", meta.get("camera_angle_x")))
        views.append(View(image, np.asarray(frame["transform_matrix"], dtype=np.float64),
                          fov, frame.get("name", frame["file_path"])))
    return views, float(meta.get("mesh_scale", 1.0))


def to_view_space(points: np.ndarray, frame=PIXAL3D_FRAME, mesh_scale: float = 1.0) -> np.ndarray:
    """GLB coordinates -> the coordinate frame the view cameras live in."""
    perm, signs = frame
    return np.stack([signs[i] * points[:, perm[i]] for i in range(3)], axis=1) / mesh_scale


def project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points in view space -> (x, y) pixel coordinates and depth in front of the camera.

    Pixel centres sit at integer + 0.5, matching Pillow's and the upstream formula's S/2
    principal point. Depth is positive in front of the camera.
    """
    w2c = np.linalg.inv(view.cam_to_world)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    height, width = view.image.shape[:2]
    x = view.focal_px * cam[:, 0] / safe + width / 2.0
    y = -view.focal_px * cam[:, 1] / safe + height / 2.0
    return x, y, depth


def rasterize(tri_xy: np.ndarray, tri_depth: np.ndarray, size: tuple[int, int],
              chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Draw triangles into a (height, width) grid, keeping the nearest at each pixel.

    `tri_xy` is (F, 3, 2) in pixel coordinates, `tri_depth` (F, 3). A pixel is covered when
    its centre is inside the triangle. Returns the triangle index per pixel (-1 where none)
    and the barycentric weights (H, W, 3) of that triangle at the pixel centre.
    """
    height, width = size
    best_depth = np.full(height * width, np.inf)
    face_of = np.full(height * width, -1, dtype=np.int64)
    bary = np.zeros((height * width, 3))
    for start in range(0, len(tri_xy), chunk):
        xy = tri_xy[start:start + chunk]
        dz = tri_depth[start:start + chunk]
        lo = np.clip(np.floor(xy.min(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        hi = np.clip(np.ceil(xy.max(axis=1) - 0.5), 0, [width - 1, height - 1]).astype(np.int64)
        extent = hi - lo + 1
        counts = extent[:, 0] * extent[:, 1]
        counts[(xy.max(axis=1) < 0).any(axis=1)
               | (xy.min(axis=1)[:, 0] > width) | (xy.min(axis=1)[:, 1] > height)] = 0
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
        eps = -1e-9
        inside = keep & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        tri, w0, w1, w2 = tri[inside], w0[inside], w1[inside], w2[inside]
        pixel = py[inside] * width + px[inside]
        z = w0 * dz[tri, 0] + w1 * dz[tri, 1] + w2 * dz[tri, 2]
        # Nearest candidate per pixel within this chunk, then against what is already drawn.
        order = np.lexsort((z, pixel))
        pixel, z, tri, w0, w1, w2 = (v[order] for v in (pixel, z, tri, w0, w1, w2))
        first = np.ones(len(pixel), dtype=bool)
        first[1:] = pixel[1:] != pixel[:-1]
        pixel, z, tri, w0, w1, w2 = (v[first] for v in (pixel, z, tri, w0, w1, w2))
        nearer = z < best_depth[pixel]
        pixel = pixel[nearer]
        best_depth[pixel] = z[nearer]
        face_of[pixel] = tri[nearer] + start
        bary[pixel] = np.stack([w0[nearer], w1[nearer], w2[nearer]], axis=1)
    return face_of.reshape(height, width), bary.reshape(height, width, 3)


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. Winding may be inconsistent in generated meshes, so
    callers use these only up to sign."""
    tri = positions[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def outward_sign(positions: np.ndarray, faces: np.ndarray) -> float:
    """+1 if `vertex_normals` point out of the mesh, -1 if the winding is inside out, 0 if
    the mesh is open and has no inside to tell.

    Closed means almost no edge belongs to a single triangle (a hole's rim), counted after
    welding vertices that a UV seam split apart. Edges shared by three or more triangles
    are untidy but not holes. Then the signed volume gives the winding.
    """
    _, weld = np.unique(np.round(positions, 6), axis=0, return_inverse=True)
    welded = weld.reshape(-1)[faces]
    edges = np.sort(np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]]), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    if (uses == 1).mean() > OPEN_EDGE_SHARE:
        return 0.0
    centred = positions - positions.mean(axis=0)
    v0, v1, v2 = (centred[faces[:, k]] for k in range(3))
    volume = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()
    return 1.0 if volume > 0 else -1.0 if volume < 0 else 0.0


# Generated meshes measure 0-0.02% hole edges (2026-10-12); a loose sheet is all rim.
OPEN_EDGE_SHARE = 0.02


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Distance, in whole pixels up to `pixels`, from each True pixel to the nearest False.

    Returns an int array: 0 outside the mask, 1 on its edge, growing inward, capped.
    """
    depth = np.zeros(mask.shape, dtype=np.int32)
    current = mask.copy()
    for step in range(1, pixels + 1):
        depth[current] = step
        shrunk = current.copy()
        shrunk[1:, :] &= current[:-1, :]
        shrunk[:-1, :] &= current[1:, :]
        shrunk[:, 1:] &= current[:, :-1]
        shrunk[:, :-1] &= current[:, 1:]
        shrunk[0, :] = shrunk[-1, :] = False
        shrunk[:, 0] = shrunk[:, -1] = False
        current = shrunk
    return depth


def inner_edges(zbuffer: np.ndarray, jump: float) -> np.ndarray:
    """Pixels where the surface steps in depth by more than `jump` to a neighbour that is
    also surface (one part in front of another). The outer silhouette, where surface meets
    background (inf), is not counted: `edge_px` handles that."""
    seen = np.isfinite(zbuffer)
    depth = np.where(seen, zbuffer, 0.0)
    cut = np.zeros(zbuffer.shape, dtype=bool)
    for here, there in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                        ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        both = seen[here] & seen[there]
        step = both & (np.abs(depth[here] - depth[there]) > jump)
        cut[here] |= step
        cut[there] |= step
    return cut


def smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_rgb(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear RGB lookup at pixel coordinates (pixel centres at +0.5), clamped."""
    height, width = image.shape[:2]
    fx = np.clip(x - 0.5, 0, width - 1)
    fy = np.clip(y - 0.5, 0, height - 1)
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
    rgb = image[..., :3].astype(np.float64)
    top = rgb[y0, x0] * (1 - tx) + rgb[y0, x1] * tx
    bottom = rgb[y1, x0] * (1 - tx) + rgb[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


@dataclass(frozen=True)
class Settings:
    # Facing: dot(normal, direction to camera). Below `facing_low` the photo is not used
    # at all (grazing angles smear a few pixels over a large area); full trust from
    # `facing_high`.
    facing_low: float = 0.25
    facing_high: float = 0.6
    # Fade in from the model's silhouette edge over this many view pixels. The edge is
    # where a matte bleeds backdrop colour and where a slightly-off camera misses.
    edge_px: int = 6
    # Fade in from the picture's own cut-out edge over this share of its width (8 px at
    # 768). Edge pixels are half backdrop and land a few pixels off the model, so they
    # paint thin dark lines on heads, ears and cheeks. 0 turns it off.
    cutout_edge: float = 8 / 768
    # Fade in from inner depth edges (an ear in front of a head, a collar in front of a
    # neck) over this share of the view's width. The drawn edge and the model's edge never
    # line up exactly, so colour from one side spills onto the other. 0 turns it off.
    inner_edge: float = 8 / 768
    # A depth step between neighbouring view pixels bigger than this share of the model's
    # size counts as an inner edge; a tilted surface changes depth far more gently.
    inner_jump: float = 0.02
    # A texel counts as seen when its depth is within this fraction of the model's size
    # of the nearest surface in the view.
    depth_tolerance: float = 0.004
    # Grow painted texels this many texels into the UV gutter, so filtering at island
    # borders does not pull the old paint back in as a seam.
    gutter_px: int = 4
    # Shift the model's own paint to the photo's palette (see fit_colour_match), so the
    # parts no photo sees do not stay in the repaint's more saturated colours.
    match_colour: bool = True
    # Match locally rather than with one shift for the whole model: the gap between the
    # model's paint and the picture is measured spot by spot where the picture is trusted,
    # spread smoothly over the surface, and added everywhere. Sides then meet each picture
    # in colour (a robe's side next to a pale back, a belt beside a dark front). Falls back
    # to the global match when too little of the picture is trusted.
    local_colour: bool = True
    # Nudge each picture onto the model before painting (see view_align): the cut-out onto
    # the model's outline, then detailed spots onto where the model built them. Only moves
    # what it is sure of, a few pixels at most.
    align: bool = True


DEFAULT_SETTINGS = Settings()


def view_weights(texel_points: np.ndarray, texel_normals: np.ndarray, view: View,
                 view_positions: np.ndarray, faces: np.ndarray,
                 settings: Settings | None = None, outward: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """How much to trust this view's pixel at each texel, and that pixel's colour.

    Everything is in view space. `view_positions` are the mesh's vertices (for the
    camera's own depth buffer), `texel_points` / `texel_normals` the surface at each texel.
    """
    settings = settings or DEFAULT_SETTINGS
    height, width = view.image.shape[:2]
    vx, vy, vz = project(view_positions, view)
    tri_xy = np.stack([vx, vy], axis=1)[faces]
    face_of, bary = rasterize(tri_xy, vz[faces], (height, width))
    # The camera's nearest depth per pixel, from the winning triangle.
    seen = face_of >= 0
    zbuffer = np.full((height, width), np.inf)
    zbuffer[seen] = (bary[seen] * vz[faces][face_of[seen]]).sum(axis=1)

    x, y, depth = project(texel_points, view)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (depth > 0)
    ix, iy = np.clip(ix, 0, width - 1), np.clip(iy, 0, height - 1)

    extent = np.ptp(view_positions, axis=0).max()
    visible = inside & (depth <= zbuffer[iy, ix] + settings.depth_tolerance * extent)

    to_camera = view.position - texel_points
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    facing = (texel_normals * to_camera).sum(axis=1)
    # On a closed mesh, only surfaces facing the camera take its paint: the far side of a
    # thin sheet (cloak, ponytail) sits inside the depth tolerance and would otherwise get
    # the near side's colours. An open mesh has no outside to tell, so either side counts.
    if outward is None:
        outward = outward_sign(view_positions, faces)
    facing = np.maximum(facing * outward, 0.0) if outward else np.abs(facing)

    edge = erode(seen, settings.edge_px)[iy, ix] / max(settings.edge_px, 1)
    inner_px = round(settings.inner_edge * width)
    if inner_px > 0:
        away = erode(~inner_edges(zbuffer, settings.inner_jump * extent), inner_px)
        edge = edge * smoothstep(away[iy, ix] / inner_px, 0.0, 1.0)
    matte = view.image[iy, ix, 3] / 255.0
    cutout_px = round(settings.cutout_edge * width)
    if cutout_px > 0:
        inside = erode(view.image[..., 3] > 127, cutout_px)[iy, ix] / cutout_px
        matte = matte * smoothstep(inside, 0.0, 1.0)

    weight = visible * smoothstep(facing, settings.facing_low, settings.facing_high) \
        * smoothstep(edge, 0.0, 1.0) * matte
    return weight, sample_rgb(view.image, x, y)


def paint_texture(texture: np.ndarray, positions: np.ndarray, uvs: np.ndarray,
                  faces: np.ndarray, views: list[View], mesh_scale: float = 1.0,
                  frame=PIXAL3D_FRAME, settings: Settings | None = None,
                  align_reports: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Blend the views' pixels into `texture` (H x W x 3 or 4, uint8).

    `uvs` follow glTF: (0, 0) is the image's top-left corner. Returns the new texture and
    the per-texel weight actually applied (0..1), which is the debug map worth looking at.
    Pass a list as `align_reports` to get one view_align.AlignReport per view.
    """
    settings = settings or DEFAULT_SETTINGS
    if settings.align:
        from image_to_3dlab import view_align  # imports this module; import late
        aligned = [view_align.align_view(view, positions, uvs, faces, texture, frame, mesh_scale)
                   for view in views]
        views = [view for view, _ in aligned]
        if align_reports is not None:
            align_reports.extend(report for _, report in aligned)
    height, width = texture.shape[:2]
    uv_xy = np.stack([uvs[:, 0] * width, uvs[:, 1] * height], axis=1)[faces]
    face_of, bary = rasterize(uv_xy, np.zeros((len(faces), 3)), (height, width))
    covered = face_of >= 0
    texel_face = face_of[covered]
    texel_bary = bary[covered]

    view_pos = to_view_space(positions, frame, mesh_scale)
    normals = vertex_normals(view_pos, faces)
    corners = faces[texel_face]
    points = (texel_bary[:, :, None] * view_pos[corners]).sum(axis=1)
    texel_n = (texel_bary[:, :, None] * normals[corners]).sum(axis=1)
    texel_n /= np.maximum(np.linalg.norm(texel_n, axis=1, keepdims=True), 1e-12)

    total_w = np.zeros(len(points))
    colour = np.zeros((len(points), 3))
    strongest = np.zeros(len(points))
    outward = outward_sign(view_pos, faces)
    for view in views:
        weight, rgb = view_weights(points, texel_n, view, view_pos, faces, settings, outward)
        total_w += weight
        colour += weight[:, None] * rgb
        strongest = np.maximum(strongest, weight)
    blend = strongest
    photo = colour / np.maximum(total_w, 1e-12)[:, None]

    out = texture.astype(np.float64).copy()
    flat = out.reshape(-1, out.shape[2])
    index = np.flatnonzero(covered.ravel())
    local = None
    if settings.match_colour and settings.local_colour:
        local = local_colour_offset(positions, faces, texel_face, texel_bary,
                                    flat[index, :3], photo, blend)
    if local is not None:
        flat[index, :3] = np.clip(flat[index, :3] + local, 0, 255)
    elif settings.match_colour:
        # The whole image, gutters included, or island borders keep the old palette.
        gain, offset = fit_colour_match(flat[index, :3], photo, blend)
        flat[:, :3] = np.clip(flat[:, :3] * gain + offset, 0, 255)
    own = flat[index, :3]
    flat[index, :3] = own * (1 - blend[:, None]) + photo * blend[:, None]
    weight_map = np.zeros(height * width)
    weight_map[index] = blend
    weight_map = weight_map.reshape(height, width)
    if local is not None:
        # Every texel changed colour, seen or not, so grow all of them into the gutter.
        out, _ = _grow_into_gutter(out, covered.astype(np.float64), covered, settings.gutter_px)
        _, weight_map = _grow_into_gutter(out.copy(), weight_map, covered, settings.gutter_px)
    else:
        out, weight_map = _grow_into_gutter(out, weight_map, covered, settings.gutter_px)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), weight_map


def local_colour_offset(positions: np.ndarray, faces: np.ndarray, texel_face: np.ndarray,
                        texel_bary: np.ndarray, own: np.ndarray, photo: np.ndarray,
                        weight: np.ndarray, trusted: float = 0.9, min_texels: int = 500,
                        smooth: float = 0.05, outlier: float = 3.0) -> np.ndarray | None:
    """Per-texel colour offset taking the model's own paint towards the pictures, locally.

    The gap (picture - own) is measured where a picture is fully trusted, splatted onto
    the mesh's vertices (welded across UV seams so it flows over them), and spread to the
    rest by a smooth interpolation over the mesh's edges: a graph Laplacian with the
    measured vertices held loosely (`smooth`). What is spread is each spot's difference
    from the typical gap, so a part no measurement can reach (a loose piece) still gets
    the typical gap rather than none. A gap far from the typical one (more than
    `outlier` x the median absolute deviation) is a mismatched feature, such as a rope the
    model painted as robe, not colour drift, and is left out so it cannot smear.
    Returns None when too little is trusted to measure.
    """
    mask = weight >= trusted
    if mask.sum() < min_texels:
        return None
    gap = photo - own
    typical = np.median(gap[mask], axis=0)
    deviation = np.linalg.norm(gap - typical, axis=1)
    limit = max(np.median(deviation[mask]) * 1.4826 * outlier, 1.0)
    mask &= deviation <= limit

    _, weld = np.unique(np.round(positions, 6), axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    count = weld.max() + 1
    corners = weld[faces]
    total = np.zeros((count, 3))
    seen = np.zeros(count)
    for k in range(3):
        vertex = corners[texel_face[mask], k]
        share = texel_bary[mask, k]
        np.add.at(seen, vertex, share)
        np.add.at(total, vertex, share[:, None] * gap[mask])
    known = seen > 0.5
    target = np.zeros((count, 3))
    target[known] = total[known] / seen[known][:, None] - typical

    edges = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
    hold = known * smooth
    system = (diags(np.asarray(adjacency.sum(axis=1)).ravel() + hold) - adjacency).tocsr()
    precondition = diags(1.0 / system.diagonal())
    field = np.zeros((count, 3))
    for channel in range(3):
        field[:, channel], _ = cg(system, hold * target[:, channel], M=precondition,
                                  maxiter=3000, rtol=1e-6)
    return typical + (texel_bary[:, :, None] * field[corners[texel_face]]).sum(axis=1)


def fit_colour_match(own: np.ndarray, photo: np.ndarray, weight: np.ndarray,
                     trusted: float = 0.9, min_texels: int = 500
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain and offset taking the model's own paint to the photo's palette.

    Fitted only where the photo is fully trusted, so both colours describe the same
    surface, then applied to the whole texture: the sides and back the photo cannot see
    take on its palette instead of the repaint's (which runs more saturated). Plain
    mean/spread matching per channel, clamped, so a strange fit cannot wreck the colours;
    too little overlap returns the identity.
    """
    mask = weight >= trusted
    if mask.sum() < min_texels:
        return np.ones(3), np.zeros(3)
    a, b = own[mask], photo[mask]
    spread_a, spread_b = a.std(axis=0), b.std(axis=0)
    gain = np.clip(np.where(spread_a > 1e-6, spread_b / np.maximum(spread_a, 1e-6), 1.0),
                   0.5, 2.0)
    offset = np.clip(b.mean(axis=0) - gain * a.mean(axis=0), -64.0, 64.0)
    return gain, offset


def _grow_into_gutter(image: np.ndarray, weight: np.ndarray, covered: np.ndarray,
                      steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Copy painted texels outward into uncovered (gutter) texels, a ring at a time."""
    filled = covered & (weight > 0)
    for _ in range(steps):
        grown = np.zeros_like(filled)
        source = np.zeros(image.shape[:2] + (2,), dtype=np.int64)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.zeros_like(filled)
            ys, xs = np.nonzero(filled)
            ty, tx = ys + dy, xs + dx
            ok = (ty >= 0) & (ty < filled.shape[0]) & (tx >= 0) & (tx < filled.shape[1])
            ty, tx, ys, xs = ty[ok], tx[ok], ys[ok], xs[ok]
            target = ~covered[ty, tx] & ~filled[ty, tx] & ~grown[ty, tx]
            ty, tx, ys, xs = ty[target], tx[target], ys[target], xs[target]
            shifted[ty, tx] = True
            source[ty, tx] = np.stack([ys, xs], axis=1)
            grown |= shifted
        ys, xs = np.nonzero(grown)
        sy, sx = source[ys, xs, 0], source[ys, xs, 1]
        image[ys, xs, :3] = image[sy, sx, :3]
        weight[ys, xs] = weight[sy, sx]
        filled |= grown
        covered = covered | grown
    return image, weight


# --- GLB in and out ------------------------------------------------------------------------


def _split_glb(data: bytes) -> tuple[dict, bytes]:
    if data[:4] != b"glTF":
        raise ValueError("not a GLB file")
    json_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + json_len])
    offset = 20 + json_len
    binary = b""
    if offset < len(data):
        bin_len = struct.unpack_from("<I", data, offset)[0]
        binary = data[offset + 8:offset + 8 + bin_len]
    return doc, binary


def _join_glb(doc: dict, binary: bytes) -> bytes:
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    out = struct.pack("<III", 0x46546C67, 2, length)
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    if binary:
        out += struct.pack("<II", len(binary), 0x004E4942) + binary
    return out


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride")
    item = np.dtype(dtype).itemsize * width
    if stride and stride != item:
        raise ValueError("interleaved buffers are not supported")
    return np.frombuffer(binary, dtype=dtype, count=accessor["count"] * width,
                         offset=start).reshape(accessor["count"], width)


def base_colour_image_index(doc: dict) -> int:
    """The image index the (single) material uses as base colour."""
    materials = doc.get("materials") or []
    if len(materials) != 1:
        raise ValueError(f"expected one material, found {len(materials)}")
    texture = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if texture is None:
        raise ValueError("the material has no base colour texture to paint")
    entry = doc["textures"][texture["index"]]
    if "source" in entry:
        return entry["source"]
    # Pixal3D's raw output carries only a WebP image, reachable through the extension.
    webp = entry.get("extensions", {}).get("EXT_texture_webp", {})
    if "source" not in webp:
        raise ValueError("the base colour texture has no image this reader understands")
    return webp["source"]


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positions, glTF UVs, triangle indices and the base colour image of a one-mesh GLB.

    Node transforms are not applied: Finish writes a single node with none, and a GLB
    that has one is refused rather than painted in the wrong place.
    """
    doc, binary = _split_glb(Path(path).read_bytes())
    for node in doc.get("nodes", []):
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("node transforms are not supported; expected Finish's output")
    primitives = [p for mesh in doc["meshes"] for p in mesh["primitives"]]
    if len(primitives) != 1:
        raise ValueError(f"expected one primitive, found {len(primitives)}")
    attributes = primitives[0]["attributes"]
    positions = _accessor(doc, binary, attributes["POSITION"]).astype(np.float64)
    uvs = _accessor(doc, binary, attributes["TEXCOORD_0"]).astype(np.float64)
    faces = _accessor(doc, binary, primitives[0]["indices"]).reshape(-1, 3).astype(np.int64)
    image = doc["images"][base_colour_image_index(doc)]
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    texture = np.asarray(picture.convert("RGBA" if "A" in picture.getbands() else "RGB"))
    return positions, uvs, faces, texture


def replace_base_colour(data: bytes, png: bytes) -> bytes:
    """Return the GLB with its base colour image swapped for `png`, all else unchanged.

    Buffer views are repacked in order with glTF's 4-byte alignment; only the image's own
    view changes size.
    """
    doc, binary = _split_glb(data)
    target = doc["images"][base_colour_image_index(doc)]["bufferView"]
    chunks = []
    for index, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunks.append(png if index == target else binary[start:start + view["byteLength"]])
    packed = b""
    for view, chunk in zip(doc["bufferViews"], chunks):
        packed += b"\0" * (-len(packed) % 4)
        view["byteOffset"] = len(packed)
        view["byteLength"] = len(chunk)
        packed += chunk
    doc["buffers"][0]["byteLength"] = len(packed)
    image = base_colour_image_index(doc)
    doc["images"][image]["mimeType"] = "image/png"
    _drop_webp_extension(doc, image)
    return _join_glb(doc, packed)


def _drop_webp_extension(doc: dict, image: int) -> None:
    """Point every texture that showed `image` through EXT_texture_webp straight at it,
    now that it is a PNG, and stop declaring the extension once nothing uses it."""
    for texture in doc.get("textures", []):
        extensions = texture.get("extensions", {})
        if extensions.get("EXT_texture_webp", {}).get("source") == image:
            texture["source"] = image
            del extensions["EXT_texture_webp"]
            if not extensions:
                texture.pop("extensions")
    if any("EXT_texture_webp" in t.get("extensions", {}) for t in doc.get("textures", [])):
        return
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in doc:
            doc[key] = [name for name in doc[key] if name != "EXT_texture_webp"]
            if not doc[key]:
                del doc[key]


def encode_png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()
