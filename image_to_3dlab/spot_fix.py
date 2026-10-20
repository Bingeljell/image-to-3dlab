"""Fix a spot the user points at: cut the triangles around it and cap the opening.

The studio's "Fix a spot" brush. Pixal3D sometimes leaves pockets (a pupil, under a brow,
in a beard) that read as holes; finding them automatically also catches real shape (gaps
between fur tufts or mohawk spikes), so a person picks the spot and this only does the
how: every triangle whose centre lies within `radius` of the point goes, and each opening
left behind is capped with a fan, its centre relaxed toward the rim so the patch sits
flush. The cap takes the paint, UVs and tangents of the rim vertex it starts from, so it
reads as the colour around it. Pure numpy on the GLB's own buffers: textures, materials
and everything else in the file are kept as they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import surface_calm as sc

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
FORMATS = {5126: ("<f4", np.float32), 5125: ("<u4", np.uint32), 5123: ("<u2", np.uint16),
           5121: ("<u1", np.uint8)}
# Brush sizes as a share of the model's height.
RADII = {"small": 0.012, "medium": 0.025, "large": 0.045}


def _read(doc: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    width = COMPONENTS[accessor["type"]]
    dtype = FORMATS[accessor["componentType"]][0]
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride", item)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    out = np.ndarray((accessor["count"], width), dtype=dtype, buffer=binary, offset=start,
                     strides=(stride, np.dtype(dtype).itemsize))
    return np.array(out)


def _boundary_loops(faces: np.ndarray, key: np.ndarray) -> list[list[tuple[int, int]]]:
    """Open edges of `faces` (welded by `key`), as loops of (from, to) original indices."""
    welded = key[faces]
    edges = {}
    for f, wf in zip(faces, welded):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            ka, kb = int(wf[i]), int(wf[(i + 1) % 3])
            if ka == kb:
                continue
            edges.setdefault((min(ka, kb), max(ka, kb)), []).append((a, b, ka, kb))
    open_edges = [e[0] for e in edges.values() if len(e) == 1]
    by_start: dict[int, list[tuple[int, int, int, int]]] = {}
    for edge in open_edges:
        by_start.setdefault(edge[2], []).append(edge)
    loops, used = [], set()
    for edge in open_edges:
        if id(edge) in used:
            continue
        loop, current = [], edge
        while current is not None and id(current) not in used:
            used.add(id(current))
            loop.append((current[0], current[1]))
            nxt = [e for e in by_start.get(current[3], []) if id(e) not in used]
            current = nxt[0] if nxt else None
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Which 2D points lie inside the closed 2D polygon (even-odd ray casting)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None], polygon[:, 1][None]
    x1, y1 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return crosses.sum(1) % 2 == 1


def _edge_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    t = np.clip(((points[:, None] - a[None]) * ab[None]).sum(2) / np.maximum((ab ** 2).sum(1), 1e-12)[None], 0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None] - nearest, axis=2).min(1)


def _even_patch(rim: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Fill a hole with evenly spaced triangles about the size of its rim's edges.

    `rim` is the hole's edge in loop order (n x 3). Returns (interior points, triangles over
    rim 0..n-1 then interior n..), wound opposite to the rim's direction so they continue the
    surface the rim belongs to, or None when the hole is too twisted to lay out flat.
    """
    from scipy.spatial import Delaunay

    centre = rim.mean(0)
    _s, _v, axes = np.linalg.svd(rim - centre)
    flat = (rim - centre) @ axes[:2].T
    spacing = float(np.linalg.norm(np.roll(rim, -1, axis=0) - rim, axis=1).mean())
    lo, hi = flat.min(0), flat.max(0)
    xs = np.arange(lo[0], hi[0] + spacing, spacing)
    rows = []
    for i, y in enumerate(np.arange(lo[1], hi[1] + spacing, spacing * 0.866)):
        rows.append(np.stack([xs + (spacing / 2 if i % 2 else 0), np.full_like(xs, y)], 1))
    grid = np.vstack(rows) if rows else np.zeros((0, 2))
    if len(grid):
        grid = grid[_inside(grid, flat) & (_edge_distance(grid, flat) > 0.55 * spacing)]
    points = np.vstack([flat, grid])
    if len(points) < 3:
        return None
    try:
        triangles = Delaunay(points).simplices
    except Exception:
        return None
    triangles = triangles[_inside(points[triangles].mean(1), flat)]
    n = len(rim)
    have = {tuple(sorted(e)) for t in triangles for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    if any(tuple(sorted((i, (i + 1) % n))) not in have for i in range(n)):
        return None   # the rim is not all there: a twisted hole, so the fan does it
    # wind every triangle against the rim's own direction (it runs the other way round the kept face)
    loop_area = (flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1]).sum()
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = np.sign(area) == np.sign(loop_area)
    triangles[flip] = triangles[flip][:, ::-1]
    interior = centre + grid @ axes[:2]
    return interior, triangles


def _fair(positions: np.ndarray, faces: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Positions with `free` vertices moved to the smoothest surface that continues their
    surroundings (minimum bending: the uniform Laplacian squared is zero over them)."""
    from scipy.sparse import coo_matrix, identity
    from scipy.sparse.linalg import spsolve

    key = sc.weld(positions)
    count = key.max() + 1
    welded = np.zeros((count, 3)); welded[key] = positions
    wf = key[faces]
    edges = np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]])
    edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], 1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    adjacency = coo_matrix((np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))),
                           shape=(count, count)).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = (identity(count, format="csr") - coo_matrix(
        (1 / np.maximum(degree[np.concatenate([a, b])], 1), (np.concatenate([a, b]), np.concatenate([b, a]))),
        shape=(count, count)).tocsr())
    free_w = np.unique(key[free])
    rows = np.unique(np.concatenate([free_w, adjacency[free_w].indices]))
    system = laplacian[rows]
    fixed = np.setdiff1d(np.arange(count), free_w)
    a_free, a_fixed = system[:, free_w], system[:, fixed]
    lhs = (a_free.T @ a_free).tocsc()
    for axis in range(3):
        rhs = -(a_free.T @ (a_fixed @ welded[fixed, axis]))
        welded[free_w, axis] = spsolve(lhs, rhs)
    return welded[key]


def fix_spot(source: Path, output: Path, point: np.ndarray, radius: float,
             views_dir: Path | None = None) -> dict:
    """Cut triangles with centres within `radius` (model units) of `point` and cap the opening.

    Each cap first reads one texel, the rim's most typical colour. Then, where the texture has
    room, each cap gets its own small square of it (every map filled from that texel, so the
    normal and metal-roughness maps stay sensible), and with `views_dir` (the run's Pixel
    Match pictures and cameras) only those squares are repainted from the picture: an eye
    whose pocket was capped gets its eye back.
    """
    doc, binary = pp._split_glb(Path(source).read_bytes())
    primitive = doc["meshes"][0]["primitives"][0]
    attributes = {name: _read(doc, binary, index) for name, index in primitive["attributes"].items()}
    faces = _read(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    positions = attributes["POSITION"].astype(np.float64)
    centres = positions[faces].mean(1)
    cut = np.linalg.norm(centres - np.asarray(point, dtype=np.float64), axis=1) < radius
    if not cut.any():
        return {"cut": 0, "capped": 0}
    keep = faces[~cut]
    key = sc.weld(positions)
    loops = _boundary_loops(keep, key)
    # only the openings this cut made: every rim vertex was on a removed triangle
    touched = set(np.unique(key[faces[cut]]).tolist())
    loops = [loop for loop in loops if all(int(key[a]) in touched for a, _ in loop)]
    # The cap gets its own vertices (same places as the rim, so the shape stays closed), all
    # reading ONE texel: the rim's most typical colour. Stretching the rim's own UVs across a
    # fan smeared everything between them into stripes (2026-10-20, gnoll pupils).
    texture = pp.read_glb(source)[3]
    th, tw = texture.shape[:2]
    uvs = attributes.get("TEXCOORD_0")
    new_rows = {name: [] for name in attributes}
    new_faces = []
    centres_added = []
    caps = []          # (vertex indices of the cap, its normal, the rim vertex whose texel it reads)
    even = 0
    count = len(positions)

    def add(name_values: dict) -> int:
        for name, value in name_values.items():
            new_rows[name].append(value)
        return count + len(new_rows["POSITION"]) - 1

    for loop in loops:
        ring = np.array([a for a, _ in loop])
        pick = int(ring[0])
        if uvs is not None:
            px = np.clip((uvs[ring, 0] * tw).astype(int), 0, tw - 1)
            py = np.clip((uvs[ring, 1] * th).astype(int), 0, th - 1)
            colours = texture[py, px, :3].astype(np.float64)
            pick = int(ring[np.argmin(np.linalg.norm(colours - np.median(colours, axis=0), axis=1))])
        if "NORMAL" in attributes:
            normal = attributes["NORMAL"][ring].astype(np.float64).mean(0)
        else:   # the rim's own facing (Newell): sum of cross products round the loop
            rim = positions[ring]
            normal = np.cross(rim, np.roll(rim, -1, axis=0)).sum(0)

        def vertex(position, source_index, normal=normal, pick=pick):
            row = {}
            for name, values in attributes.items():
                if name == "POSITION":
                    row[name] = position
                elif name == "NORMAL":
                    row[name] = normal / max(np.linalg.norm(normal), 1e-12)
                elif name == "TEXCOORD_0":
                    row[name] = values[pick]
                else:
                    row[name] = values[source_index]
            return row

        # one copy per rim POINT: at a UV seam the rim passes through two vertices at one place
        copies = {}
        for a in ring:
            copies.setdefault(int(key[a]), add(vertex(positions[a], a)))
        patch = _even_patch(positions[ring])
        if patch is not None:
            interior, triangles = patch
            inner = [add(vertex(p, pick)) for p in interior]
            centres_added.extend(inner)
            local = [copies[int(key[a])] for a in ring] + inner
            new_faces.extend((local[i], local[j], local[k]) for i, j, k in triangles)
            caps.append((sorted(set(copies.values())) + inner, normal, pick, ring))
            even += 1
            continue
        centre_index = add(vertex(positions[ring].mean(0), pick))
        centres_added.append(centre_index)
        caps.append((sorted(set(copies.values())) + [centre_index], normal, pick, ring))
        for a, b in loop:
            new_faces.append((copies[int(key[b])], copies[int(key[a])], centre_index))   # kept face runs a->b
    for name, rows in new_rows.items():
        if rows:
            attributes[name] = np.vstack([attributes[name], np.array(rows, dtype=attributes[name].dtype)])
    faces_out = np.vstack([keep, np.array(new_faces, dtype=np.int64).reshape(-1, 3)])
    # bend the patches to continue the surface around them (the rim and its copies stay put)
    if centres_added:
        faired = _fair(attributes["POSITION"].astype(np.float64), faces_out, np.array(centres_added))
        attributes["POSITION"] = faired.astype(np.float32)
        if "NORMAL" in attributes:
            weld = sc.weld(faired)
            welded = np.zeros((weld.max() + 1, 3)); welded[weld] = faired
            smooth = sc.vertex_normals(welded, weld[faces_out])[weld]
            attributes["NORMAL"][count:] = smooth[count:].astype(attributes["NORMAL"].dtype)
    repainted = False
    images = None
    if uvs is not None and caps:
        images, repainted = _own_squares(doc, binary, attributes, faces_out, caps, views_dir)
    _write(doc, binary, primitive, attributes, faces_out, output, images)
    return {"cut": int(cut.sum()), "capped": len(loops), "even": even, "repainted": repainted}


GUTTER = 3               # free pixels kept around each square
CELL_MIN, CELL_MAX = 16, 256
STRIP_SHARE = 1 / 8      # the strip added under a full texture, as a share of its height


def _material_images(doc: dict) -> dict[str, int]:
    """Which image each of the first material's maps uses."""
    material = doc.get("materials", [{}])[0]
    pbr = material.get("pbrMetallicRoughness", {})
    slots = {"base": pbr.get("baseColorTexture"), "mr": pbr.get("metallicRoughnessTexture"),
             "normal": material.get("normalTexture"), "occlusion": material.get("occlusionTexture"),
             "emissive": material.get("emissiveTexture")}
    def source(texture: dict) -> int:   # a WebP texture names its image in an extension
        return texture.get("source", texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
    return {name: source(doc["textures"][slot["index"]]) for name, slot in slots.items() if slot}


def _decode(doc: dict, binary: bytes, image: int) -> np.ndarray:
    import io

    from PIL import Image

    view = doc["bufferViews"][doc["images"][image]["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    return np.array(picture.convert("RGBA" if "A" in picture.getbands() else "RGB"))


def _free_cells(uvs: np.ndarray, faces: np.ndarray, height: int, width: int, cell: int) -> list[tuple[int, int]]:
    """Top-left corners of cell-sized squares no UV triangle touches (with a gutter)."""
    small_h, small_w = max(32, height // 2), max(32, width // 2)
    xy = uvs[faces] * np.array([small_w, small_h])
    face_of, _ = pp.rasterize(xy, np.zeros((len(faces), 3)), (small_h, small_w))
    covered = face_of.reshape(small_h, small_w) >= 0
    step = cell + 2 * GUTTER
    out = []
    # from the bottom up, so an added strip (always at the bottom) is never stepped over
    for y in range(height - step, -1, -step):
        for x in range(0, width - step + 1, step):
            y0, y1 = y * small_h // height, -(-(y + step) * small_h // height)
            x0, x1 = x * small_w // width, -(-(x + step) * small_w // width)
            if not covered[y0:y1, x0:x1].any():
                out.append((x + GUTTER, y + GUTTER))
    return out


def _cap_layout(positions, vertices, normal):
    """A cap's vertices laid flat on its own plane, in model units (origin at its corner)."""
    n = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    flat = np.stack([positions[vertices] @ u, positions[vertices] @ v], 1)
    return flat - flat.min(0)


def _own_squares(doc, binary, attributes, faces, caps, views_dir):
    """Give each cap its own square of the texture, as sharp as the paint around it, and
    repaint the squares from the picture.

    Pixal3D packs its atlas tightly, so when the gaps between islands are too small a strip
    is added under every map (existing pixels are kept, not resampled; UVs move with them),
    and later fixes find room in that strip. Returns ({image: PNG}, repainted).
    """
    import io

    from PIL import Image

    maps = _material_images(doc)
    if "base" not in maps:
        return None, False
    pictures = {image: _decode(doc, binary, image) for image in set(maps.values())}
    height, width = pictures[maps["base"]].shape[:2]
    uvs = attributes["TEXCOORD_0"]
    positions = attributes["POSITION"].astype(np.float64)
    # texels per model unit around each rim, so a patch is as sharp as its surroundings
    layouts, cell = [], CELL_MIN
    for vertices, normal, _pick, rim in caps:
        step_uv = np.linalg.norm((np.roll(uvs[rim], -1, axis=0) - uvs[rim]) * np.array([width, height]), axis=1)
        step_xyz = np.linalg.norm(np.roll(positions[rim], -1, axis=0) - positions[rim], axis=1)
        density = float(np.median(step_uv / np.maximum(step_xyz, 1e-12)))
        flat = _cap_layout(positions, vertices, normal)
        layouts.append(flat)
        cell = max(cell, int(np.ceil(flat.max() * density)) + 4)
    cell = min(cell, CELL_MAX)
    cap_vertices = {v for vertices, *_rest in caps for v in vertices}
    others = faces[~np.isin(faces, list(cap_vertices)).all(1)]
    cells = _free_cells(uvs.astype(np.float64), others, height, width, cell)
    if len(cells) < len(caps):
        strip = max(cell + 2 * GUTTER, int(height * STRIP_SHARE))
        strip += -strip % 4
        moved = uvs.astype(np.float64)
        moved[:, 1] *= height / (height + strip)
        cells = _free_cells(moved, others, height + strip, width, cell)
        if len(cells) < len(caps):
            return None, False     # nothing changed: the caps keep their one shared texel
        # only now that it fits: move the UVs with the pixels and grow every map
        uvs[:, 1] = moved[:, 1].astype(uvs.dtype)
        for image, picture in list(pictures.items()):
            extra = int(round(strip * picture.shape[0] / height))
            # the strip repeats each map's last row, so filtering at the old edge pulls in no black
            pictures[image] = np.concatenate([picture, np.repeat(picture[-1:], extra, axis=0)], 0)
        height += strip
    base = pictures[maps["base"]]
    squares = []
    for (vertices, _normal, pick, _rim), flat, (x, y) in zip(caps, layouts, cells):
        flat = flat / max(flat.max(), 1e-12)
        pad = 2
        for vertex, (fu, fv) in zip(vertices, flat):
            uvs[vertex] = ((x + pad + fu * (cell - 2 * pad)) / width, (y + pad + fv * (cell - 2 * pad)) / height)
        # fill every map's square from the texel the cap read before, so nothing goes black; the
        # normal map is flat there (the patch is smooth, its rim's bumps would tilt its light)
        for image, picture in pictures.items():
            h, w = picture.shape[:2]
            old = uvs[pick]
            texel = picture[min(int(old[1] * h), h - 1), min(int(old[0] * w), w - 1)].copy()
            if image == maps.get("normal"):
                texel[:3] = (128, 128, 255)
            sx, sy = x * w // width, y * h // height
            picture[sy:sy + cell * h // height, sx:sx + cell * w // width] = texel
        squares.append((x, y))
    repainted = False
    if views_dir is not None and (Path(views_dir) / "transforms.json").is_file():
        views, mesh_scale = pp.load_views(Path(views_dir))
        # the picture's own pixels, as they are: matching colours to the surrounding paint has
        # nothing to solve on a brand-new square (it divided by zero and painted them grey)
        plain = pp.Settings(match_colour=False, align=False)
        painted, weight = pp.paint_texture(base, positions, uvs.astype(np.float64), faces, views, mesh_scale,
                                           settings=plain)
        for x, y in squares:
            base[y:y + cell, x:x + cell] = painted[y:y + cell, x:x + cell]
        repainted = bool(any((weight[y:y + cell, x:x + cell] > 0.2).any() for x, y in squares))
    out = {}
    for image, picture in pictures.items():
        buffer = io.BytesIO()
        Image.fromarray(picture).save(buffer, "PNG", compress_level=1)
        out[image] = buffer.getvalue()
    return out, repainted


# Inside a spot the user marked, the picture is trusted fully: no fading near edges where one
# surface sits in front of another, a low facing threshold, and replace rather than blend.
# Finish's own Pixel Match is cautious there, which is exactly what left the shards
# (2026-10-20, the gnoll's brow).
TRUST_PICTURE = {"match_colour": False, "local_colour": False, "align": False, "inner_edge": 0.0,
                 "cutout_edge": 0.0, "edge_px": 1, "facing_low": 0.05, "facing_high": 0.2}   # edge_px 0 zeroes every weight
SEAM_REACH = 1.3          # past each spot's ring, new paint fades into the old out to this, so no hard seam


def repaint_area(source: Path, output: Path, spots: list[tuple[np.ndarray, float]], views_dir: Path) -> dict:
    """Paint only: repaint from the picture, in one pass, everything inside any (point, radius)
    of `spots`, fading into the old paint out to SEAM_REACH x radius; the shape is untouched.
    Inside the ring is fully new: a filled patch's own paint is only a placeholder."""
    import io

    from PIL import Image

    doc, binary = pp._split_glb(Path(source).read_bytes())
    primitive = doc["meshes"][0]["primitives"][0]
    positions, uvs, faces, _tex = pp.read_glb(source)
    faces = faces.reshape(-1, 3).astype(np.int64)
    centres = positions[faces].mean(1)
    near = np.zeros(len(faces), bool)
    for point, radius in spots:
        near |= np.linalg.norm(centres - np.asarray(point, dtype=np.float64), axis=1) < radius * SEAM_REACH
    maps = _material_images(doc)
    if not near.any() or "base" not in maps or not (Path(views_dir) / "transforms.json").is_file():
        return {"repainted_faces": 0, "repainted_texels": 0}
    base = _decode(doc, binary, maps["base"])
    height, width = base.shape[:2]
    face_of, bary = pp.rasterize(uvs[faces[near]] * np.array([width, height]), np.zeros((int(near.sum()), 3)),
                                 (height, width))
    face_of, bary = face_of.reshape(height, width), bary.reshape(height, width, 3)
    mask = face_of >= 0
    # 1 inside a ring, falling to 0 at SEAM_REACH x its radius
    where = (positions[faces[near][face_of[mask]]] * bary[mask][..., None]).sum(1)
    blend = np.zeros(len(where))
    for point, radius in spots:
        reach = radius * SEAM_REACH
        inside = (reach - np.linalg.norm(where - np.asarray(point, dtype=np.float64), axis=1)) / (reach - radius)
        blend = np.maximum(blend, np.clip(inside, 0.0, 1.0))
    views, mesh_scale = pp.load_views(Path(views_dir))
    painted, weight = pp.paint_texture(base, positions, uvs, faces, views, mesh_scale,
                                       settings=pp.Settings(**TRUST_PICTURE))
    blend *= weight[mask] > 0
    mixed = base[mask].astype(np.float64) * (1 - blend[:, None]) + painted[mask].astype(np.float64) * blend[:, None]
    base[mask] = np.rint(mixed).astype(np.uint8)
    mask[mask] = blend > 0
    buffer = io.BytesIO()
    Image.fromarray(base).save(buffer, "PNG", compress_level=1)
    attributes = {name: _read(doc, binary, index) for name, index in primitive["attributes"].items()}
    _write(doc, binary, primitive, attributes, faces, output, {maps["base"]: buffer.getvalue()})
    return {"repainted_faces": int(near.sum()), "repainted_texels": int(mask.sum())}


def repaint_spot(source: Path, output: Path, point: np.ndarray, radius: float, views_dir: Path) -> dict:
    """Paint only, one spot (see repaint_area)."""
    return repaint_area(source, output, [(point, radius)], views_dir)


MODES = ("fill", "repaint")


def apply_spots(source: Path, output: Path, spots: list[dict], views_dir: Path | None) -> list[dict]:
    """Apply marked spots ({point, radius in model units, mode}); the result lands at `output`.

    Shapes first (every "fill", in order, with no painting in between), then ONE repaint of
    every marked area from the picture, each a little wider so seams with the old paint are
    covered. Painting per spot made Patch slower with every mark. Returns a report per spot.
    """
    import shutil
    import tempfile

    reports = []
    with tempfile.TemporaryDirectory() as scratch:
        current = Path(scratch) / "0.glb"
        shutil.copy(source, current)
        for i, spot in enumerate(spots, 1):
            if spot["mode"] != "fill":
                reports.append({"mode": spot["mode"]})
                continue
            nxt = Path(scratch) / f"{i}.glb"
            reports.append({"mode": "fill", **fix_spot(current, nxt, spot["point"], spot["radius"], None)})
            if nxt.exists():
                current = nxt
        if views_dir is not None:
            painted = Path(scratch) / "painted.glb"
            area = [(s["point"], s["radius"]) for s in spots]
            done = repaint_area(current, painted, area, views_dir)
            if painted.exists():
                current = painted
            for report in reports:
                report["repainted"] = bool(done["repainted_texels"])
                if report["mode"] == "repaint":
                    report["repainted_faces"] = done["repainted_faces"]
        shutil.copy(current, output)
    return reports


def _write(doc: dict, binary: bytes, primitive: dict, attributes: dict, faces: np.ndarray, output: Path,
           images: dict[int, bytes] | None = None) -> None:
    """The GLB with the mesh's buffers (and any `images`, as PNG) replaced; the rest kept."""
    mesh_accessors = set(primitive["attributes"].values()) | {primitive["indices"]}
    mesh_views = {doc["accessors"][i]["bufferView"] for i in mesh_accessors}
    image_views = {doc["images"][i]["bufferView"]: png for i, png in (images or {}).items()}
    chunks = []
    for index, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunks.append(b"" if index in mesh_views else image_views.get(index, binary[start:start + view["byteLength"]]))
    for image in images or {}:
        doc["images"][image]["mimeType"] = "image/png"
        pp._drop_webp_extension(doc, image)
    for name, index in primitive["attributes"].items():
        accessor = doc["accessors"][index]
        values = np.ascontiguousarray(attributes[name], dtype=FORMATS[accessor["componentType"]][0])
        doc["bufferViews"].append({"buffer": 0, "byteLength": values.nbytes})
        chunks.append(values.tobytes())
        accessor.update(bufferView=len(doc["bufferViews"]) - 1, byteOffset=0, count=len(values))
        if name == "POSITION":
            accessor["min"], accessor["max"] = values.min(0).tolist(), values.max(0).tolist()
    indices = doc["accessors"][primitive["indices"]]
    data = np.ascontiguousarray(faces.ravel(), dtype="<u4")
    doc["bufferViews"].append({"buffer": 0, "byteLength": data.nbytes})
    chunks.append(data.tobytes())
    indices.update(bufferView=len(doc["bufferViews"]) - 1, byteOffset=0, count=int(data.size), componentType=5125)
    indices.pop("min", None)
    indices.pop("max", None)
    packed = b""
    for view, chunk in zip(doc["bufferViews"], chunks):
        packed += b"\0" * (-len(packed) % 4)
        view["byteOffset"] = len(packed)
        view["byteLength"] = len(chunk)
        view.pop("byteStride", None)
        packed += chunk
    doc["buffers"][0]["byteLength"] = len(packed)
    Path(output).write_bytes(pp._join_glb(doc, packed))


def model_height(source: Path) -> float:
    positions = pp.read_glb(source)[0]
    return float(np.ptp(positions[:, 1]))

