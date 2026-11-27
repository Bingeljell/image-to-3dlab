"""Find the parts of a model that stick out past the pictures it was made from.

A generator only sees the picture. Where it builds a part longer than the picture shows (a
tail past its tip, a second sole under a boot), that part is its guess, and often a wrong
one: Pixal3D paints such parts a pale filler grey, and a figure can end up standing on a
glued-on platform shoe. Both show the same way: laid over the picture with the camera the
generator used, the model reaches past the figure's outline.

This lays the model over every picture of its run (Pixel Match's own projection,
`photo_paint.project`), measures how far each vertex lies past the picture's cut-out, joins
the vertices past it along the mesh into parts, and says for each part how far it reaches,
which way, and where on the model. It changes nothing: the right repair differs (a colour,
a cut, a squashed sole), so it only reports.

Pure numpy + scipy + Pillow (trimesh to read the model): no Blender, no GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from image_to_3dlab import photo_paint as pp

PIXAL3D_FRAME = pp.PIXAL3D_FRAME

# How far past the cut-out a vertex must lie to count, as a share of the picture's width:
# a matte's edge and the camera fit are good to about a pixel or two, and Pixel Match treats
# 2 % past the cut-out as "far past" too.
TOLERANCE = 0.02
# Parts of fewer vertices are specks, not parts.
MIN_VERTICES = 3

# The mesh Finish keeps in the pictures' frame, newest first: oriented, calmed, remeshed.
FINISH_MESHES = ("steps/1_orient.glb", "steps/1_calm.glb", "steps/1_retopo.glb", "input/source.glb")

RED = np.array([230, 40, 40], dtype=np.float64)
AMBER = np.array([240, 170, 40], dtype=np.float64)


@dataclass(frozen=True)
class Part:
    """One piece of the model past a picture's outline."""

    view: str        # the picture it sticks out of: "front", "back", or the file's name
    side: str        # which way, in the picture: "below", "above", "left" or "right"
    reach: float     # how far past the outline, as a share of the model's height
    low: float       # where on the model: its lowest and highest point,
    high: float      # as shares of the model's height from the bottom
    vertices: int


def view_label(name: str) -> str:
    """'view_000.png' -> 'front', 'view_180.png' -> 'back', 'view_090.png' -> 'side (90°)';
    other names as they are."""
    stem = Path(name).stem
    labels = {"view_000": "front", "view_180": "back", "view_090": "side (90°)", "view_270": "side (270°)"}
    return labels.get(stem, stem)


def _past_cutout(view: pp.View) -> tuple[np.ndarray, np.ndarray] | None:
    """Per pixel: how many pixels past the cut-out, and the nearest cut-out pixel (2, H, W)."""
    outside = view.image[..., 3] <= 127
    if outside.all():
        return None
    distance, nearest = distance_transform_edt(outside, return_indices=True)
    return distance, nearest


def _vertex_past(x: np.ndarray, y: np.ndarray, distance: np.ndarray, nearest: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels past the cut-out for points at (x, y), points off the picture included, and
    the direction (dx, dy) from the nearest cut-out pixel to each point."""
    height, width = distance.shape
    cx = np.clip(np.floor(x).astype(np.int64), 0, width - 1)
    cy = np.clip(np.floor(y).astype(np.int64), 0, height - 1)
    off = np.hypot(np.maximum(0.0, np.maximum(-x, x - width)), np.maximum(0.0, np.maximum(-y, y - height)))
    past = distance[cy, cx] + off
    my, mx = nearest[0][cy, cx], nearest[1][cy, cx]
    return past, x - (mx + 0.5), y - (my + 0.5)


def _side(dx: float, dy: float) -> str:
    if abs(dy) >= abs(dx):
        return "below" if dy > 0 else "above"   # image y grows downward
    return "right" if dx > 0 else "left"


def parts_past(positions: np.ndarray, faces: np.ndarray, views: list[pp.View], mesh_scale: float = 1.0,
               frame=None, tolerance: float = TOLERANCE, min_vertices: int = MIN_VERTICES) -> list[Part]:
    """Every part of the mesh that lies more than `tolerance` (a share of the picture's
    width) past a picture's cut-out, farthest first.

    `positions` are the GLB's, mapped into the cameras' frame with `frame` (Pixal3D's by
    default) and `mesh_scale`, as Pixel Match does. A part is a piece of the mesh joined by
    its edges whose every vertex is that far past; its reach is how far its farthest vertex
    lies past the outline, measured at that vertex's depth.
    """
    frame = frame or PIXAL3D_FRAME
    faces = np.asarray(faces, dtype=np.int64)
    points = pp.to_view_space(np.asarray(positions, dtype=np.float64), frame, mesh_scale)
    used = np.zeros(len(points), dtype=bool)
    used[faces.ravel()] = True
    if not used.any() or not views:
        return []
    up = views[0].cam_to_world[:3, 1]
    up = up / np.linalg.norm(up)
    heights = points @ up
    bottom, top = heights[used].min(), heights[used].max()
    size = max(top - bottom, 1e-12)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    parts: list[Part] = []
    for view in views:
        found = _past_cutout(view)
        if found is None:
            continue
        distance, nearest = found
        x, y, depth = pp.project(points, view)
        past_px, dx, dy = _vertex_past(x, y, distance, nearest)
        width = view.image.shape[1]
        past = used & (depth > 0) & (past_px > tolerance * width)
        if not past.any():
            continue
        keep = past[edges[:, 0]] & past[edges[:, 1]]
        graph = coo_matrix((np.ones(int(keep.sum())), (edges[keep, 0], edges[keep, 1])),
                           shape=(len(points), len(points)))
        _, label = connected_components(graph, directed=False)
        reach_units = past_px * depth / view.focal_px
        for group in np.unique(label[past]):
            members = np.flatnonzero(past & (label == group))
            if len(members) < min_vertices:
                continue
            far = members[np.argmax(reach_units[members])]
            parts.append(Part(
                view=view_label(view.name),
                side=_side(float(dx[far]), float(dy[far])),
                reach=round(float(reach_units[far] / size), 4),
                low=round(float((heights[members].min() - bottom) / size), 3),
                high=round(float((heights[members].max() - bottom) / size), 3),
                vertices=int(len(members)),
            ))
    return sorted(parts, key=lambda part: -part.reach)


def _where(part: Part) -> str:
    if part.high <= 0.2:
        return "at the bottom of the model"
    if part.low >= 0.8:
        return "at the top of the model"
    return f"at {part.low:.0%}-{part.high:.0%} of the model's height"


SIDES = {"below": "below", "above": "above", "left": "to the left of", "right": "to the right of"}
# The message names the farthest parts; the report's JSON keeps every one.
NAMED = 3


def describe(part: Part, height_m: float | None = None) -> str:
    """One part in plain words: '9 cm below the outline in the front picture, at the bottom ...'."""
    reach = f"{part.reach * height_m * 100:.0f} cm" if height_m else f"{part.reach:.0%} of the model's height"
    return f"{reach} {SIDES[part.side]} the outline in the {part.view} picture, {_where(part)}"


def report(parts: list[Part], views: int, height_m: float | None = None, tolerance: float = TOLERANCE) -> dict:
    """The check's result as plain JSON: clean or not, each part, and a message for a person."""
    if not parts:
        message = (f"Nothing sticks out past the {views} picture(s) the model was made from "
                   f"(more than {tolerance:.0%} of a picture's width).")
    else:
        count = "1 part sticks out" if len(parts) == 1 else f"{len(parts)} parts stick out"
        named = "; ".join(describe(part, height_m) for part in parts[:NAMED])
        rest = len(parts) - NAMED
        if rest > 0:
            named += f"; and {rest} smaller {'one' if rest == 1 else 'ones'}"
        message = (f"{count} past the pictures the model was made from, so the shape there is the "
                   f"generator's guess: {named}. Look at it before using the model.")
    return {"clean": not parts, "views": views, "tolerance": tolerance, "height_m": height_m,
            "parts": [asdict(part) for part in parts], "message": message}


def marked_picture(positions: np.ndarray, faces: np.ndarray, views: list[pp.View], mesh_scale: float = 1.0,
                   frame=None, tolerance: float = TOLERANCE) -> np.ndarray:
    """Each picture with the model laid over it: red where the model lies more than
    `tolerance` past the cut-out, amber where it is just past; the pictures side by side."""
    frame = frame or PIXAL3D_FRAME
    faces = np.asarray(faces, dtype=np.int64)
    points = pp.to_view_space(np.asarray(positions, dtype=np.float64), frame, mesh_scale)
    panels = []
    for view in views:
        image = view.image.astype(np.float64)
        alpha = image[..., 3:4] / 255.0
        panel = image[..., :3] * alpha + 40.0 * (1 - alpha)
        found = _past_cutout(view)
        x, y, depth = pp.project(points, view)
        face_of, _ = pp.rasterize(np.stack([x, y], axis=1)[faces], depth[faces], view.image.shape[:2])
        covered = face_of >= 0
        if found is not None:
            distance = found[0]
            far = covered & (distance > tolerance * view.image.shape[1])
            near = covered & (distance > 0) & ~far
            panel[near] = 0.4 * panel[near] + 0.6 * AMBER
            panel[far] = 0.15 * panel[far] + 0.85 * RED
        panels.append(np.clip(panel, 0, 255).astype(np.uint8))
    height = max(panel.shape[0] for panel in panels)
    panels = [np.pad(panel, ((0, height - panel.shape[0]), (0, 0), (0, 0))) for panel in panels]
    return np.concatenate(panels, axis=1)


def load_mesh(model: Path) -> tuple[np.ndarray, np.ndarray]:
    """A GLB's vertices and triangles, welded by position so a part's pieces join up."""
    import trimesh

    mesh = trimesh.load(model, force="mesh", process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def find_inputs(target: Path) -> tuple[Path, Path]:
    """The model and its pictures for a Finish run folder or a generated model.

    A Finish run keeps its pictures in `input/source.views/` and its mesh, still in the
    pictures' frame, under `steps/`; a Pixal3D model keeps its pictures beside it as
    `<model>.svviews/`.
    """
    target = Path(target)
    if target.is_dir():
        views = target / "input" / "source.views"
        meshes = [target / name for name in FINISH_MESHES if (target / name).is_file()]
        if (views / "transforms.json").is_file() and meshes:
            return meshes[0], views
        raise FileNotFoundError(f"{target} is no Finish run with saved pictures (input/source.views)")
    views = target.with_suffix(".svviews")
    if (views / "transforms.json").is_file():
        return target, views
    raise FileNotFoundError(f"no saved pictures and cameras for {target.name} (expected {views.name}/); "
                            "only Pixal3D models made in this lab keep them, or pass the folder with --views")


def check(model: Path, views_dir: Path, height_m: float | None = None, frame=None,
          tolerance: float = TOLERANCE, picture: Path | None = None) -> dict:
    """Read `model` and the pictures in `views_dir` (a transforms.json folder) and report."""
    positions, faces = load_mesh(Path(model))
    views, mesh_scale = pp.load_views(Path(views_dir))
    parts = parts_past(positions, faces, views, mesh_scale, frame, tolerance)
    if picture is not None:
        Image.fromarray(marked_picture(positions, faces, views, mesh_scale, frame, tolerance)).save(picture)
    return report(parts, len(views), height_m, tolerance)
