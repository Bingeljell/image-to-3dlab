"""Turn round the triangles a picture camera sees from behind.

Pixal3D's meshes come out of simplification with about 1% of triangles wound the wrong way
(measured 2026-10-20: 0.5-1.6% on every Finish). Pixel Match skips surfaces facing away
from the camera, so a wrong-way triangle on the front keeps Pixal3D's own paint (the gnoll's
"shards"), and a game engine that hides back faces shows it as a hole. Pixal3D models are
closed shells, so a triangle a camera can see must face it: turn round those that don't.
Each turned-round triangle gets its own corners, lit flat: it usually sits in a fold, and
corners shared with neighbours facing the other way average to "away" (black chips).
Runs in Finish after the clean-up and before Pixel Match and the bake, which then make the
paint, normal map and tangents for the corrected triangles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix as sf

RESOLUTION = 1024


def facing_away(positions: np.ndarray, faces: np.ndarray, views_and_scale) -> np.ndarray:
    """Indices of triangles some view sees front-most while they face away from it."""
    views, mesh_scale = views_and_scale
    view_pos = pp.to_view_space(positions, mesh_scale=mesh_scale)
    away = np.zeros(len(faces), bool)
    for view in views:
        camera = pp.View(np.zeros((RESOLUTION, RESOLUTION, 4), np.uint8), view.cam_to_world, view.fov_x, view.name)
        x, y, z = pp.project(view_pos, camera)
        face_of, _ = pp.rasterize(np.stack([x, y], 1)[faces], z[faces], (RESOLUTION, RESOLUTION))
        seen = np.unique(face_of[face_of >= 0])
        a, b, c = view_pos[faces[seen, 0]], view_pos[faces[seen, 1]], view_pos[faces[seen, 2]]
        normal = np.cross(b - a, c - a)
        to_camera = view.position - (a + b + c) / 3
        away[seen[(normal * to_camera).sum(1) < 0]] = True
    return np.flatnonzero(away)


def face_the_cameras(source: Path, output: Path, views_dir: Path) -> dict:
    """Write `source` with every seen-from-behind triangle turned round."""
    doc, binary = pp._split_glb(Path(source).read_bytes())
    primitive = doc["meshes"][0]["primitives"][0]
    attributes = {name: sf._read(doc, binary, index) for name, index in primitive["attributes"].items()}
    faces = sf._read(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    positions = attributes["POSITION"].astype(np.float64)
    flip = facing_away(positions, faces, pp.load_views(Path(views_dir)))
    if len(flip):
        corners = faces[flip][:, [0, 2, 1]].reshape(-1)
        start = len(positions)
        for name, values in attributes.items():
            attributes[name] = np.concatenate([values, values[corners]])
        faces[flip] = np.arange(start, start + len(corners)).reshape(-1, 3)
        if "NORMAL" in attributes:
            p = attributes["POSITION"].astype(np.float64)[faces[flip]]
            normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
            attributes["NORMAL"][start:] = np.repeat(normal, 3, axis=0).astype(attributes["NORMAL"].dtype)
    sf._write(doc, binary, primitive, attributes, faces, output)
    return {"flipped": int(len(flip)), "triangles": int(len(faces))}
