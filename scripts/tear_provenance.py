#!/usr/bin/env python3
"""Ask of every large tear: could the input view see it?

The repair pass (`remove_loose_parts.py` then `fill_holes.py`) closed the pinholes and
small holes and deliberately skipped the large tears, on the argument that they are
missing evidence rather than artefacts. That argument was asserted, never tested. This
script tests it.

For each boundary loop above the size threshold it reports two things:

* **facing** — the rim's outward plane normal against the input camera direction.
  A tear on a surface turned away from the camera was never photographed.
* **occluded** — a ray from the tear toward the camera, does other geometry block it?
  A tear on a camera-facing surface that sits behind the animal's own leg is equally
  unseen.

A tear that is both camera-facing and unoccluded is the interesting case: the decoder
had the evidence and tore anyway. Those are a decode or decimation defect and more
views will not help them.

On the hero fox, only **5 of 96** are that case. Sweeping the camera over 36 directions
does not rescue the rest either — the best any single view manages is 11 tears and 11.6%
of tear perimeter. So "one more view would have fixed this" is not supported: these
holes are unseeable from anywhere, which points at the surface being sheet-like rather
than at the capture being unlucky.

The camera direction is a **stated assumption, not a measurement**: `--view` defaults to
the `front_three_quarter` preset in `blender_render_asset.py`, converted from Blender's
Z-up back to glTF's Y-up. If the asset was generated from a different view, pass the
right one — the whole classification hangs on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fill_holes import boundary_loops
from mesh_health import weld_by_position

# blender_render_asset.py puts its "front_three_quarter" camera at
# (-0.9, -0.45, ~0) * distance in Blender's Z-up space. The glTF importer maps
# gltf (x, y, z) -> blender (x, -z, y), so blender -Y is glTF +Z.
DEFAULT_VIEW = (-0.9, 0.0, 0.45)


def loop_geometry(
    vertices: np.ndarray, faces: np.ndarray, loops: list[list[int]]
) -> list[dict]:
    """Per-loop perimeter, centroid and outward rim normal.

    The rim normal is the loop's own best-fit plane normal (Newell's method), which is
    what "which way does this hole face" means. Averaging the surrounding faces instead
    is wrong in the symmetric case — around a box's missing face the four neighbours
    cancel exactly to zero — so the neighbours are used only to choose the sign.

    When they cancel, the sign comes from the rim's position against the *local*
    surface around it, never against the mesh's overall centre: a second component
    somewhere else in the file (a blocking limb, a loose speck) moves the global
    centroid and would silently invert the verdict.
    """
    normals = _face_normals(vertices, faces)
    centres = vertices[faces].mean(axis=1)
    touching = _faces_touching(faces, len(vertices))

    out = []
    for loop in loops:
        ring = vertices[loop]
        perimeter = float(np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1).sum())
        centroid = ring.mean(axis=0)

        normal = _newell_normal(ring)

        face_ids = sorted({f for v in loop for f in touching.get(int(v), ())})
        reference = normals[face_ids].sum(axis=0) if face_ids else np.zeros(3)
        if np.linalg.norm(reference) < 1e-9 and face_ids:
            reference = centroid - centres[face_ids].mean(axis=0)
        if np.dot(normal, reference) < 0:
            normal = -normal

        out.append(
            {
                "vertices": len(loop),
                "perimeter": perimeter,
                "centroid": centroid,
                "normal": normal,
            }
        )
    return out


def _newell_normal(ring: np.ndarray) -> np.ndarray:
    """Best-fit plane normal of a closed polygon; robust to non-planar rims."""
    nxt = np.roll(ring, -1, axis=0)
    normal = np.cross(ring - ring.mean(axis=0), nxt - ring.mean(axis=0)).sum(axis=0)
    length = float(np.linalg.norm(normal))
    return normal / length if length > 1e-12 else np.zeros(3)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)


def _faces_touching(faces: np.ndarray, n_vertices: int) -> dict[int, list[int]]:
    touching: dict[int, list[int]] = {}
    for index, face in enumerate(faces):
        for vertex in face:
            touching.setdefault(int(vertex), []).append(index)
    return touching


def classify(
    mesh: trimesh.Trimesh,
    view: np.ndarray,
    min_perimeter: float,
    ray_offset: float = 1e-3,
) -> dict:
    """Classify every boundary loop, and the large ones by what the camera could see."""
    welded = weld_by_position(mesh)
    vertices, faces = np.asarray(welded.vertices), np.asarray(welded.faces)
    scale = float(np.ptp(vertices, axis=0).max())
    limit = scale * min_perimeter

    view = np.asarray(view, dtype=float)
    view = view / np.linalg.norm(view)

    loops = loop_geometry(vertices, faces, boundary_loops(vertices, faces))
    large = [loop for loop in loops if loop["perimeter"] > limit]

    # One batched ray cast: from each tear toward the camera, offset off the surface so
    # the ray does not immediately re-hit the face it started on.
    seen_flags: list[bool] = []
    if large:
        origins = np.array([loop["centroid"] for loop in large]) + view * scale * ray_offset
        directions = np.tile(view, (len(large), 1))
        blocked = welded.ray.intersects_any(
            ray_origins=origins, ray_directions=directions
        )
        seen_flags = [not bool(hit) for hit in blocked]

    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    extent = np.where((upper - lower) > 1e-12, upper - lower, 1.0)

    tears = []
    for loop, unoccluded in zip(large, seen_flags):
        facing = float(np.dot(loop["normal"], view))
        tears.append(
            {
                "perimeter": loop["perimeter"],
                "perimeter_rel": loop["perimeter"] / scale,
                "vertices": loop["vertices"],
                "centroid": [float(v) for v in loop["centroid"]],
                "centroid_rel": [
                    float(v) for v in (loop["centroid"] - lower) / extent
                ],
                "facing_camera": facing > 0.0,
                "facing_dot": facing,
                "occluded": not unoccluded,
                # The decoder had evidence here and tore anyway.
                "was_visible": bool(facing > 0.0 and unoccluded),
            }
        )
    tears.sort(key=lambda t: t["perimeter"], reverse=True)

    visible = [t for t in tears if t["was_visible"]]
    return {
        "scale": scale,
        "min_perimeter": min_perimeter,
        "view": [float(v) for v in view],
        "loops_total": len(loops),
        "loops_large": len(large),
        "tears": tears,
        "summary": {
            "large_tears": len(tears),
            "visible_to_input_view": len(visible),
            "facing_away": sum(1 for t in tears if not t["facing_camera"]),
            "occluded": sum(1 for t in tears if t["occluded"] and t["facing_camera"]),
            "visible_perimeter_share": (
                sum(t["perimeter"] for t in visible) / sum(t["perimeter"] for t in tears)
                if tears
                else 0.0
            ),
        },
    }


def report(result: dict) -> None:
    s = result["summary"]
    print(f"boundary loops: {result['loops_total']}  large: {result['loops_large']}")
    print(f"view direction: {[round(v, 3) for v in result['view']]}")
    print()
    print(f"  {s['large_tears']:4d} large tears")
    print(f"  {s['facing_away']:4d} face away from the input view — never photographed")
    print(f"  {s['occluded']:4d} face the camera but are occluded by other geometry")
    print(f"  {s['visible_to_input_view']:4d} were VISIBLE — evidence existed, mesh tore anyway")
    print(f"       ({s['visible_perimeter_share'] * 100:.0f}% of total tear perimeter)")
    print()
    print("largest ten, by perimeter:")
    print(f"  {'perim':>7}  {'x':>5} {'y':>5} {'z':>5}  {'facing':>7}  state")
    for tear in result["tears"][:10]:
        x, y, z = tear["centroid_rel"]
        state = (
            "VISIBLE" if tear["was_visible"]
            else ("occluded" if tear["facing_camera"] else "facing away")
        )
        print(
            f"  {tear['perimeter_rel']:7.3f}  {x:5.2f} {y:5.2f} {z:5.2f}  "
            f"{tear['facing_dot']:7.2f}  {state}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument(
        "--min-perimeter", type=float, default=0.15,
        help="A loop counts as a large tear above this fraction of the asset's size. "
             "Matches fill_holes.py's default skip threshold, so this measures exactly "
             "the holes the repair pass declined to fill",
    )
    parser.add_argument(
        "--view", type=float, nargs=3, default=DEFAULT_VIEW, metavar=("X", "Y", "Z"),
        help="Direction from the subject toward the input camera, in mesh space",
    )
    parser.add_argument("--json", type=Path, help="Write the full per-tear record here")
    args = parser.parse_args()

    # force="mesh" matches remove_loose_parts.py and fill_holes.py: one Trimesh across
    # trimesh versions, without Scene.dump/to_geometry churn.
    mesh = trimesh.load(args.asset.expanduser().resolve(), force="mesh", process=False)

    result = classify(mesh, np.array(args.view), args.min_perimeter)
    report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
