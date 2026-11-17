#!/usr/bin/env python3
"""Say whether a GLB's shading breaks at its triangle edges, as a normal map baked against flat shading makes it.

A normal map bends the shading of a low-poly mesh toward the shape it was baked from. If the
bake used flat shading on the low-poly, the map holds each triangle's flat normal: rendered
with smooth shading, as every game does, the shading jumps at every edge by the angle between
the two triangles, and the figure looks like a 1990s model up close however good its colour.
A map baked from a high-poly against smooth shading follows the high-poly's surface instead,
which runs on across the edges.

So this looks at the edges. On each side of an interior edge (not a UV seam) it takes the
normal the engine would shade with: the map's value (glTF's convention: tangents from the UVs,
green up) in the smooth tangent frame. How far the two sides differ, against how far the two
flat faces differ, is the measure: near 1 the map repeats the facets, near 0 the shading runs
on. Read 5 % into each face, even a neutral map keeps about 0.1. Measured on real figures:
a ComfyUI pipeline whose facets show plainly scores 0.33 to 0.47, figures that look smooth
0.14 to 0.22, so 0.3 is the line. Hard normals in the mesh itself (split at every edge)
show the same way, and look the same in the game. Reports only; it changes nothing.

    python scripts/normal_map_facets.py model.glb [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MIN_DIHEDRAL = np.radians(3.0)  # flatter edges show no facet either way
MIN_EDGES = 100                 # fewer edges to judge than this: no verdict
FACETED = 0.3                   # jump over dihedral, median over the edges (see the module docstring)
INSET = 0.05                    # how far into each face the two sides are read


def face_tangents(positions: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each face's tangent and bitangent: the directions in which u and v grow across it."""
    p0, p1, p2 = (positions[faces[:, i]] for i in range(3))
    t0, t1, t2 = (uv[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1.0, det), 0.0)[:, None]
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    return tangent, bitangent


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(length > 1e-12, length, 1.0)


def tangent_frame(normal: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tangent and bitangent made orthogonal to `normal`, the bitangent's side kept (mirrored UVs)."""
    t = tangent - (tangent * normal).sum(1, keepdims=True) * normal
    degenerate = np.linalg.norm(t, axis=1) < 1e-9
    if degenerate.any():   # a UV-less or collapsed face: any direction in the plane will do
        helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        t[degenerate] = np.cross(helper[degenerate], normal[degenerate])
    t = _unit(t)
    side = np.sign((np.cross(normal, t) * bitangent).sum(1, keepdims=True))
    return t, np.cross(normal, t) * np.where(side == 0, 1.0, side)


def shading_normal(tangent: np.ndarray, bitangent: np.ndarray, normal: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    """The normal the engine shades with: the map's tangent-space value in the smooth frame."""
    return _unit(tangent * mapped[:, :1] + bitangent * mapped[:, 1:2] + normal * mapped[:, 2:3])


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip((a * b).sum(1), -1.0, 1.0))


def facet_score(jumps: np.ndarray, dihedrals: np.ndarray) -> dict:
    """How much of each edge's angle the shading still shows, over edges sharp enough to tell."""
    keep = dihedrals > MIN_DIHEDRAL
    if keep.sum() < MIN_EDGES:
        return {"edges": int(keep.sum()), "jump_share": None}
    return {"edges": int(keep.sum()), "jump_share": round(float(np.median(jumps[keep] / dihedrals[keep])), 3)}


def verdict(score: dict) -> str:
    if score["jump_share"] is None:
        return "no edges to judge"
    return "faceted" if score["jump_share"] >= FACETED else "smooth"


def _normal_image(material) -> np.ndarray | None:
    image = getattr(material, "normalTexture", None)
    if image is None:
        return None
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0 * 2.0 - 1.0


def _lookup(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.floor((uv[:, 0] % 1.0) * w).astype(int).clip(0, w - 1)
    y = np.floor((1.0 - uv[:, 1] % 1.0) * h).astype(int).clip(0, h - 1)
    return image[y, x]


def interior_edges(positions: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(face, face, corner, corner, corner, corner) for each edge two faces share in space and in UV.

    Corners are 0-2 within each face: the edge's two ends in the first face, then in the second.
    glTF splits vertices at UV seams and hard edges, so faces are matched by position.
    """
    _, weld = np.unique(np.round(positions, 6), axis=0, return_inverse=True)
    weld = weld.ravel()
    owner: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for f, face in enumerate(faces):
        for i in range(3):
            j = (i + 1) % 3
            a, b = weld[face[i]], weld[face[j]]
            key = (a, b) if a < b else (b, a)
            owner.setdefault(key, []).append((f, i, j) if a < b else (f, j, i))
    edges = []
    for pair in owner.values():
        if len(pair) != 2:
            continue
        (f, i, j), (g, k, m) = pair
        if np.allclose(uv[faces[f, i]], uv[faces[g, k]], atol=1e-5) and np.allclose(uv[faces[f, j]], uv[faces[g, m]], atol=1e-5):
            edges.append((f, g, i, j, k, m))
    return np.array(edges, dtype=np.int64).reshape(-1, 6)


def edge_jumps(mesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """(shading jump, dihedral) in radians for up to `count` interior edges of one mesh."""
    image = _normal_image(getattr(mesh.visual, "material", None))
    uv = getattr(mesh.visual, "uv", None)
    if image is None or uv is None or len(uv) != len(mesh.vertices):
        return None
    positions, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    uv = np.asarray(uv, float)
    normals = _unit(np.asarray(mesh.vertex_normals, float))
    edges = interior_edges(positions, faces, uv)
    if len(edges) > count:
        edges = edges[rng.choice(len(edges), size=count, replace=False)]
    if not len(edges):
        return np.zeros(0), np.zeros(0)
    tangent, bitangent = face_tangents(positions, faces, uv)
    flat = _unit(np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]]))
    flat *= np.where((flat * normals[faces].mean(1)).sum(1, keepdims=True) < 0, -1.0, 1.0)
    sides = []
    for face_col, a_col, b_col in ((0, 2, 3), (1, 4, 5)):
        f = edges[:, face_col]
        bary = np.zeros((len(edges), 3))
        rows = np.arange(len(edges))
        bary[rows, edges[:, a_col]] += (1 - INSET) / 2
        bary[rows, edges[:, b_col]] += (1 - INSET) / 2
        bary[rows, 3 - edges[:, a_col] - edges[:, b_col]] += INSET
        tri = faces[f]
        smooth = _unit((bary[:, :, None] * normals[tri]).sum(1))
        t, b = tangent_frame(smooth, tangent[f], bitangent[f])
        mapped = _lookup(image, (bary[:, :, None] * uv[tri]).sum(1))
        sides.append(shading_normal(t, b, smooth, mapped))
    return _angle(sides[0], sides[1]), _angle(flat[edges[:, 0]], flat[edges[:, 1]])


def check(path: Path, count: int = 20000, seed: int = 0) -> dict:
    """The facet report of one GLB: every mesh with a normal map, its edges judged together."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    rng = np.random.default_rng(seed)
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    found = [edge_jumps(g, count, rng) for g in meshes]
    found = [x for x in found if x is not None]
    report = {"model": str(path), "meshes": len(meshes), "with_normal_map": len(found)}
    if not found:
        return {**report, "verdict": "no normal map", "edges": 0, "jump_share": None}
    score = facet_score(np.concatenate([x[0] for x in found]), np.concatenate([x[1] for x in found]))
    return {**report, **score, "verdict": verdict(score)}


SAYS = {
    "faceted": "The shading jumps at the triangle edges: in the game every triangle shows. Bake the normal map "
               "again against smooth shading (and keep the mesh's normals smooth), the shading the engine uses.",
    "smooth": "The shading runs on across the triangle edges: no facets.",
    "no edges to judge": "Too few edges with an angle to judge.",
    "no normal map": "No normal map to check.",
}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over the tested functions
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--count", type=int, default=20000, help="edges to read (default 20000)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = check(args.model, args.count)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        detail = (f" (the shading keeps {report['jump_share']:.0%} of the edges' angle, {report['edges']} edges)"
                  if report["jump_share"] is not None else "")
        print(f"{args.model.name}: {report['verdict']}{detail}. {SAYS[report['verdict']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
