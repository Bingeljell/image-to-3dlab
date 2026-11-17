"""Tests for the facet check (scripts/normal_map_facets.py).

A normal map baked against flat shading makes the shading jump at every triangle edge, which
is what a player sees as facets. The check reads both sides of each edge; these tests pin the
frame maths on hand-worked cases and run the whole check on a small bumpy patch baked both ways.
"""

from __future__ import annotations

import math

import normal_map_facets as nmf
import numpy as np
import trimesh
from PIL import Image


def test_the_tangent_frame_follows_the_uv_directions():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tangent, bitangent = nmf.face_tangents(positions, np.array([[0, 1, 2]]), positions[:, :2].copy())
    t, b = nmf.tangent_frame(np.array([[0.0, 0.0, 1.0]]), tangent, bitangent)
    assert np.allclose(t, [[1, 0, 0]]) and np.allclose(b, [[0, 1, 0]])


def test_a_mirrored_uv_island_flips_the_bitangent():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uv = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])   # u runs against x
    tangent, bitangent = nmf.face_tangents(positions, np.array([[0, 1, 2]]), uv)
    t, b = nmf.tangent_frame(np.array([[0.0, 0.0, 1.0]]), tangent, bitangent)
    assert np.allclose(t, [[-1, 0, 0]]) and np.allclose(b, [[0, 1, 0]])


def test_a_neutral_map_shades_with_the_smooth_normal_and_a_tilted_one_leans():
    a = math.radians(20)
    smooth = np.array([[0.0, math.sin(a), math.cos(a)]])
    t, b = nmf.tangent_frame(smooth, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    assert np.allclose(nmf.shading_normal(t, b, smooth, np.array([[0.0, 0.0, 1.0]])), smooth)
    # the face's own flat normal written in that frame gives the flat normal back
    flat = np.array([[0.0, 0.0, 1.0]])
    written = np.c_[(flat * t).sum(1), (flat * b).sum(1), (flat * smooth).sum(1)]
    assert np.allclose(nmf.shading_normal(t, b, smooth, written), flat)


def test_edges_are_shared_in_space_and_in_uv_and_seams_do_not_count():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = positions[:, :2].copy()
    assert len(nmf.interior_edges(positions, faces, uv)) == 1
    # the same square split at a UV seam: the second face's corners are copies with other UVs
    positions2 = np.vstack([positions, positions[[0, 2]]])
    faces2 = np.array([[0, 1, 2], [4, 5, 3]])
    uv2 = np.vstack([uv, uv[[0, 2]] + 0.5])
    assert len(nmf.interior_edges(positions2, faces2, uv2)) == 0


def test_shading_that_keeps_the_edges_angle_reads_as_faceted():
    rng = np.random.default_rng(0)
    dihedral = rng.uniform(0.1, 0.4, 500)
    assert nmf.verdict(nmf.facet_score(dihedral * rng.uniform(0.9, 1.0, 500), dihedral)) == "faceted"


def test_shading_that_runs_on_reads_as_smooth():
    rng = np.random.default_rng(1)
    dihedral = rng.uniform(0.1, 0.4, 500)
    score = nmf.facet_score(dihedral * rng.uniform(0.0, 0.2, 500), dihedral)
    assert score["jump_share"] < 0.2 and nmf.verdict(score) == "smooth"


def test_a_mesh_without_angled_edges_is_not_judged():
    flat = np.full(500, np.radians(1.0))
    assert nmf.verdict(nmf.facet_score(flat, flat)) == "no edges to judge"


def _patch(tmp_path, faceted: bool, cells: int = 12, size: int = 128):
    """A bumpy square patch, smooth normals from the surface, one seamless UV map; the normal map
    baked against flat shading (each triangle's own normal) or neutral."""
    xs = np.linspace(0, 1, cells + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    z = 0.12 * np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy)
    dzdx = 0.12 * 2 * np.pi * np.cos(2 * np.pi * gx) * np.cos(2 * np.pi * gy)
    dzdy = -0.12 * 2 * np.pi * np.sin(2 * np.pi * gx) * np.sin(2 * np.pi * gy)
    positions = np.c_[gx.ravel(), gy.ravel(), z.ravel()]
    normals = np.c_[-dzdx.ravel(), -dzdy.ravel(), np.ones(positions.shape[0])]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    idx = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    faces = []
    for i in range(cells):
        for j in range(cells):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [[a, b, c], [a, c, d]]
    faces = np.array(faces)
    uv = positions[:, :2].copy()
    image = np.full((size, size, 3), (128, 128, 255), dtype=np.uint8)
    if faceted:
        flat = trimesh.Trimesh(positions, faces, process=False).face_normals
        tangent, bitangent = nmf.face_tangents(positions, faces, uv)
        ys, xs_ = np.mgrid[0:size, 0:size]
        pix_uv = np.c_[(xs_.ravel() + 0.5) / size, 1 - (ys.ravel() + 0.5) / size]
        for k, face in enumerate(faces):
            a, b, c = uv[face]
            w = np.linalg.solve(np.c_[b - a, c - a], (pix_uv - a).T).T
            inside = (w[:, 0] >= -1e-9) & (w[:, 1] >= -1e-9) & (w.sum(1) <= 1 + 1e-9)
            bary = np.c_[1 - w[inside].sum(1), w[inside]]
            smooth = bary @ normals[face]
            smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
            t, bt = nmf.tangent_frame(smooth, np.repeat(tangent[k:k + 1], len(smooth), 0),
                                      np.repeat(bitangent[k:k + 1], len(smooth), 0))
            n = np.repeat(flat[k:k + 1], len(smooth), 0)
            enc = np.c_[(n * t).sum(1), (n * bt).sum(1), (n * smooth).sum(1)]
            image.reshape(-1, 3)[np.flatnonzero(inside)] = np.clip((enc * 0.5 + 0.5) * 255 + 0.5, 0, 255)
    material = trimesh.visual.material.PBRMaterial(normalTexture=Image.fromarray(image))
    mesh = trimesh.Trimesh(positions, faces, vertex_normals=normals, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    path = tmp_path / ("flat.glb" if faceted else "neutral.glb")
    mesh.export(path)
    return path


def test_a_patch_baked_against_flat_shading_is_caught(tmp_path):
    report = nmf.check(_patch(tmp_path, faceted=True))
    assert report["verdict"] == "faceted", report
    assert report["jump_share"] > 0.7


def test_a_patch_with_a_neutral_map_passes(tmp_path):
    report = nmf.check(_patch(tmp_path, faceted=False))
    assert report["verdict"] == "smooth", report
    assert report["jump_share"] < 0.2   # read 5 % into each face, the smooth normals differ a little too


def test_a_model_without_a_normal_map_says_so(tmp_path):
    path = tmp_path / "bare.glb"
    trimesh.creation.icosphere(subdivisions=1).export(path)
    assert nmf.check(path)["verdict"] == "no normal map"
