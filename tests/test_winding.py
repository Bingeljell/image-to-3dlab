"""Finish's orient step: triangles a picture camera sees must face that camera."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix as sf
from image_to_3dlab import winding


def _views(tmp_path, colour=(0, 0, 0)):
    """A front and a back camera, Pixal3D style (view space Z up, front camera at -Y)."""
    folder = tmp_path / "views"
    folder.mkdir()
    front = [[1, 0, 0, 0], [0, 0, -1, -3.0], [0, 1, 0, 0], [0, 0, 0, 1]]
    back = [[-1, 0, 0, 0], [0, 0, 1, 3.0], [0, 1, 0, 0], [0, 0, 0, 1]]
    frames = []
    for name, matrix in (("view_000.png", front), ("view_180.png", back)):
        Image.new("RGBA", (64, 64), (*colour, 255)).save(folder / name)
        frames.append({"file_path": name, "transform_matrix": matrix, "camera_angle_x": 0.8})
    (folder / "transforms.json").write_text(json.dumps({"camera_angle_x": 0.8, "mesh_scale": 1, "frames": frames}))
    return folder


def _sphere(tmp_path, flip=(), normals=False):
    from test_spot_fix import _sphere_with_pocket

    source, pos, _front = _sphere_with_pocket(tmp_path, n=32)
    doc, binary = pp._split_glb(source.read_bytes())
    prim = doc["meshes"][0]["primitives"][0]
    faces = sf._read(doc, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
    faces[list(flip)] = faces[list(flip)][:, [0, 2, 1]]
    attrs = {n: sf._read(doc, binary, i) for n, i in prim["attributes"].items()}
    if normals:   # a unit sphere: each point's position is its outward direction
        attrs["NORMAL"] = attrs["POSITION"] / np.linalg.norm(attrs["POSITION"], axis=1, keepdims=True)
        doc["accessors"].append({k: v for k, v in doc["accessors"][prim["attributes"]["POSITION"]].items()
                                 if k not in ("min", "max")})
        prim["attributes"]["NORMAL"] = len(doc["accessors"]) - 1
    path = tmp_path / "flipped.glb"
    sf._write(doc, binary, prim, attrs, faces, path)
    return path, pos, faces


def _facing_away(path, views_dir):
    pos, _uv, faces, _t = pp.read_glb(path)
    return winding.facing_away(pos, faces.reshape(-1, 3).astype(np.int64), pp.load_views(views_dir))


def test_visible_triangles_facing_away_are_turned_round(tmp_path):
    views = _views(tmp_path)
    clean, _pos, faces = _sphere(tmp_path)
    # pick triangles on the front of the sphere (glTF +Z faces the front camera)
    pos = pp.read_glb(clean)[0]
    front = np.flatnonzero(pos[faces].mean(1)[:, 2] > 0.8)[:6]
    flipped, _p, _f = _sphere(tmp_path, flip=front)
    assert len(_facing_away(flipped, views)) >= len(front)
    report = winding.face_the_cameras(flipped, tmp_path / "out.glb", views)
    assert report["flipped"] >= len(front)
    assert len(_facing_away(tmp_path / "out.glb", views)) == 0


def test_a_clean_model_is_left_as_it_is(tmp_path):
    views = _views(tmp_path)
    clean, _pos, _faces = _sphere(tmp_path)
    report = winding.face_the_cameras(clean, tmp_path / "out.glb", views)
    assert report["flipped"] == 0
    assert np.array_equal(pp.read_glb(clean)[2], pp.read_glb(tmp_path / "out.glb")[2])


def test_the_orient_step_runs_after_the_clean_up_and_before_pixel_match():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rr", Path(__file__).resolve().parents[1] / "scripts" / "retopo_repaint.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    assert rr.stage_plan(skip_paint=True, skip_compress=False, photo=True, calm=True) == [
        "retopologise", "calm", "orient", "photo", "bake", "compress"]
    assert "orient" not in rr.stage_plan(skip_paint=True, skip_compress=False, photo=False)
    assert "orient" not in rr.stage_plan(skip_paint=True, skip_compress=False, photo=True, keep_mesh=True)


def test_turned_round_triangles_are_lit_the_way_they_now_face(tmp_path):
    views = _views(tmp_path)
    clean, _pos, faces = _sphere(tmp_path)
    pos = pp.read_glb(clean)[0]
    front = np.flatnonzero(pos[faces].mean(1)[:, 2] > 0.8)[:6]
    flipped, _p, _f = _sphere(tmp_path, flip=front, normals=True)
    winding.face_the_cameras(flipped, tmp_path / "out.glb", views)
    doc, binary = pp._split_glb((tmp_path / "out.glb").read_bytes())
    prim = doc["meshes"][0]["primitives"][0]
    normals = sf._read(doc, binary, prim["attributes"]["NORMAL"])
    out = sf._read(doc, binary, prim["indices"]).reshape(-1, 3)
    assert (normals[out[front]][..., 2] > 0.5).all(), "front triangles must be lit facing the front"
