"""Tests for keeping a model's cameras and pictures inside its GLB."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import glb_views, photo_paint


def _glb() -> bytes:
    box = trimesh.creation.box()
    return trimesh.Scene(box).export(file_type="glb")


def _views(folder, sides=(0, 180), missing=()):
    folder.mkdir()
    rng = np.random.default_rng(1)
    frames = []
    for angle in sides:
        name = f"view_{angle:03d}.png"
        pixels = rng.integers(0, 256, (24, 24, 4), dtype=np.uint8)
        pixels[:4, :, 3] = 0                      # see-through rows whose colour must survive too
        if angle not in missing:
            Image.fromarray(pixels).save(folder / name)
        frames.append({"file_path": name, "camera_angle_x": 0.35,
                       "transform_matrix": np.eye(4).tolist(), "paint": "all"})
    (folder / "transforms.json").write_text(json.dumps({"camera_angle_x": 0.35, "mesh_scale": 1,
                                                        "frames": frames}))
    return folder


def test_cameras_and_pictures_come_back_exactly(tmp_path):
    views = _views(tmp_path / "m.svviews")
    data = glb_views.embed(_glb(), views)
    assert glb_views.has_views(data) and not glb_views.has_views(_glb())
    assert glb_views.extract(data, tmp_path / "out")
    meta = json.loads((tmp_path / "out" / "transforms.json").read_text())
    assert [f["file_path"] for f in meta["frames"]] == ["view_000.webp", "view_180.webp"]
    assert meta["frames"][0]["paint"] == "all" and meta["mesh_scale"] == 1
    for angle in (0, 180):
        before = np.asarray(Image.open(views / f"view_{angle:03d}.png"))
        after = np.asarray(Image.open(tmp_path / "out" / f"view_{angle:03d}.webp").convert("RGBA"))
        assert np.array_equal(before, after)
    loaded, scale = photo_paint.load_views(tmp_path / "out")
    assert len(loaded) == 2 and scale == 1


def test_the_model_still_opens_unchanged(tmp_path):
    plain = _glb()
    data = glb_views.embed(plain, _views(tmp_path / "m.svviews"))
    before = trimesh.load(io.BytesIO(plain), file_type="glb").to_geometry()
    after = trimesh.load(io.BytesIO(data), file_type="glb").to_geometry()
    assert np.allclose(before.vertices, after.vertices) and np.array_equal(before.faces, after.faces)
    doc, binary = glb_views._split(data)
    assert doc["buffers"][0]["byteLength"] == len(binary)
    assert len(data) % 4 == 0


def test_a_camera_without_its_picture_is_left_out(tmp_path):
    data = glb_views.embed(_glb(), _views(tmp_path / "m.svviews", missing=(180,)))
    glb_views.extract(data, tmp_path / "out")
    assert [f["file_path"] for f in json.loads((tmp_path / "out" / "transforms.json").read_text())["frames"]] \
        == ["view_000.webp"]
    with pytest.raises(ValueError):
        glb_views.embed(_glb(), _views(tmp_path / "none.svviews", missing=(0, 180)))


def test_a_model_without_cameras_extracts_nothing(tmp_path):
    assert glb_views.extract(_glb(), tmp_path / "out") is False
    assert not (tmp_path / "out").exists()
