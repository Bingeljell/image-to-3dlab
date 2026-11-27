"""Verify CPU mesh extraction and standalone paint export helpers."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hunyuan_mlx.shape.hy3dmlx.pipeline import Hunyuan3DShapePipeline


def test_grid_to_mesh_scales_surface_to_world_bounds():
    grid = np.broadcast_to(np.array([-1, 0, 1])[:, None, None], (3, 4, 5))
    low = np.array([-2, 1, 10])
    high = np.array([4, 7, 14])
    mesh = Hunyuan3DShapePipeline._grid_to_mesh(grid, low, high, (2, 3, 4), 0)
    np.testing.assert_allclose(mesh.bounds, [[1, 1, 10], [1, 7, 14]])
    assert mesh.area == pytest.approx(24)
    assert mesh.is_winding_consistent
    assert mesh.nondegenerate_faces().all()
    # Check the exported orientation as well as the location of the plane.
    np.testing.assert_array_equal(mesh.face_normals[:, 0], -np.ones(len(mesh.faces)))


def test_grid_to_mesh_rejects_missing_level():
    grid = np.arange(27).reshape(3, 3, 3)
    with pytest.raises(ValueError, match="Surface level"):
        Hunyuan3DShapePipeline._grid_to_mesh(grid, np.zeros(3), np.ones(3), (2, 2, 2), 50)


@pytest.fixture(params=[("run_paint", "prep_img"), ("run_paint_pbr", "prep_rgb")])
def paint_entry(request, monkeypatch, tmp_path):
    name, prep_function = request.param
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PAINT_OUT", raising=False)
    module = importlib.import_module(f"hunyuan_mlx.paint.scripts.{name}")
    return module, getattr(module, prep_function)


def test_paint_prepares_rgb_and_alpha_against_white(paint_entry):
    _, prepare = paint_entry
    rgb = prepare(Image.new("RGB", (1, 1), (0, 64, 255)), size=2)
    alpha = prepare(Image.new("RGBA", (1, 1), (0, 0, 255, 128)), size=2)
    assert rgb.shape == alpha.shape == (2, 2, 3)
    assert rgb.dtype == alpha.dtype == np.float32
    np.testing.assert_allclose(rgb[0, 0], [0, 64 / 255, 1])
    np.testing.assert_allclose(alpha[0, 0], [127 / 255, 127 / 255, 1])


def test_standalone_paint_export_writes_provenance(paint_entry, monkeypatch, tmp_path):
    module, _ = paint_entry
    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(image)
    monkeypatch.setattr(module, "IMG", str(image))
    repo = module._repo_root()
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo])
    out = Path(module._export_dir())
    assert sys.path[0] == repo
    assert out.parent == tmp_path / "outputs"
    assert out.name.startswith("paint-")
    model = out / "result.glb"
    model.write_bytes(b"mesh")
    sys.path.remove(repo)
    module._write_sidecar(str(model))
    assert sys.path[0] == repo
    data = json.loads(model.with_suffix(".provenance.json").read_text())
    assert data["backend"] in ("hunyuan-mlx-paint", "hunyuan-mlx-paint-pbr")
    assert data["input"]["path"] == str(image)
    assert data["output"]["path"] == str(model)
    assert len(data["components"]) == 1
    assert "paint-stage" in data["components"][0]["purpose"]


def test_parent_chosen_output_leaves_provenance_to_parent(paint_entry, monkeypatch, tmp_path):
    module, _ = paint_entry
    chosen = tmp_path / "parent" / "paint"
    monkeypatch.setenv("PAINT_OUT", str(chosen))
    out = Path(module._export_dir())
    assert out == chosen
    assert out.is_dir()
    model = out / "result.glb"
    model.write_bytes(b"mesh")
    module._write_sidecar(str(model))
    assert not model.with_suffix(".provenance.json").exists()


def test_pbr_export_preserves_geometry_uv_and_both_texture_slots(tmp_path, monkeypatch):
    import trimesh

    from hunyuan_mlx.paint.scripts import run_paint_pbr

    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0., 1.], [1., 1.], [0., 0.]])
    albedo = np.ones((2, 2, 3), np.float32) * [1, 0, 0]
    mr = np.ones((2, 2, 3), np.float32) * [0, 0.5, 1]
    monkeypatch.setenv("PAINT_OUT", str(tmp_path))
    output = run_paint_pbr.export_textures(tmp_path, vertices, faces, uv, albedo, mr)
    assert Path(output).name == "textured_mesh_pbr.glb"
    assert not Path(output).with_suffix(".provenance.json").exists()
    mesh = trimesh.load(output, force="mesh", process=False)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.visual.uv, [[0, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(uv, [[0, 1], [1, 1], [0, 0]])
    material = mesh.visual.material
    assert material.roughnessFactor == material.metallicFactor == 1
    for filename, texture, expected in (
        ("pbr_albedo_texture.png", material.baseColorTexture, [255, 0, 0]),
        ("pbr_mr_texture.png", material.metallicRoughnessTexture, [0, 127, 255]),
    ):
        with Image.open(tmp_path / filename) as png:
            assert list(png.getpixel((0, 0))) == expected
        assert list(texture.getpixel((0, 0)))[:3] == expected


def test_rgb_export_preserves_geometry_uv_and_texture(tmp_path, monkeypatch):
    import trimesh

    from hunyuan_mlx.paint.scripts import run_paint

    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0., 1.], [1., 1.], [0., 0.]])
    texture = np.broadcast_to([0, 1, 0], (2, 2, 3))
    monkeypatch.setenv("PAINT_OUT", str(tmp_path))
    output = run_paint.export_texture(tmp_path, vertices, faces, uv, texture)
    assert Path(output).name == "textured_mesh.glb"
    mesh = trimesh.load(output, force="mesh", process=False)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.visual.uv, [[0, 0], [1, 0], [0, 1]])
    assert list(mesh.visual.material.baseColorTexture.getpixel((0, 0)))[:3] == [0, 255, 0]
    with Image.open(tmp_path / "paint_texture.png") as png:
        assert png.getpixel((0, 0)) == (0, 255, 0)


def test_shape_model_selects_flow_or_consistency_schedule_from_config():
    from hunyuan_mlx.shape.hy3dmlx.pipeline import sampling_schedule

    flow, _ = sampling_schedule({}, 3)
    np.testing.assert_array_equal(flow, [0, 0.5, 1])
    config = {"scheduler": {"target": "schedulers.ConsistencyFlowMatchEulerDiscreteScheduler",
                            "params": {"pcm_timesteps": 4}}}
    turbo, terminal = sampling_schedule(config, 3)
    np.testing.assert_allclose(turbo, np.array([0, 249, 499]) / 999)
    assert terminal[-1] == 1
    config["scheduler"].pop("params")
    default_turbo, _ = sampling_schedule(config, 3)
    np.testing.assert_allclose(default_turbo, np.array([0, 329, 659]) / 999)
