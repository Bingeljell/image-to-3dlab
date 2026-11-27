"""Exercise the Hunyuan launchers offline with CPU MLX and local test models."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from mlx import core as mx
from PIL import Image

pytestmark = pytest.mark.mlx_unit

REPO = Path(__file__).resolve().parents[2]
SHAPE_ROOT = REPO / "hunyuan_mlx" / "shape"
RUNNER = Path(__file__).with_name("acceptance_launchers") / "launch.py"


@pytest.fixture
def launch_inputs(tmp_path):
    assert mx.default_device() == mx.Device(mx.cpu)
    image = tmp_path / "front.png"
    Image.new("RGB", (4, 4), (255, 0, 128)).save(image)
    return image, tmp_path / "result.glb"


def launch(mode, cwd, *args):
    # Start from a small environment. The shared process guard adds offline and coverage settings.
    env = {
        "PATH": os.defpath,
        "HOME": str(Path(args[0]).parent),
        "PYTHONUNBUFFERED": "1",
    }
    return subprocess.run(
        [sys.executable, str(RUNNER), mode, *map(str, args)],
        cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=45,
    )


def read_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    assert len(mesh.vertices) >= 3
    assert np.isfinite(mesh.vertices).all()
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    assert len(mesh.faces) > 0
    assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
    assert (mesh.area_faces > 0).all()
    return mesh


def read_record(image, output, backend):
    record = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == 1 and record["backend"] == backend
    assert record["input"]["path"] == str(image)
    assert record["output"]["path"] == str(output)
    for key, path in (("input", image), ("output", output)):
        assert record[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def test_shape_module_exports_mesh_and_shape_only_provenance(launch_inputs):
    image, output = launch_inputs
    proc = launch("shape", SHAPE_ROOT, image, output, "valid")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "shape model inference on Device(cpu, 0)" in proc.stdout
    assert "[denoise] 3 steps" in proc.stdout
    mesh = read_mesh(output)
    np.testing.assert_allclose(mesh.bounds, [[-1.01, -1.01, 0], [1.01, 1.01, 0]], atol=1e-6)
    record = read_record(image, output, "hunyuan-mlx-shape")
    assert [entry["purpose"] for entry in record["components"]] == ["shape-stage diffusion model"]
    assert record["components"][0]["weights"] == str(image.parent / "shape-weights")
    assert record["components"][0]["commercial_status"] == "territory-restricted"


def test_shape_module_refuses_empty_model_directory_before_inference(launch_inputs):
    image, output = launch_inputs
    proc = launch("shape", SHAPE_ROOT, image, output, "empty")

    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr and "shape-weights/config.yaml" in proc.stderr
    assert "shape model inference" not in proc.stdout
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_paint_wrapper_child_exports_bound_material_and_painted_provenance(launch_inputs):
    image, output = launch_inputs
    proc = launch("wrapper", REPO, image, output, "valid")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "paint model inference on Device(cpu, 0)" in proc.stdout
    assert "[2.1 PBR] res=8 steps=1" in proc.stdout
    mesh = read_mesh(output)
    np.testing.assert_allclose(mesh.bounds, [[-1.01, -1.01, 0], [1.01, 1.01, 0]], atol=1e-6)
    assert mesh.visual.uv.shape == (len(mesh.vertices), 2)
    assert np.isfinite(mesh.visual.uv).all()
    material = mesh.visual.material
    for texture, color in (
        (material.baseColorTexture, [51, 153, 255]),
        (material.metallicRoughnessTexture, [255, 0, 76]),
    ):
        assert texture.size == (9, 9)
        assert texture.getpixel((4, 4))[:3] == pytest.approx(color, abs=1)
    record = read_record(image, output, "hunyuan-mlx-xiong")
    assert [entry["purpose"] for entry in record["components"] if entry["license"] != "MIT"] == [
        "shape-stage diffusion model", "paint-stage texture generation model",
    ]
    parameters = record["model"]["parameters"]
    assert (parameters["paint_seed"], parameters["paint_res"], parameters["paint_steps"],
            parameters["paint_tex"]) == (7, 8, 1, 9)


def test_paint_wrapper_child_refuses_bad_local_config_before_inference(launch_inputs):
    image, output = launch_inputs
    proc = launch("wrapper", REPO, image, output, "bad-config")

    assert proc.returncode != 0
    assert "JSONDecodeError" in proc.stderr and "paint stage exited with code" in proc.stderr
    assert "paint model inference" not in proc.stdout
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()
    assert not output.with_suffix(".json").exists()
    assert not list((image.parent / "paint").rglob("textured_mesh_pbr.glb"))
