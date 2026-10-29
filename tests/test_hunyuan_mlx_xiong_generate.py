"""Tests for scripts/hunyuan_mlx_xiong_generate.py licence sidecar."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import hunyuan_mlx_xiong_generate as xiong

from image_to_3dlab.provenance import sha256_file


class _Mesh:
    """Minimal mesh stand-in for the shape/remesh stubs."""

    def __init__(self) -> None:
        self.vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.faces = [(0, 1, 2)]

    def export(self, path: str) -> None:
        Path(path).write_text("stub mesh")


def _png(path: Path) -> Path:
    """Write a tiny but valid-enough PNG for sha256 checks."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return path


def test_write_provenance_creates_sidecar_with_expected_licences(tmp_path):
    image = _png(tmp_path / "in.png")
    output = tmp_path / "out.glb"
    output.write_bytes(b"glb bytes")
    manifest = {
        "schema_version": 1,
        "backend": "hunyuan-mlx-xiong",
        "parameters": {"model": "2.0"},
    }

    sidecar = xiong.write_provenance(output, image, manifest)

    assert sidecar.is_file()
    record = json.loads(sidecar.read_text())
    assert record["schema_version"] == 1
    assert record["backend"] == "hunyuan-mlx-xiong"
    assert record["input"]["sha256"] == sha256_file(image)
    assert record["output"]["sha256"] == sha256_file(output)
    assert record["model"]["parameters"]["model"] == "2.0"
    assert record["license"] == {
        "name": xiong.TENCENT_LICENSE_NAME,
        "url": xiong.TENCENT_LICENSE_URL,
        "conditions": list(xiong.TENCENT_CONDITIONS),
    }

    mit_component = {
        "component": "ZimengXiong/hunyuan3d-mlx (in-repo shape+paint port)",
        "purpose": "MLX generation code tracked in this repository",
        "license": "MIT",
        "license_url": xiong.MIT_LICENSE_PATH,
        "commercial_status": "commercial-clear",
    }
    tencent_component = {
        "license": xiong.TENCENT_LICENSE_NAME,
        "license_url": xiong.TENCENT_LICENSE_URL,
        "commercial_status": "territory-restricted",
        "conditions": list(xiong.TENCENT_CONDITIONS),
    }
    assert record["components"] == [
        mit_component,
        {
            "component": "Tencent Hunyuan3D-2/2.1 shape weights",
            "purpose": "shape-stage diffusion model",
            **tencent_component,
        },
        {
            "component": "Tencent Hunyuan3D-2/2.1 paint weights (MLX-ready conversion)",
            "purpose": "paint-stage texture generation model",
            **tencent_component,
        },
    ]


def test_main_writes_manifest_and_provenance(tmp_path, monkeypatch):
    image = _png(tmp_path / "in.png")
    output = tmp_path / "out.glb"

    # Keep the main() body from importing real mlx/torch.
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = types.SimpleNamespace(clear_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx.core)

    calls = {}

    def fake_shape(*args, **kwargs):
        calls["shape"] = True
        return _Mesh()

    def fake_remesh(mesh, target, t0):
        calls["remesh"] = True
        return mesh, False

    def fake_paint(*args, **kwargs):
        calls["paint"] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"glb")

    monkeypatch.setattr(xiong, "run_shape", fake_shape)
    monkeypatch.setattr(xiong, "run_remesh", fake_remesh)
    monkeypatch.setattr(xiong, "run_paint", fake_paint)

    monkeypatch.setattr(
        sys,
        "argv",
        ["hunyuan_mlx_xiong_generate.py", str(image), str(output), "--model", "2.0"],
    )
    xiong.main()

    assert calls == {"shape": True, "remesh": True, "paint": True}
    assert (output.with_suffix(".json")).is_file()
    sidecar = output.with_suffix(".provenance.json")
    assert sidecar.is_file()
    record = json.loads(sidecar.read_text())
    assert record["backend"] == "hunyuan-mlx-xiong"
    assert record["input"]["sha256"] == sha256_file(image)
    assert record["output"]["sha256"] == sha256_file(output)
