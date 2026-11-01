"""Hunyuan exports write a licence sidecar beside the mesh."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hunyuan_mlx_generate as generate
import hunyuan_mlx_xiong_generate as xiong

from image_to_3dlab.provenance import LICENSES

TERRITORY = LICENSES["hunyuan-comfyui"]


class _Mesh:
    def __init__(self) -> None:
        self.vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.faces = [(0, 1, 2)]

    def export(self, path: str) -> None:
        Path(path).write_text("stub mesh")


def _png(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return path


def _licence(record: dict) -> None:
    assert record["schema_version"] == 1
    assert record["license"]["name"] == TERRITORY.license_name
    assert record["license"]["conditions"] == list(TERRITORY.conditions)


def test_generate_main_writes_a_licence_sidecar(tmp_path, monkeypatch):
    image = _png(tmp_path / "in.png")
    output = tmp_path / "out.glb"

    def fake_paint(*args, **kwargs):
        output.write_bytes(b"glb")

    monkeypatch.setattr(generate, "run_shape", lambda *args, **kwargs: _Mesh())
    monkeypatch.setattr(generate, "run_remesh", lambda mesh, target, t0: (mesh, False))
    monkeypatch.setattr(generate, "run_paint", fake_paint)
    monkeypatch.setattr(
        sys, "argv", ["hunyuan_mlx_generate.py", str(image), str(output)]
    )

    generate.main()

    sidecar = output.with_suffix(".provenance.json")
    assert sidecar.is_file()
    record = json.loads(sidecar.read_text())
    _licence(record)
    weights = [item.get("weights") for item in record["components"]]
    assert "dgrauet/hunyuan3d-2.1-mlx" in weights
    assert "weights/hunyuan3d-paintpbr-v2-1" in weights


def test_shape_only_sidecar_does_not_claim_paint(tmp_path):
    from image_to_3dlab.provenance import write_hunyuan_sidecar

    image = _png(tmp_path / "in.png")
    output = tmp_path / "shape.glb"
    output.write_bytes(b"glb")

    sidecar = write_hunyuan_sidecar(
        output,
        image,
        backend="hunyuan-mlx-shape",
        shape=True,
        shape_weights="weights/Hunyuan3D-2/hunyuan3d-dit-v2-0",
    )

    record = json.loads(sidecar.read_text())
    _licence(record)
    text = json.dumps(record["components"]).lower()
    assert "shape" in text
    assert "paint" not in text
    assert record["components"][0]["weights"] == "weights/Hunyuan3D-2/hunyuan3d-dit-v2-0"


def test_paint_sidecar_names_the_weights_it_loaded(tmp_path):
    from image_to_3dlab.provenance import write_hunyuan_sidecar

    image = _png(tmp_path / "in.png")
    output = tmp_path / "painted.glb"
    output.write_bytes(b"glb")

    sidecar = write_hunyuan_sidecar(
        output,
        image,
        backend="hunyuan-mlx-paint",
        paint=True,
        paint_weights="weights/hunyuan3d-paint-v2-0",
    )

    record = json.loads(sidecar.read_text())
    _licence(record)
    text = json.dumps(record["components"]).lower()
    assert "paint" in text
    assert "shape" not in text
    assert record["components"][0]["weights"] == "weights/hunyuan3d-paint-v2-0"


def test_two_paint_runs_do_not_share_an_output_file(tmp_path, monkeypatch):
    from image_to_3dlab.provenance import hunyuan_paint_run_dir

    monkeypatch.setattr("os.getpid", lambda: 111)
    first = hunyuan_paint_run_dir(tmp_path, "creature")
    monkeypatch.setattr("os.getpid", lambda: 222)
    second = hunyuan_paint_run_dir(tmp_path, "creature")

    assert first != second
    assert first / "textured_mesh_pbr.glb" != second / "textured_mesh_pbr.glb"
    assert first.parent == tmp_path
    assert second.parent == tmp_path


def _stub_paint(module, tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd, env, check):
        seen["out"] = env["PAINT_OUT"]
        dest = Path(env["PAINT_OUT"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "textured_mesh_pbr.glb").write_bytes(b"glb")
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(module, "PAINT_ROOT", tmp_path / "paint")
    monkeypatch.setattr(module, "PAINT_PYTHON", tmp_path / "python")
    monkeypatch.setattr(module, "PAINT_SCRIPT", tmp_path / "run_paint_pbr.py")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return seen


def test_generate_run_paint_reads_its_own_directory(tmp_path, monkeypatch):
    seen = _stub_paint(generate, tmp_path, monkeypatch)
    output = tmp_path / "final.glb"
    generate.run_paint(tmp_path / "mesh.obj", tmp_path / "in.png", output, 0, 512, 15, 4096, 0.0)

    assert output.read_bytes() == b"glb"
    shared = tmp_path / "paint" / "outputs" / "textured_mesh_pbr.glb"
    assert Path(seen["out"]) / "textured_mesh_pbr.glb" != shared


def test_xiong_run_paint_reads_its_own_directory(tmp_path, monkeypatch):
    seen = _stub_paint(xiong, tmp_path, monkeypatch)
    output = tmp_path / "final.glb"
    xiong.run_paint(tmp_path / "mesh.obj", tmp_path / "in.png", output, 0, 512, 15, 4096, 0.0)

    assert output.read_bytes() == b"glb"
    shared = tmp_path / "paint" / "outputs" / "textured_mesh_pbr.glb"
    assert Path(seen["out"]) / "textured_mesh_pbr.glb" != shared
