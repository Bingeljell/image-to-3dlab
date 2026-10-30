import json
from argparse import Namespace
from pathlib import Path

import pytest

from image_to_3dlab.cli import (
    _extract_workflow_model_identity,
    _resolve_model_identity,
    main,
)


def test_extract_workflow_identity_finds_checkpoint_loader(tmp_path):
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "tencent/Hunyuan3D-2.1"},
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow))
    assert _extract_workflow_model_identity(path) == "tencent/Hunyuan3D-2.1"


def test_extract_workflow_identity_joins_multiple_loaders(tmp_path):
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "tencent/Hunyuan3D-2.1"},
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "another/checkpoint.safetensors"},
        },
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow))
    assert _extract_workflow_model_identity(path) == (
        "tencent/Hunyuan3D-2.1, another/checkpoint.safetensors"
    )


def test_extract_workflow_identity_returns_none_when_no_loader(tmp_path):
    workflow = {"2": {"class_type": "LoadImage", "inputs": {"image": "input.png"}}}
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow))
    assert _extract_workflow_model_identity(path) is None


def test_extract_workflow_identity_rejects_ui_workflow_array(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps([{"class_type": "CheckpointLoaderSimple"}]))
    with pytest.raises(ValueError, match="API-format object"):
        _extract_workflow_model_identity(path)


def test_resolve_model_identity_for_sf3d_uses_model_arg():
    args = Namespace(fast=True, trellis=False, quality=False, model="stabilityai/stable-fast-3d")
    assert _resolve_model_identity(args) == "stabilityai/stable-fast-3d"


def test_resolve_model_identity_for_trellis_is_canonical():
    args = Namespace(fast=False, trellis=True, quality=False, workflow=None)
    assert _resolve_model_identity(args) == "microsoft/TRELLIS.2-4B"


def test_resolve_model_identity_for_quality_reads_workflow(tmp_path):
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "tencent/Hunyuan3D-2.1"},
        }
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow))
    args = Namespace(
        fast=False, trellis=False, quality=True, workflow=path, model="unused"
    )
    assert _resolve_model_identity(args) == "tencent/Hunyuan3D-2.1"


def test_resolve_model_identity_for_quality_without_workflow_is_none():
    args = Namespace(
        fast=False, trellis=False, quality=True, workflow=None, model="unused"
    )
    assert _resolve_model_identity(args) is None


def test_main_with_missing_quality_workflow_returns_2_without_raising():
    result = main(
        ["--quality", "input.png", "--workflow", "does-not-exist-workflow.json"]
    )
    assert result == 2


def _hunyuan_manifest(tmp_path, workflow_body: dict, *, allow_unverified: bool = False) -> Path:
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    (tmp_path / "wf.json").write_text(json.dumps(workflow_body))
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "use_case": "showcase",
        "distribution": "private",
        "commercial_intent": False,
        "input": {"path": "in.png"},
        "workflow": "wf.json",
        "model": {"backend": "hunyuan-comfyui"},
        "license_policy": {"allow_conditional": True, "allow_unverified": allow_unverified},
        "output": {"directory": str(tmp_path / "out")},
    }))
    return manifest


def test_manifest_workflow_is_checked_before_a_working_dir(tmp_path, monkeypatch, capsys):
    manifest = _hunyuan_manifest(tmp_path, {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "other/model"},
        },
    })

    def boom(*_args, **_kwargs):
        raise AssertionError("generate must not run")

    monkeypatch.setattr("image_to_3dlab.comfyui_backend.ComfyUIClient.generate", boom)
    assert main(["--run-manifest", str(manifest)]) == 2
    assert "other/model" in capsys.readouterr().err
    assert not (tmp_path / "out" / ".working").exists()


def test_unet_loader_workflow_does_not_get_a_tencent_sidecar(tmp_path, monkeypatch):
    manifest = _hunyuan_manifest(
        tmp_path,
        {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "some.safetensors"}}},
        allow_unverified=True,
    )

    def fake_generate(self, image, workflow_path, output_dir, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        asset = Path(output_dir) / "mesh.glb"
        asset.write_bytes(b"glb")
        return asset

    monkeypatch.setattr("image_to_3dlab.comfyui_backend.ComfyUIClient.generate", fake_generate)
    assert main(["--run-manifest", str(manifest)]) == 0
    sidecars = list((tmp_path / "out").rglob("*.provenance.json"))
    assert len(sidecars) == 1
    record = json.loads(sidecars[0].read_text())
    assert record["output"]["classification"] == "unverified"
    assert record["license"]["verified"] is False
    assert "Tencent" not in record["license"]["name"]


def test_bad_material_mode_refuses_before_generate(tmp_path, monkeypatch, capsys):
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "use_case": "showcase",
        "distribution": "private",
        "commercial_intent": False,
        "input": {"path": "in.png"},
        "model": {
            "backend": "trellis2",
            "id": "microsoft/TRELLIS.2-4B",
            "parameters": {"material_mode": "nope"},
        },
        "license_policy": {"allow_conditional": True},
        "output": {"directory": str(tmp_path / "out")},
    }))

    def boom(*_args, **_kwargs):
        raise AssertionError("generate_trellis must not run")

    monkeypatch.setattr("image_to_3dlab.trellis_backend.generate_trellis", boom)
    assert main(["--run-manifest", str(manifest)]) == 2
    assert "material_mode" in capsys.readouterr().err
    assert not (tmp_path / "out" / ".working").exists()
