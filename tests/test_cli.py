import json
from argparse import Namespace

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
