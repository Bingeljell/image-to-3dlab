"""Exercise Hunyuan's local-weight guards without importing models."""

import os
import sys
from pathlib import Path

import pytest

from scripts import hunyuan_cuda_generate as cuda
from scripts import hunyuan_rocm_generate as rocm


def test_cuda_weight_check_uses_same_models_folder_as_loader(tmp_path, monkeypatch):
    default_models = tmp_path / "default-models"
    monkeypatch.setattr(cuda.bootstrap, "MODELS", default_models)
    vendor = tmp_path / "custom-vendor"
    for model in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        (default_models / cuda.bootstrap.HUNYUAN_REPO / model).mkdir(parents=True)
    absent = cuda.missing_weights(vendor, tmp_path / "hub", shape_only=True)
    assert len(absent) == 2
    expected = Path(cuda.configure_environment(vendor, {})["HY3DGEN_MODELS"])
    for model in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        (expected / cuda.bootstrap.HUNYUAN_REPO / model).mkdir(parents=True)
    assert cuda.missing_weights(vendor, tmp_path / "hub", shape_only=True) == []


@pytest.mark.parametrize("module", [cuda, rocm])
def test_enter_upstream_disables_hub_access(module, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HY3DGEN_MODELS", "fixture")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    module._enter_upstream(tmp_path)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
