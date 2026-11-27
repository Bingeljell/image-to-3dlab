"""Verify local model selection and loading without fetching weights."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from image_to_3dlab import upscale, weight_gate


@pytest.mark.parametrize("key", ["params_ema", "params", None])
def test_upscaler_reads_only_the_selected_checkpoint(tmp_path, key):
    path = tmp_path / "small.pth"
    values = {"bias": torch.tensor([1., 2.])}
    other = {"bias": torch.tensor([9., 9.])}
    payload = {"params": other, key: values} if key == "params_ema" else {key: values} if key else values
    torch.save(payload, path)
    result = upscale.state_dict(path)
    assert set(result) == {"bias"}
    torch.testing.assert_close(result["bias"], values["bias"])


def test_upscaler_npz_loader_uses_local_state_and_evaluation_mode(tmp_path, monkeypatch):
    path = tmp_path / "small.npz"
    np.savez(path, weight=np.array([[2.]], np.float32))
    monkeypatch.setattr(upscale, "network", lambda: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = upscale.load(path)
    assert not result.training
    assert result.weight.device.type == "cpu"
    torch.testing.assert_close(result(torch.tensor([[3.]])), torch.tensor([[6.]]))


@pytest.mark.parametrize(("mps", "cuda", "device"), [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")])
def test_upscaler_device_prefers_gpu_when_available(monkeypatch, mps, cuda, device):
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: mps))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert upscale.device().type == device


def test_upscaler_missing_weights_refuses_local_load(tmp_path):
    with pytest.raises(FileNotFoundError, match="67 MB"):
        upscale.load(tmp_path / "absent.pth")


@pytest.mark.parametrize("ref", ["active", "../outside", "missing", None])
def test_model_cache_prefers_checked_reference_then_newest_complete_snapshot(tmp_path, monkeypatch, ref):
    model = tmp_path / "models--org--demo"
    snapshots = model / "snapshots"
    snapshots.mkdir(parents=True)
    for name, mtime in (("active", 1), ("new", 2), ("outside", 3), ("empty", 4)):
        folder = (model if name == "outside" else snapshots) / name
        folder.mkdir()
        if name != "empty":
            (folder / "weights").touch()
        os.utime(folder, (mtime, mtime))
    if ref is not None:
        (model / "refs").mkdir()
        (model / "refs/main").write_text(ref)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    result = weight_gate.require_local("org/demo", backend="Demo", route="local", size="0 GB", hub=tmp_path)
    assert result == str(snapshots / ("active" if ref == "active" else "new"))
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_empty_relative_model_is_not_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty").mkdir()
    assert weight_gate.snapshot_dir("empty", tmp_path) is None


def test_missing_weights_can_name_how_to_install_them():
    with pytest.raises(SystemExit, match=r"Demo \(local\).*0 GB.*\nOpen Setup"):
        weight_gate.refuse_hub_download(backend="Demo", route="local", size="0 GB", detail="Open Setup")


def test_unknown_remover_is_refused_before_resolving_any_file():
    from image_to_3dlab.matte import _verified_model_file

    with pytest.raises(RuntimeError, match="No verified offline background remover"):
        _verified_model_file("unreviewed-model")


def test_missing_cache_env_uses_default_huggingface_home(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert weight_gate.hub_dir() == tmp_path / ".cache/huggingface/hub"
