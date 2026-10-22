"""The Hero face upscaler: Real-ESRGAN x4plus in the project's own PyTorch.

Its output was checked against the MLX copy it replaces (mean 0.5 of 255 apart, max 4);
these tests pin what can be checked without the 67 MB weights: the network's shape, tiling,
and an installer that never keeps a bad file or downloads without a yes.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from image_to_3dlab import upscale
from scripts import bootstrap_upscaler


def test_network_matches_the_release_s_weights_layout():
    keys = upscale.network().state_dict().keys()
    assert len(keys) == 702                                   # the release has 702 tensors
    assert "conv_first.weight" in keys and "body.22.rdb3.conv5.bias" in keys


def test_tiles_join_without_seams():
    torch.manual_seed(0)
    model = upscale.network().eval()
    picture = np.random.default_rng(0).random((40, 36, 3)).astype(np.float32)
    whole = upscale.upscale(model, picture, tile=64)
    tiled = upscale.upscale(model, picture, tile=16, overlap=10)
    assert whole.shape == (160, 144, 3)
    assert np.abs(whole - tiled).max() < 1e-3


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_a_rejected_download_leaves_nothing_behind(tmp_path):
    target = tmp_path / "RealESRGAN_x4plus.pth"
    with pytest.raises(SystemExit, match="nothing was installed"):
        bootstrap_upscaler.download(target, opener=lambda url: _Response(b"not weights"),
                                    check=lambda path: "wrong size")
    assert not target.exists() and not list(tmp_path.iterdir())


def test_a_good_download_is_moved_into_place(tmp_path):
    target = tmp_path / "w" / "RealESRGAN_x4plus.pth"
    bootstrap_upscaler.download(target, opener=lambda url: _Response(b"weights"), check=lambda path: None)
    assert target.read_bytes() == b"weights"


def test_nothing_downloads_without_a_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(upscale, "WEIGHTS_FILE", tmp_path / "missing.pth")
    fetched = []
    assert bootstrap_upscaler.main([], ask=lambda prompt: "n", fetch=fetched.append) == 1
    assert bootstrap_upscaler.main([], ask=lambda prompt: (_ for _ in ()).throw(EOFError()), fetch=fetched.append) == 1
    assert fetched == []
    assert bootstrap_upscaler.main(["--yes"], fetch=fetched.append) == 0 and fetched
