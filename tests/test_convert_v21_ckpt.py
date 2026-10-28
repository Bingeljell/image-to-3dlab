"""Tests for the v2-1 checkpoint converter.

The converter normally runs in an "oracle venv" with torch/safetensors, but the repo
venv only has torch, so the test mocks both libraries via ``sys.modules`` before loading
the script. No real or malicious checkpoint is deserialized.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "hunyuan_mlx" / "shape" / "scripts" / "convert_v21_ckpt.py"


class _FakeTensor:
    """Minimal tensor stand-in for the half/contiguous pass."""

    def __init__(self, name: str, *, floating: bool = True):
        self.name = name
        self._floating = floating
        self.halved = False
        self.contiguized = False

    def is_floating_point(self) -> bool:
        return self._floating

    def half(self):
        self.halved = True
        return self

    def contiguous(self):
        self.contiguized = True
        return self


def _load_converter():
    """Import the converter with torch/safetensors mocked."""
    torch_mod = MagicMock()
    torch_mod.is_tensor = lambda v: isinstance(v, _FakeTensor)
    torch_mod.load = MagicMock()

    safetensors_pkg = MagicMock()
    safetensors_torch = MagicMock()
    safetensors_torch.save_file = MagicMock()
    safetensors_pkg.torch = safetensors_torch

    sys.modules["torch"] = torch_mod
    sys.modules["safetensors"] = safetensors_pkg
    sys.modules["safetensors.torch"] = safetensors_torch

    spec = importlib.util.spec_from_file_location("convert_v21_ckpt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_v21_ckpt"] = module
    spec.loader.exec_module(module)
    return module, torch_mod, safetensors_torch


CVT, TORCH_MOCK, SAVE_MOCK = _load_converter()


@pytest.fixture(autouse=True)
def _reset_mocks():
    TORCH_MOCK.reset_mock()
    TORCH_MOCK.load.side_effect = None
    SAVE_MOCK.reset_mock()
    SAVE_MOCK.save_file.side_effect = None
    yield


def _make_ckpt(tmp_path: Path) -> Path:
    ckpt = tmp_path / "input.ckpt"
    ckpt.write_text("dummy")
    return ckpt


def test_convert_checkpoint_uses_explicit_paths_and_weights_only_true(tmp_path, capsys):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "nested" / "output.safetensors"

    fake = {
        "model": {"a": _FakeTensor("a"), "b": _FakeTensor("b")},
        "vae": {"c": _FakeTensor("c", floating=False)},
    }
    TORCH_MOCK.load.return_value = fake

    assert CVT.convert_checkpoint(ckpt, out) == 0

    TORCH_MOCK.load.assert_called_once_with(ckpt, map_location="cpu", weights_only=True)
    SAVE_MOCK.save_file.assert_called_once()
    saved, saved_path = SAVE_MOCK.save_file.call_args[0]
    assert saved_path == str(out)
    assert set(saved.keys()) == {"model.a", "model.b", "vae.c"}
    assert saved["model.a"].halved
    assert saved["model.a"].contiguized
    assert not saved["vae.c"].halved
    assert saved["vae.c"].contiguized

    printed = capsys.readouterr().out
    assert str(ckpt) in printed
    assert str(out) in printed


def test_convert_checkpoint_fails_closed_on_pickle_error(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"
    TORCH_MOCK.load.side_effect = Exception("pickle bomb")

    with pytest.raises(RuntimeError, match="weights_only=True"):
        CVT.convert_checkpoint(ckpt, out)

    assert TORCH_MOCK.load.call_count == 1
    assert TORCH_MOCK.load.call_args.kwargs["weights_only"] is True
    SAVE_MOCK.save_file.assert_not_called()


def test_convert_checkpoint_does_not_retry_with_allow_pickle_false(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"
    TORCH_MOCK.load.side_effect = Exception("pickle bomb")

    with pytest.raises(RuntimeError):
        CVT.convert_checkpoint(ckpt, out, allow_pickle=False)

    assert TORCH_MOCK.load.call_count == 1


def test_convert_checkpoint_warns_and_retries_with_allow_pickle(tmp_path, capsys):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    fake = {"model": {"a": _FakeTensor("a")}}
    TORCH_MOCK.load.side_effect = [Exception("pickle bomb"), fake]

    assert CVT.convert_checkpoint(ckpt, out, allow_pickle=True) == 0

    assert TORCH_MOCK.load.call_count == 2
    assert TORCH_MOCK.load.call_args_list[0].kwargs["weights_only"] is True
    assert TORCH_MOCK.load.call_args_list[1].kwargs["weights_only"] is False
    SAVE_MOCK.save_file.assert_called_once()

    printed = capsys.readouterr().out
    assert "retrying False" in printed


def test_convert_checkpoint_unwraps_state_dict_wrapper(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    inner = {"model": {"a": _FakeTensor("a")}}
    TORCH_MOCK.load.return_value = {"state_dict": inner}

    assert CVT.convert_checkpoint(ckpt, out) == 0
    saved, _ = SAVE_MOCK.save_file.call_args[0]
    assert "model.a" in saved


def test_convert_checkpoint_handles_flat_layout(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    TORCH_MOCK.load.return_value = {"a": _FakeTensor("a"), "b": _FakeTensor("b", floating=False)}

    assert CVT.convert_checkpoint(ckpt, out) == 0
    saved, _ = SAVE_MOCK.save_file.call_args[0]
    assert set(saved.keys()) == {"a", "b"}


def test_cli_defaults_match_documented_no_arg_invocation():
    args = CVT.build_parser().parse_args([])
    assert args.input == CVT.DEFAULT_CKPT
    assert args.output == CVT.DEFAULT_OUT
    assert args.allow_pickle is False


def test_cli_positional_args_override_defaults(tmp_path):
    args = CVT.build_parser().parse_args([
        str(tmp_path / "in.ckpt"),
        str(tmp_path / "out.safetensors"),
    ])
    assert args.input == str(tmp_path / "in.ckpt")
    assert args.output == str(tmp_path / "out.safetensors")


def test_cli_allow_pickle_flag():
    args = CVT.build_parser().parse_args(["in.ckpt", "out.safetensors", "--allow-pickle"])
    assert args.allow_pickle is True
