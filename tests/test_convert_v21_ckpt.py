"""Tests for the v2-1 checkpoint converter.

The converter normally runs in an "oracle venv" with torch/safetensors, but the repo
venv only has torch, so each test mocks both libraries via ``monkeypatch.setitem``.
No real or malicious checkpoint is deserialized.
"""
from __future__ import annotations

import importlib.util
import pickle
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


def _load_converter_once():
    """Import the converter once; it lazy-imports torch/safetensors at call time."""
    spec = importlib.util.spec_from_file_location("convert_v21_ckpt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_v21_ckpt"] = module
    spec.loader.exec_module(module)
    return module


CVT = _load_converter_once()


@pytest.fixture
def converter_mocks(monkeypatch):
    """Replace torch/safetensors in ``sys.modules`` for one test, restoring after."""
    torch_mod = MagicMock()
    torch_mod.is_tensor = lambda v: isinstance(v, _FakeTensor)
    torch_mod.load = MagicMock()

    safetensors_pkg = MagicMock()
    safetensors_torch = MagicMock()
    safetensors_torch.save_file = MagicMock()
    safetensors_pkg.torch = safetensors_torch

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_pkg)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    yield torch_mod, safetensors_torch


def _make_ckpt(tmp_path: Path) -> Path:
    ckpt = tmp_path / "input.ckpt"
    ckpt.write_text("dummy")
    return ckpt


def test_convert_checkpoint_uses_explicit_paths_and_weights_only_true(
    tmp_path, capsys, converter_mocks
):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "nested" / "output.safetensors"

    fake = {
        "model": {"a": _FakeTensor("a"), "b": _FakeTensor("b")},
        "vae": {"c": _FakeTensor("c", floating=False)},
    }
    torch_mod.load.return_value = fake

    assert CVT.convert_checkpoint(ckpt, out) == 0

    torch_mod.load.assert_called_once_with(ckpt, map_location="cpu", weights_only=True)
    save_mock.save_file.assert_called_once()
    saved, saved_path = save_mock.save_file.call_args[0]
    assert saved_path == str(out)
    assert set(saved.keys()) == {"model.a", "model.b", "vae.c"}
    assert saved["model.a"].halved
    assert saved["model.a"].contiguized
    assert not saved["vae.c"].halved
    assert saved["vae.c"].contiguized

    printed = capsys.readouterr().out
    assert str(ckpt) in printed
    assert str(out) in printed


def test_convert_checkpoint_fails_closed_on_unpickling_error(tmp_path, converter_mocks):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"
    torch_mod.load.side_effect = pickle.UnpicklingError("pickle bomb")

    with pytest.raises(RuntimeError, match="weights_only=True"):
        CVT.convert_checkpoint(ckpt, out)

    assert torch_mod.load.call_count == 1
    assert torch_mod.load.call_args.kwargs["weights_only"] is True
    save_mock.save_file.assert_not_called()


def test_convert_checkpoint_does_not_retry_with_allow_pickle_false(tmp_path, converter_mocks):
    torch_mod, _save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"
    torch_mod.load.side_effect = pickle.UnpicklingError("pickle bomb")

    with pytest.raises(RuntimeError):
        CVT.convert_checkpoint(ckpt, out, allow_pickle=False)

    assert torch_mod.load.call_count == 1


def test_non_unpickling_errors_propagate_unchanged(tmp_path, converter_mocks):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"
    torch_mod.load.side_effect = OSError("disk failed")

    with pytest.raises(OSError, match="disk failed"):
        CVT.convert_checkpoint(ckpt, out)

    assert torch_mod.load.call_count == 1
    save_mock.save_file.assert_not_called()


def test_convert_checkpoint_warns_and_retries_with_allow_pickle(tmp_path, capsys, converter_mocks):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    fake = {"model": {"a": _FakeTensor("a")}}
    torch_mod.load.side_effect = [pickle.UnpicklingError("pickle bomb"), fake]

    assert CVT.convert_checkpoint(ckpt, out, allow_pickle=True) == 0

    assert torch_mod.load.call_count == 2
    assert torch_mod.load.call_args_list[0].kwargs["weights_only"] is True
    assert torch_mod.load.call_args_list[1].kwargs["weights_only"] is False
    save_mock.save_file.assert_called_once()

    printed = capsys.readouterr().out
    assert "retrying False" in printed


def test_convert_checkpoint_unwraps_state_dict_wrapper(tmp_path, converter_mocks):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    inner = {"model": {"a": _FakeTensor("a")}}
    torch_mod.load.return_value = {"state_dict": inner}

    assert CVT.convert_checkpoint(ckpt, out) == 0
    saved, _ = save_mock.save_file.call_args[0]
    assert "model.a" in saved


def test_convert_checkpoint_handles_flat_layout(tmp_path, converter_mocks):
    torch_mod, save_mock = converter_mocks
    ckpt = _make_ckpt(tmp_path)
    out = tmp_path / "output.safetensors"

    torch_mod.load.return_value = {
        "a": _FakeTensor("a"),
        "b": _FakeTensor("b", floating=False),
    }

    assert CVT.convert_checkpoint(ckpt, out) == 0
    saved, _ = save_mock.save_file.call_args[0]
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
