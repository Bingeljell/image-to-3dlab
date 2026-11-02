"""Tests for the Hunyuan weight downloader's post-conversion cleanup.

`download_shape` fetches a `.ckpt`, converts it to `.safetensors`, and used to leave both
on disk. That stored the same 6.9 GB model twice for every user, and went unnoticed until
a disk audit on 2026-09-21.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "hunyuan_mlx" / "download_weights.py"
PAINT = REPO / "hunyuan_mlx" / "paint"


def _load():
    spec = importlib.util.spec_from_file_location("download_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_weights"] = module
    spec.loader.exec_module(module)
    return module


dw = _load()


def test_the_checkpoint_goes_once_the_conversion_exists(tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    converted = tmp_path / "model.fp16.safetensors"
    ckpt.write_bytes(b"\0" * 4096)
    converted.write_bytes(b"\0" * 4096)

    assert dw.discard_converted_checkpoint(ckpt, converted) is True
    assert not ckpt.exists()
    assert converted.is_file()


def test_an_unfinished_conversion_keeps_the_checkpoint(tmp_path):
    # Deleting the source next to a zero-byte output would leave nothing to retry from.
    ckpt = tmp_path / "model.fp16.ckpt"
    converted = tmp_path / "model.fp16.safetensors"
    ckpt.write_bytes(b"\0" * 4096)
    converted.write_bytes(b"")

    assert dw.discard_converted_checkpoint(ckpt, converted) is False
    assert ckpt.is_file()


def test_a_missing_conversion_keeps_the_checkpoint(tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    ckpt.write_bytes(b"\0" * 4096)
    assert dw.discard_converted_checkpoint(ckpt, tmp_path / "absent.safetensors") is False
    assert ckpt.is_file()


def test_rerunning_after_cleanup_is_harmless(tmp_path):
    converted = tmp_path / "model.fp16.safetensors"
    converted.write_bytes(b"\0" * 4096)
    assert dw.discard_converted_checkpoint(tmp_path / "model.fp16.ckpt", converted) is False


def test_the_2_1_route_is_the_only_one_that_converts():
    # 2.1 ships only a .ckpt on HF; the others ship safetensors directly, so only 2.1 has
    # a checkpoint to clean up afterwards.
    assert set(dw.SHAPE_HF_SOURCES) == {"2.1", "2.0", "2.0-turbo"}
    assert dw.SHAPE_HF_SOURCES["2.1"][0] == "tencent/Hunyuan3D-2.1"


def test_only_the_default_model_downloads_unless_all_is_asked_for():
    """The bug: `--model` defaulted to None and None meant *every* model.

    The README documents this exact command and calls it "~13 GB", while all three shape
    checkpoints plus paint come to ~24 GB. Nobody chose the extra two, and the default
    route never loads them.
    """
    assert dw.DEFAULT_MODEL == "2.0"
    bare = dw.build_parser().parse_args([])
    assert bare.model == "2.0"
    assert bare.all is False

    every = dw.build_parser().parse_args(["--all"])
    assert every.all is True

    one = dw.build_parser().parse_args(["--model", "2.1"])
    assert one.model == "2.1"


def test_the_default_route_costs_what_the_readme_says(capsys):
    total = dw.announce([dw.DEFAULT_MODEL], paint=True)
    assert 12.0 <= total <= 14.0, f"README promises ~13 GB, announce says {total}"
    printed = capsys.readouterr().out
    assert "13.3 GB" in printed
    # The territorial restriction is stated before anything is fetched, not afterwards.
    assert "EU" in printed and "South Korea" in printed


def test_every_model_is_visibly_more_expensive(capsys):
    total = dw.announce(sorted(dw.SHAPE_HF_SOURCES), paint=True)
    assert total > 24.0
    assert "total" in capsys.readouterr().out


def test_skipping_paint_is_reflected_in_the_total():
    with_paint = dw.announce([dw.DEFAULT_MODEL], paint=True)
    without = dw.announce([dw.DEFAULT_MODEL], paint=False)
    assert with_paint - without == dw.APPROX_GB["paint"]


def test_every_shape_model_has_a_stated_size():
    # A model that downloads silently because nobody gave it a number is the failure mode.
    for model in dw.SHAPE_HF_SOURCES:
        assert dw.APPROX_GB.get(model, 0) > 0, model


def test_paint_comes_from_the_mlx_ready_repo_at_a_pinned_revision():
    """Issue #65: Tencent's paint folder has no DINOv2, no v2-0 VAE and a .bin UNet, so
    a fresh install could shape but not paint. A branch name instead of a commit would let
    a re-upload change what new installs get without anyone noticing."""
    assert dw.PAINT_HF_REPO == "zimengxiong/hunyuan3d-mlx-paint-large"
    assert re.fullmatch(r"[0-9a-f]{40}", dw.PAINT_HF_REVISION)
    assert dw.PAINT_LOCAL_DIR == "hunyuan3d-paintpbr-v2-1"


def test_every_file_the_paint_script_opens_is_downloaded():
    # Read from the real paint script and RealESRGAN loader, not a copy: the bug was the
    # loader and the downloader drifting apart.
    script = (PAINT / "scripts" / "run_paint_pbr.py").read_text()
    assert 'PBR = "weights/hunyuan3d-paintpbr-v2-1"' in script
    opened = set(re.findall(r'\{PBR\}/([\w./-]+)"', script))
    assert opened, "the paint script no longer loads from {PBR}/; update this test"
    loader = (PAINT / "hy3dpaint_mlx" / "realesrgan.py").read_text()
    opened |= set(re.findall(r'"weights/hunyuan3d-paintpbr-v2-1/([\w./-]+)"', loader))
    assert "realesrgan/rrdbnet_mlx.safetensors" in opened
    missing = opened - set(dw.PAINT_FILES)
    assert not missing, f"paint loads files the downloader never fetches: {missing}"


def test_no_paint_weight_lives_outside_the_downloaded_folder():
    script = (PAINT / "scripts" / "run_paint_pbr.py").read_text()
    assert "hunyuan3d-paint-v2-0" not in script


def test_main_refuses_to_download_without_yes(monkeypatch, capsys):
    """A silent multi-gigabyte fetch is the failure mode; default answer is No."""
    calls = []
    monkeypatch.setattr(dw, "download_shape", lambda m: calls.append(("shape", m)))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr("builtins.input", lambda: "")

    assert dw.main() != 0
    assert calls == []
    out = capsys.readouterr().out
    assert "13.3 GB" in out
    assert "EU" in out and "South Korea" in out


def test_main_downloads_when_given_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(dw, "download_shape", lambda m: calls.append(("shape", m)))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--yes"])

    assert dw.main() == 0
    assert ("shape", dw.DEFAULT_MODEL) in calls
    assert "paint" in calls


def test_main_downloads_on_interactive_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(dw, "download_shape", lambda m: calls.append(("shape", m)))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr("builtins.input", lambda: "yes")

    assert dw.main() == 0
    assert ("shape", dw.DEFAULT_MODEL) in calls
    assert "paint" in calls


def test_model_2_1_refuses_before_download_when_the_converter_cannot_import_torch(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(dw, "download_shape", lambda model: calls.append(model))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(dw, "converter_can_run", lambda: False)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--yes", "--model", "2.1", "--skip-paint"])

    assert dw.main() == 1
    assert calls == []
    captured = capsys.readouterr()
    assert "convert_v21_ckpt.py" in captured.out + captured.err


def test_the_default_model_does_not_need_the_converter(monkeypatch):
    calls = []

    def boom():
        raise AssertionError("2.0 must not probe torch")

    monkeypatch.setattr(dw, "download_shape", lambda model: calls.append(model))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(dw, "converter_can_run", boom)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--yes", "--skip-paint"])

    assert dw.main() == 0
    assert calls == [dw.DEFAULT_MODEL]


def test_main_refuses_on_eof_without_yes(monkeypatch, capsys):
    """Non-interactive pipes have nobody to ask; --yes is required."""
    calls = []
    monkeypatch.setattr(dw, "download_shape", lambda m: calls.append(("shape", m)))
    monkeypatch.setattr(dw, "download_paint", lambda: calls.append("paint"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    def eof():
        raise EOFError()

    monkeypatch.setattr("builtins.input", eof)

    assert dw.main() != 0
    assert calls == []
    out = capsys.readouterr().out
    assert "Refusing to download without --yes" in out
