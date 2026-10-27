"""Tests for the RealESRGAN converter's download gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "hunyuan_mlx" / "paint" / "scripts" / "convert_realesrgan.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_realesrgan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_realesrgan"] = module
    spec.loader.exec_module(module)
    return module


crg = _load()


def test_missing_input_without_fetch_exits_naming_the_url(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(crg, "convert", lambda pth, out: calls.append(("convert", pth, out)))
    out = tmp_path / "rrdbnet.npz"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(out)])

    assert crg.main() != 0
    assert calls == []
    captured = capsys.readouterr().out
    assert crg.RELEASE_URL in captured
    assert "MB" in captured or "GB" in captured


def test_fetch_downloads_the_default_input(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(crg, "convert", lambda pth, out: calls.append(("convert", pth, out)))
    monkeypatch.setattr(
        crg.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(b"pth"),
    )
    out = tmp_path / "rrdbnet.npz"
    pth = out.parent / "RealESRGAN_x4plus.pth"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(out), "--fetch"])

    assert crg.main() == 0
    assert calls == [("convert", pth, out)]
    assert pth.read_bytes() == b"pth"


def test_explicit_local_input_needs_no_fetch(monkeypatch, tmp_path):
    calls = []
    local = tmp_path / "RealESRGAN_x4plus.pth"
    local.write_bytes(b"local")
    out = tmp_path / "out.npz"
    monkeypatch.setattr(crg, "convert", lambda pth, out_path: calls.append(("convert", pth, out_path)))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--pth", str(local), "--out", str(out)])

    assert crg.main() == 0
    assert calls == [("convert", local, out)]
