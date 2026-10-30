"""The shape-port ModelScope downloaders ask first, and a short file is not done."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "shape" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dl_modelscope as modelscope


class _Body:
    def __init__(self, status: int, payload: bytes):
        self.status = status
        self._payload = payload
        self._at = 0

    def read(self, count: int) -> bytes:
        chunk = self._payload[self._at : self._at + count]
        self._at += count
        return chunk


def test_a_short_file_is_not_success_and_a_near_one_is(tmp_path, monkeypatch):
    def offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(modelscope, "ROOT", str(tmp_path))
    monkeypatch.setattr(modelscope.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(modelscope.opener, "open", offline)
    short = tmp_path / "weights" / "model.safetensors"
    short.parent.mkdir()
    short.write_bytes(b"x" * 100)
    assert modelscope.download("weights/model.safetensors", 1_000_000) is False

    near = tmp_path / "weights" / "near.safetensors"
    near.write_bytes(b"x" * (10_000 - 4096))
    assert modelscope.download("weights/near.safetensors", 10_000) is True


def test_only_a_206_body_is_appended(tmp_path):
    path = tmp_path / "part.bin"
    path.write_bytes(b"OLD")
    modelscope.write_body(_Body(200, b"NEW"), str(path), 3)
    assert path.read_bytes() == b"NEW"

    path.write_bytes(b"OLD")
    modelscope.write_body(_Body(206, b"TAIL"), str(path), 3)
    assert path.read_bytes() == b"OLDTAIL"


def test_missing_yes_asks_and_eof_refuses(monkeypatch, capsys):
    def eof():
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert modelscope.confirm(
        False, route="ModelScope Hunyuan3D-2mini", size="about 4.2 GB"
    ) is False
    text = capsys.readouterr().out
    assert "Hunyuan" in text
    assert "ModelScope Hunyuan3D-2mini" in text
    assert "about 4.2 GB" in text
    assert "EU" in text and "UK" in text and "South Korea" in text


def test_no_does_not_download(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "n")
    monkeypatch.setattr(modelscope, "download", lambda *_a, **_k: pytest.fail("fetched"))
    assert modelscope.main([]) == 1


def test_dl_any_does_no_work_at_import(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["dl_any.py", "repo", "file.bin", str(tmp_path / "out.bin")])

    def boom(*_args, **_kwargs):
        raise AssertionError("downloaded at import")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", boom)
    sys.modules.pop("dl_any", None)
    module = importlib.import_module("dl_any")

    path = tmp_path / "part.bin"
    path.write_bytes(b"OLD")
    module.write_body(_Body(200, b"NEW"), str(path), 3)
    assert path.read_bytes() == b"NEW"
    path.write_bytes(b"OLD")
    module.write_body(_Body(206, b"TAIL"), str(path), 3)
    assert path.read_bytes() == b"OLDTAIL"
