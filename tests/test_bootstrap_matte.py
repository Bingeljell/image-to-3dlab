"""The BiRefNet-lite installer: never downloads without a yes, never keeps a bad file."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap_matte as bm

from image_to_3dlab import matte


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    return tmp_path


def _opener(payload: bytes):
    def opener(url):
        assert url == matte.LITE_URL
        return io.BytesIO(payload)
    return opener


def test_the_announcement_names_the_file_its_size_and_where_it_goes(home):
    text = bm.announcement()
    assert "BiRefNet-lite" in text and "224 MB" in text
    assert matte.LITE_URL in text and str(home) in text


def test_saying_no_downloads_nothing(home, monkeypatch):
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    assert bm.main([], ask=lambda _: "n") == 1
    assert not any(home.iterdir())


def test_check_reports_without_downloading(home, monkeypatch):
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    assert bm.main(["--check"]) == 1
    (home / "birefnet-general-lite.onnx").write_bytes(b"x")
    assert bm.main(["--check"]) == 0


def test_a_file_with_the_right_checksum_is_installed(home, monkeypatch):
    payload = b"model bytes"
    monkeypatch.setattr(matte, "LITE_MD5", hashlib.md5(payload).hexdigest())
    target = bm.download(home / "birefnet-general-lite.onnx", opener=_opener(payload))
    assert target.read_bytes() == payload
    assert not (home / "birefnet-general-lite.part").exists()


def test_a_file_with_the_wrong_checksum_is_thrown_away(home):
    with pytest.raises(SystemExit, match="checksum mismatch"):
        bm.download(home / "birefnet-general-lite.onnx", opener=_opener(b"corrupt"))
    assert not any(home.iterdir())


def test_no_terminal_means_no_download_rather_than_a_crash(home, monkeypatch, capsys):
    """Run without --yes and without a keyboard, input() raises EOFError."""
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))

    def no_terminal(prompt):
        raise EOFError

    assert bm.main([], ask=no_terminal) == 1
    assert "nobody to ask" in capsys.readouterr().out
