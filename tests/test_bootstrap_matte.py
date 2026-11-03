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
        assert url == matte.LEAN_URL
        return io.BytesIO(payload)
    return opener


def test_the_announcement_names_the_file_its_size_and_where_it_goes(home):
    text = bm.announcement()
    assert "BiRefNet-lite" in text and "181 MB" in text
    assert matte.LEAN_URL in text and str(home) in text


def test_saying_no_downloads_nothing(home, monkeypatch):
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    assert bm.main([], ask=lambda _: "n") == 1
    assert not any(home.iterdir())


def test_check_reports_without_downloading(home, monkeypatch):
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    assert bm.main(["--check"]) == 1
    (home / "birefnet-lite-lean.onnx").write_bytes(b"x")
    assert bm.main(["--check"]) == 0


def test_a_file_with_the_right_checksum_is_installed(home, monkeypatch):
    payload = b"model bytes"
    monkeypatch.setattr(matte, "LEAN_SHA256", hashlib.sha256(payload).hexdigest())
    target = bm.download(home / "birefnet-lite-lean.onnx", opener=_opener(payload))
    assert target.read_bytes() == payload
    assert not (home / "birefnet-lite-lean.part").exists()


def test_a_file_with_the_wrong_checksum_is_thrown_away(home):
    with pytest.raises(SystemExit, match="checksum mismatch"):
        bm.download(home / "birefnet-lite-lean.onnx", opener=_opener(b"corrupt"))
    assert not any(home.iterdir())


def test_no_terminal_means_no_download_rather_than_a_crash(home, monkeypatch, capsys):
    """Run without --yes and without a keyboard, input() raises EOFError."""
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))

    def no_terminal(prompt):
        raise EOFError

    assert bm.main([], ask=no_terminal) == 1
    assert "nobody to ask" in capsys.readouterr().out


def test_rembgs_older_copy_does_not_count_as_installed(home, monkeypatch):
    """A machine with only rembg's 9.5 GB-per-cut copy is offered the lean file on Setup."""
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    (home / "birefnet-general-lite.onnx").write_bytes(b"x")
    assert bm.main(["--check"]) == 1


def _fake_download(target, opener=None):
    target.write_bytes(b"lean")
    return target


def test_the_announcement_says_the_older_copy_goes(home):
    assert "removes" not in bm.announcement()
    (home / "birefnet-general-lite.onnx").write_bytes(b"x")
    text = bm.announcement()
    assert "removes the older copy" in text and "224 MB" in text
    assert str(home / "birefnet-general-lite.onnx") in text


def test_installing_the_lighter_file_removes_the_older_one(home, monkeypatch):
    older = home / "birefnet-general-lite.onnx"
    older.write_bytes(b"x")
    monkeypatch.setattr(bm, "download", _fake_download)
    monkeypatch.setattr(matte, "LeanSession", lambda path: object())
    assert bm.main(["--yes"]) == 0
    assert matte.model_file(matte.LEAN_MODEL).is_file()
    assert not older.exists()


def test_an_engine_setup_also_removes_the_older_one(home, monkeypatch):
    older = home / "birefnet-general-lite.onnx"
    older.write_bytes(b"x")
    monkeypatch.setattr(matte, "LeanSession", lambda path: object())
    bm.install_if_missing(download=_fake_download)
    assert not older.exists()


def test_a_leftover_older_copy_goes_when_setup_runs_again(home, monkeypatch):
    """Installed the lean file on 0.5.5 or 0.5.6: the old one is still on disk."""
    older = home / "birefnet-general-lite.onnx"
    older.write_bytes(b"x")
    matte.model_file(matte.LEAN_MODEL).write_bytes(b"lean")
    monkeypatch.setattr(bm, "download", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(matte, "LeanSession", lambda path: object())
    assert bm.main(["--check"]) == 0 and older.exists()  # --check only reports
    assert bm.main([]) == 0 and not older.exists()
