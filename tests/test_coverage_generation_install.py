"""Use tiny local archives and fake download responses to test setup commands."""

import hashlib
import io
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import bootstrap_blender as blender
from scripts import bootstrap_gltfpack as gltfpack


@pytest.fixture
def linux_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(blender.platform, "system", lambda: "Linux")
    monkeypatch.setattr(blender.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(gltfpack, "target_dir", lambda: tmp_path / "vendor/gltfpack")
    return tmp_path


@pytest.mark.parametrize("mode", ["noninteractive", "decline", "success", "broken"])
def test_blender_setup_fetches_archive_only_after_confirmation(mode, linux_setup, monkeypatch, capsys):
    home = linux_setup
    previous = home / "blender-lts"
    previous.mkdir()
    (previous / "stale").touch()
    tarball = "blender-4.2.15-linux-x64.tar.xz"
    listing = f"blender-4.2.9-linux-x64.tar.xz {tarball} blender-4.2.20-linux-x64.tar.xz.sha256"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        payload = b"test blender"
        binary = tarfile.TarInfo("blender-4.2.15-linux-x64/blender")
        binary.size = len(payload)
        archive.addfile(binary, io.BytesIO(payload))
    calls = []

    def urlopen(request, timeout):
        assert request.get_header("User-agent") == blender.USER_AGENT
        assert timeout == 60
        calls.append((request.full_url, request.method))
        if request.full_url == blender.RELEASES:
            return io.BytesIO(listing.encode())
        assert request.full_url == blender.RELEASES + tarball
        if request.method == "HEAD":
            response = io.BytesIO()
            response.headers = {"Content-Length": str(250 * 1024**2)}
            return response
        return io.BytesIO(buffer.getvalue())

    def run(argv, **kwargs):
        assert argv == [str(previous / "blender"), "--background", "--version"]
        assert (previous / "blender").read_bytes() == b"test blender"
        assert not (previous / "stale").exists()
        assert not list(home.glob("*.partial"))
        if mode == "broken":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="cannot start\n")
        return subprocess.CompletedProcess(argv, 0, stdout="Blender 4.2.15\n", stderr="")

    monkeypatch.setattr(blender.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(blender.subprocess, "run", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: mode != "noninteractive")

    def answer(prompt):
        assert prompt == "Continue? [y/N] "
        assert "250 MB download" in capsys.readouterr().out
        return "no" if mode == "decline" else " Yes "

    monkeypatch.setattr("builtins.input", answer)
    assert blender.main(["--yes"] if mode == "broken" else []) == int(mode != "success")
    expected = [(blender.RELEASES, "GET"), (blender.RELEASES + tarball, "HEAD")]
    if mode in {"success", "broken"}:
        expected.append((blender.RELEASES + tarball, "GET"))
    else:
        assert (previous / "stale").is_file()
    assert calls == expected
    out = capsys.readouterr().out
    assert {
        "noninteractive": "Refusing to download",
        "decline": "Nothing downloaded",
        "success": "Done: Blender 4.2.15",
        "broken": "cannot start",
    }[mode] in out


@pytest.mark.parametrize("mode", ["noninteractive", "decline", "success", "broken"])
def test_gltfpack_setup_verifies_archive_and_checks_binary(mode, linux_setup, monkeypatch, capsys):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("gltfpack", b"local fake binary")
    asset = replace(
        gltfpack.ASSETS["linux", "x86_64"],
        sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
    )
    monkeypatch.setitem(gltfpack.ASSETS, ("linux", "x86_64"), asset)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: mode != "noninteractive")
    monkeypatch.setattr("builtins.input", lambda _: "no" if mode == "decline" else "y")
    calls = []

    def urlopen(request, timeout):
        assert request.full_url == gltfpack.RELEASES + asset.zip
        assert request.get_header("User-agent") == gltfpack.USER_AGENT
        assert timeout == 60
        calls.append(request.full_url)
        return io.BytesIO(buf.getvalue())

    def run(argv, **kwargs):
        binary = gltfpack.target_dir() / "gltfpack"
        assert argv == [str(binary), "-h"]
        assert binary.read_bytes() == b"local fake binary"
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="gltfpack help" if mode == "success" else ""
        )

    monkeypatch.setattr(gltfpack.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gltfpack.subprocess, "run", run)
    assert gltfpack.main(["--yes"] if mode == "broken" else []) == int(mode != "success")
    assert len(calls) == int(mode in {"success", "broken"})
    assert gltfpack.target_dir().is_dir() == bool(calls)
    out = capsys.readouterr().out
    assert f"size   {asset.size / 1024**2:.1f} MB" in out
    assert {
        "noninteractive": "Refusing to download",
        "decline": "Nothing downloaded",
        "success": "Done: gltfpack",
        "broken": "did not start",
    }[mode] in out


def test_gltfpack_rejects_matching_archive_without_binary(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("README", "no binary")
    asset = gltfpack.Asset("fixture.zip", hashlib.sha256(buf.getvalue()).hexdigest(), 0, "gltfpack")
    with pytest.raises(SystemExit, match="has no gltfpack"):
        gltfpack.install(buf.getvalue(), asset, tmp_path)
    assert list(tmp_path.iterdir()) == []
