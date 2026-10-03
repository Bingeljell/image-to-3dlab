"""scripts/bootstrap_gltfpack.py: gltfpack for the Props tab, from Setup & Status.

Without gltfpack the Props tab's LODs stay uncompressed, and the only fix it offered was
"put a native release on PATH": jargon, and a by-hand install (props test, 2026-10-03).
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_gltfpack as boot  # noqa: E402
import finish_props  # noqa: E402


def _zip(name: str, payload: bytes = b"#!/bin/sh\necho gltfpack\n") -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, payload)
    return data.getvalue()


def _asset(archive: bytes, binary: str = "gltfpack") -> boot.Asset:
    return boot.Asset("gltfpack-test.zip", hashlib.sha256(archive).hexdigest(),
                      len(archive), binary)


@pytest.mark.parametrize("system, machine, zip_name", [
    ("Darwin", "arm64", "gltfpack-macos.zip"),
    ("Linux", "x86_64", "gltfpack-ubuntu.zip"),
    ("Windows", "AMD64", "gltfpack-windows.zip"),
])
def test_picks_the_release_build_for_each_machine(system, machine, zip_name):
    assert boot.asset_for(system, machine).zip == zip_name


@pytest.mark.parametrize("system, machine", [("Darwin", "x86_64"), ("Linux", "aarch64")])
def test_no_build_for_a_machine_is_none_not_a_guess(system, machine):
    # The macOS zip is Apple Silicon only and the Linux one x86_64 only (checked on v1.3).
    assert boot.asset_for(system, machine) is None


def test_every_pinned_build_has_a_checksum_and_a_size():
    for asset in boot.ASSETS.values():
        assert len(asset.sha256) == 64 and asset.size > 0


def test_announcement_names_the_tool_size_licence_and_destination():
    text = boot.announcement(boot.ASSETS[("darwin", "arm64")])
    assert "gltfpack" in text and boot.VERSION in text
    assert "MB" in text and "MIT" in text and "vendor/gltfpack" in text


def test_installs_where_the_props_tab_already_looks(tmp_path, monkeypatch):
    archive = _zip("gltfpack")
    binary = boot.install(archive, _asset(archive), boot.target_dir(tmp_path))
    assert binary.stat().st_mode & 0o111
    monkeypatch.setattr(finish_props, "REPO", tmp_path)
    assert finish_props.find_gltfpack(which=lambda _name: None) == binary


def test_the_props_tab_finds_the_windows_exe_too(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_props, "REPO", tmp_path)
    vendor = tmp_path / "vendor" / "gltfpack"
    vendor.mkdir(parents=True)
    (vendor / "gltfpack.exe").write_bytes(b"")
    assert finish_props.find_gltfpack(which=lambda _name: None) == vendor / "gltfpack.exe"


def test_a_download_that_does_not_match_the_checksum_is_refused(tmp_path):
    archive = _zip("gltfpack")
    tampered = _zip("gltfpack", b"something else")
    with pytest.raises(SystemExit, match="checksum"):
        boot.install(tampered, _asset(archive), tmp_path)
    assert not (tmp_path / "gltfpack").exists()


def test_a_zip_without_the_binary_is_an_error(tmp_path):
    archive = _zip("README.txt")
    with pytest.raises(SystemExit, match="gltfpack"):
        boot.install(archive, _asset(archive), tmp_path)


def test_without_yes_and_nobody_to_ask_it_downloads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(boot.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(boot.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(boot, "download", lambda _url: pytest.fail("downloaded"))
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out
