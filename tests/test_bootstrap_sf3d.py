"""The SF3D installer: Mac only, gated weights fetched only on a yes."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import backend_catalog
import bootstrap_sf3d as boot
import pytest


def test_weight_total_matches_the_catalogue():
    """Including DINOv2, which SF3D otherwise downloads silently on its first run."""
    sf3d = next(b for b in backend_catalog.CATALOG if b.id == "sf3d")
    assert sum(w.bytes_expected for w in sf3d.weights) == pytest.approx(
        boot.total_gb() * backend_catalog.GB, rel=0.02)
    assert {w.source for w in sf3d.weights} == {repo for repo, _, _ in boot.WEIGHTS}


def test_announcement_names_backend_route_size_and_gating(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    text = boot.announcement()
    for needle in ("Stable Fast 3D", "4.9 GB", "facebook/dinov2-large", "gated",
                   "Stability AI Community License"):
        assert needle in text


@pytest.mark.parametrize("key", [None, "windows-nvidia", "linux-nvidia"])
def test_unsupported_machines_are_refused_before_anything(monkeypatch, capsys, key):
    monkeypatch.setattr(boot, "target", lambda: key)
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("installed"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_no_yes_and_no_terminal_means_no_download(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("installed"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main([]) == 1


def test_code_and_weights_halves(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    called = []
    monkeypatch.setattr(boot, "install_code", lambda *a: called.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: called.append("weights"))
    assert boot.main(["--yes", "--code-only"]) == 0
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert boot.main(["--yes"]) == 0
    assert called == ["code", "weights", "code", "weights"]


def test_gated_weights_explain_how_to_get_access(monkeypatch, capsys):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")

    def refused():
        raise boot.GatedAccess("stabilityai/stable-fast-3d")

    monkeypatch.setattr(boot, "install_weights", refused)
    assert boot.main(["--yes", "--weights-only"]) == 1
    out = capsys.readouterr().out
    assert "huggingface.co/stabilityai/stable-fast-3d" in out and "hf auth login" in out


def test_the_viewer_can_run_it():
    import download_api

    command = download_api.COMMANDS["sf3d"]
    assert command[1].endswith("bootstrap_sf3d.py") and "--yes" in command


def test_mac_sf3d_installs_into_its_own_venv(monkeypatch):
    monkeypatch.setattr(boot, "install_interpreter",
                        lambda: Path("/vendor/stable-fast-3d/.venv/bin/python"))
    monkeypatch.setattr(boot.host, "find_uv", lambda: "/mise/uv")
    command = boot.pip_install_command()
    assert command[command.index("--python") + 1] == "/vendor/stable-fast-3d/.venv/bin/python"
    assert sys.executable not in command


def test_packages_go_in_with_uv_when_the_environment_has_no_pip(monkeypatch):
    """The one-line installer builds the environment with uv, which ships no pip. On the
    second NVIDIA pod `python -m pip` failed with 'No module named pip'."""
    monkeypatch.setattr(boot.host, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(boot, "has_pip", lambda: False)
    monkeypatch.setattr(boot, "install_interpreter",
                        lambda: Path("/vendor/stable-fast-3d/.venv/bin/python"))
    command = boot.pip_install_command()
    assert command[:3] == ["/usr/bin/uv", "pip", "install"]
    assert "--python" in command and str(boot.install_interpreter()) in command
    assert sys.executable not in command


def test_pip_is_used_when_there_is_no_uv(monkeypatch):
    monkeypatch.setattr(boot.host, "find_uv", lambda: None)
    monkeypatch.setattr(boot, "has_pip", lambda: True)
    monkeypatch.setattr(boot, "install_interpreter",
                        lambda: Path("/vendor/stable-fast-3d/.venv/bin/python"))
    assert boot.pip_install_command() == [str(boot.install_interpreter()), "-m", "pip", "install"]


def test_neither_uv_nor_pip_says_how_to_fix_it(monkeypatch):
    monkeypatch.setattr(boot.host, "find_uv", lambda: None)
    monkeypatch.setattr(boot, "has_pip", lambda: False)
    monkeypatch.setattr(boot, "install_interpreter",
                        lambda: Path("/vendor/stable-fast-3d/.venv/bin/python"))
    with pytest.raises(SystemExit, match="uv"):
        boot.pip_install_command()


def test_build_tools_go_in_before_the_extensions(monkeypatch, tmp_path):
    """--no-build-isolation builds with what is already installed, and a fresh uv
    environment has no setuptools or wheel."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(boot, "VENDOR", tmp_path)
    monkeypatch.setattr(boot, "install_interpreter",
                        lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(boot, "pip_install_command", lambda: ["pip", "install"])
    monkeypatch.setattr(boot.host, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(boot.host, "mise_python",
                        lambda version, **kw: Path(f"/mise/python{version}"))
    monkeypatch.setattr(boot.host, "venv_argv",
                        lambda uv, dest, python: [uv, "venv", str(dest), "--python", str(python)])
    monkeypatch.setattr(boot.Path, "exists", lambda self: True)  # libomp
    calls = []
    monkeypatch.setattr(boot.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    boot.install_code("macos-arm64")
    assert calls[0][0] == ["/usr/bin/uv", "venv", str(tmp_path / ".venv"),
                           "--python", "/mise/python3.11"]
    assert calls[1][0] == ["pip", "install", "torch", "torchvision"]
    assert calls[2][0] == ["pip", "install", "setuptools", "wheel"]
    assert calls[3][0][-2:] == ["-r", "requirements.txt"]
    assert calls[3][1]["env"]["USE_METAL"] == "1" and calls[3][1]["env"]["USE_CUDA"] == "0"


def test_nvidia_is_not_offered():
    assert boot.route("linux-nvidia") is None
    sf3d = backend_catalog.resolve("sf3d")
    assert not sf3d.runs_here(backend_catalog.NVIDIA, "linux")


def test_fetches_exactly_the_pinned_commit(tmp_path):
    commands = boot.fetch_commands(tmp_path)
    assert ["git", "-C", str(tmp_path), "fetch", "-q", "--depth", "1", "origin",
            boot.COMMIT] in commands
    assert len(boot.COMMIT) == 40


