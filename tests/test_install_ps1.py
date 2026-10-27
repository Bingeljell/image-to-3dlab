"""install.ps1: what can be read without running it.

Not in test_install_script.py, whose module is skipped on Windows, the one system that
runs install.ps1.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "install.ps1"


def test_the_windows_installer_uses_mise_and_uv_sync():
    ps1 = PS1.read_text(encoding="utf-8")
    assert "mise install" in ps1
    assert "uv sync --frozen --no-dev" in ps1
    assert "requirements.lock" not in ps1
    assert "& uv pip install" not in ps1


def test_the_windows_installer_bootstraps_mise_when_missing():
    ps1 = PS1.read_text(encoding="utf-8")
    assert "irm https://mise.run" in ps1
    assert "Get-Command mise" in ps1
