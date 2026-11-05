"""Pins for the lab's uv project. Python stays 3.11; mise is not part of this."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_lab_is_a_uv_project_on_python_3_11():
    project = (REPO / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert "rembg==2.0.69" in project
    assert '"numpy<2"' in project
    assert "ruff==0.16.1" in project
    assert (REPO / "uv.lock").is_file()
    assert (REPO / ".python-version").read_text().strip() == "3.11"
    assert not (REPO / "mise.toml").exists()
    assert not (REPO / "requirements.txt").exists()
    assert not (REPO / "requirements-dev.txt").exists()
    assert not (REPO / "requirements.lock").exists()


def test_the_version_is_the_one_in_the_changelog():
    """The version now lives in two places. A release has to move both, so check they agree."""
    project = (REPO / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', project, re.MULTILINE).group(1)
    assert f"## [{version}]" in (REPO / "CHANGELOG.md").read_text()
