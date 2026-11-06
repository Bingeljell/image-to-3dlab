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


def test_coverage_measures_the_package_and_the_viewer():
    """Opt-in line coverage for our code. scripts/ stays out: most of those files run
    inside Blender, and one combined number would mostly describe them."""
    project = (REPO / "pyproject.toml").read_text()
    assert "pytest-cov>=7" in project
    assert 'source = ["image_to_3dlab", "viewer"]' in project
    assert 'directory = "htmlcov"' in project
    ignored = (REPO / ".gitignore").read_text().splitlines()
    assert ".coverage" in ignored
    assert "htmlcov/" in ignored
    assert 'name = "pytest-cov"' in (REPO / "uv.lock").read_text()
    command = "--cov-config=pyproject.toml --cov --cov-report=term-missing"
    assert command in (REPO / "README.md").read_text()
    assert command in (REPO / "AGENTS.md").read_text()


def test_the_version_is_the_one_in_the_changelog():
    """The version now lives in two places. A release has to move both, so check they agree."""
    project = (REPO / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', project, re.MULTILINE).group(1)
    assert f"## [{version}]" in (REPO / "CHANGELOG.md").read_text()
