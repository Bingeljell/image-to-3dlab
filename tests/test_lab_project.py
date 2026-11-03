"""Pins for the lab's uv project. Python stays 3.11; mise is not part of this."""

from __future__ import annotations

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
