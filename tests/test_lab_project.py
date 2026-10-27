"""Pins for mise and the lab's uv project."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_mise_pins_and_the_lab_project():
    root = (REPO / "mise.toml").read_text()
    assert 'python = ["3.14", "3.12", "3.11"]' in root
    assert 'python.uv_venv_auto = "source"' in root
    assert (REPO / "hunyuan_mlx" / "shape" / "mise.toml").read_text().count('python = "3.12"') == 1
    assert (REPO / "hunyuan_mlx" / "paint" / "mise.toml").read_text().count('python = "3.11"') == 1
    assert (REPO / ".python-version").read_text().strip() == "3.14"
    project = (REPO / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.14,<3.15"' in project
    assert "rembg==2.0.86" in project
    assert "numpy>=2.3.2" in project
    assert "ruff==0.16.1" in project
    assert (REPO / "uv.lock").is_file()
    assert (REPO / "mise.lock").is_file()
