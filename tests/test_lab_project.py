"""Pins for the lab's uv project. Python stays 3.11; mise is not part of this."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.measure_coverage import inventory

REPO = Path(__file__).resolve().parents[1]


def test_the_lab_is_a_uv_project_on_python_3_11():
    project = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert "rembg==2.0.69" in project
    assert '"numpy<2"' in project
    assert "ruff==0.16.1" in project
    assert (REPO / "uv.lock").is_file()
    assert (REPO / ".python-version").read_text(encoding="utf-8").strip() == "3.11"
    assert not (REPO / "mise.toml").exists()
    assert not (REPO / "requirements.txt").exists()
    assert not (REPO / "requirements-dev.txt").exists()
    assert not (REPO / "requirements.lock").exists()


def test_pytest_puts_the_repo_root_on_the_path():
    """The project is not installed, so pytest must be told where the package is."""
    assert "pythonpath = ." in (REPO / "pytest.ini").read_text(encoding="utf-8")
    for name in ("AGENTS.md", "CHANGELOG.md"):
        assert "PYTHONPATH=. pytest" not in (REPO / name).read_text(encoding="utf-8")


def test_coverage_inventory_includes_all_maintained_areas():
    files = inventory(REPO)
    assert {
        "pipeline.py", "image_to_3dlab/cli.py", "viewer/generate_api.py",
        "scripts/measure_coverage.py", "hunyuan_mlx/shape/hy3dmlx/sampler.py",
        "hunyuan_mlx/paint/hy3dpaint_mlx/scheduler.py",
    } <= set(files["python"])
    assert {"understanding_3d/mesh-winding-demo.js", "viewer/studio/app.js"} <= set(files["javascript"])
    assert "viewer/vendor/three.module.js" not in files["javascript"]


def test_the_version_is_the_one_in_the_changelog():
    """The version now lives in two places. A release has to move both, so check they agree."""
    project = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', project, re.MULTILINE).group(1)
    assert f"## [{version}]" in (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
