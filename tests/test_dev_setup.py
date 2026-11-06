from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_lab_no_longer_tells_anyone_to_pip_install_requirements():
    roots = [REPO / "scripts", REPO / "viewer", REPO / "image_to_3dlab",
             REPO / "AGENTS.md", REPO / "README.md", REPO / "docs" / "agents.md"]
    hits = []
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if path.suffix not in {".py", ".md"}:
                continue
            if "pip install -r requirements-dev.txt" in path.read_text(errors="ignore"):
                hits.append(str(path.relative_to(REPO)))
    assert hits == []


def test_no_setup_message_points_at_a_deleted_requirements_file():
    """The root requirements files are gone, so a setup script that tells someone to
    `pip install -r requirements.txt` sends them to a file that is not there. Upstream
    checkouts still have their own requirements.txt; only the message class is checked."""
    hits = []
    for path in (REPO / "scripts").rglob("*.py"):
        text = path.read_text(errors="ignore")
        if "is not installed. pip install -r requirements" in text:
            hits.append(str(path.relative_to(REPO)))
    assert hits == []
