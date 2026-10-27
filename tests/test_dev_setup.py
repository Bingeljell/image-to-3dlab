from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_lab_no_longer_tells_anyone_to_pip_install():
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
    assert not (REPO / "requirements.lock").exists()
