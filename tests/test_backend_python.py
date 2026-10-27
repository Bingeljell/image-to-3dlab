"""Every backend bootstrap must hand uv a real interpreter path, not a bare version."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_no_backend_passes_a_bare_minor_version_to_uv_venv():
    offenders = []
    for path in (REPO / "scripts").glob("bootstrap_*.py"):
        text = path.read_text()
        if ('"--python", "3.' in text or '"--python", PYTHON_VERSION' in text
                or '"--python", SOURCE_PYTHON' in text):
            offenders.append(path.name)
    assert offenders == []
