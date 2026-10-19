"""The Pixal3D patch scripts read and write upstream's C++ as UTF-8 on every OS.

2026-10-19: our Windows NVIDIA build died in patch_pixal3d_steps.py with UnicodeDecodeError,
because Windows' Python reads text in the local code page unless told otherwise. Python's
own EncodingWarning catches any read or write that does not name its encoding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "scripts").glob("patch_pixal3d_*.py"))


def test_there_are_patch_scripts_to_check():
    assert len(SCRIPTS) >= 5


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_every_read_and_write_names_utf8(script, tmp_path):
    target = tmp_path / "source.cpp"
    target.write_text("// upstream comment with a dash — and an arrow →\nint main() {}\n",
                      encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning",
         str(script), "--check", str(target)],
        capture_output=True, text=True, encoding="utf-8", check=False)
    assert "EncodingWarning" not in result.stderr, result.stderr
    assert "UnicodeDecodeError" not in result.stderr, result.stderr
