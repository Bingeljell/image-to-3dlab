"""Focused tests for dependency-free browser modules that can run under Node."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_job_progress_formats_durations_from_the_shipped_module():
    module_url = (REPO / "viewer" / "components" / "job-progress.js").as_uri()
    program = (
        f"import {{ formatDuration }} from {json.dumps(module_url)};"
        "console.log(JSON.stringify(["
        "formatDuration(null), formatDuration(18.4), formatDuration(120), formatDuration(3720)"
        "]));"
    )

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["estimating…", "18s", "2 min", "1h 2m"]
