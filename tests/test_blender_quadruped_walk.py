"""The walk saver must not replace the blend Blender was opened with."""

from __future__ import annotations

from pathlib import Path

import blender_quadruped_walk as walk
import pytest


def test_refuses_to_save_over_the_opened_blend(tmp_path: Path):
    opened = tmp_path / "rigged.blend"
    opened.write_bytes(b"blend")
    with pytest.raises(SystemExit, match="opened"):
        walk.refuse_open_blend(opened, str(opened))
    walk.refuse_open_blend(tmp_path / "walked.blend", str(opened))
