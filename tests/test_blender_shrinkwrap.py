"""Tests for shrinkwrap argument handling.

Only the pure parts are testable without Blender. The argument parsing matters because a
dropped path silently wraps the wrong mesh, and the run is long enough that the mistake
surfaces minutes later.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_shrinkwrap.py"


def _load():
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("blender_shrinkwrap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load()


def test_parses_three_paths_and_offset():
    argv = ["blender", "--", "remesh.glb", "decode.glb", "out.glb", "0.001"]
    assert bs.parse_args(argv) == ("remesh.glb", "decode.glb", "out.glb", 0.001)


def test_offset_defaults_to_zero():
    argv = ["blender", "--", "a.glb", "b.glb", "c.glb"]
    assert bs.parse_args(argv)[3] == pytest.approx(0.0)


def test_order_is_remeshed_then_decode():
    """Reversing these wraps the 12.8M-face decode onto the low-poly mesh - the wrong way."""
    source, target, out, _ = bs.parse_args(["b", "--", "low.glb", "high.glb", "o.glb"])
    assert source == "low.glb"
    assert target == "high.glb"
    assert out == "o.glb"


def test_missing_separator_is_refused():
    with pytest.raises(SystemExit, match="after --"):
        bs.parse_args(["blender", "low.glb", "high.glb", "o.glb"])


def test_too_few_paths_is_refused():
    with pytest.raises(SystemExit, match="usage"):
        bs.parse_args(["blender", "--", "low.glb", "high.glb"])
