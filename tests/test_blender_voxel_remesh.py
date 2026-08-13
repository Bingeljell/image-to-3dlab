"""Tests for the voxel-remesh argument handling.

Only the pure parts are testable — everything else is `bpy` calls that need Blender. That
split is deliberate: CLAUDE.md's rule is to pull the logic out of the string/script so it
can be imported, and a wrong voxel size or a silently-dropped argument is exactly the kind
of mistake that costs a long headless run before it surfaces.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_voxel_remesh.py"


def _load():
    # bpy only exists inside Blender; stub it so the module imports here.
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("blender_voxel_remesh", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bvr = _load()


def test_parses_paths_and_voxel_size():
    argv = ["blender", "--background", "--python", "s.py", "--", "in.glb", "out.glb", "0.002"]
    assert bvr.parse_args(argv) == ("in.glb", "out.glb", 0.002)


def test_voxel_size_defaults_when_omitted():
    argv = ["blender", "--", "in.glb", "out.glb"]
    source, target, size = bvr.parse_args(argv)
    assert (source, target) == ("in.glb", "out.glb")
    assert size == pytest.approx(0.004)


def test_missing_separator_is_refused():
    """Without --, Blender eats the arguments and the run would remesh nothing."""
    with pytest.raises(SystemExit, match="after --"):
        bvr.parse_args(["blender", "--background", "in.glb"])


def test_too_few_arguments_is_refused():
    with pytest.raises(SystemExit, match="usage"):
        bvr.parse_args(["blender", "--", "in.glb"])


def test_arguments_before_the_separator_are_ignored():
    argv = ["blender", "-b", "--python", "x.py", "--", "a.glb", "b.glb", "0.01"]
    assert bvr.parse_args(argv)[0] == "a.glb"
