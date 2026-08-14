"""Tests for lifting the 200k face cap.

Patch scripts have a specific failure mode: the anchor drifts upstream, the patch quietly
matches nothing, and the pipeline goes on shipping damaged meshes while everyone believes
it is fixed. So: assert the anchor is found, assert re-running is idempotent, and assert a
missing anchor is loud.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_face_cap.py"


def _load():
    spec = importlib.util.spec_from_file_location("face_cap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap = _load()

REAL = """
                import fast_simplification
                verts_np = mesh_out.vertices.cpu().numpy()
                faces_np = mesh_out.faces.cpu().numpy()
                target_faces = min(args.bake_target_faces, 200000, len(faces_np))
                if len(faces_np) > target_faces:
"""


def test_finds_the_current_cap():
    assert cap.find_cap(REAL) == 200000


def test_unpatched_source_is_reported_as_unpatched():
    assert not cap.is_patched(REAL)


def test_apply_raises_the_ceiling():
    out = cap.apply(REAL, 1_000_000)
    assert cap.find_cap(out) == 1_000_000
    assert cap.is_patched(out)


def test_apply_keeps_bake_target_faces_in_the_min():
    """The manifest must still be able to ask for FEWER faces than the ceiling."""
    out = cap.apply(REAL, 1_000_000)
    assert "args.bake_target_faces" in out
    assert "len(faces_np)" in out


def test_apply_is_idempotent():
    once = cap.apply(REAL, 1_000_000)
    twice = cap.apply(once, 1_000_000)
    assert once == twice


def test_apply_can_retarget_an_older_patch():
    once = cap.apply(REAL, 1_000_000)
    retargeted = cap.apply(once, 16_777_216)
    assert cap.find_cap(retargeted) == 16_777_216
    assert retargeted != once


def test_apply_leaves_surrounding_code_intact():
    out = cap.apply(REAL, 300000)
    assert "import fast_simplification" in out
    assert "if len(faces_np) > target_faces:" in out


def test_missing_anchor_is_loud_not_silent():
    """A silent no-op here means shipping damaged meshes while believing it is fixed."""
    with pytest.raises(SystemExit):
        cap.apply("nothing that looks like the anchor", 1_000_000)


def test_patched_line_records_why():
    """The next reader needs to know 200000 was not an upstream default but our own cap."""
    line = cap.patched_line(1_000_000)
    assert "200000" in line
    assert cap.MARKER in line


def test_default_ceiling_matches_the_official_demo_pre_simplification():
    assert cap.DEFAULT_CEILING == 16_777_216
