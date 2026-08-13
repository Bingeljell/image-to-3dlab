"""Tests for re-enabling the decode-time cleanup stubs.

Patch scripts get three things wrong in this repo's history, so all three are pinned here:
they assume an anchor that has moved, they are not idempotent, and they cannot be undone.
The last one matters most for this patch — the stub exists because Metal `cumesh` segfaulted,
so `--revert` is the escape hatch when it segfaults again, and it must work without a
re-bootstrap.

Tested against the real vendored text, not a paraphrase: the fixture is copied verbatim from
`trellis2/representations/mesh/base.py`, because a patch matched against re-typed source is
testing something that is not what ships.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_enable_cleanup.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_enable_cleanup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pec = _load()


# Verbatim from the vendored port, stubs included.
STUBBED = '''\
class MeshBase:
    def fill_holes(self, max_hole_perimeter=3e-2):
        return  # Skip — Metal cumesh segfaults on large decode meshes
        vertices = self.vertices.to(self.device)
        faces = self.faces.to(self.device)

    def remove_faces(self, face_mask: torch.Tensor):
        return
        vertices = self.vertices.to(self.device)
        faces = self.faces.to(self.device)

    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        return
        vertices = self.vertices.to(self.device)
        faces = self.faces.to(self.device)
'''


def test_detects_all_three_stubs():
    assert len(pec.find_stubs(STUBBED)) == 3


def test_enable_removes_every_early_return():
    enabled, changed = pec.enable(STUBBED)
    assert len(changed) == 3
    assert pec.find_stubs(enabled) == []
    # The bodies must survive untouched - they were never deleted, only made unreachable.
    assert enabled.count("vertices = self.vertices.to(self.device)") == 3
    assert "Skip — Metal cumesh segfaults" not in enabled


def test_enable_is_idempotent():
    once, _ = pec.enable(STUBBED)
    twice, changed = pec.enable(once)
    assert changed == []
    assert twice == once


def test_revert_restores_the_stubs_exactly():
    """The escape hatch: a segfault must be undoable without re-bootstrapping vendor/."""
    enabled, _ = pec.enable(STUBBED)
    reverted, changed = pec.revert(enabled)
    assert len(changed) == 3
    assert reverted == STUBBED


def test_revert_is_idempotent():
    _reverted, changed = pec.revert(STUBBED)
    assert changed == []


def test_missing_anchor_raises_rather_than_guessing():
    """If the port changes shape, refuse - a silently mis-applied patch is worse."""
    with pytest.raises(RuntimeError, match="anchor missing"):
        pec.find_stubs("class MeshBase:\n    def something_else(self):\n        pass\n")


def test_enabled_source_is_still_valid_python():
    import ast

    enabled, _ = pec.enable(STUBBED)
    ast.parse(enabled)


def test_marker_identifies_our_edit():
    """So a future reader can tell this line came from us, not from the port."""
    enabled, _ = pec.enable(STUBBED)
    assert enabled.count("patch_trellis_enable_cleanup.py") == 3
