"""Failing-first tests for two review findings: the weld patch's --repo, the weights audit.

The audit has no weight-size threshold of its own, so these tests assume 1 MiB.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


weld = _load("patch_ovoxel_weld_before_simplify")
audit = _load("audit_model_weights")

# --- 5. weld patch --repo ------------------------------------------------------------

_SOURCE = "import torch\n\n\ndef to_glb(mesh):\n    mesh.simplify(0.5)\n"


class _Stop(Exception):
    pass


class _Mesh:
    def read(self):
        raise _Stop  # reached only once the helper has found the checkout


def _run_helper(patched: str, fake_file: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"__file__": str(fake_file), "__name__": "o_voxel.postprocess"}
    exec(compile(patched, str(fake_file), "exec"), namespace)
    namespace[weld.MARKER](_Mesh())


def test_weld_target_outside_the_repo_works_at_run_time_or_is_refused(
    tmp_path, monkeypatch
):
    target = tmp_path / "elsewhere" / "site-packages" / "o_voxel" / "postprocess.py"
    target.parent.mkdir(parents=True)
    target.write_text(_SOURCE)
    monkeypatch.setattr(
        sys, "argv", ["patch", "--target", str(target), "--repo", str(ROOT)]
    )
    try:
        weld.main()
    except SystemExit as exit_:
        assert exit_.code not in (0, None), "refusal must be non-zero"
        assert target.read_text() == _SOURCE, (
            "a refused patch must not touch the target"
        )
        return
    with pytest.raises(_Stop):
        _run_helper(target.read_text(), target, monkeypatch)
