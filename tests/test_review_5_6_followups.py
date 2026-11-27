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


# --- 6. weights audit ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tiny_gb(monkeypatch):
    monkeypatch.setattr(audit, "GB", 1024)


def _blob(hub: Path, repo: str, rel: str, data: bytes) -> Path:
    base = hub / f"models--{repo.replace('/', '--')}"
    blob = base / "blobs" / f"{repo}-{rel}".replace("/", "_")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    link = base / "snapshots" / "rev" / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(blob)
    return link


def _local(root: Path, rel: str, data: bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_entry_with_all_big_files_matching_is_deletable_and_names_small_files(tmp_path):
    hub, local = tmp_path / "hub", tmp_path / "local"
    big = b"a" * MIB
    _blob(hub, "org/m", "model.safetensors", big)
    _blob(hub, "org/m", "README.md", b"readme")
    _blob(hub, "org/m", "config.json", b"{}")
    _local(local, "model.safetensors", big)
    (found,) = audit.find_cached_duplicates(hub, roots=(local,))
    assert found.deletable is True, (
        f"big files all match but not deletable: {found.reason}"
    )
    for small in ("README.md", "config.json"):
        assert small in found.reason, (
            f"small file {small} not listed in: {found.reason}"
        )


def test_big_file_matching_by_size_but_not_bytes_is_not_deletable(tmp_path):
    hub, local = tmp_path / "hub", tmp_path / "local"
    _blob(hub, "org/m", "model.safetensors", b"a" * MIB)
    _local(local, "model.safetensors", b"b" * MIB)
    found = audit.find_cached_duplicates(hub, roots=(local,))
    assert all(not c.deletable for c in found)


def test_each_file_is_hashed_once_across_entries_sharing_a_candidate(
    tmp_path, monkeypatch
):
    hub, local = tmp_path / "hub", tmp_path / "local"
    big = b"a" * MIB
    _blob(hub, "org/one", "model.safetensors", big)
    _blob(hub, "org/two", "model.safetensors", big)
    _local(local, "model.safetensors", big)
    real, calls = audit._sha256, []

    def spy(path):
        calls.append(Path(path).resolve())
        return real(path)

    monkeypatch.setattr(audit, "_sha256", spy)
    audit.find_cached_duplicates(hub, roots=(local,))
    repeated = {p.name for p in calls if calls.count(p) > 1}
    assert not repeated, f"hashed more than once: {repeated} ({len(calls)} calls)"
