"""Bugs from the old review commit 90362af that are still present on this branch.

Each test fails today for the reason in its docstring and passes once that bug is fixed.
No Blender, no network, no weights: tiny synthetic inputs only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import arm_clearance as ac
from scripts import audit_model_weights as audit
from scripts import clean_skin_weights, project_markings


def _weights(root: Path, name: str, size: int) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def _cache_entry(hub: Path, repo_id: str, files: dict[str, int]) -> Path:
    entry = hub / ("models--" + repo_id.replace("/", "--"))
    blobs, snap = entry / "blobs", entry / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for index, (name, size) in enumerate(files.items()):
        blob = blobs / f"blob{index}"
        with blob.open("wb") as handle:
            handle.truncate(size)
        (snap / name).symlink_to(blob)
    return entry


# 2. audit_model_weights --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tiny_gb(monkeypatch):
    """Shrink the audit's "GB" to 1 KiB so no test here writes a large file.

    Truncated files are sparse on macOS and Linux but take real space on Windows NTFS,
    and the audit hashes every size-matched file in full.
    """
    monkeypatch.setattr(audit, "GB", 1024)


@pytest.mark.symlinks
def test_ninety_percent_by_size_is_not_offered_for_deletion(tmp_path):
    """9 of 10 GB matched: the other 1 GB exists nowhere else, so it must not be deletable."""
    hub, local = tmp_path / "hub", tmp_path / "local"
    _cache_entry(
        hub,
        "org/model",
        {"shared.safetensors": 9 * audit.GB, "only-here.bin": audit.GB},
    )
    _weights(local, "shared.safetensors", 9 * audit.GB)

    found = audit.find_cached_duplicates(hub=hub, roots=(local,))

    assert len(found) == 1
    assert found[0].deletable is False
    unique = next((hub / "models--org--model").rglob("only-here.bin"))
    audit.report(found, apply=True)
    assert unique.exists() or unique.is_symlink()


@pytest.mark.symlinks
def test_same_name_and_size_but_different_bytes_is_not_a_duplicate(tmp_path):
    """Name and size alone call two different files 'the same'; the bytes must be compared."""
    hub, local = tmp_path / "hub", tmp_path / "local"
    entry = _cache_entry(hub, "org/model", {"w.bin": 2048})
    next((entry / "blobs").iterdir()).write_bytes(b"a" * 2048)
    (local / "w.bin").parent.mkdir(parents=True)
    (local / "w.bin").write_bytes(b"b" * 2048)

    found = audit.find_cached_duplicates(hub=hub, roots=(local,))

    assert [c for c in found if c.deletable] == []


# 3. clean_skin_weights ---------------------------------------------------------------


class _Objects(list):
    def remove(self, obj, do_unlink=True):
        super().remove(obj)


def _fake_bpy(names):
    class Obj:
        type = "MESH"

        def __init__(self, name):
            self.name = name

        def keys(self):
            return []

        def find_armature(self):
            return None

    store = _Objects(Obj(n) for n in names)
    nothing = types.SimpleNamespace(
        read_factory_settings=lambda **k: None, gltf=lambda **k: None
    )
    bpy = types.ModuleType("bpy")
    bpy.ops = types.SimpleNamespace(
        wm=nothing, import_scene=nothing, export_scene=nothing
    )
    bpy.data = types.SimpleNamespace(objects=store)
    return bpy, store


def test_only_the_exact_helper_name_is_removed(tmp_path, monkeypatch):
    """`startswith("Icosphere")` deletes a real body that is named Icosphere_body."""
    bpy, store = _fake_bpy(["Icosphere", "Icosphere_body", "Body"])
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_skin_weights",
            "--in",
            str(tmp_path / "in.glb"),
            "--out",
            str(tmp_path / "out.glb"),
        ],
    )

    clean_skin_weights.main()

    assert sorted(o.name for o in store) == ["Body", "Icosphere_body"]


# 4. project_markings --mirror --------------------------------------------------------


class _Stop(Exception):
    pass


def test_mirror_reflects_through_the_model_middle_not_the_origin(tmp_path, monkeypatch):
    """A model spanning x 2..4 must mirror to x 2..4 (swapped), not to x -4..-2."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_translation((3.0, 0.0, 0.0))
    uv = np.zeros((len(box.vertices), 2))
    box.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=Image.new("RGB", (4, 4), (128, 128, 128))
    )
    mesh_path, source_path = tmp_path / "m.glb", tmp_path / "s.png"
    box.export(mesh_path)
    Image.new("RGBA", (8, 8), (200, 50, 50, 255)).save(source_path)

    seen = []

    def fake_project(mesh, *args, **kwargs):
        seen.append(np.array(mesh.vertices))
        n = len(mesh.vertices)
        zeros, none = np.zeros(n), None
        if len(seen) == 2:
            raise _Stop
        return none, np.ones(n, bool), zeros, zeros, none

    monkeypatch.setattr(project_markings, "project", fake_project)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_markings",
            str(mesh_path),
            str(source_path),
            str(tmp_path / "o.glb"),
            "--mirror",
        ],
    )
    with pytest.raises(_Stop):
        project_markings.main()

    original, mirrored = seen
    assert mirrored[:, 0].min() == pytest.approx(original[:, 0].min())
    assert mirrored[:, 0].max() == pytest.approx(original[:, 0].max())


# 5. arm_clearance --------------------------------------------------------------------


def test_opposite_directions_make_a_finite_half_turn():
    """c = -1 divides by (1 + c) = 0, so the matrix comes back full of inf/nan."""
    a, b = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
    with np.errstate(
        all="raise"
    ):  # a divide-by-zero is the bug, not just its nan result
        turned = ac._rotation_between(a, b)
    assert np.isfinite(turned).all()
    assert turned @ a == pytest.approx(b)
