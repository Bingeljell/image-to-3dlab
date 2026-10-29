"""Tests for the duplicate-weights audit.

This script deletes multi-gigabyte files, so the tests worth having are the ones that
stop it deleting the wrong thing: a dry run must remove nothing, a partial match must not
be offered for deletion, and the "nothing references this" rule must never delete at all.
Both threshold bugs below were real and silent, found the day the script was written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_model_weights.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_model_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_model_weights"] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def _weights(root: Path, name: str, size: int) -> Path:
    """A file that *reports* `size` without occupying it.

    The thresholds under test are in gigabytes, and writing real zeros would add minutes
    of I/O per run for no extra coverage, so these are sparse.
    """
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def _valid_safetensors(path: Path) -> Path:
    """Write a minimal valid single-tensor safetensors file.

    This builds the same bytes ``safetensors.torch.save_file`` would emit for a
    one-element float32 tensor, without pulling in the library.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b'{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    data = b"\x00\x00\x00\x00"
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(data)
    return path


def test_a_checkpoint_beside_its_conversion_is_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    _valid_safetensors(tmp_path / "m/model.fp16.safetensors")
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is True


def test_a_checkpoint_with_no_conversion_is_left_alone(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    assert audit.find_superseded_checkpoints(roots=(tmp_path,)) == []


def test_a_checkpoint_beside_a_zero_byte_conversion_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    _weights(tmp_path, "m/model.fp16.safetensors", 0)
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_a_checkpoint_beside_garbage_conversion_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    (tmp_path / "m/model.fp16.safetensors").write_bytes(b"not a safetensors file")
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_a_checkpoint_beside_a_truncated_conversion_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    # Header length claims 1000 bytes, but the file ends right after that claim.
    (tmp_path / "m/model.fp16.safetensors").write_bytes(
        (1000).to_bytes(8, "little") + b"partial header"
    )
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_a_checkpoint_beside_an_overlong_header_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    # Header length claims 200 MB, far above the 100 MB cap, but the file is tiny.
    (tmp_path / "m/model.fp16.safetensors").write_bytes(
        (200 * 1024 * 1024).to_bytes(8, "little") + b"{}"
    )
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_a_checkpoint_beside_a_dict_without_tensor_offsets_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    # Valid JSON dict, but its values are not safetensors tensor metadata.
    header = b'{"t":{"not":"tensor"}}'
    (tmp_path / "m/model.fp16.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_a_checkpoint_beside_inverted_offsets_is_not_reclaimable(tmp_path):
    _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    # data_offsets start is greater than end.
    header = b'{"t":{"dtype":"F32","shape":[1],"data_offsets":[8,0]}}'
    (tmp_path / "m/model.fp16.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert [c.path.name for c in found] == ["model.fp16.ckpt"]
    assert found[0].deletable is False
    assert "conversion unproven" in found[0].reason


def test_symlinks_are_counted_once(tmp_path):
    # The HF cache keeps one copy in blobs/ and links to it from snapshots/; following
    # both doubles every figure the report prints.
    (tmp_path / "blobs").mkdir()
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "blobs" / "w.bin").write_bytes(b"\0" * 1000)
    (tmp_path / "snapshots" / "w.bin").symlink_to(tmp_path / "blobs" / "w.bin")
    assert audit.directory_size(tmp_path) == 1000


def _cache_entry(hub: Path, repo_id: str, files: dict[str, int]) -> Path:
    entry = hub / ("models--" + repo_id.replace("/", "--"))
    blobs, snap = entry / "blobs", entry / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for index, (name, size) in enumerate(files.items()):
        blob = blobs / f"blob{index}"
        with blob.open("wb") as handle:
            handle.truncate(size)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return entry


def test_a_fully_duplicated_cache_entry_is_offered_for_deletion(tmp_path):
    hub, local = tmp_path / "hub", tmp_path / "local"
    _cache_entry(hub, "org/model", {"big.safetensors": 4 * audit.GB})
    _weights(local, "big.safetensors", 4 * audit.GB)

    found = audit.find_cached_duplicates(hub=hub, roots=(local,))
    assert len(found) == 1
    assert found[0].deletable is True


def test_a_partly_duplicated_cache_entry_is_reported_but_not_deletable(tmp_path):
    """Half the bytes exist only in the cache, so deleting the entry would lose them.

    This is the real Hunyuan case: the checkpoint is duplicated in the repo, but a second
    file was converted to a different name, so matching by name under-counts.
    """
    hub, local = tmp_path / "hub", tmp_path / "local"
    _cache_entry(hub, "org/model", {
        "shared.ckpt": 4 * audit.GB,
        "converted.bin": 3 * audit.GB,
    })
    _weights(local, "shared.ckpt", 4 * audit.GB)

    found = audit.find_cached_duplicates(hub=hub, roots=(local,))
    assert len(found) == 1
    assert found[0].deletable is False
    assert "PARTIAL" in found[0].reason


def test_a_cache_entry_sharing_nothing_is_not_reported(tmp_path):
    hub, local = tmp_path / "hub", tmp_path / "local"
    _cache_entry(hub, "org/model", {"only.safetensors": 4 * audit.GB})
    _weights(local, "different.safetensors", 32)
    assert audit.find_cached_duplicates(hub=hub, roots=(local,)) == []


def test_unreferenced_entries_are_never_deletable(tmp_path):
    hub = tmp_path / "hub"
    _cache_entry(hub, "nobody/uses-this", {"w.bin": 1024})
    found = audit.find_unreferenced(hub=hub, repo=tmp_path)
    assert found and all(c.deletable is False for c in found)


def test_a_dry_run_deletes_nothing(tmp_path, capsys):
    ckpt = _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    _valid_safetensors(tmp_path / "m/model.fp16.safetensors")
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    assert found and found[0].deletable is True

    audit.report(found, apply=False)
    assert ckpt.is_file(), "a dry run removed a file"
    assert "dry run" in capsys.readouterr().out


def test_apply_deletes_only_the_deletable(tmp_path):
    ckpt = _weights(tmp_path, "m/model.fp16.ckpt", 2048)
    keep = _valid_safetensors(tmp_path / "m/model.fp16.safetensors")
    found = audit.find_superseded_checkpoints(roots=(tmp_path,))
    found.append(audit.Candidate("unreferenced", keep, keep.stat().st_size, "review only", deletable=False))

    audit.report(found, apply=True)
    assert not ckpt.exists()
    assert keep.is_file(), "apply removed an entry marked review-only"
