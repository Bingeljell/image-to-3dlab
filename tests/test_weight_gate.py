"""The shared refusal: name the backend, the route and the size, then stop."""

import os
from pathlib import Path

import pytest

from image_to_3dlab.weight_gate import refuse_hub_download, require_local, snapshot_dir


def test_refuse_names_the_backend_the_route_and_the_size():
    with pytest.raises(SystemExit) as raised:
        refuse_hub_download(backend="TRELLIS.2", route="Mac clean port", size="about 16.5 GB")
    message = str(raised.value)
    assert "TRELLIS.2" in message
    assert "Mac clean port" in message
    assert "about 16.5 GB" in message
    assert "does not fetch" in message


def test_an_empty_cache_is_not_a_snapshot(tmp_path):
    assert snapshot_dir("microsoft/TRELLIS.2-4B", hub=tmp_path) is None


def test_snapshot_dir_prefers_the_hub_main_ref(tmp_path, monkeypatch):
    model = tmp_path / "models--microsoft--TRELLIS.2-4B"
    current = model / "snapshots" / "current"
    stale = model / "snapshots" / "stale"
    current.mkdir(parents=True)
    stale.mkdir(parents=True)
    (current / "pipeline.json").write_text("{}\n")
    (stale / "pipeline.json").write_text("{}\n")
    refs = model / "refs"
    refs.mkdir()
    (refs / "main").write_text("current\n")

    original_iterdir = Path.iterdir

    def unordered_snapshots(path):
        if path == model / "snapshots":
            return iter((current, stale))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", unordered_snapshots)

    assert snapshot_dir("microsoft/TRELLIS.2-4B", hub=tmp_path) == current


def test_snapshot_dir_falls_back_to_the_newest_snapshot(tmp_path):
    snapshots = tmp_path / "models--microsoft--TRELLIS.2-4B" / "snapshots"
    old = snapshots / "old"
    newest = snapshots / "newest"
    old.mkdir(parents=True)
    newest.mkdir(parents=True)
    (old / "pipeline.json").write_text("{}\n")
    (newest / "pipeline.json").write_text("{}\n")
    os.utime(old, (1, 1))
    os.utime(newest, (2, 2))

    assert snapshot_dir("microsoft/TRELLIS.2-4B", hub=tmp_path) == newest


def test_require_local_returns_the_snapshot_and_not_the_hub_id(tmp_path, monkeypatch):
    snap = tmp_path / "models--microsoft--TRELLIS.2-4B" / "snapshots" / "main"
    snap.mkdir(parents=True)
    (snap / "pipeline.json").write_text("{}\n")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    found = require_local(
        "microsoft/TRELLIS.2-4B", backend="TRELLIS.2", route="Mac clean port",
        size="about 16.5 GB", hub=tmp_path,
    )
    assert found == str(snap)
    assert "microsoft/TRELLIS.2-4B" not in found
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"


def test_require_local_refuses_before_any_hub_call(tmp_path):
    with pytest.raises(SystemExit, match="about 16.5 GB"):
        require_local(
            "microsoft/TRELLIS.2-4B", backend="TRELLIS.2", route="Mac clean port",
            size="about 16.5 GB", hub=tmp_path,
        )
