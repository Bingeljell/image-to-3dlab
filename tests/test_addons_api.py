"""Tests for the studio add-on API.

Uses a fake runner and a local index file; no network, no real git in unit tests.
"""

from __future__ import annotations

import json
import time

import pytest

from tests.test_addons import (
    listing,
    make_bare_repo,
    make_installed_addon,
    manifest,
    repo_head,
)

viewer = pytest.importorskip("viewer.addons_api", reason="viewer not importable")


def _set_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "ADDONS_DIR", tmp_path / "addons")
    monkeypatch.setattr(viewer, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(viewer, "REPO", tmp_path)


def test_addons_payload_shows_installed_when_list_unreachable(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(viewer, "INDEX_FILE", str(tmp_path / "missing.json"))
    make_installed_addon(tmp_path, manifest())
    payload = viewer.addons_payload()
    assert payload["online"] is False
    assert payload["error"] is not None
    assert len(payload["addons"]) == 1
    assert payload["addons"][0]["id"] == "vehicle-kit"
    assert payload["addons"][0]["installed"] is True


def test_addons_payload_shows_listed_and_installed(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    payload = viewer.addons_payload()
    assert payload["online"] is True
    assert len(payload["addons"]) == 1
    assert payload["addons"][0]["listed"] is True
    assert payload["addons"][0]["installed"] is False


def test_addons_payload_marks_no_longer_listed(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    make_installed_addon(tmp_path, manifest())
    payload = viewer.addons_payload()
    assert payload["addons"][0]["listed"] is False
    assert payload["addons"][0].get("no_longer_listed") is True


def test_install_payload_returns_plan(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
    registry = tmp_path / "index.json"
    registry.write_text(
        json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
    )
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    payload = viewer.install_payload("vehicle-kit")
    assert payload["author"] == "Ahnyth"
    assert payload["licence"] == "Apache-2.0"
    assert "reviewed by AssetFurnace" in payload["warning"]


def test_apply_install_and_remove(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
    registry = tmp_path / "index.json"
    registry.write_text(
        json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
    )
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    viewer.apply_install("vehicle-kit")
    assert (tmp_path / "addons" / "vehicle-kit" / ".addon-install.json").is_file()
    viewer.apply_remove("vehicle-kit")
    assert not (tmp_path / "addons" / "vehicle-kit").exists()


def test_run_payload_starts_job(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    addon_dir = make_installed_addon(tmp_path, manifest())
    script = addon_dir / "shape_guide.py"
    script.write_text(
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "name = out.name\n"
        "(out / f'{name}.glb').write_text('glb', encoding='utf-8')\n",
        encoding="utf-8",
    )
    payload = viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"})
    assert "job_id" in payload
    job = viewer.ADDON_JOBS.get(payload["job_id"])
    assert job is not None
    # Wait for the short-running job to finish.
    deadline = time.monotonic() + 5
    while job.status not in ("done", "error") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.status == "done"
    assert job.result_path is not None


def test_run_payload_refuses_second_run(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    addon_dir = make_installed_addon(tmp_path, manifest())
    script = addon_dir / "shape_guide.py"
    script.write_text(
        "import sys, pathlib, time\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "name = out.name\n"
        "time.sleep(2)\n"
        "(out / f'{name}.glb').write_text('glb', encoding='utf-8')\n",
        encoding="utf-8",
    )
    first = viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"})
    with pytest.raises(RuntimeError, match="already in progress"):
        viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"})
    job = viewer.ADDON_JOBS.get(first["job_id"])
    assert job is not None
    viewer.ADDON_JOBS.cancel(first["job_id"])
