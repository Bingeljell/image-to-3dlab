"""Tests for the importable add-on install/remove/run helpers.

These import the real functions from image_to_3dlab.addon_install and never touch the
network. Helpers are reused from tests.test_addons.
"""

from __future__ import annotations

import json

import pytest

from image_to_3dlab import addon_install as ai
from tests.test_addons import (
    listing,
    make_bare_repo,
    make_installed_addon,
    manifest,
    repo_head,
)


def test_install_addon_without_confirm_returns_plan_and_does_not_install(tmp_path):
    registry = tmp_path / "index.json"
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
    registry.write_text(
        json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
    )
    result = ai.install_addon(
        addon_id="vehicle-kit",
        index=str(registry),
        confirm=False,
        addons_dir=tmp_path / "addons",
        repo_dir=tmp_path,
    )
    assert "plan" in result
    assert result["plan"]["author"] == "Ahnyth"
    assert not (tmp_path / "addons" / "vehicle-kit").exists()


def test_install_addon_with_confirm_installs_and_writes_record(tmp_path):
    registry = tmp_path / "index.json"
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
    registry.write_text(
        json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
    )
    result = ai.install_addon(
        addon_id="vehicle-kit",
        index=str(registry),
        confirm=True,
        addons_dir=tmp_path / "addons",
        repo_dir=tmp_path,
    )
    assert result["installed"] is True
    install_json = tmp_path / "addons" / "vehicle-kit" / ".addon-install.json"
    assert install_json.is_file()
    data = json.loads(install_json.read_text(encoding="utf-8"))
    assert data["id"] == "vehicle-kit"
    assert data["commit"] == repo_head(repo)
    assert data["reviewed"] is True


def test_install_addon_refuses_bad_id(tmp_path):
    with pytest.raises(ValueError, match="invalid add-on id"):
        ai.install_addon(addon_id="../bad", addons_dir=tmp_path / "addons", repo_dir=tmp_path)


def test_install_addon_refuses_unlisted(tmp_path):
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    with pytest.raises(ValueError, match="not on the reviewed registry"):
        ai.install_addon(
            addon_id="not-listed",
            index=str(registry),
            addons_dir=tmp_path / "addons",
            repo_dir=tmp_path,
        )


def test_remove_addon_removes_installed(tmp_path):
    make_installed_addon(tmp_path, manifest())
    ai.remove_addon("vehicle-kit", addons_dir=tmp_path / "addons")
    assert not (tmp_path / "addons" / "vehicle-kit").exists()


def test_remove_addon_refuses_missing(tmp_path):
    with pytest.raises(ValueError, match="not installed"):
        ai.remove_addon("vehicle-kit", addons_dir=tmp_path / "addons")


def test_remove_addon_refuses_path_escape(tmp_path):
    with pytest.raises(ValueError, match="invalid add-on id"):
        ai.remove_addon("../output", addons_dir=tmp_path / "addons")


def test_run_addon_runs_tool_and_returns_result_path(tmp_path):
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
    result = ai.run_addon(
        "vehicle-kit",
        "build",
        {"preset": "bus_12m"},
        run_stem="run-abc",
        output_root=tmp_path,
        addons_dir=tmp_path / "addons",
        repo_dir=tmp_path,
    )
    assert result == (tmp_path / "output" / "addon-vehicle-kit" / "run-abc" / "run-abc.glb").resolve()
    assert result.is_file()
    assert result.with_suffix(".provenance.json").is_file()


def test_run_addon_refuses_missing_tool(tmp_path):
    make_installed_addon(tmp_path, manifest())
    with pytest.raises(ValueError, match="tool"):
        ai.run_addon(
            "vehicle-kit",
            "missing",
            {"preset": "bus_12m"},
            addons_dir=tmp_path / "addons",
            repo_dir=tmp_path,
        )


def test_load_registry_reads_local_file(tmp_path):
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    entries = ai.load_registry(str(registry))
    assert len(entries) == 1
    assert entries[0].id == "vehicle-kit"


def test_list_installed_returns_installed_ids(tmp_path):
    make_installed_addon(tmp_path, manifest())
    assert ai.list_installed(addons_dir=tmp_path / "addons") == ["vehicle-kit"]
