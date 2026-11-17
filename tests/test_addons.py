"""Add-on core: manifest validation, command building, and provenance records.

These tests import the real functions and never touch the network or a real git
clone. A local bare repository in tmp_path stands in for a listed add-on repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from image_to_3dlab import addons

RUN_PYTHON = sys.executable


def manifest(**overrides) -> dict:
    """A valid schema-1 add-on manifest, with overrides applied."""
    base = {
        "schema": 1,
        "id": "vehicle-kit",
        "name": "Vehicle kit",
        "version": "0.1.0",
        "author": {"name": "Ahnyth", "url": "https://github.com/Ahnyth"},
        "licence": "Apache-2.0",
        "summary": "Buses, vans and metro cars from a few numbers.",
        "needs": {"assetfurnace": ">=0.6.0", "blender": ">=4.2"},
        "setup": [RUN_PYTHON, "-c", "print('setup ok')"],
        "models": [],
        "tools": [
            {
                "id": "build",
                "label": "Build a vehicle",
                "hint": "A bus, van or metro car with doors and wheels.",
                "inputs": [
                    {
                        "id": "preset",
                        "type": "choice",
                        "label": "Vehicle",
                        "choices": ["bus_12m", "van_long"],
                    },
                    {
                        "id": "length",
                        "type": "number",
                        "label": "Length (m)",
                        "min": 4,
                        "max": 20,
                        "optional": True,
                    },
                ],
                "run": [RUN_PYTHON, "shape_guide.py", "{out}", "--preset", "{preset}"],
                "makes": "glb",
            }
        ],
    }
    base.update(overrides)
    return base


def listing(**overrides) -> dict:
    """A valid registry listing, with overrides applied."""
    base = {
        "id": "vehicle-kit",
        "name": "Vehicle kit",
        "repo": "https://github.com/Ahnyth/vehicle-kit.git",
        "commit": "a" * 40,
        "author": {"name": "Ahnyth", "url": "https://github.com/Ahnyth"},
        "summary": "Buses, vans and metro cars from a few numbers.",
        "type": "tool",
        "links": {"repo": "https://github.com/Ahnyth/vehicle-kit", "site": "https://vehiclekit.example"},
        "rating": 4.5,
        "ratings_count": 12,
    }
    base.update(overrides)
    return base


def write_manifest(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _manifest_object(tmp_path: Path, data: dict) -> addons.Manifest:
    """Build a fully-typed Manifest from a dict via read_manifest."""
    return addons.read_manifest(write_manifest(tmp_path / "addon.json", data))


class TestIdValidation:
    def test_accepts_valid_ids(self):
        for addon_id in ("vehicle-kit", "a1", "foo_bar", "x"):
            addons.validate_addon_id(addon_id)

    def test_rejects_path_like_ids(self):
        for addon_id in ("../output", "foo/bar", "..", "."):
            with pytest.raises(ValueError, match="invalid add-on id"):
                addons.validate_addon_id(addon_id)

    def test_rejects_uppercase_and_dots(self):
        for addon_id in ("Vehicle-Kit", "foo.bar", "-start", "_start"):
            with pytest.raises(ValueError, match="invalid add-on id"):
                addons.validate_addon_id(addon_id)


class TestReadManifest:
    def test_loads_a_valid_manifest(self, tmp_path):
        path = write_manifest(tmp_path / "addon.json", manifest())
        m = addons.read_manifest(path)
        assert m.id == "vehicle-kit"
        assert m.schema == 1
        assert m.tools[0].id == "build"
        assert m.tools[0].inputs[0].id == "preset"

    def test_rejects_non_object_json(self, tmp_path):
        path = tmp_path / "addon.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="addon.json must be an object"):
            addons.read_manifest(path)

    def test_rejects_missing_schema(self, tmp_path):
        data = manifest()
        del data["schema"]
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="schema"):
            addons.read_manifest(path)

    def test_rejects_wrong_schema_version(self, tmp_path):
        path = write_manifest(tmp_path / "addon.json", manifest(schema=2))
        with pytest.raises(ValueError, match="schema"):
            addons.read_manifest(path)

    def test_rejects_missing_id(self, tmp_path):
        data = manifest()
        del data["id"]
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="id"):
            addons.read_manifest(path)

    def test_rejects_tool_without_id(self, tmp_path):
        data = manifest()
        del data["tools"][0]["id"]
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="tools\\[0\\].*'id'"):
            addons.read_manifest(path)

    def test_rejects_input_without_type(self, tmp_path):
        data = manifest()
        del data["tools"][0]["inputs"][0]["type"]
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="tools\\[0\\].inputs\\[0\\].*'type'"):
            addons.read_manifest(path)

    def test_rejects_unknown_input_type(self, tmp_path):
        data = manifest()
        data["tools"][0]["inputs"][0]["type"] = "colour"
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="tools\\[0\\].inputs\\[0\\].*type.*'colour'"):
            addons.read_manifest(path)

    def test_rejects_choice_without_choices(self, tmp_path):
        data = manifest()
        del data["tools"][0]["inputs"][0]["choices"]
        path = write_manifest(tmp_path / "addon.json", data)
        with pytest.raises(ValueError, match="tools\\[0\\].inputs\\[0\\].*choices"):
            addons.read_manifest(path)


class TestReadRegistry:
    def test_reads_registry_entries(self):
        entries = addons.read_registry([listing(), listing(id="mac-wrapper", type="app")])
        assert len(entries) == 2
        assert entries[0].id == "vehicle-kit"
        assert entries[0].type == "tool"
        assert entries[1].id == "mac-wrapper"
        assert entries[1].type == "app"

    def test_reads_links_and_rating(self):
        entry = addons.read_registry([listing()])[0]
        assert entry.links == {"repo": "https://github.com/Ahnyth/vehicle-kit",
                               "site": "https://vehiclekit.example"}
        assert entry.rating == 4.5
        assert entry.ratings_count == 12

    def test_missing_links_and_rating_default_safely(self):
        entry = addons.read_registry([listing(links=None, rating=None, ratings_count=None)])[0]
        assert entry.links == {}
        assert entry.rating is None
        assert entry.ratings_count == 0

    def test_registry_entry_requires_commit(self):
        with pytest.raises(ValueError, match="commit"):
            addons.read_registry([listing(commit=None)])

    def test_registry_entry_requires_type(self):
        with pytest.raises(ValueError, match="type"):
            addons.read_registry([listing(type=None)])

    def test_registry_entry_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="type"):
            addons.read_registry([listing(type="library")])

    def test_registry_rejects_bad_id(self):
        with pytest.raises(ValueError, match="invalid add-on id"):
            addons.read_registry([listing(id="../bad")])


class TestInstallPlan:
    def test_plan_includes_review_line(self, tmp_path):
        m = _manifest_object(tmp_path, manifest())
        entry = addons.Listing(**listing())
        plan = addons.install_plan(entry, m)
        assert plan["author"] == "Ahnyth"
        assert plan["licence"] == "Apache-2.0"
        assert "reviewed by AssetFurnace" in plan["warning"]
        assert entry.commit in plan["warning"]

    def test_plan_says_not_reviewed_for_unreviewed(self, tmp_path):
        m = _manifest_object(tmp_path, manifest())
        entry = addons.Listing(**listing())
        plan = addons.install_plan(entry, m, reviewed=False)
        assert "not reviewed" in plan["warning"].lower()
        assert "not reviewed by AssetFurnace" in plan["warning"]

    def test_plan_lists_models(self, tmp_path):
        m = _manifest_object(
            tmp_path,
            manifest(
                models=[
                    {
                        "name": "shape predictor",
                        "licence": "MIT",
                        "licence_url": "https://mit.edu/",
                        "size": "12 MB",
                    }
                ]
            ),
        )
        entry = addons.Listing(**listing())
        plan = addons.install_plan(entry, m)
        assert len(plan["models"]) == 1
        assert plan["models"][0]["name"] == "shape predictor"


class TestParseInputs:
    def test_text_stays_string(self):
        data = manifest()
        data["tools"][0]["inputs"] = [{"id": "code", "type": "text", "label": "Code"}]
        tool = addons.tool_from_dict(data["tools"][0])
        parsed = addons.parse_inputs({"code": "007"}, tool)
        assert parsed == {"code": "007"}

    def test_number_coerces_int(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        parsed = addons.parse_inputs({"preset": "bus_12m", "length": "12"}, tool)
        assert parsed["length"] == 12
        assert isinstance(parsed["length"], int)

    def test_number_coerces_float(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        parsed = addons.parse_inputs({"preset": "bus_12m", "length": "12.5"}, tool)
        assert parsed["length"] == 12.5

    def test_toggle_coerces_true_and_false(self):
        data = manifest()
        data["tools"][0]["inputs"] = [{"id": "flag", "type": "toggle", "label": "Flag"}]
        tool = addons.tool_from_dict(data["tools"][0])
        assert addons.parse_inputs({"flag": "true"}, tool)["flag"] is True
        assert addons.parse_inputs({"flag": "no"}, tool)["flag"] is False

    def test_picture_resolves_and_checks_exists(self, tmp_path):
        picture = tmp_path / "photo.png"
        picture.write_text("png", encoding="utf-8")
        data = manifest()
        data["tools"][0]["inputs"] = [{"id": "photo", "type": "picture", "label": "Photo"}]
        tool = addons.tool_from_dict(data["tools"][0])
        parsed = addons.parse_inputs({"photo": "photo.png"}, tool, cwd=tmp_path)
        assert parsed["photo"] == str(picture.resolve())

    def test_picture_refuses_missing_file(self, tmp_path):
        data = manifest()
        data["tools"][0]["inputs"] = [{"id": "photo", "type": "picture", "label": "Photo"}]
        tool = addons.tool_from_dict(data["tools"][0])
        with pytest.raises(ValueError, match="does not exist"):
            addons.parse_inputs({"photo": "missing.png"}, tool, cwd=tmp_path)


class TestBuildCommand:
    def test_fills_placeholders(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        argv = addons.build_command(
            tool, {"preset": "bus_12m"}, out_dir="/tmp/out", blender="/usr/bin/blender"
        )
        assert argv == [
            RUN_PYTHON,
            "shape_guide.py",
            "/tmp/out",
            "--preset",
            "bus_12m",
        ]

    def test_uses_blender_placeholder(self):
        data = manifest()
        data["tools"][0]["run"] = ["{blender}", "--background", "--python", "export.py"]
        data["tools"][0]["inputs"] = []
        tool = addons.tool_from_dict(data["tools"][0])
        argv = addons.build_command(tool, {}, out_dir="/tmp/out", blender="/apps/blender")
        assert argv[0] == "/apps/blender"

    def test_refuses_unknown_input(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        with pytest.raises(ValueError, match="unknown input"):
            addons.build_command(
                tool, {"preset": "bus_12m", "colour": "red"}, out_dir="/tmp/out", blender="/b"
            )

    def test_refuses_missing_required_input(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        with pytest.raises(ValueError, match="missing"):
            addons.build_command(tool, {}, out_dir="/tmp/out", blender="/b")

    def test_refuses_choice_not_offered(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        with pytest.raises(ValueError, match="preset"):
            addons.build_command(
                tool, {"preset": "spaceship"}, out_dir="/tmp/out", blender="/b"
            )

    def test_refuses_number_below_min(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        with pytest.raises(ValueError, match="length"):
            addons.build_command(
                tool, {"preset": "bus_12m", "length": 2}, out_dir="/tmp/out", blender="/b"
            )

    def test_refuses_number_above_max(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        with pytest.raises(ValueError, match="length"):
            addons.build_command(
                tool, {"preset": "bus_12m", "length": 25}, out_dir="/tmp/out", blender="/b"
            )

    def test_optional_input_can_be_omitted(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        argv = addons.build_command(
            tool, {"preset": "bus_12m"}, out_dir="/tmp/out", blender="/b"
        )
        assert "--length" not in argv and "length" not in argv

    def test_never_returns_shell_string(self):
        tool = addons.tool_from_dict(manifest()["tools"][0])
        argv = addons.build_command(
            tool, {"preset": "bus_12m"}, out_dir="/tmp/out", blender="/b"
        )
        assert isinstance(argv, list)
        assert all(isinstance(a, str) for a in argv)


class TestOutputDir:
    def test_output_dir_shape(self):
        path = addons.output_dir("vehicle-kit", "run-123")
        assert path == Path("addon-vehicle-kit/run-123")


class TestRunRecord:
    def test_record_matches_provenance_expectations(self, tmp_path):
        m = _manifest_object(tmp_path, manifest())
        entry = addons.Listing(**listing())
        tool = m.tools[0]
        record = addons.run_record(entry, m, tool, {"preset": "bus_12m"}, reviewed=True)
        assert record["schema_version"] == 1
        assert record["addon"]["id"] == "vehicle-kit"
        assert record["addon"]["commit"] == entry.commit
        assert record["addon"]["licence"] == "Apache-2.0"
        assert record["addon"]["reviewed"] is True
        assert record["tool"]["id"] == "build"
        assert record["models"] == []

    def test_record_includes_declared_models(self, tmp_path):
        m = _manifest_object(
            tmp_path,
            manifest(
                models=[
                    {
                        "name": "shape predictor",
                        "licence": "MIT",
                        "licence_url": "https://mit.edu/",
                        "size": "12 MB",
                    }
                ]
            ),
        )
        entry = addons.Listing(**listing())
        record = addons.run_record(entry, m, m.tools[0], {"preset": "bus_12m"}, reviewed=False)
        assert record["models"][0]["name"] == "shape predictor"
        assert record["models"][0]["licence"] == "MIT"
        assert record["addon"]["reviewed"] is False


class TestScriptRefusals:
    def test_unlisted_id_is_refused(self, tmp_path):
        registry = tmp_path / "index.json"
        registry.write_text(json.dumps([listing()]), encoding="utf-8")
        result = run_addon_script(
            tmp_path, ["add", "not-listed", "--index", str(registry), "--yes"]
        )
        assert result.returncode != 0
        assert "not on the reviewed registry" in result.stderr.lower()

    def test_head_mismatch_is_refused(self, tmp_path):
        registry = tmp_path / "index.json"
        registry.write_text(json.dumps([listing()]), encoding="utf-8")
        repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
        # Point the registry at a different commit than the repo's HEAD.
        entries = [listing(repo=str(repo), commit="b" * 40)]
        registry.write_text(json.dumps(entries), encoding="utf-8")
        result = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry), "--yes"]
        )
        assert result.returncode != 0
        assert "head" in result.stderr.lower() or "commit" in result.stderr.lower()

    def test_no_install_without_yes(self, tmp_path):
        registry = tmp_path / "index.json"
        repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
        registry.write_text(
            json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
        )
        result = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry)]
        )
        assert result.returncode != 0
        assert not (tmp_path / "addons" / "vehicle-kit").exists()

    def test_reinstall_keeps_existing_until_yes(self, tmp_path):
        registry = tmp_path / "index.json"
        repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
        registry.write_text(
            json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
        )
        # First install.
        first = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry), "--yes"]
        )
        assert first.returncode == 0, first.stderr
        marker = tmp_path / "addons" / "vehicle-kit" / "installed.txt"
        marker.write_text("old", encoding="utf-8")
        # Second install without --yes leaves the existing folder alone.
        second = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry)]
        )
        assert second.returncode != 0
        assert marker.read_text(encoding="utf-8") == "old"

    def test_manifest_id_mismatch_is_refused(self, tmp_path):
        registry = tmp_path / "index.json"
        repo = make_bare_repo(
            tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest(id="wrong-id"))}
        )
        registry.write_text(
            json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
        )
        result = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry), "--yes"]
        )
        assert result.returncode != 0
        assert "does not match" in result.stderr.lower()

    def test_remove_refuses_path_escape(self, tmp_path):
        result = run_addon_script(tmp_path, ["remove", "../output"])
        assert result.returncode != 0
        assert "invalid add-on id" in result.stderr.lower()

    def test_run_refuses_path_escape(self, tmp_path):
        result = run_addon_script(tmp_path, ["run", "../output", "build"])
        assert result.returncode != 0
        assert "invalid add-on id" in result.stderr.lower()

    def test_add_refuses_path_escape(self, tmp_path):
        result = run_addon_script(tmp_path, ["add", "../output"])
        assert result.returncode != 0
        assert "invalid add-on id" in result.stderr.lower()

    def test_unreviewed_derived_id_is_validated(self, tmp_path):
        result = run_addon_script(
            tmp_path,
            ["add", "--unreviewed", "https://example.com/../Bad-Name.git@" + "a" * 40, "--yes"],
        )
        assert result.returncode != 0
        assert "invalid add-on id" in result.stderr.lower()

    def test_missing_result_file_is_refused(self, tmp_path):
        addon_dir = make_installed_addon(tmp_path, manifest())
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'wrong-name.glb').write_text('glb', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "run-123",
                "--yes",
            ],
        )
        assert result.returncode != 0
        assert "expected result" in result.stderr.lower()

    def test_files_outside_out_dir_are_ignored(self, tmp_path):
        addon_dir = make_installed_addon(tmp_path, manifest())
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'run-123.glb').write_text('glb', encoding='utf-8')\n"
            "# write something outside out_dir; run should ignore it\n"
            "(pathlib.Path(sys.argv[1]).parent / 'escape.glb').write_text('bad', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "run-123",
                "--yes",
            ],
        )
        assert result.returncode == 0, result.stderr


class TestScriptHappyPath:
    def test_run_writes_provenance(self, tmp_path):
        addon_dir = make_installed_addon(tmp_path, manifest())
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'run-abc.glb').write_text('glb', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "run-abc",
                "--yes",
            ],
        )
        assert result.returncode == 0, result.stderr
        run_dir = tmp_path / "addon-vehicle-kit" / "run-abc"
        assert (run_dir / "run-abc.glb").is_file()
        assert (run_dir / "run-abc.provenance.json").is_file()

    def test_two_default_runs_get_separate_provenance_files(self, tmp_path):
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
        first = run_addon_script(
            tmp_path,
            ["run", "vehicle-kit", "build", "--input", "preset=bus_12m", "--yes"],
        )
        assert first.returncode == 0, first.stderr
        time.sleep(1)
        second = run_addon_script(
            tmp_path,
            ["run", "vehicle-kit", "build", "--input", "preset=bus_12m", "--yes"],
        )
        assert second.returncode == 0, second.stderr
        run_dirs = list((tmp_path / "addon-vehicle-kit").iterdir())
        assert len(run_dirs) == 2
        for run_dir in run_dirs:
            assert any(p.suffix == ".glb" for p in run_dir.iterdir())
            assert any(p.suffix == ".json" for p in run_dir.iterdir())

    def test_explicit_run_stem_overwrite_accepted_when_modified(self, tmp_path):
        addon_dir = make_installed_addon(tmp_path, manifest())
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'same.glb').write_text('glb', encoding='utf-8')\n",
            encoding="utf-8",
        )
        first = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "same",
                "--yes",
            ],
        )
        assert first.returncode == 0, first.stderr
        second = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "same",
                "--yes",
            ],
        )
        assert second.returncode == 0, second.stderr

    def test_install_writes_install_json(self, tmp_path):
        registry = tmp_path / "index.json"
        repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
        registry.write_text(
            json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8"
        )
        result = run_addon_script(
            tmp_path, ["add", "vehicle-kit", "--index", str(registry), "--yes"]
        )
        assert result.returncode == 0, result.stderr
        install_json = tmp_path / "addons" / "vehicle-kit" / ".addon-install.json"
        assert install_json.is_file()
        data = json.loads(install_json.read_text(encoding="utf-8"))
        assert data["id"] == "vehicle-kit"
        assert data["repo"] == str(repo)
        assert data["commit"] == repo_head(repo)
        assert data["reviewed"] is True

    def test_unreviewed_install_marks_not_reviewed(self, tmp_path):
        repo = make_bare_repo(tmp_path, "pedestal", files={"addon.json": json.dumps(manifest(id="pedestal"))})
        commit = repo_head(repo)
        result = run_addon_script(
            tmp_path,
            [
                "add",
                "--unreviewed",
                f"{repo}@{commit}",
                "--yes",
            ],
        )
        assert result.returncode == 0, result.stderr
        install_json = tmp_path / "addons" / "pedestal" / ".addon-install.json"
        data = json.loads(install_json.read_text(encoding="utf-8"))
        assert data["reviewed"] is False
        assert "not reviewed" in result.stdout.lower()

    def test_run_reads_install_json_not_registry(self, tmp_path):
        addon_dir = make_installed_addon(
            tmp_path,
            manifest(),
            install={"repo": "https://example.com/test.git", "commit": "b" * 40, "reviewed": False},
        )
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'run-abc.glb').write_text('glb', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "run-abc",
                "--yes",
            ],
        )
        assert result.returncode == 0, result.stderr
        record = json.loads(
            (tmp_path / "addon-vehicle-kit" / "run-abc" / "run-abc.provenance.json")
            .read_text(encoding="utf-8")
        )
        assert record["addon"]["repo"] == "https://example.com/test.git"
        assert record["addon"]["commit"] == "b" * 40
        assert record["addon"]["reviewed"] is False

    def test_run_defaults_missing_reviewed_to_false(self, tmp_path):
        addon_dir = make_installed_addon(
            tmp_path,
            manifest(),
            install={"repo": "https://example.com/test.git", "commit": "b" * 40},
        )
        install_json = addon_dir / ".addon-install.json"
        assert "reviewed" not in json.loads(install_json.read_text(encoding="utf-8"))
        script = addon_dir / "shape_guide.py"
        script.write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'run-abc.glb').write_text('glb', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_addon_script(
            tmp_path,
            [
                "run",
                "vehicle-kit",
                "build",
                "--input",
                "preset=bus_12m",
                "--run-stem",
                "run-abc",
                "--yes",
            ],
        )
        assert result.returncode == 0, result.stderr
        record = json.loads(
            (tmp_path / "addon-vehicle-kit" / "run-abc" / "run-abc.provenance.json")
            .read_text(encoding="utf-8")
        )
        assert record["addon"]["reviewed"] is False

    def test_list_shows_author_links_and_rating(self, tmp_path):
        registry = tmp_path / "index.json"
        registry.write_text(json.dumps([listing()]), encoding="utf-8")
        result = run_addon_script(tmp_path, ["list", "--index", str(registry)])
        assert result.returncode == 0, result.stderr
        assert "Vehicle kit by Ahnyth" in result.stdout
        assert "* 4.5 (12)" in result.stdout
        assert "author: https://github.com/Ahnyth" in result.stdout
        assert "site: https://vehiclekit.example" in result.stdout


def make_bare_repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    """Create a local bare git repo with the given files committed."""
    repo = tmp_path / f"{name}.git"
    repo.mkdir()
    subprocess.run(["git", "init", "--bare", str(repo)], check=True, capture_output=True)
    work = tmp_path / f"{name}-work"
    work.mkdir()
    subprocess.run(["git", "clone", str(repo), str(work)], check=True, capture_output=True)
    for rel, content in files.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(work), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(work), "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(work), "push", "origin", "HEAD"], check=True, capture_output=True)
    return repo


def repo_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_installed_addon(tmp_path: Path, data: dict, install: dict | None = None) -> Path:
    """Create a fake already-installed add-on directory with a valid manifest."""
    addon_dir = tmp_path / "addons" / data["id"]
    addon_dir.mkdir(parents=True)
    write_manifest(addon_dir / "addon.json", data)
    install_record = {
        "id": data["id"],
        "repo": install.get("repo", "") if install else "",
        "commit": install.get("commit", "") if install else "",
        "installed_at": "2026-01-01T00:00:00+00:00",
    }
    if install and "reviewed" in install:
        install_record["reviewed"] = install["reviewed"]
    (addon_dir / ".addon-install.json").write_text(
        json.dumps(install_record, indent=2), encoding="utf-8"
    )
    return addon_dir


def run_addon_script(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run scripts/addon.py with a temporary addons/output dir and no network."""
    repo = Path(__file__).resolve().parents[1]
    # Inherit the parent environment so the child can find git on PATH and
    # resolve Path.home() on Windows, while still overriding our own dirs.
    env = os.environ.copy()
    env.update({
        "ADDONS_DIR": str(tmp_path / "addons"),
        "OUTPUT_DIR": str(tmp_path),
        "PYTHONPATH": str(repo),
    })
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "addon.py"), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
