"""Tests for the importable add-on install/remove/run helpers.

These import the real functions from image_to_3dlab.addon_install and never touch the
network. Helpers are reused from tests.test_addons.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from image_to_3dlab import addon_install as ai
from tests.test_addons import (
    RUN_PYTHON,
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
    # nor its temporary clone: on Windows its read-only git objects kept every planned install's clone in addons/
    addons = tmp_path / "addons"
    assert not addons.exists() or not any(addons.iterdir())


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
    assert data["tier"] == "listed"


def test_install_addon_refuses_bad_id(tmp_path):
    with pytest.raises(ValueError, match="invalid add-on id"):
        ai.install_addon(addon_id="../bad", addons_dir=tmp_path / "addons", repo_dir=tmp_path)


def test_install_addon_refuses_unlisted(tmp_path):
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    with pytest.raises(ValueError, match="not on the add-on list"):
        ai.install_addon(
            addon_id="not-listed",
            index=str(registry),
            addons_dir=tmp_path / "addons",
            repo_dir=tmp_path,
        )


def test_install_addon_again_replaces_a_read_only_clone(tmp_path):
    # Windows leaves a cloned repo's git objects read-only, so installing a new version over the old one stopped with
    # "Access is denied" (PermissionError) while removing it worked
    import stat

    registry = tmp_path / "index.json"
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest())})
    registry.write_text(json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8")

    def install():
        return ai.install_addon(addon_id="vehicle-kit", index=str(registry), confirm=True,
                                addons_dir=tmp_path / "addons", repo_dir=tmp_path)

    install()
    stuck = tmp_path / "addons" / "vehicle-kit" / "object"
    stuck.write_text("git object", encoding="utf-8")
    os.chmod(stuck, stat.S_IREAD)
    assert install()["installed"] is True
    assert not stuck.exists()


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
    assert result == (tmp_path / "addon-vehicle-kit" / "run-abc" / "run-abc.glb").resolve()
    assert result.is_file()
    assert result.with_suffix(".provenance.json").is_file()


def test_run_addon_starts_the_tool_in_the_add_ons_own_folder_for_PWD_too(tmp_path):
    """Blender on Linux finds a relative `-P script.py` from $PWD, so $PWD must name the add-on's folder, not ours."""
    addon_dir = make_installed_addon(tmp_path, manifest())
    (addon_dir / "shape_guide.py").write_text(
        "import os, sys, pathlib\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'where.txt').write_text(os.getcwd() + '\\n' + os.environ['PWD'], encoding='utf-8')\n"
        "(out / f'{out.name}.glb').write_text('glb', encoding='utf-8')\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old_pwd, old_cwd = os.environ.get("PWD"), os.getcwd()
    os.chdir(elsewhere)
    os.environ["PWD"] = str(elsewhere)
    try:
        result = ai.run_addon("vehicle-kit", "build", {"preset": "bus_12m"}, run_stem="run-pwd", output_root=tmp_path,
                              addons_dir=tmp_path / "addons", repo_dir=tmp_path)
    finally:
        os.chdir(old_cwd)
        if old_pwd is None:
            os.environ.pop("PWD", None)
        else:
            os.environ["PWD"] = old_pwd
    cwd, pwd = (result.parent / "where.txt").read_text(encoding="utf-8").splitlines()
    assert Path(cwd).resolve() == addon_dir.resolve() and Path(pwd).resolve() == addon_dir.resolve()


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


def test_result_is_fresh_true_for_new_file(tmp_path):
    started = 0.0
    path = tmp_path / "new.glb"
    path.write_text("glb", encoding="utf-8")
    assert ai.result_is_fresh(path, started) is True


def test_result_is_fresh_true_for_overwritten_file(tmp_path):
    path = tmp_path / "old.glb"
    path.write_text("old", encoding="utf-8")
    started = path.stat().st_mtime
    path.write_text("new", encoding="utf-8")
    assert ai.result_is_fresh(path, started) is True


def test_result_is_fresh_false_for_unchanged_file(tmp_path):
    path = tmp_path / "old.glb"
    path.write_text("old", encoding="utf-8")
    started = path.stat().st_mtime + 1.0
    assert ai.result_is_fresh(path, started) is False


def test_cancel_stops_a_run_that_prints_nothing():
    import sys
    import time

    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    result = ai._run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        Path.cwd(), dict(os.environ), None, cancel_event=cancel,
    )
    assert time.monotonic() - started < 10
    assert result.returncode != 0


# "reading its output" needs the runner to fail decoding the byte \xff. Text mode uses the locale
# encoding: on Windows that is cp1252, where \xff decodes fine, so nothing raises there.
@pytest.mark.parametrize("fails", [
    "writing its log",
    pytest.param("reading its output", marks=pytest.mark.posix),
])
def test_a_run_that_fails_on_the_way_stops_its_tool_and_its_cancel_watcher(fails):
    # A line that could not be logged (a full disk) or read (output that is not text) ended the
    # run with an error. The tool ran on with nobody reading its output, and the thread that
    # watches for Cancel kept watching for good: nothing was left that could stop either.
    import sys
    import time

    said = "print('started', flush=True)" if fails == "writing its log" else "sys.stdout.buffer.write(b'\\xff\\n')"
    started, before = [], set(threading.enumerate())

    def log(_line):
        raise OSError("No space left on device")

    with pytest.raises((OSError, UnicodeDecodeError)):
        ai._run_subprocess(
            [sys.executable, "-c", f"import sys, time; {said}; sys.stdout.flush(); time.sleep(30)"],
            Path.cwd(), {**os.environ, "PYTHONIOENCODING": "utf-8"}, log,
            cancel_event=threading.Event(), on_start=started.append)
    [tool] = started
    try:
        deadline = time.monotonic() + 5
        while (tool.poll() is None or set(threading.enumerate()) - before) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert tool.poll() is not None, "the tool ran on"
        assert not set(threading.enumerate()) - before, "the Cancel watcher ran on"
    finally:
        if tool.poll() is None:
            tool.kill()


@pytest.mark.posix
def test_a_failed_run_whose_tool_cannot_be_stopped_still_raises_its_error_with_the_output_closed(monkeypatch):
    # When the group could not be stopped (kill_group swallows that), the run waited for the
    # tool for good: the error never surfaced, the add-on's slot stayed taken, and the
    # tool's output pipe stayed open. It now waits only the grace period, then raises.
    import sys

    monkeypatch.setattr(ai.processes, "kill_group", lambda pid: None)
    monkeypatch.setattr(ai, "CANCEL_GRACE_SECONDS", 0.5)
    started, raised = [], []

    def log(_line):
        raise OSError("No space left on device")

    def run():
        try:
            ai._run_subprocess(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                Path.cwd(), {**os.environ, "PYTHONIOENCODING": "utf-8"}, log,
                cancel_event=threading.Event(), on_start=started.append)
        except BaseException as error:
            raised.append(error)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        runner.join(timeout=5)
        assert not runner.is_alive(), "the run waited on a tool that could not be stopped"
        [error] = raised
        assert isinstance(error, OSError)
        [tool] = started
        assert tool.stdout.closed, "the tool's output stayed open"
    finally:
        for tool in started:
            if tool.poll() is None:
                tool.kill()
            tool.wait()
            if tool.stdout is not None:
                tool.stdout.close()


def _model_tool_addon(tmp_path):
    data = manifest(tools=[{
        "id": "tweak", "label": "Tweak", "hint": "Changes a model.",
        "inputs": [{"id": "src", "type": "model", "label": "Model"}],
        "run": [RUN_PYTHON, "tweak.py", "{out}", "{src}"], "makes": "glb",
    }])
    addon_dir = make_installed_addon(tmp_path, data)
    (addon_dir / "tweak.py").write_text(
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "(out / f'{out.name}.glb').write_text('glb', encoding='utf-8')\n",
        encoding="utf-8",
    )
    return addon_dir


def _run_tweak(tmp_path, source):
    return ai.run_addon("vehicle-kit", "tweak", {"src": str(source)}, run_stem="run-abc",
                        output_root=tmp_path, addons_dir=tmp_path / "addons", repo_dir=tmp_path)


def test_result_keeps_the_licence_restriction_of_the_model_that_went_in(tmp_path):
    _model_tool_addon(tmp_path)
    source = tmp_path / "hunyuan.glb"
    source.write_text("glb", encoding="utf-8")
    source.with_suffix(".provenance.json").write_text(json.dumps({
        "license": {"name": "Tencent Hunyuan Community", "conditions": ["Not licensed in the EU, UK or South Korea."]},
        "output": {"classification": "restricted"},
    }), encoding="utf-8")
    result = _run_tweak(tmp_path, source)
    record = json.loads(result.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert "Not licensed in the EU, UK or South Korea." in record["license"]["conditions"]
    assert record["addon"]["id"] == "vehicle-kit"
    assert record["output"]["sha256"]
    assert record["output"]["source"] == "add-on"


def test_result_from_a_model_with_no_record_counts_as_unverified(tmp_path):
    _model_tool_addon(tmp_path)
    source = tmp_path / "mystery.glb"
    source.write_text("glb", encoding="utf-8")
    result = _run_tweak(tmp_path, source)
    record = json.loads(result.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert record["license"]["verified"] is False
    assert record["output"]["classification"] == "unverified"

def test_unlisted_install_takes_its_id_from_the_manifest(tmp_path):
    """A repo called foo-addon whose manifest says foo installs as foo, marked not reviewed."""
    repo = make_bare_repo(tmp_path, "pedestal-addon", files={"addon.json": json.dumps(manifest(id="pedestal"))})
    result = ai.install_addon(
        unlisted=f"{repo}@{repo_head(repo)}",
        confirm=True,
        addons_dir=tmp_path / "addons",
        repo_dir=tmp_path,
    )
    assert result["id"] == "pedestal"
    record = json.loads((tmp_path / "addons" / "pedestal" / ".addon-install.json").read_text(encoding="utf-8"))
    assert record["id"] == "pedestal"
    assert record["tier"] == "unknown"


def test_listed_install_still_refuses_a_manifest_with_another_id(tmp_path):
    registry = tmp_path / "index.json"
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(manifest(id="other-thing"))})
    registry.write_text(json.dumps([listing(repo=str(repo), commit=repo_head(repo))]), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ai.install_addon(addon_id="vehicle-kit", index=str(registry), confirm=True,
                         addons_dir=tmp_path / "addons", repo_dir=tmp_path)


def test_the_list_is_read_from_the_public_list_repo(monkeypatch):
    """With no local file given, load_registry reads the published index and understands it."""
    seen = []
    entry = {"id": "pedestal", "name": "Pedestal", "repo": "https://github.com/AssetFurnace/pedestal-addon.git",
             "commit": "a" * 40, "author": {"name": "AssetFurnace"}, "summary": "A stand.", "tier": "listed",
             "runs": "local", "links": {"repo": "https://github.com/AssetFurnace/pedestal-addon"}}
    monkeypatch.setattr(ai, "_fetch_json", lambda url: seen.append(url) or [entry])
    listings = ai.load_registry(None)
    assert seen == ["https://raw.githubusercontent.com/AssetFurnace/assetfurnace-addons/main/index.json"]
    assert listings[0].id == "pedestal"
