"""Exercise add-on consent, broken installs and owned subprocess cleanup."""

import io
import json
import os
import subprocess
import sys
import threading

import pytest

from image_to_3dlab import addon_install as addons
from tests.test_addons import (
    listing,
    make_bare_repo,
    make_installed_addon,
    manifest,
    repo_head,
)


def test_unlisted_install_plan_uses_manifest_identity(tmp_path):
    repo = make_bare_repo(tmp_path, "different", files={"addon.json": json.dumps(manifest())})
    result = addons.install_addon(unlisted=f"{repo}@{repo_head(repo)}", addons_dir=tmp_path / "addons", repo_dir=tmp_path)
    assert result["listing"].id == "vehicle-kit"
    assert result["plan"]["tier"] == "unknown"
    assert list((tmp_path / "addons").iterdir()) == []


@pytest.mark.parametrize("unlisted", [None, "https://example.invalid/tool@main"])
def test_install_requires_identity_and_pinned_commit(tmp_path, unlisted):
    with pytest.raises(ValueError, match="requires an id|must be <git url>@<commit>"):
        addons.install_addon(unlisted=unlisted, addons_dir=tmp_path / "addons", repo_dir=tmp_path)
    assert not (tmp_path / "addons").exists()


def test_rejected_checkout_removes_temporary_clone(tmp_path, monkeypatch):
    dest = tmp_path / "clone"
    dest.mkdir()
    (dest / "old").touch()

    def git(argv, **kwargs):
        assert argv[0] == "git"
        if argv[1] == "clone":
            assert not dest.exists()
            dest.mkdir()
        return subprocess.CompletedProcess(argv, 0, stdout="wrong")

    monkeypatch.setattr(addons.subprocess, "run", git)
    with pytest.raises(RuntimeError, match="HEAD mismatch"):
        addons._clone("https://example.invalid/addon", "expected", dest, tmp_path)
    assert not dest.exists()


def test_setup_failure_is_reported_after_confirmed_install(tmp_path):
    payload = manifest(setup=[sys.executable, "-c", "raise SystemExit(7)"])
    repo = make_bare_repo(tmp_path, "vehicle-kit", files={"addon.json": json.dumps(payload)})
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing(repo=str(repo), commit=repo_head(repo))]))
    directory = tmp_path / "addons"
    with pytest.raises(RuntimeError, match="setup failed"):
        addons.install_addon("vehicle-kit", index=registry, confirm=True, addons_dir=directory, repo_dir=tmp_path)
    assert sorted(path.name for path in directory.iterdir()) == ["vehicle-kit"]


def test_cleanup_removes_read_only_file_even_if_chmod_fails(tmp_path, monkeypatch):
    target = tmp_path / "read-only"
    target.touch()

    def refuse(*args):
        raise OSError("chmod refused")

    monkeypatch.setattr(addons.os, "chmod", refuse)
    addons._remove_readonly(os.unlink, target, None)
    assert not target.exists()
    addons._remove_tree(target, quiet=True)
    with pytest.raises(FileNotFoundError):
        addons._remove_tree(target)


def test_registry_fetch_parses_offline_http_response(monkeypatch):
    response = [listing()]

    def open_url(request, timeout):
        assert request.full_url == addons.REGISTRY_URL
        assert request.get_header("User-agent") == addons.USER_AGENT
        assert timeout == 60
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(addons, "urlopen", open_url)
    assert addons.load_registry(None)[0].id == response[0]["id"]


def test_run_addon_refuses_missing_install_and_missing_result(tmp_path):
    with pytest.raises(ValueError, match="is not installed"):
        addons.run_addon("vehicle-kit", "build", {}, addons_dir=tmp_path / "addons")
    dest = make_installed_addon(tmp_path, manifest())
    (dest / "shape_guide.py").write_text("print('finished with no file')")
    with pytest.raises(RuntimeError, match="expected result.*not written"):
        addons.run_addon("vehicle-kit", "build", {"preset": "bus_12m"}, run_stem="no-result",
                         addons_dir=tmp_path / "addons", repo_dir=tmp_path, blender="blender", output_root=tmp_path)


def test_started_callback_receives_stream_worker_with_no_logger(tmp_path):
    argv = [sys.executable, "-c", "print('hello')"]
    children = []
    result = addons._run_subprocess(argv, tmp_path, os.environ.copy(), None, on_start=children.append)
    assert result.returncode == 0
    assert len(children) == 1 and children[0].poll() == 0
    children[0].stdout.close()


def test_log_callback_error_stops_unresponsive_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    killed = []
    stream = io.StringIO("line\n")

    def wait(timeout):
        assert timeout == addons.CANCEL_GRACE_SECONDS
        raise subprocess.TimeoutExpired(["worker"], timeout)

    worker = SimpleNamespace(stdout=stream, pid=123, wait=wait)

    def log(_line):
        raise ValueError("log cannot be written")

    monkeypatch.setattr(addons.subprocess, "Popen", lambda *args, **kwargs: worker)
    monkeypatch.setattr(addons.processes, "kill_group", killed.append)
    with pytest.raises(ValueError, match="log cannot be written"):
        addons._run_subprocess(["worker"], tmp_path, {}, log)
    assert stream.closed
    assert killed == [123]


def test_cancelled_addon_reports_cancellation_before_result_checks(tmp_path):
    dest = make_installed_addon(tmp_path, manifest())
    (dest / "shape_guide.py").write_text("print('done')")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        addons.run_addon("vehicle-kit", "build", {"preset": "bus_12m"}, run_stem="stopped",
                         addons_dir=tmp_path / "addons", repo_dir=tmp_path, blender="blender",
                         output_root=tmp_path, cancel_event=cancel)
