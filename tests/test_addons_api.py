"""Tests for the studio add-on API.

Uses a fake runner and a local index file; no network, no real git in unit tests.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

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


def test_addons_payload_says_whether_we_test_ran_it(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing(), listing(id="two", run_check={"status": "passed"}),
                                    listing(id="three", run_check={"status": "partial", "skipped": ["fit"]})]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    payload = {a["id"]: a for a in viewer.addons_payload()["addons"]}
    flags = {k: v["run_checked"] for k, v in payload.items()}
    assert sorted(flags.values()) == [False, True, True] and flags["two"] is True and flags["three"] is True
    assert payload["three"]["run_skipped"] == ["fit"] and payload["two"]["run_skipped"] == []


def test_addons_payload_includes_links_and_rating(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    payload = viewer.addons_payload()
    addon = payload["addons"][0]
    assert addon["links"] == {"repo": "https://github.com/Ahnyth/vehicle-kit",
                              "site": "https://vehiclekit.example"}
    assert addon["rating"] == 4.5
    assert addon["ratings_count"] == 12


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
    assert payload["version"] == "0.1.0"
    assert payload["tier"] == "listed"
    assert "automatic checks" in payload["warning"]


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


def test_two_runs_asked_for_at_the_same_moment_do_not_both_start(tmp_path, monkeypatch):
    """Asking "is one running?" and putting the job on the books were two steps, with the
    request's own checks between them. A second request arriving in between started a second
    tool, and the manager then knew only the later one: stopping the studio left the first running."""
    import threading

    _set_paths(tmp_path, monkeypatch)
    make_installed_addon(tmp_path, manifest())
    manager = viewer.AddonJobManager()
    monkeypatch.setattr(viewer, "ADDON_JOBS", manager)
    monkeypatch.setattr(viewer, "_run_addon_job", lambda job, _manager: None)     # no tool is run here
    reached, let_go = threading.Event(), threading.Event()
    check_inputs = viewer.ai.parse_inputs

    def held_on_the_way(*args, **more):      # the first request stops here, past "is one running?"
        if not reached.is_set():
            reached.set()
            assert let_go.wait(20)
        return check_inputs(*args, **more)
    monkeypatch.setattr(viewer.ai, "parse_inputs", held_on_the_way)
    answers = []

    def first():
        try:
            answers.append(viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"}))
        except RuntimeError as refused:
            answers.append(refused)
    asking = threading.Thread(target=first, daemon=True)
    asking.start()
    assert reached.wait(20)
    second = viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"})    # all the way through
    let_go.set()
    asking.join(20)

    assert not asking.is_alive()
    assert isinstance(answers[0], RuntimeError) and "already in progress" in str(answers[0])
    assert list(manager.jobs) == [second["job_id"]] and manager.active == second["job_id"]


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


def _alive(pid: int) -> bool:
    """Whether the process still runs. One that has ended and waits to be collected (a zombie)
    has not outlived anything: where nothing collects orphans, as in a bare container, it
    would otherwise count as alive for good."""
    import subprocess

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_until(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return bool(condition())


def _start_tool(tmp_path, monkeypatch, script: str):
    """Install the test add-on with `script` as its tool, start it, and wait until it says "started"."""
    _set_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(viewer, "ADDON_JOBS", viewer.AddonJobManager())
    addon_dir = make_installed_addon(tmp_path, manifest())
    (addon_dir / "shape_guide.py").write_text(script, encoding="utf-8")
    job = viewer.ADDON_JOBS.get(viewer.run_payload("vehicle-kit", "build", {"preset": "bus_12m"})["job_id"])
    assert _wait_until(lambda: "started" in job.log_lines, seconds=20), job.log_lines
    return addon_dir, job


@pytest.mark.posix
def test_cancel_stops_the_tool_and_everything_it_started(tmp_path, monkeypatch):
    # Cancel stopped the tool alone. What the tool had started (Blender's helpers) ran on and
    # kept its output open, so the run stayed "cancelling" until that ended by itself.
    addon_dir, job = _start_tool(tmp_path, monkeypatch, (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])\n"
        "pathlib.Path('child.pid').write_text(str(child.pid), encoding='utf-8')\n"
        "print('started', flush=True)\n"
        "time.sleep(20)\n"))
    child = int((addon_dir / "child.pid").read_text(encoding="utf-8"))
    try:
        viewer.ADDON_JOBS.cancel(job.id)

        assert _wait_until(lambda: job.status == "cancelled", seconds=8), job.status
        assert _wait_until(lambda: not _alive(child)), "what the tool started outlived Cancel"
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)


@pytest.mark.posix
def test_a_running_tool_leaves_a_note_of_itself_that_the_next_studio_start_can_find(tmp_path, monkeypatch):
    # A tool leads a process group of its own, so it outlives a studio that is killed. Every
    # other job leaves `pid` in its run folder for the next start to find and stop what was
    # left; an add-on run left nothing, and its tool ran on with nobody knowing.
    _, job = _start_tool(tmp_path, monkeypatch, "import time\nprint('started', flush=True)\ntime.sleep(20)\n")
    try:
        [note] = (tmp_path / "output").rglob("pid")
        assert note.read_text(encoding="utf-8").split() == [str(job.process.pid), str(os.getpid())]
        assert job.run_stem in note.parent.name

        viewer.ADDON_JOBS.cancel(job.id)

        assert _wait_until(lambda: job.status == "cancelled", seconds=8), job.status
        assert _wait_until(lambda: not note.exists()), "the note outlived the run"
    finally:
        if job.process is not None and job.process.poll() is None:
            os.kill(job.process.pid, signal.SIGKILL)


@pytest.mark.posix
def test_cancel_stops_a_tool_that_ignores_being_asked(tmp_path, monkeypatch):
    # Cancel only asked. A tool that ignored it stayed "cancelling" until it ended by itself,
    # and all that time it held the add-on's place, which every other big job waits for.
    monkeypatch.setattr(viewer.ai, "CANCEL_GRACE_SECONDS", 0.5)
    _, job = _start_tool(tmp_path, monkeypatch, (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"))
    try:
        viewer.ADDON_JOBS.cancel(job.id)

        assert _wait_until(lambda: job.status == "cancelled", seconds=8), job.status
        assert not viewer.ADDON_JOBS.busy()
    finally:
        if job.process is not None and job.process.poll() is None:
            os.kill(job.process.pid, signal.SIGKILL)


@pytest.mark.posix
def test_a_running_tool_is_known_to_its_job_and_leads_its_own_group(tmp_path, monkeypatch):
    # A server stop ends each job by its process group. The job did not know its process, so a
    # stop never reached a running add-on tool.
    _, job = _start_tool(tmp_path, monkeypatch, "import time\nprint('started', flush=True)\ntime.sleep(20)\n")
    try:
        process = getattr(job, "process", None)
        assert process is not None, "the job does not know the process it started"
        assert os.getpgid(process.pid) == process.pid != os.getpgid(0)
    finally:
        viewer.ADDON_JOBS.cancel(job.id)
        assert _wait_until(lambda: job.status == "cancelled", seconds=8), job.status


def test_addons_page_sends_the_studio_request_header():
    """The page's POSTs are refused without the header that request-header.js adds."""
    html = (Path(__file__).resolve().parents[1] / "viewer" / "addons.html").read_text(encoding="utf-8")
    assert './request-header.js' in html
    assert html.index('request-header.js') < html.index('addons_page.js')


def test_installed_copy_without_a_list_entry_stays_unknown_even_if_its_id_is_listed(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    make_installed_addon(tmp_path, manifest(), install={"tier": "unknown"})
    entry = viewer.addons_payload()["addons"][0]
    assert entry["installed"] is True
    assert entry["tier"] == "unknown"


def test_a_verified_list_entry_shows_verified_and_an_installed_copy_shows_its_own_tier(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing()]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    make_installed_addon(tmp_path, manifest(), install={"tier": "listed"})
    assert viewer.addons_payload()["addons"][0]["tier"] == "listed"



def test_payload_carries_runs_for_listed_and_installed(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    registry.write_text(json.dumps([listing(runs="cloud")]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    assert viewer.addons_payload()["addons"][0]["runs"] == "cloud"
    make_installed_addon(tmp_path, manifest(runs="cloud"))
    assert viewer.addons_payload()["addons"][0]["runs"] == "cloud"


def test_payload_carries_the_picture_link(tmp_path, monkeypatch):
    _set_paths(tmp_path, monkeypatch)
    registry = tmp_path / "index.json"
    link = "https://example.test/pictures/vehicle-kit.jpg"
    registry.write_text(json.dumps([listing(picture=link)]), encoding="utf-8")
    monkeypatch.setattr(viewer, "INDEX_FILE", str(registry))
    assert viewer.addons_payload()["addons"][0]["picture"] == link
