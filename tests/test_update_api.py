"""The update check: at most one GitHub request a day, silent when offline, off on request."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import update_api

DAY = 24 * 3600


def release(tag="v0.3.0"):
    return {"tag_name": tag, "html_url": f"https://github.com/x/releases/tag/{tag}"}


class Fetch:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = 0, result or release(), error

    def __call__(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def check(tmp_path, fetch, now=1_000_000.0, current="0.2.0", **kw):
    return update_api.check(current=current, now=now, cache=tmp_path / "check.json",
                            fetch=fetch, env=kw.get("env", {}), repo=Path("/lab"),
                            family=kw.get("family", "macos"))


def test_a_newer_release_is_reported_with_the_command_for_this_folder(tmp_path):
    result = check(tmp_path, Fetch())
    assert result["newer"] is True and result["latest"] == "0.3.0"
    assert f"install.sh | bash -s -- --dir {Path('/lab')}" in result["command"]
    assert result["url"].endswith("v0.3.0")


def test_windows_gets_the_powershell_command(tmp_path):
    result = check(tmp_path, Fetch(), family="windows")
    assert "install.ps1" in result["command"] and "I3D_DIR" in result["command"]


def test_same_or_older_release_is_not_newer(tmp_path):
    assert check(tmp_path, Fetch(release("v0.2.0")))["newer"] is False
    assert check(tmp_path / "b", Fetch(release("v0.1.9")), current="0.2.0")["newer"] is False


def test_github_is_asked_at_most_once_a_day(tmp_path):
    fetch = Fetch()
    check(tmp_path, fetch, now=1_000_000.0)
    check(tmp_path, fetch, now=1_000_000.0 + DAY - 60)
    assert fetch.calls == 1
    check(tmp_path, fetch, now=1_000_000.0 + DAY + 60)
    assert fetch.calls == 2


def test_offline_is_silent_and_keeps_the_last_answer(tmp_path):
    check(tmp_path, Fetch(), now=1_000_000.0)
    offline = Fetch(error=urllib.error.URLError("no network"))
    result = check(tmp_path, offline, now=1_000_000.0 + 2 * DAY)
    assert offline.calls == 1
    assert result["latest"] == "0.3.0" and result["newer"] is True


def test_offline_with_nothing_cached_reports_nothing(tmp_path):
    result = check(tmp_path, Fetch(error=TimeoutError()))
    assert result["newer"] is False and result["latest"] is None


def test_a_failed_check_is_not_retried_every_page_load(tmp_path):
    offline = Fetch(error=urllib.error.URLError("no network"))
    check(tmp_path, offline, now=1_000_000.0)
    check(tmp_path, offline, now=1_000_000.0 + 3600)
    assert offline.calls == 1


def test_junk_tags_and_a_corrupt_cache_do_not_break_the_viewer(tmp_path):
    assert check(tmp_path, Fetch(release("nightly")))["newer"] is False
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "check.json").write_text("{not json")
    result = check(tmp_path / "c", Fetch())
    assert result["newer"] is True


def test_the_environment_can_turn_it_off_without_any_request(tmp_path):
    fetch = Fetch()
    result = check(tmp_path, fetch, env={"I3D_NO_UPDATE_CHECK": "1"})
    assert fetch.calls == 0 and result["enabled"] is False and result["newer"] is False


def test_the_cache_lives_in_the_git_ignored_output_folder():
    assert update_api.CACHE.parent.name == "output"


def test_the_request_goes_to_this_repo_releases_only():
    assert update_api.LATEST_URL == (
        "https://api.github.com/repos/Bingeljell/image-to-3dlab/releases/latest")


def test_cache_file_records_when_it_asked(tmp_path):
    check(tmp_path, Fetch(), now=1_234.0)
    assert json.loads((tmp_path / "check.json").read_text())["checked_at"] == 1_234.0
