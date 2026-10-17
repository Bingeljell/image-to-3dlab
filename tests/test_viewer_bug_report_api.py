"""Report a bug: the report the studio fills in, and the three ways to send it."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import bug_report_api as br
from activity_api import HISTORY_NAME

HOME = "/Users/ada"
FAILED = {"time": "2026-10-17T14:02:11", "chain": "c1", "title": "blocky knight", "prompt": "a blocky knight, A-pose",
          "step": "finished", "result": "error", "seconds": 19.0, "note": "Finish stopped: Blender crashed",
          "log": "\n".join(f"line {i}" for i in range(100)) + f"\nOSError: {HOME}/AssetFurnace/output/x.glb"}
DONE = {**FAILED, "chain": "c2", "result": "done", "note": ""}


def _history(tmp_path, *records):
    (tmp_path / HISTORY_NAME).write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp_path


def test_the_home_folder_becomes_a_tilde_everywhere():
    assert br.scrub(f"{HOME}/AssetFurnace and {HOME}", HOME) == "~/AssetFurnace and ~"
    assert br.scrub(r"C:\Users\ada\AssetFurnace", r"C:\Users\ada") == r"~\AssetFurnace"


def test_the_newest_failure_is_found(tmp_path):
    out = _history(tmp_path, FAILED, DONE)
    assert br.find_failure(out)["chain"] == "c1"


def test_a_named_failure_is_found_and_a_finished_step_is_not(tmp_path):
    out = _history(tmp_path, FAILED, DONE)
    assert br.find_failure(out, chain="c1", step="finished")["note"].startswith("Finish stopped")
    assert br.find_failure(out, chain="c2", step="finished") is None


def test_details_name_the_machine_the_step_and_the_error_without_the_home_folder(tmp_path):
    out = _history(tmp_path, FAILED)
    details = br.details(out, home=HOME, machine="macOS 26.0 · Apple M5", version="0.4.5")
    assert "AssetFurnace 0.4.5" in details and "macOS 26.0 · Apple M5" in details
    assert "Step: Finish" in details and "Finish stopped: Blender crashed" in details
    assert "~/AssetFurnace/output/x.glb" in details and HOME not in details
    assert "line 99" in details and "line 0\n" not in details   # the tail, not the whole log


def test_the_prompt_stays_out_unless_asked_for(tmp_path):
    out = _history(tmp_path, FAILED)
    assert "blocky knight, A-pose" not in br.details(out, home=HOME, machine="m", version="v")
    assert "blocky knight, A-pose" in br.details(out, home=HOME, machine="m", version="v", include_prompt=True)


def test_no_failure_still_gives_the_machine(tmp_path):
    details = br.details(tmp_path, home=HOME, machine="Linux · RX 7900 XTX", version="0.4.5")
    assert "Linux · RX 7900 XTX" in details and "Step:" not in details


def _draft_from(site_url):
    encoded = urlparse(site_url).fragment.split("=", 1)[1]
    return json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))


def test_the_site_link_carries_the_report_after_the_hash():
    links = br.links("Finish crashed — naïve", "AssetFurnace 0.4.5\nOSError", "0.4.5")
    assert links["site"].startswith(br.SITE_REPORT_URL + "#report=")
    assert _draft_from(links["site"]) == {"message": "Finish crashed — naïve", "details": "AssetFurnace 0.4.5\nOSError",
                                          "version": "0.4.5"}


def test_the_github_link_opens_a_new_issue_with_the_report():
    links = br.links("Finish crashed on my knight\nmore words", "AssetFurnace 0.4.5", "0.4.5")
    url = urlparse(links["github"])
    assert url.netloc == "github.com" and url.path.endswith("/issues/new")
    query = parse_qs(url.query)
    assert query["title"] == ["Finish crashed on my knight"]
    assert "AssetFurnace 0.4.5" in query["body"][0] and "more words" in query["body"][0]


def test_a_huge_log_is_trimmed_to_fit_a_github_link():
    links = br.links("broke", "x" * 50_000, "0.4.5")
    assert len(links["github"]) <= br.GITHUB_URL_MAX


def test_the_copied_text_has_the_message_and_the_details():
    text = br.links("broke", "details here", "0.4.5")["text"]
    assert text.startswith("broke") and "details here" in text


@pytest.mark.parametrize("system, expected", [("darwin", "macOS"), ("linux", "Linux"), ("win32", "Windows")])
def test_the_machine_line_starts_with_the_system(system, expected):
    assert br.machine_summary(sys_platform=system, chip=lambda: "Some GPU").startswith(expected)
