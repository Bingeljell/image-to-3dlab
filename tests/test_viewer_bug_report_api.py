"""Report a bug: the report the studio fills in, and the three ways to send it."""

from __future__ import annotations

import base64
import json
import subprocess
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


def test_windows_names_its_nvidia_card(monkeypatch):
    # The NVIDIA driver puts nvidia-smi on the PATH on Windows too. Without it a Windows report
    # said only "Windows · AMD64" (#192), and the card is the first thing a GPU bug needs.
    asked = []

    def nvidia_smi(command, **kwargs):
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="NVIDIA GeForce RTX 4070 Ti SUPER\n")

    monkeypatch.setattr(br.subprocess, "run", nvidia_smi)
    assert br._chip("windows") == "NVIDIA GeForce RTX 4070 Ti SUPER"
    assert asked[0][0] == "nvidia-smi"


def test_windows_without_nvidia_smi_names_no_card(monkeypatch):
    # An AMD or Intel card, or no driver: the report goes out without a card, as before.
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(br.subprocess, "run", missing)
    assert br._chip("windows") == ""


def test_every_spelling_of_a_windows_home_folder_is_hidden():
    # A failed command prints its paths doubled (C:\\Users\\ada); tools also write C:/Users/ada,
    # and Windows does not care about case.
    home = r"C:\Users\ada"
    text = (r"File C:\Users\ada\lab.py; ['C:\\Users\\ada\\.local\\bin\\uv.EXE']; "
            r"C:/Users/ada/x; c:\users\ADA\y")
    scrubbed = br.scrub(text, home)
    assert "ada" not in scrubbed.lower()
    assert scrubbed.startswith(r"File ~\lab.py; ['~\\.local")


def test_a_pasted_message_is_scrubbed_too():
    # A Setup failure copied into the message box carried the user's name in every path.
    home = r"C:\Users\ada"
    links = br.links(r"Setup failed: C:\Users\ada\.local\bin\uv.EXE", r"log at C:\\Users\\ada\\x",
                     "0.5.1", home=home)
    assert "ada" not in links["text"] and "ada" not in links["github"]
    assert links["text"].startswith(r"Setup failed: ~\.local")


def test_scrub_hides_only_the_whole_home_folder():
    assert br.scrub(r"C:\Users\ada\x and C:\Users\adam\y", r"C:\Users\ada") == r"~\x and C:\Users\adam\y"
    assert br.scrub("/Users/ada/x /Users/adam/y /Users/ada.b in /Users/ada.", "/Users/ada") == \
        "~/x /Users/adam/y /Users/ada.b in ~."
    assert br.scrub("/Users/ada/x", "/Users/ada/") == "~/x"           # a trailing slash changes nothing


def test_scrub_leaves_text_alone_when_home_is_a_root():
    """Some services run with home at / or C:\\: replacing that would turn every path into ~."""
    for home in ("/", "C:\\", "", "C:"):
        assert br.scrub("/opt/lab/run.log C:\\lab\\x", home) == "/opt/lab/run.log C:\\lab\\x"


@pytest.mark.parametrize("message,diagnostics", [
    ("x" * 8000, "diagnostics"), ("x" * 8000, ""), ("🙂" * 4000, "x"),
    ("漢字" * 2000, "xx"), ("x" * 8000, "xxx"), ("small\nnext", "é" * 20000),
])
def test_long_message_has_bounded_github_url(message, diagnostics):
    import os
    import subprocess

    # Run the real function out of process: the original infinite loop must not hang pytest.
    result = subprocess.run(
        [sys.executable, "-c", ("import json,sys; import bug_report_api as br; "
         "print(json.dumps(br.links(*json.load(sys.stdin), version='test', home='/Users/ada')))")],
        input=json.dumps([message, diagnostics]), text=True, capture_output=True, timeout=5,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, check=True,
    )
    links = json.loads(result.stdout)
    assert len(links["github"]) <= br.GITHUB_URL_MAX
    body = parse_qs(urlparse(links["github"]).query)["body"][0]
    assert "…" in body
    assert _draft_from(links["site"])["message"] == message
    assert _draft_from(links["site"])["details"] == diagnostics
    assert links["text"] == f"{message}\n\n{diagnostics}"
