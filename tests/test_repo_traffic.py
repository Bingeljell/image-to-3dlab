"""scripts/repo_traffic.py: GitHub's 14-day traffic and the install counts, kept as one CSV."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import repo_traffic as rt

VIEWS = {"count": 9, "uniques": 5, "views": [
    {"timestamp": "2026-10-09T00:00:00Z", "count": 6, "uniques": 3},
    {"timestamp": "2026-10-10T00:00:00Z", "count": 3, "uniques": 2},
]}
CLONES = {"count": 4, "uniques": 2, "clones": [
    {"timestamp": "2026-10-10T00:00:00Z", "count": 4, "uniques": 2},
]}


def test_github_payloads_become_one_row_per_day():
    rows = rt.rows_from_github(VIEWS, CLONES)
    assert rows == {
        "2026-10-09": {"views": 6, "unique_views": 3},
        "2026-10-10": {"views": 3, "unique_views": 2, "clones": 4, "unique_clones": 2},
    }


def test_install_counts_become_columns():
    assert rt.rows_from_installs({"2026-10-10": {"sh": 4, "ps1": 1}}) == {
        "2026-10-10": {"installs_sh": 4, "installs_ps1": 1}}


def test_fresh_numbers_correct_a_partial_day_but_keep_what_they_do_not_report():
    saved = {"2026-10-01": {"views": 50}, "2026-10-10": {"views": 1, "installs_sh": 7}}
    out = rt.merge(saved, rt.rows_from_github(VIEWS, CLONES))
    assert out["2026-10-01"] == {"views": 50}                  # older than GitHub's window: kept
    assert out["2026-10-10"]["views"] == 3                     # partial day corrected
    assert out["2026-10-10"]["installs_sh"] == 7               # GitHub doesn't report it: kept


def test_csv_round_trips_and_rerunning_changes_nothing(tmp_path):
    path = tmp_path / "traffic.csv"
    rows = rt.merge({}, rt.rows_from_github(VIEWS, CLONES), rt.rows_from_installs({"2026-10-10": {"sh": 2}}))
    rt.write_csv(path, rows)
    first = path.read_text()
    assert rt.read_csv(path) == rows
    rt.write_csv(path, rt.merge(rt.read_csv(path), rt.rows_from_github(VIEWS, CLONES)))
    assert path.read_text() == first
    assert first.splitlines()[0] == ",".join(rt.FIELDS)


def test_write_csv_replaces_a_finished_file(tmp_path, monkeypatch):
    path = tmp_path / "traffic.csv"
    path.write_text("old\n")
    replaced = []
    real = os.replace

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    rt.write_csv(path, {"2026-10-10": {"views": 1}})
    assert replaced
    assert replaced[0][0] != path
    assert replaced[0][1] == path
    assert path.read_text() != "old\n"
    assert not replaced[0][0].exists()


def test_week_summary_sums_the_newest_days_and_treats_gaps_as_zero():
    rows = {f"2026-10-{d:02d}": {"unique_views": 1} for d in range(1, 11)}
    rows["2026-10-10"]["installs_sh"] = 3
    week = rt.last_days(rows)
    assert week["unique_views"] == 7
    assert week["installs_sh"] == 3 and week["clones"] == 0


def test_a_missing_token_skips_installs_without_failing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rt, "gh_api", lambda p: VIEWS if p.endswith("views") else CLONES)
    out = tmp_path / "t.csv"
    assert rt.main(["--repo", "o/n", "--out", str(out), "--token-file", str(tmp_path / "none")]) == 0
    assert "installs skipped" in capsys.readouterr().err
    assert rt.read_csv(out)["2026-10-10"]["clones"] == 4


def test_traffic_cli_fetches_installs_and_uses_the_current_repository(tmp_path, monkeypatch, capsys):
    import io
    import json
    import runpy
    from types import SimpleNamespace

    script = Path(rt.__file__)
    output, token = tmp_path / "stats" / "daily.csv", tmp_path / "token"
    token.write_text(" fixture-token\n")
    url = "https://example.invalid/install-counts"
    monkeypatch.setattr(sys, "argv", [
        str(script), "--out", str(output), "--token-file", str(token), "--installs-url", url,
    ])
    expected = [
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        ["gh", "api", "repos/org/project/traffic/views"],
        ["gh", "api", "repos/org/project/traffic/clones"],
    ]
    replies = iter(["org/project\n", json.dumps(VIEWS), json.dumps(CLONES)])
    issued = []

    def run(args, *, check, capture_output, text):
        assert check and capture_output and text
        issued.append(args)
        return SimpleNamespace(stdout=next(replies))

    def read_installs(request, *, timeout):
        assert request.full_url == url and timeout == 20
        assert request.get_header("Authorization") == "Bearer fixture-token"
        return io.BytesIO(b'{"2026-10-10": {"sh": 3, "ps1": 2}}')

    monkeypatch.setattr(rt.subprocess, "run", run)
    monkeypatch.setattr(rt.urllib.request, "urlopen", read_installs)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 0
    assert issued == expected
    assert rt.read_csv(output)["2026-10-10"] == {
        "views": 3, "unique_views": 2, "clones": 4, "unique_clones": 2,
        "installs_sh": 3, "installs_ps1": 2,
    }
    assert "5 installs" in capsys.readouterr().out


def test_github_only_traffic_does_not_open_install_endpoint(tmp_path, monkeypatch, capsys):
    def unexpected_request(*args, **kwargs):
        pytest.fail("GitHub-only mode should not contact the install endpoint")

    monkeypatch.setattr(rt.urllib.request, "urlopen", unexpected_request)
    monkeypatch.setattr(rt, "gh_api", lambda path: VIEWS if path.endswith("views") else CLONES)
    output = tmp_path / "traffic.csv"
    assert rt.main(["--repo", "org/project", "--out", str(output), "--no-installs"]) == 0
    assert rt.read_csv(output)["2026-10-10"]["clones"] == 4
    assert not capsys.readouterr().err


def test_traffic_preserves_install_history_if_the_endpoint_is_unavailable(tmp_path, monkeypatch, capsys):
    def offline(*args, **kwargs):
        raise OSError("fixture endpoint unavailable")

    output, token = tmp_path / "daily.csv", tmp_path / "token"
    token.write_text("fixture-token")
    rt.write_csv(output, {"2026-10-10": {"installs_sh": 12}})
    monkeypatch.setattr(rt.urllib.request, "urlopen", offline)
    monkeypatch.setattr(rt, "gh_api", lambda path: VIEWS if path.endswith("views") else CLONES)
    assert rt.main(["--repo", "org/project", "--out", str(output), "--token-file", str(token)]) == 0
    assert "fixture endpoint unavailable" in capsys.readouterr().err
    assert rt.read_csv(output)["2026-10-10"]["installs_sh"] == 12
    assert rt.read_csv(output)["2026-10-10"]["views"] == 3
