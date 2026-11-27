"""Check weight cache reports and cleanup using tiny local files."""

import json
import subprocess
import sys

import pytest

from scripts import audit_model_weights as audit


@pytest.mark.parametrize("header", [[], {}, {"__metadata__": {}}, {"a": -1}])
def test_conversion_rejects_headers_that_describe_no_valid_tensor(tmp_path, header):
    path = tmp_path / "weights.safetensors"
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"1234")
    assert not audit.conversion_proven(path)
    assert not audit.conversion_proven(tmp_path / "missing.safetensors")


def test_cached_duplicates_skip_small_entries_and_report_unmatched_repo(tmp_path, monkeypatch):
    root = tmp_path / "local"
    hub = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(audit, "GB", 5)
    local_file = root / "same.bin"
    local_file.write_bytes(b"small")
    complete = hub / "models--org--complete"
    snapshot = complete / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "same.bin").write_bytes(b"small")
    below = hub / "models--org--small" / "snapshots" / "head"
    below.mkdir(parents=True)
    (below / "small.bin").write_bytes(b"abc")
    (root / "small.bin").write_bytes(b"abc")
    (hub / "models--org--empty").mkdir()
    (hub / "models--file--ignored").write_text("non-directory")

    candidates = audit.find_cached_duplicates(hub, [tmp_path / "missing", root])
    assert [(c.path, c.bytes_, c.deletable) for c in candidates] == [(complete, 5, True)]
    assert audit.find_cached_duplicates(hub, [tmp_path / "missing"]) == []
    assert audit.cache_repos(tmp_path / "missing") == []

    def grep(command, **kwargs):
        argv = command[:4]
        assert argv[:3] == ["git", "grep", "-lF"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, "scripts/loader.py" if argv[-1] == "org/complete" else "")

    monkeypatch.setattr(audit.subprocess, "run", grep)
    unmatched = audit.find_unreferenced(hub, repo=tmp_path)
    assert {c.path.name for c in unmatched} == {"models--org--empty", "models--org--small"}
    assert all(not candidate.deletable for candidate in unmatched)


def test_unreferenced_lookup_keeps_models_if_git_is_unavailable(monkeypatch, tmp_path):
    checkout = tmp_path / "models--org--known"
    checkout.mkdir()

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(audit.subprocess, "run", unavailable)
    assert audit.find_unreferenced(tmp_path) == []


def test_audit_cli_uses_selected_rule_and_keeps_review_only_files(monkeypatch, tmp_path, capsys):
    cache = tmp_path / "hub"
    cache.mkdir()
    unused = cache / "models--org--unused"
    unused.mkdir()
    (unused / "note").write_text("do not delete")

    def grep(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "")

    monkeypatch.setattr(audit.subprocess, "run", grep)
    monkeypatch.setattr(sys, "argv", ["audit_model_weights.py", "--rule", "unreferenced",
                                     "--apply", "--hf-cache", str(cache)])
    assert audit.main() == 0
    assert "needs review (not deleted)" in capsys.readouterr().out
    assert (unused / "note").read_text() == "do not delete"


def test_audit_apply_reports_unlink_errors_and_keeps_going(tmp_path, capsys):
    nonexistent = tmp_path / "gone.bin"
    exists = tmp_path / "keeps-audit-going.bin"
    exists.write_bytes(b"123")
    found = [
        audit.Candidate("ckpt", nonexistent, 5, "was already removed"),
        audit.Candidate("ckpt", exists, 3, "safe local fixture"),
    ]
    assert audit.report([], apply=True) == 0
    assert "nothing to reclaim" in capsys.readouterr().out
    assert audit.report(found, apply=True) == 0
    printed = capsys.readouterr().out
    assert "could not remove" in printed
    assert f"removed {exists}" in printed
    assert not exists.exists()
