"""Pins for the commit hooks."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_pre_commit_uses_the_python_runner_and_pinned_tools():
    text = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "https://github.com/pre-commit/pre-commit-hooks" in text
    assert "repo: builtin" not in text
    assert "rev: v6.0.0" in text
    assert "rev: v0.16.1" in text
    assert "rev: 0.13.7" in text
    assert "https://github.com/compilerla/conventional-pre-commit" in text
    assert "rev: v4.4.0" in text
    assert "id: conventional-pre-commit" in text
    assert "[--fix, --exit-non-zero-on-fix]" in text
    assert "id: trailing-whitespace" not in text
    assert "id: end-of-file-fixer" not in text
    assert "id: ruff-format" not in text
    assert "pytest" not in text
    assert "hooks/check_commit_msg.py" not in text
    assert "commit-msg" in text
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "uv tool install pre-commit --with pre-commit-uv" in readme
    assert "pre-commit install" in readme


def test_pull_requests_run_the_file_checks():
    """GitHub Actions runs the file hooks. The pull request title is checked locally."""
    workflow = (REPO / ".github" / "workflows" / "pre-commit.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert "pull_request_target" not in workflow
    assert "actions/checkout@v7" in workflow
    assert "actions/checkout@v4" not in workflow
    assert "tox-dev/action-pre-commit-uv@v1" in workflow
    pre_commit_job = workflow.split("\n  pytest:")[0]
    assert "pytest" not in pre_commit_job
    assert "uv run pytest --tb=short -ra --color=yes" in workflow
    assert "actions/cache@v6" in workflow
    assert "enable-cache: false" in workflow
    assert "path: ~/.cache/uv" in workflow
    assert "hashFiles('uv.lock')" in workflow
    assert "\n    name: Linters\n" in workflow
    assert "\n    name: Tests\n" in workflow
    assert "\n    needs: pre-commit\n" in workflow.split("\n  pytest:", 1)[1]
    note = (REPO / ".github" / "workflows" / "pr-status.yml").read_text(encoding="utf-8")
    assert "name: CI\n" in workflow
    assert "workflows: [CI]" in note
    assert "conclusion == 'failure'" in note
    assert "delete: true" in note
    assert "marocchino/sticky-pull-request-comment@5770ad5eb8f42dd2c4f34da00c94c5381e49af88" in note
    assert "sticky-pull-request-comment@v3" not in note
    assert "pull_requests[0]" not in note
    assert "head_repository.owner.login" in note
    assert "head_branch" in note
    assert "head.sha" in note
    assert "Thank you for your contribution!" in note
    assert "A few of the checks did not pass:" in note
    assert "green light" in note
    assert "sticky-pull-request-comment" not in workflow
    config = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "stages: [commit-msg]" in config
    assert "action-semantic-pull-request" not in config
    assert "\n  title:" not in workflow
    assert "# title:" not in workflow
    assert "action-semantic-pull-request" not in workflow


def test_the_suite_also_runs_on_windows_and_blocks_merges():
    workflow = (REPO / ".github" / "workflows" / "pre-commit.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  pytest-windows:", 1)[1]
    assert "\n    name: Tests (Windows)\n" in job
    assert "\n    runs-on: windows-latest\n" in job
    assert "\n    needs: pre-commit\n" in job
    assert "\n    continue-on-error:" not in job
    assert "uv run pytest --tb=short -ra --color=yes" in job
    assert "PYTHONUTF8: " not in job   # a read without an encoding must fail here, as on a user's machine


def test_the_suite_also_runs_on_a_mac_and_a_failure_there_fails_the_run():
    """Apple-specific paths must pass on macOS before a merge."""
    workflow = (REPO / ".github" / "workflows" / "pre-commit.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  pytest-macos:", 1)[1].split("\n  pytest-windows:", 1)[0]
    assert "\n    name: Tests (macOS)\n" in job
    assert "\n    runs-on: macos-latest\n" in job
    assert "\n    needs: pre-commit\n" in job
    assert "\n    continue-on-error:" not in job
    assert "uv sync --frozen" in job
    assert "uv run --frozen python scripts/measure_coverage.py coverage-report -- --tb=short -ra --color=yes" in job
    assert "name: unit-coverage-macos" in job
    assert "path: ~/.cache/uv" in job and "hashFiles('uv.lock')" in job
    assert 'version: "0.13.7"' in job       # the same uv as the other two jobs


def test_only_successful_main_runs_publish_coverage_badges():
    import yaml

    path = REPO / ".github/workflows/pre-commit.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    publish = jobs["publish-coverage-badges"]
    assert publish["needs"] == ["merge-gate"]
    assert "success()" in publish["if"]
    assert "github.event_name == 'push'" in publish["if"]
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert publish["permissions"] == {"contents": "write"}
    assert all(not job.get("permissions") for name, job in jobs.items() if name != "publish-coverage-badges")
    assert publish["concurrency"]["cancel-in-progress"] is False
    shell = next(step["run"] for step in publish["steps"] if "run" in step)
    assert "git push origin HEAD:refs/heads/coverage-badges" in shell and "--force" not in shell

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for language in ("python", "javascript"):
        for metric in ("lines", "branches"):
            assert f"/coverage-badges/{language}-{metric}.svg" in readme
