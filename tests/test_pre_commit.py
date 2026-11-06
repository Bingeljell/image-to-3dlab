"""Pins for the commit hooks."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_pre_commit_uses_the_python_runner_and_pinned_tools():
    text = (REPO / ".pre-commit-config.yaml").read_text()
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
    readme = (REPO / "README.md").read_text()
    assert "uv tool install pre-commit --with pre-commit-uv" in readme
    assert "pre-commit install" in readme


def test_pull_requests_run_the_file_checks():
    """GitHub Actions runs the file hooks. The pull request title check is commented out."""
    workflow = (REPO / ".github" / "workflows" / "pre-commit.yml").read_text()
    assert "pull_request:" in workflow
    assert "branches: [main, feat/tests-infra]" in workflow
    assert "pull_request_target" not in workflow
    assert "tox-dev/action-pre-commit-uv@v1" in workflow
    pre_commit_job = workflow.split("\n  pytest:")[0]
    assert "pytest" not in pre_commit_job
    assert "uv run pytest -q" in workflow
    assert "\n    needs: pre-commit\n" in workflow.split("\n  pytest:", 1)[1]
    config = (REPO / ".pre-commit-config.yaml").read_text()
    assert "stages: [commit-msg]" in config
    assert "action-semantic-pull-request@v6" in config
    assert "\n  title:" not in workflow
    assert "# title:" in workflow
