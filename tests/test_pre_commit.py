"""The committed pre-commit config matches the lab's Ruff pin and skips third-party trees."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_pre_commit_uses_the_pinned_ruff():
    text = (REPO / ".pre-commit-config.yaml").read_text()
    dev = (REPO / "requirements-dev.txt").read_text()
    assert "ruff==0.16.1" in dev
    assert "pre-commit" in dev
    assert "https://github.com/astral-sh/ruff-pre-commit" in text
    assert "rev: v0.16.1" in text
    assert "id: ruff-check" in text
    # ruff format would rewrite the tree; the lab only runs `ruff check`.
    assert "id: ruff-format" not in text
    assert "hunyuan_mlx/" in text
    assert "vendor/" in text
    # uv.lock is not on main yet; that hook belongs after #217.
    assert "uv-lock" not in text
