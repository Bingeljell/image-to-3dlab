"""Hugging Face sign-in from the Setup page (viewer/hf_api.py).

A new user's path is curl, then the viewer: signing in to Hugging Face for the gated
models was the one step that still needed a terminal (`hf auth login`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

import hf_api


class Refused(Exception):
    pass


def _deps(user="ada", granted=("facebook/dinov3-vitl16-pretrain-lvd1689m",), token="hf_x"):
    saved = []

    def whoami(token=None):
        if token in (None, "bad"):
            raise Refused
        return {"name": user}

    def has_access(repo, token=None):
        return repo in granted

    return {"whoami": whoami, "has_access": has_access, "current_token": lambda: token,
            "save_token": saved.append}, saved


def test_signed_out_lists_what_needs_a_sign_in():
    deps, _ = _deps(token=None)
    status = hf_api.status(**{k: v for k, v in deps.items() if k != "save_token"})
    assert status["signed_in"] is False and status["user"] is None
    repos = {r["repo"]: r for r in status["repos"]}
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" in repos
    assert all(r["access"] == "unknown" for r in repos.values())
    assert all(r["request_url"].startswith("https://huggingface.co/") for r in repos.values())


def test_signed_in_reports_access_per_gated_model():
    deps, _ = _deps()
    status = hf_api.status(**{k: v for k, v in deps.items() if k != "save_token"})
    repos = {r["repo"]: r["access"] for r in status["repos"]}
    assert status["signed_in"] is True and status["user"] == "ada"
    assert repos["facebook/dinov3-vitl16-pretrain-lvd1689m"] == "yes"
    assert repos["stabilityai/stable-fast-3d"] == "no"


def test_a_good_token_is_checked_then_saved():
    deps, saved = _deps(token=None)
    result = hf_api.sign_in("hf_good", **deps)
    assert saved == ["hf_good"] and "error" not in result


def test_a_refused_token_is_not_saved_and_is_never_echoed():
    deps, saved = _deps(token=None)
    result = hf_api.sign_in("bad", **deps)
    assert saved == [] and "error" in result
    assert "bad" not in str(result)


def test_an_empty_token_is_refused_before_any_network_call():
    deps, saved = _deps(token=None)
    assert "error" in hf_api.sign_in("   ", **deps) and saved == []
