"""Hugging Face sign-in for the Setup page.

A new user's path is: run the install command, then do everything in the viewer. Signing in
to Hugging Face for the gated models was the one step that still needed a terminal
(`hf auth login`). This checks a pasted token with Hugging Face, saves it where
`hf auth login` would, and says which gated models the account can reach.

The token is never logged, stored anywhere else, or sent back to the page.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The gated models this lab fetches, and what needs each one.
GATED = (
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "TRELLIS.2 (its image encoder)"),
    ("stabilityai/stable-fast-3d", "Stable Fast 3D"),
)


def _whoami(token: str | None = None) -> dict[str, Any]:
    from huggingface_hub import whoami

    return whoami(token=token)


def _has_access(repo: str, token: str | None = None) -> bool:
    """Whether Hugging Face will let this account download `repo`. Seconds, no download."""
    from huggingface_hub import auth_check
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        auth_check(repo, token=token)
    except (GatedRepoError, RepositoryNotFoundError):
        return False
    return True


def _current_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def _save_token(token: str) -> None:
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def status(whoami: Callable = _whoami, has_access: Callable = _has_access,
           current_token: Callable = _current_token) -> dict[str, Any]:
    """Signed in or not, as whom, and access to each gated model ("yes", "no", "unknown")."""
    token = current_token()
    user = None
    if token:
        try:
            user = whoami(token=token).get("name")
        except Exception:  # an expired or revoked token reads as signed out
            user = None
    repos = []
    for repo, needed_by in GATED:
        if user is None:
            access = "unknown"
        else:
            try:
                access = "yes" if has_access(repo, token=token) else "no"
            except Exception:
                access = "unknown"
        repos.append({"repo": repo, "for": needed_by, "access": access,
                      "request_url": f"https://huggingface.co/{repo}"})
    return {"signed_in": user is not None, "user": user, "repos": repos}


def sign_in(token: str, whoami: Callable = _whoami, has_access: Callable = _has_access,
            current_token: Callable = _current_token,
            save_token: Callable = _save_token) -> dict[str, Any]:
    """Check `token` with Hugging Face, save it if it is good, and return the new status."""
    token = (token or "").strip()
    if not token:
        return {"error": "Paste a token first: huggingface.co/settings/tokens, type Read."}
    try:
        whoami(token=token)
    except Exception:
        return {"error": "Hugging Face refused that token. Check you copied all of it, "
                         "and that it has not been revoked."}
    save_token(token)
    return status(whoami=whoami, has_access=has_access, current_token=lambda: token)
