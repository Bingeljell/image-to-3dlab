"""Stop a generate command from fetching weights on its own.

Setup is the download. A loader that cannot see a local snapshot calls
``refuse_hub_download`` and does not call ``from_pretrained`` on a hub id.
"""

from __future__ import annotations

import os
from pathlib import Path


def refuse_hub_download(*, backend: str, route: str, size: str, detail: str = "") -> None:
    """Exit. The message names the backend, the route and the size."""
    message = (
        f"{backend} ({route}) needs its weights ({size}) on disk before a run. "
        "Generation does not fetch them."
    )
    if detail:
        message = f"{message}\n{detail}"
    raise SystemExit(message)


def hub_dir(hub: Path | None = None) -> Path:
    """The Hugging Face hub cache, honouring the usual variables."""
    if hub is not None:
        return hub
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def snapshot_dir(repo_id: str, hub: Path | None = None) -> Path | None:
    """A local snapshot of ``repo_id``, or None.

    A directory that already holds files is itself (a path the caller built).
    Otherwise the hub cache is searched. An empty folder is not a snapshot.
    """
    direct = Path(repo_id)
    if direct.is_dir() and any(direct.rglob("*")):
        return direct
    if "/" not in repo_id:
        return None
    snapshots = hub_dir(hub) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    found = [snap for snap in snapshots.iterdir()
             if snap.is_dir() and any(snap.rglob("*"))]
    return found[-1] if found else None


def require_local(repo_id: str, *, backend: str, route: str, size: str,
                  hub: Path | None = None) -> str:
    """The local snapshot path. Refuses when there is none.

    Also pins the process offline, so a later import cannot fetch the rest.
    """
    local = snapshot_dir(repo_id, hub)
    if local is None:
        refuse_hub_download(backend=backend, route=route, size=size)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return str(local)
