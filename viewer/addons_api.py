"""Add-on HTTP API for the studio.

Works offline: if the reviewed list cannot be fetched, installed add-ons are still
shown and the UI can say the list is unreachable.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import addon_install as ai
from image_to_3dlab.addons import Listing, read_manifest

ADDONS_DIR = Path(os.environ.get("ADDONS_DIR", REPO / "addons"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO / "output"))
INDEX_FILE = os.environ.get("ADDONS_INDEX")


def _addon_dir(addon_id: str) -> Path:
    return ADDONS_DIR / addon_id


def _input_spec(spec: Any) -> dict[str, Any]:
    """Convert an Input dataclass to a JSON-friendly dict for the studio."""
    return {
        "id": spec.id,
        "type": spec.type,
        "label": spec.label,
        "optional": spec.optional,
        "choices": list(spec.choices) if spec.choices is not None else None,
        "min": spec.min,
        "max": spec.max,
    }


def _tool_spec(tool: Any) -> dict[str, Any]:
    return {
        "id": tool.id,
        "label": tool.label,
        "hint": tool.hint,
        "inputs": [_input_spec(i) for i in tool.inputs],
        "makes": tool.makes,
    }


def _installed_addon_payload(addon_id: str) -> dict[str, Any] | None:
    addon_dir = _addon_dir(addon_id)
    if not addon_dir.is_dir():
        return None
    try:
        manifest = read_manifest(addon_dir / "addon.json")
        install_info = ai._read_install_json(addon_dir)
    except (OSError, ValueError):
        return None
    return {
        "id": manifest.id,
        "name": manifest.name,
        "author": manifest.author,
        "summary": manifest.summary,
        "type": manifest.type,
        "installed": True,
        "listed": False,
        "reviewed": bool(install_info.get("reviewed", False)),
        "licence": manifest.licence,
        "tools": [_tool_spec(t) for t in manifest.tools],
    }


def _listed_addon_payload(listing: Listing, installed_ids: set[str]) -> dict[str, Any]:
    payload = {
        "id": listing.id,
        "name": listing.name,
        "author": listing.author,
        "summary": listing.summary,
        "type": listing.type,
        "installed": listing.id in installed_ids,
        "listed": True,
        "reviewed": True,  # the list entry itself; an installed copy overrides this below
        "links": listing.links,
        "rating": listing.rating,
        "ratings_count": listing.ratings_count,
    }
    if listing.id in installed_ids:
        installed = _installed_addon_payload(listing.id)
        if installed is not None:
            # What was actually installed decides, not the name: a copy installed without
            # review stays "not reviewed" even if a listed add-on shares its id.
            payload["reviewed"] = installed["reviewed"]
            payload["licence"] = installed["licence"]
            payload["tools"] = installed["tools"]
    return payload


def addons_payload() -> dict[str, Any]:
    """Everything the Add-ons section of Setup and the Add-ons sheet need."""
    installed_ids = set(ai.list_installed(ADDONS_DIR))
    addons: list[dict[str, Any]] = []
    online = True
    error: str | None = None

    try:
        registry = ai.load_registry(INDEX_FILE)
        listed_ids = {entry.id for entry in registry}
        for entry in registry:
            addons.append(_listed_addon_payload(entry, installed_ids))
        for addon_id in sorted(installed_ids - listed_ids):
            installed = _installed_addon_payload(addon_id)
            if installed is not None:
                installed["no_longer_listed"] = True
                addons.append(installed)
    except Exception as exc:
        online = False
        error = f"The add-on list is unreachable: {exc}"
        for addon_id in sorted(installed_ids):
            installed = _installed_addon_payload(addon_id)
            if installed is not None:
                addons.append(installed)

    return {"online": online, "error": error, "addons": addons}


def install_payload(addon_id: str) -> dict[str, Any]:
    """Return the plan for installing a listed add-on."""
    result = ai.install_addon(
        addon_id=addon_id,
        index=INDEX_FILE,
        confirm=False,
        addons_dir=ADDONS_DIR,
        repo_dir=REPO,
    )
    plan = result["plan"]
    listing: Listing = result["listing"]
    return {
        "id": listing.id,
        "name": result["manifest"].name,
        "author": plan["author"],
        "licence": plan["licence"],
        "models": plan["models"],
        "setup": plan["setup"],
        "warning": plan["warning"],
        "reviewed": True,
    }


def apply_install(addon_id: str) -> dict[str, Any]:
    """Install a listed add-on after the user confirmed."""
    result = ai.install_addon(
        addon_id=addon_id,
        index=INDEX_FILE,
        confirm=True,
        addons_dir=ADDONS_DIR,
        repo_dir=REPO,
    )
    return {"installed": True, "id": result["id"]}


def apply_remove(addon_id: str) -> dict[str, Any]:
    """Remove an installed add-on."""
    ai.remove_addon(addon_id, ADDONS_DIR)
    return {"removed": True, "id": addon_id}


class AddonJob:
    def __init__(self, job_id: str, addon_id: str, tool_id: str, inputs: dict[str, str],
                 run_stem: str):
        self.id = job_id
        self.addon_id = addon_id
        self.tool_id = tool_id
        self.inputs = inputs
        self.run_stem = run_stem
        self.status = "queued"
        self.log_lines: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.result_path: str | None = None
        self.error_message: str | None = None
        self.started = time.monotonic()
        self.condition = threading.Condition()
        self.cancel_event = threading.Event()

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        self.emit({"phase": "log", "message": line})


JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = {"done", "error", "cancelled"}


class AddonJobManager:
    def __init__(self):
        self.jobs: dict[str, AddonJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def busy(self) -> bool:
        active = self.jobs.get(self.active) if self.active else None
        return active is not None and active.status not in TERMINAL

    def create(self, addon_id: str, tool_id: str, inputs: dict[str, str]) -> AddonJob:
        """Validate the request and create a job without starting it."""
        addon_dir = _addon_dir(addon_id)
        if not addon_dir.is_dir():
            raise ValueError(f"{addon_id} is not installed")
        manifest = read_manifest(addon_dir / "addon.json")
        tool = ai._tool_by_id(manifest, tool_id)
        # Validate inputs early so the API can refuse before queueing the job.
        ai.parse_inputs(inputs, tool, cwd=REPO)
        run_stem = ai._default_run_stem()
        job = AddonJob(uuid.uuid4().hex, addon_id, tool_id, dict(inputs), run_stem)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def _claim(self, job: AddonJob) -> AddonJob:
        with self.lock:
            self.active = job.id
        return job

    def run(self, job: AddonJob) -> None:
        self._claim(job)
        threading.Thread(
            target=_run_addon_job,
            args=(job, self),
            daemon=True,
            name=f"addon-{job.id[:8]}",
        ).start()

    def get(self, job_id: str) -> AddonJob | None:
        if not JOB_ID_RE.fullmatch(job_id):
            return None
        return self.jobs.get(job_id)

    def finish(self, job: AddonJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            raise ValueError("unknown job")
        if job.status in TERMINAL:
            raise ValueError(f"job is already {job.status}")
        job.status = "cancelling"
        job.cancel_event.set()


ADDON_JOBS = AddonJobManager()


def _run_addon_job(job: AddonJob, manager: AddonJobManager) -> None:
    try:
        job.status = "running"
        result = ai.run_addon(
            job.addon_id,
            job.tool_id,
            job.inputs,
            run_stem=job.run_stem,
            output_root=OUTPUT_DIR,
            addons_dir=ADDONS_DIR,
            repo_dir=REPO,
            logger=job.append_log,
            cancel_event=job.cancel_event,
        )
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.emit({"phase": "cancelled"})
            return
        job.result_path = str(result)
        job.status = "done"
        job.emit({"phase": "done", "result": job.result_path})
    except Exception as exc:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.emit({"phase": "cancelled"})
            return
        job.status = "error"
        job.error_message = str(exc)
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        manager.finish(job)


def run_payload(addon_id: str, tool_id: str, inputs: dict[str, str]) -> dict[str, Any]:
    """Start an add-on run and return the job id."""
    if ADDON_JOBS.busy():
        raise RuntimeError("another add-on run is already in progress")
    job = ADDON_JOBS.create(addon_id, tool_id, inputs)
    ADDON_JOBS.run(job)
    return {"job_id": job.id, "status": "running"}


def status_payload(job: AddonJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "addon_id": job.addon_id,
        "tool_id": job.tool_id,
        "log_tail": "\n".join(job.log_lines)[-4000:],
        "result": job.result_path,
        "error": job.error_message,
        "last_event": job.events[-1] if job.events else None,
    }


