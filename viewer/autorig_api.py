"""The Animate tab: give a humanoid a skeleton, then play a preset clip on it.

Two jobs, both short and both subprocesses so a crash never takes the viewer down:

* **rig**: SkinTokens (`vendor/SkinTokens/demo.py`) fits a skeleton and skin weights to a
  T-posed humanoid. About a minute and ~8 GB on an Apple Silicon Mac.
* **animate**: `scripts/kimodo_retarget.py` plays a preset clip on that skeleton and
  writes an animated GLB. About ten seconds, CPU only. It runs in SkinTokens' venv,
  which already carries Blender's `bpy`.

Each rigged model gets one folder, so its animations sit beside it:

    output/animate/<name>__rig__<stamp>/
        input/source.glb          what was rigged (copied, so the run never depends on it)
        <name>_rigged.glb          the skeleton + weights, no motion
        <name>_<preset>.glb        one per preset played on it
        *.provenance.json          licences: the source model's, plus SkinTokens (MIT),
                                   plus Kimodo's for clips (NVIDIA Open Model License)

The preset clips are pre-made with Kimodo and shipped in `image_to_3dlab/motions/`. Users
never run Kimodo: no Llama download, no 24 GB text encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import processes

OUTPUT_ROOT = REPO / "output" / "animate"
SKINTOKENS = REPO / "vendor" / "SkinTokens"
RETARGET = REPO / "scripts" / "kimodo_retarget.py"
MOTIONS = REPO / "image_to_3dlab" / "motions"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
RUN_DIRECTORY = re.compile(r"^[A-Za-z0-9_-]{1,80}__rig__\d{8}-\d{6}(?:-\d+)?$")
PRESET_ID = re.compile(r"^[a-z0-9_]{1,40}$")
TERMINAL = {"done", "error", "cancelled"}
# Measured on an M-series Mac (2026-10-05): 64 s for a 200k-face knight. Only drives the
# progress bar's estimate, which says "about" for that reason.
RIG_SECONDS = 70.0
ANIMATE_SECONDS = 12.0
MAX_ARM_SPREAD = 30.0

SKINTOKENS_LICENSE = {
    "name": "MIT",
    "url": "https://github.com/VAST-AI-Research/SkinTokens/blob/main/LICENSE",
    "model": "VAST-AI/SkinTokens",
}
KIMODO_LICENSE = {
    "name": "NVIDIA Open Model License",
    "url": "https://www.nvidia.com/en-us/agreements/enterprise-software/"
           "nvidia-open-model-license/",
    "model": "nvidia/Kimodo-SOMA-RP-v1.1",
    "note": "Clip pre-generated with Kimodo. NVIDIA claims no ownership of outputs.",
}


def venv_python(root: Path = SKINTOKENS) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def installed(root: Path = SKINTOKENS) -> bool:
    return venv_python(root).exists() and (root / "demo.py").is_file()


# ---------------------------------------------------------------------------- presets


def presets(root: Path = MOTIONS) -> list[dict[str, Any]]:
    """The shipped clips, in the order `presets.json` lists them, that exist on disk."""
    index = _read_json(root / "presets.json") or {}
    return [dict(p) for p in index.get("presets", [])
            if PRESET_ID.fullmatch(str(p.get("id", ""))) and (root / f"{p['id']}.npz").is_file()]


# ----------------------------------------------------------------------------- models


def friendly_name(stem: str, detail: str | None = None) -> str:
    """'a-cute-anime-girl-standing-in__20261003-153554__pixal3d' -> 'a cute anime girl
    standing in', capped at 40 characters, plus an optional detail ('40k')."""
    words = stem.split("__")[0].replace("-", " ").replace("_", " ").strip() or stem
    if len(words) > 40:
        words = words[:39].rstrip() + "…"
    return f"{words} · {detail}" if detail else words


def _entry(path: Path, kind: str, output: Path, label: str | None = None) -> dict[str, Any]:
    return {"name": label or friendly_name(path.stem), "kind": kind,
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size, "modified": path.stat().st_mtime}


def pickable_models(output: Path = REPO / "output", limit: int = 40) -> list[dict[str, Any]]:
    """Models the Animate tab can start from, newest first, so nobody has to upload a file
    the lab made: rigged models (skip straight to a preset), finished models (the best
    input: cleaner topology, fewer faces) and raw generations."""
    found: list[dict[str, Any]] = []
    rigs = output / "animate"
    if rigs.is_dir():
        for run in rigs.iterdir():
            if run.is_dir() and RUN_DIRECTORY.fullmatch(run.name):
                for glb in run.glob("*_rigged.glb"):
                    found.append(_entry(glb, "rigged", output,
                                        friendly_name(glb.name.removesuffix("_rigged.glb"))))
    finished = output / "finish"
    if finished.is_dir():
        for run in finished.iterdir():
            if run.is_dir():
                # The result is the only GLB at the top of a finish run.
                for glb in run.glob("*.glb"):
                    faces = glb.stem.rsplit("_", 1)[-1]
                    detail = faces if faces[:-1].isdigit() and faces[-1] in "km" else None
                    found.append(_entry(glb, "finished", output,
                                        friendly_name(run.name, detail)))
    from props_api import generated_models  # the Props tab's list, same rules

    for model in generated_models(output, limit=limit):
        found.append(_entry(output / model["path"], "generated", output))
    found.sort(key=lambda m: m["modified"], reverse=True)
    return found[:limit]


def resolve_model(relative: str, output: Path = REPO / "output") -> Path:
    """One of `pickable_models`, by the path the page was given, and nothing else."""
    for model in pickable_models(output, limit=10_000):
        if model["path"] == relative:
            return output / relative
    raise ValueError(f"not a model this tab offers: {relative!r}")


def rig_run_of(glb: Path) -> Path | None:
    """The rig folder a `_rigged.glb` lives in, or None for anything else."""
    run = glb.parent
    if glb.name.endswith("_rigged.glb") and RUN_DIRECTORY.fullmatch(run.name):
        return run
    return None


# ------------------------------------------------------------------------- provenance


def source_record_for(glb: Path) -> dict[str, Any] | None:
    """A licence record kept beside a model: a provenance sidecar, a run manifest, or a
    finish record. The first that names a licence wins."""
    for sidecar in (glb.with_suffix(".provenance.json"), glb.with_suffix(".json"),
                    glb.with_suffix(".retopo-repaint.json")):
        record = _read_json(sidecar)
        if isinstance(record, dict) and record.get("license"):
            return record
    return None


def rig_record(name: str, data: bytes, source: dict[str, Any] | None,
               origin: str) -> dict[str, Any]:
    """Provenance for a rigged model: the source's licence carries over, and the rigger's
    is listed beside it rather than mixed in."""
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "viewer Animate tab: SkinTokens auto-rig",
        "input": {"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                  "source": origin},
        "license": (source or {}).get("license"),
        "classification": ((source or {}).get("output") or {}).get("classification")
        or (source or {}).get("classification"),
        "rig": {"tool": "SkinTokens", "license": SKINTOKENS_LICENSE},
        "source_record": source,
    }


def animation_record(rig: dict[str, Any] | None, preset: dict[str, Any],
                     arm_spread: float) -> dict[str, Any]:
    record = dict(rig or {})
    record.update({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "viewer Animate tab: scripts/kimodo_retarget.py",
        "motion": {"preset": preset["id"], "label": preset.get("label"),
                   "made_with": "Kimodo", "license": KIMODO_LICENSE,
                   "arm_spread_degrees": arm_spread},
    })
    return record


# ----------------------------------------------------------------------------- jobs


def rig_command(source: Path, result: Path) -> list[str]:
    return [str(venv_python()), "demo.py", "--input", str(source), "--output", str(result),
            "--use_transfer"]


def animate_command(rigged: Path, motion: Path, result: Path, arm_spread: float) -> list[str]:
    return [str(venv_python()), str(RETARGET), "--rig", str(rigged), "--motion", str(motion),
            "--out", str(result), "--arm-spread", f"{arm_spread:g}"]


def clamp_spread(value: Any) -> float:
    try:
        spread = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_spread must be a number of degrees") from exc
    if not 0 <= spread <= MAX_ARM_SPREAD:
        raise ValueError(f"arm_spread must be between 0 and {MAX_ARM_SPREAD:g} degrees")
    return spread


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:60] or "model"


class AnimateJob:
    def __init__(self, job_id: str, kind: str, directory: Path, result: Path,
                 command: list[str], cwd: Path, expected_seconds: float,
                 record: dict[str, Any]):
        self.id = job_id
        self.kind = kind
        self.directory = directory
        self.result = result
        self.command = command
        self.cwd = cwd
        self.expected_seconds = expected_seconds
        self.record = record
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.log_lines: deque[str] = deque(maxlen=300)

    @property
    def log_path(self) -> Path:
        return self.directory / "steps" / f"{self.result.stem}.log"

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def estimated_pct(elapsed: float, expected: float) -> int:
    """Progress from the clock, capped short of done: neither tool reports its own."""
    return min(95, int(100 * elapsed / max(expected, 1.0)))


class AnimateJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT, motions: Path = MOTIONS):
        self.output_root = output_root
        self.motions = motions
        self.jobs: dict[str, AnimateJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def busy(self) -> bool:
        active = self.jobs.get(self.active) if self.active else None
        return active is not None and active.status not in TERMINAL

    def _claim(self, job: AnimateJob) -> AnimateJob:
        self.jobs[job.id] = job
        self.active = job.id
        return job

    def create_rig(self, name: str, data: bytes, source: dict[str, Any] | None,
                   origin: str) -> AnimateJob:
        with self.lock:
            if self.busy():
                raise RuntimeError("an Animate job is already running")
            stem = _slug(Path(name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__rig__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__rig__{stamp}-{suffix}"
                suffix += 1
            (directory / "input").mkdir(parents=True)
            source_glb = directory / "input" / "source.glb"
            source_glb.write_bytes(data)
            result = directory / f"{stem}_rigged.glb"
            job = AnimateJob(uuid.uuid4().hex, "rig", directory, result,
                             rig_command(source_glb, result), SKINTOKENS, RIG_SECONDS,
                             rig_record(name, data, source, origin))
            return self._claim(job)

    def create_animation(self, rigged: Path, preset_id: str, arm_spread: Any) -> AnimateJob:
        with self.lock:
            if self.busy():
                raise RuntimeError("an Animate job is already running")
            run = rig_run_of(rigged)
            if run is None:
                raise ValueError("pick a rigged model first")
            preset = next((p for p in presets(self.motions) if p["id"] == preset_id), None)
            if preset is None:
                raise ValueError(f"no such preset: {preset_id!r}")
            spread = clamp_spread(arm_spread)
            stem = rigged.name.removesuffix("_rigged.glb")
            result = run / f"{stem}_{preset_id}.glb"
            record = animation_record(_read_json(rigged.with_suffix(".provenance.json")),
                                      preset, spread)
            job = AnimateJob(uuid.uuid4().hex, "animate", run, result,
                             animate_command(rigged, self.motions / f"{preset_id}.npz",
                                             result, spread),
                             REPO, ANIMATE_SECONDS, record)
            return self._claim(job)

    def get(self, job_id: str) -> AnimateJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: AnimateJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


ANIMATE_JOBS = AnimateJobManager()


def _tick(job: AnimateJob, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        elapsed = time.monotonic() - job.started
        job.emit({"phase": "running", "overall_pct": estimated_pct(elapsed, job.expected_seconds),
                  "message": "Fitting a skeleton" if job.kind == "rig" else "Playing the clip"})


def run_job(job: AnimateJob, manager: AnimateJobManager = ANIMATE_JOBS) -> None:
    stop = threading.Event()
    try:
        job.status = "running"
        job.emit({"phase": "queued", "overall_pct": 0,
                  "message": "Starting the auto-rig" if job.kind == "rig" else "Starting"})
        threading.Thread(target=_tick, args=(job, stop), daemon=True).start()
        job.process = subprocess.Popen(
            job.command, cwd=str(job.cwd),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            **processes.group_popen_kwargs(),
        )
        assert job.process.stdout is not None
        for raw in job.process.stdout:
            line = raw.rstrip("\n")
            if line:
                job.append_log(line)
        code = job.process.wait()
        stop.set()
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Cancelled"})
        elif code != 0 or not job.result.is_file():
            job.status = "error"
            job.emit({"phase": "error",
                      "message": f"{job.kind} step exited with code {code}"
                      if code else f"{job.kind} step wrote no GLB",
                      "log_tail": "\n".join(job.log_lines)[-6000:]})
        else:
            job.result.with_suffix(".provenance.json").write_text(
                json.dumps(job.record, indent=2))
            job.status = "done"
            job.emit({"phase": "done", "overall_pct": 100,
                      "message": "Rigged" if job.kind == "rig" else "Animated",
                      "kind": job.kind,
                      "path": job.result.relative_to(REPO / "output").as_posix()
                      if REPO / "output" in job.result.parents else None,
                      "result_url": served_url(job.result),
                      "size_bytes": job.result.stat().st_size})
    except Exception as exc:  # noqa: BLE001 - whatever broke, the job must end visibly
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        stop.set()
        manager.finish(job)


def cancel_job(job: AnimateJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: AnimateJob) -> dict[str, Any]:
    return {"status": job.status, "kind": job.kind,
            "log_tail": "\n".join(job.log_lines)[-4000:],
            "last_event": job.events[-1] if job.events else None}


def served_url(path: Path) -> str | None:
    try:
        return "/" + path.relative_to(REPO).as_posix()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
