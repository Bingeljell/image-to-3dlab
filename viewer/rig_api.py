"""Queued headless-Blender jobs for browser Rig Review rebinds."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.rig_sidecar import load_sidecar, plan_corrections, verify_asset, verify_scene  # noqa: E402

OUTPUT_ROOT = REPO / "output" / "rig-rebind"
WORKER = REPO / "scripts" / "blender_rebind.py"
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = {"done", "error", "cancelled"}
ARTIFACTS = {
    "result.glb": ("result_glb", "model/gltf-binary"),
    "scene.blend": ("result_blend", "application/octet-stream"),
    "rig.json": ("result_sidecar", "application/json"),
    "report.json": ("report_path", "application/json"),
}


def blender_executable() -> Path | None:
    configured = os.environ.get("I2L_BLENDER")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    return DEFAULT_BLENDER if DEFAULT_BLENDER.is_file() else None


class RigJob:
    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.asset_path = directory / "source.glb"
        self.scene_path = directory / "source.blend"
        self.sidecar_path = directory / "corrected.rig.json"
        self.result_glb = directory / "result.glb"
        self.result_blend = directory / "scene.blend"
        self.result_sidecar = directory / "result.rig.json"
        self.report_path = directory / "report.json"
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.log_lines: deque[str] = deque(maxlen=300)

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        with self.directory.joinpath("run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class RigJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, RigJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def create(self, asset_name: str, asset: bytes, scene: bytes, sidecar: bytes) -> RigJob:
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a rig rebind is already running")
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__rebind__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__rebind__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = RigJob(uuid.uuid4().hex, directory)
            job.asset_path.write_bytes(asset)
            job.scene_path.write_bytes(scene)
            job.sidecar_path.write_bytes(sidecar)
            try:
                parsed = load_sidecar(job.sidecar_path)
                verify_asset(parsed, job.asset_path)
                verify_scene(parsed, job.scene_path)
                corrections = plan_corrections(parsed)
                if not corrections:
                    raise ValueError("rig sidecar contains no corrections")
                missing = [item.joint_id for item in corrections if not item.targets]
                if missing:
                    raise ValueError("corrections have no Blender binding: " + ", ".join(missing))
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> RigJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: RigJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


RIG_JOBS = RigJobManager()


def build_command(job: RigJob, blender: Path) -> list[str]:
    return [
        str(blender), "--background", str(job.scene_path), "--python", str(WORKER), "--",
        str(job.asset_path), str(job.sidecar_path), str(job.result_glb),
        str(job.result_blend), str(job.result_sidecar), str(job.report_path),
    ]


def run_job(job: RigJob, manager: RigJobManager = RIG_JOBS) -> None:
    blender = blender_executable()
    if blender is None:
        job.status = "error"
        job.emit({"phase": "error", "message": "Blender is not installed or I2L_BLENDER is invalid"})
        manager.finish(job)
        return
    try:
        job.status = "running"
        job.emit({"phase": "validate", "overall_pct": 5, "message": "Inputs verified"})
        job.process = subprocess.Popen(
            build_command(job, blender), cwd=str(REPO), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        job.directory.joinpath("pid").write_text(str(job.process.pid))
        assert job.process.stdout is not None
        for raw in job.process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            job.append_log(line)
            event = _stage_event(line)
            if event:
                job.emit(event)
        return_code = job.process.wait()
        job.append_log(f"Blender exited with code {return_code}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Rebind cancelled"})
        elif return_code != 0:
            job.status = "error"
            job.emit({
                "phase": "error", "message": f"Blender exited with code {return_code}",
                "log_tail": "\n".join(job.log_lines)[-8000:],
            })
        elif all(path.is_file() for path in (
            job.result_glb, job.result_blend, job.result_sidecar, job.report_path,
        )):
            job.status = "done"
            job.emit({
                "phase": "done", "overall_pct": 100, "message": "Rebind complete",
                "result_url": f"/api/rig/rebind/{job.id}/result.glb",
                "scene_url": f"/api/rig/rebind/{job.id}/scene.blend",
                "sidecar_url": f"/api/rig/rebind/{job.id}/rig.json",
                "report_url": f"/api/rig/rebind/{job.id}/report.json",
            })
        else:
            job.status = "error"
            job.emit({"phase": "error", "message": "Blender exited without all required artifacts"})
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        job.directory.joinpath("pid").unlink(missing_ok=True)
        manager.finish(job)


def cancel_job(job: RigJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def status_payload(job: RigJob) -> dict[str, Any]:
    return {"status": job.status, "last_event": job.events[-1] if job.events else None}


def _stage_event(line: str) -> dict[str, Any] | None:
    if not line.startswith("I2L_STAGE::"):
        return None
    _, phase, message = line.split("::", 2)
    progress = {"apply": 15, "rigify": 35, "weights": 55, "export": 85}.get(phase, 10)
    return {"phase": phase, "overall_pct": progress, "message": message}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "rig"
