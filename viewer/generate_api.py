#!/usr/bin/env python3
"""Small, local-only HTTP job API for the viewer's Generate mode.

This module deliberately uses only the standard library at import time.  The expensive clean
TRELLIS interpreter is started in a separate process only after a browser submits a job; importing
the viewer server, running its tests, or serving Compare never imports torch.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / "vendor" / "upstream-audit-worktree"
WRAPPER = AUDIT_ROOT / "scripts" / "trellis_space_generate.py"
PYTHON = AUDIT_ROOT / "vendor" / "trellis-space-mac" / ".venv" / "bin" / "python"
OUTPUT_ROOT = REPO / "output" / "space_web"
BASELINE_PATH = REPO / "viewer" / "generate_baseline.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "resolution": "1024",
    "seed": 0,
    "decimation_target": 300_000,
    "texture_size": 2048,
    "allow_rembg": False,
}
VALID_RESOLUTIONS = {"512", "1024", "1536"}
VALID_TEXTURES = {1024, 2048, 3072, 4096}
STAGES = [
    "load",
    "sparse_structure",
    "shape_slat_coarse",
    "shape_slat_fine",
    "texture_slat",
    "decode",
    "bake",
]
PHASE_LABELS = {
    "load": "Load",
    "sparse_structure": "Sparse structure",
    "shape_slat_coarse": "Shape SLat coarse",
    "shape_slat_fine": "Shape SLat fine",
    "texture_slat": "Texture SLat",
    "decode": "Decode",
    "bake": "Bake / remesh",
}
TQDM_RE = re.compile(
    r"(?P<label>[^:]+):\s+(?P<pct>\d+)%.*?"
    r"(?P<step>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>\d+:\d+)"
    r"(?:<(?P<remain>\d+:\d+))?,?\s*(?P<rate>[\d.]+)s/it\]"
)
SECONDS_RE = re.compile(r"(?:in|Total:)\s*(?P<seconds>[\d.]+)s?")
JOB_RE = re.compile(r"^[0-9a-f]{32}$")


def _seconds(value: str | None) -> float | None:
    if not value:
        return None
    bits = value.split(":")
    try:
        if len(bits) == 2:
            return float(bits[0]) * 60 + float(bits[1])
        if len(bits) == 3:
            return float(bits[0]) * 3600 + float(bits[1]) * 60 + float(bits[2])
    except ValueError:
        return None
    return None


def parse_tqdm_line(line: str) -> dict[str, Any] | None:
    """Parse one tqdm update, including the carriage-return form emitted by MPS jobs."""
    match = TQDM_RE.search(line.strip())
    if not match:
        return None
    return {
        "label": match.group("label").strip(),
        "pct": int(match.group("pct")),
        "step": int(match.group("step")),
        "total": int(match.group("total")),
        "elapsed": _seconds(match.group("elapsed")),
        "remain": _seconds(match.group("remain")),
        "s_per_it": float(match.group("rate")),
    }


def validate_settings(raw: Any) -> dict[str, Any]:
    """Validate browser JSON and return a fresh, CLI-shaped settings dict."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**DEFAULT_SETTINGS, **raw}
    if str(settings["resolution"]) not in VALID_RESOLUTIONS:
        raise ValueError("resolution must be one of 512, 1024, or 1536")
    try:
        settings["seed"] = int(settings["seed"])
        settings["decimation_target"] = int(settings["decimation_target"])
        settings["texture_size"] = int(settings["texture_size"])
    except (TypeError, ValueError) as exc:
        raise ValueError("seed, decimation_target, and texture_size must be integers") from exc
    if settings["decimation_target"] <= 0:
        raise ValueError("decimation_target must be positive")
    if settings["texture_size"] not in VALID_TEXTURES:
        raise ValueError("texture_size must be one of 1024, 2048, 3072, or 4096")
    if not isinstance(settings["allow_rembg"], bool):
        raise ValueError("allow_rembg must be a boolean")
    settings["resolution"] = str(settings["resolution"])
    return settings


def image_has_transparent_alpha(path: Path) -> bool:
    """Return whether an image has an actual (not merely opaque) alpha channel."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.mode != "RGBA":
                return False
            return image.getextrema()[3][0] < 255
    except Exception:
        # A missing/undecodable alpha is deliberately conservative: the wrapper will refuse it
        # unless the user explicitly opts into BRIA rembg.
        return False


def _baseline() -> dict[str, float]:
    try:
        data = json.loads(BASELINE_PATH.read_text())
        seconds = data.get("seconds", data.get("stages", {}))
        return {str(k): float(v) for k, v in seconds.items() if v is not None}
    except (OSError, ValueError, TypeError):
        return {}


def _update_baseline(job: Job) -> None:
    """Learn real stage durations after a successful job for the next ETA estimate."""
    if not job.stage_durations:
        return
    try:
        data = json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError, TypeError):
        data = {"schema_version": 1}
    seconds = data.setdefault("seconds", {})
    seconds.update({name: round(value, 1) for name, value in job.stage_durations.items()})
    data["source"] = "learned from completed clean-port Generate jobs"
    BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _safe_id(value: str) -> bool:
    return bool(JOB_RE.fullmatch(value))


def parse_multipart(content_type: str, body: bytes) -> dict[str, dict[str, Any]]:
    """Parse the two small multipart fields without the removed Python 3.13 ``cgi`` module."""
    raw = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body)
    message = BytesParser(policy=policy.default).parsebytes(raw)
    fields: dict[str, dict[str, Any]] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        name = part.get_param("name", header="Content-Disposition")
        if not name or "form-data" not in disposition:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[str(name)] = {
            "filename": part.get_param("filename", header="Content-Disposition"),
            "data": payload,
            "value": payload.decode("utf-8", errors="replace"),
        }
    return fields


class Job:
    def __init__(self, job_id: str, directory: Path, image_path: Path, output_path: Path,
                 settings: dict[str, Any]):
        self.id = job_id
        self.directory = directory
        self.image_path = image_path
        self.output_path = output_path
        self.manifest_path = output_path.with_suffix(".json")
        self.settings = settings
        self.status = "queued"
        self.process: subprocess.Popen[bytes] | None = None
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.log_lines: deque[str] = deque(maxlen=200)
        self.condition = threading.Condition()
        self.shape_pass = 0
        self.stage_started: dict[str, float] = {}
        self.stage_durations: dict[str, float] = {}
        self.cancel_requested = False

    def emit(self, event: dict[str, Any]) -> None:
        event = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        with self.directory.joinpath("run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class JobManager:
    """One active subprocess, with completed jobs retained for reconnect/downloads."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.active: str | None = None

    def create(self, image_path: Path, settings: dict[str, Any]) -> Job:
        with self.lock:
            if self.active is not None:
                active = self.jobs.get(self.active)
                if active and active.status in {"queued", "running", "cancelling"}:
                    raise RuntimeError("a TRELLIS generation is already running")
            job_id = uuid.uuid4().hex
            directory = OUTPUT_ROOT / job_id
            directory.mkdir(parents=True, exist_ok=False)
            output_path = directory / "model.glb"
            job = Job(job_id, directory, image_path, output_path, settings)
            self.jobs[job_id] = job
            self.active = job_id
            return job

    def finish(self, job: Job) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


JOBS = JobManager()


def _phase_for_tqdm(job: Job, label: str, step: int) -> str:
    lower = label.lower()
    if "sparse" in lower:
        return "sparse_structure"
    if "shape" in lower:
        # The 1024 cascade prints this exact label twice.  A new 0/12 bar starts the fine pass.
        if job.shape_pass == 0:
            job.shape_pass = 1
        elif step <= 1 and getattr(job, "shape_last_step", 0) > 1:
            job.shape_pass = 2
        job.shape_last_step = step
        return "shape_slat_fine" if job.shape_pass >= 2 else "shape_slat_coarse"
    if "texture" in lower or "tex slat" in lower:
        return "texture_slat"
    return "bake"


def _overall_pct(phase: str, stage_pct: int) -> int:
    try:
        index = STAGES.index(phase)
    except ValueError:
        return 0
    return round((index + max(0, min(stage_pct, 100)) / 100) / len(STAGES) * 100)


def _event_from_tqdm(job: Job, parsed: dict[str, Any]) -> dict[str, Any]:
    phase = _phase_for_tqdm(job, parsed["label"], parsed["step"])
    now = time.monotonic()
    if parsed["step"] <= 1 or phase not in job.stage_started:
        job.stage_started.setdefault(phase, now)
    baseline = _baseline()
    future = STAGES[STAGES.index(phase) + 1:] if phase in STAGES else []
    future_seconds = sum(baseline.get(s, 0.0) for s in future)
    stage_eta = parsed["remain"]
    total_eta = (stage_eta or 0.0) + future_seconds
    if parsed["pct"] >= 100 and parsed["elapsed"] is not None:
        job.stage_durations[phase] = parsed["elapsed"]
    return {
        "phase": phase,
        "step": parsed["step"],
        "total": parsed["total"],
        "s_per_it": parsed["s_per_it"],
        "stage_eta_seconds": round(stage_eta, 1) if stage_eta is not None else None,
        "total_eta_seconds": round(total_eta, 1),
        "stage_pct": parsed["pct"],
        "overall_pct": _overall_pct(phase, parsed["pct"]),
        "message": PHASE_LABELS.get(phase, parsed["label"]),
    }


def _emit_banner(job: Job, line: str) -> None:
    lower = line.lower()
    event: dict[str, Any] | None = None
    if "loading trellis" in lower:
        event = {"phase": "load", "overall_pct": 0, "message": "Loading TRELLIS.2 pipeline"}
    elif "pipeline loaded" in lower:
        event = {"phase": "load", "stage_pct": 100, "overall_pct": _overall_pct("load", 100),
                 "message": line.strip()}
    elif "decode_latent done" in lower:
        event = {"phase": "decode", "stage_pct": 100, "overall_pct": _overall_pct("decode", 100),
                 "message": "Decode complete"}
    elif "to_glb + export done" in lower:
        event = {"phase": "bake", "stage_pct": 100, "overall_pct": _overall_pct("bake", 100),
                 "message": "GLB export complete"}
    if event:
        job.emit(event)


def _read_process(job: Job) -> None:
    assert job.process is not None and job.process.stdout is not None
    buffer = ""
    while True:
        chunk = job.process.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        while True:
            positions = [p for p in (buffer.find("\n"), buffer.find("\r")) if p >= 0]
            if not positions:
                break
            cut = min(positions)
            line, buffer = buffer[:cut], buffer[cut + 1:]
            if not line:
                continue
            job.append_log(line)
            parsed = parse_tqdm_line(line)
            if parsed:
                job.emit(_event_from_tqdm(job, parsed))
            else:
                _emit_banner(job, line)
    if buffer:
        job.append_log(buffer)
        parsed = parse_tqdm_line(buffer)
        if parsed:
            job.emit(_event_from_tqdm(job, parsed))


def _run_job(job: Job) -> None:
    args = [
        str(PYTHON), str(WRAPPER), str(job.image_path), str(job.output_path),
        "--vendor-root", str(AUDIT_ROOT / "vendor" / "trellis-space-mac"),
        "--resolution", job.settings["resolution"],
        "--seed", str(job.settings["seed"]),
        "--decimation-target", str(job.settings["decimation_target"]),
        "--texture-size", str(job.settings["texture_size"]),
    ]
    if job.settings["allow_rembg"]:
        args.append("--allow-rembg")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        job.status = "running"
        job.emit({"phase": "load", "overall_pct": 0, "message": "Starting clean-port job"})
        job.process = subprocess.Popen(
            args,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        reader = threading.Thread(target=_read_process, args=(job,), daemon=True)
        reader.start()
        return_code = job.process.wait()
        reader.join(timeout=5)
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Generation cancelled"})
        elif return_code == 0 and job.output_path.is_file():
            job.status = "done"
            _update_baseline(job)
            job.emit({
                "phase": "done", "overall_pct": 100, "message": "Generation complete",
                "result_url": f"/api/generate/{job.id}/result.glb",
                "manifest_url": f"/api/generate/{job.id}/manifest.json",
            })
        else:
            tail = "\n".join(job.log_lines)[-8000:]
            job.status = "error"
            job.emit({"phase": "error", "message": f"generator exited with code {return_code}",
                      "log_tail": tail})
    except Exception as exc:  # process launch errors must reach the browser, not kill the server
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        JOBS.finish(job)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


class Handler(SimpleHTTPRequestHandler):
    """Static repository server plus the local Generate job endpoints."""

    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".js": "text/javascript",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:
        if "404" in (fmt % args):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, value: Any) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _path_parts(self) -> list[str]:
        return [unquote(p) for p in urlparse(self.path).path.split("/") if p]

    def do_POST(self) -> None:
        parts = self._path_parts()
        if parts == ["api", "generate"]:
            self._create_job()
            return
        if len(parts) == 4 and parts[:2] == ["api", "generate"] and parts[3] == "cancel":
            self._cancel_job(parts[2])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parts = self._path_parts()
        if len(parts) == 4 and parts[:2] == ["api", "generate"]:
            job_id, action = parts[2], parts[3]
            if action == "events":
                self._events(job_id)
                return
            if action in {"result.glb", "manifest.json"}:
                self._artifact(job_id, action)
                return
        super().do_GET()

    def _create_job(self) -> None:
        try:
            if not PYTHON.is_file() or not WRAPPER.is_file():
                self._send_json(503, {"error": "clean-port generator is not installed"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send_json(400, {"error": "image upload is missing or larger than 50 MiB"})
                return
            body = self.rfile.read(length)
            form = parse_multipart(self.headers.get("Content-Type", ""), body)
            image_field = form.get("image")
            if image_field is None or not image_field.get("filename"):
                self._send_json(400, {"error": "multipart field 'image' is required"})
                return
            settings_value = form.get("settings", {}).get("value", "{}")
            try:
                settings = validate_settings(json.loads(settings_value))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(422, {"error": str(exc)})
                return
            suffix = Path(str(image_field["filename"])).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self._send_json(422, {"error": "unsupported image type; use PNG, JPG, WebP, or BMP"})
                return
            # Reserve the job directory only after validation, then persist the original upload.
            provisional = OUTPUT_ROOT / uuid.uuid4().hex
            provisional.mkdir(parents=True, exist_ok=False)
            image_path = provisional / f"input{suffix}"
            with image_path.open("wb") as handle:
                handle.write(image_field["data"])
            if not image_has_transparent_alpha(image_path) and not settings["allow_rembg"]:
                for child in provisional.iterdir():
                    child.unlink()
                provisional.rmdir()
                self._send_json(422, {
                    "error": "This image has no transparent alpha foreground. Enable 'allow rembg' "
                             "to use BRIA background removal, or upload a pre-masked PNG."
                })
                return
            # JobManager creates its own UUID directory. Move the upload into it so the id and
            # artifact URLs are stable, without ever accepting a client-provided path.
            provisional_image = image_path
            try:
                job = JOBS.create(provisional_image, settings)
            except RuntimeError as exc:
                provisional_image.unlink(missing_ok=True)
                provisional.rmdir()
                self._send_json(409, {"error": str(exc)})
                return
            final_image = job.directory / image_path.name
            provisional_image.replace(final_image)
            provisional.rmdir()
            job.image_path = final_image
            threading.Thread(target=_run_job, args=(job,), daemon=True).start()
            self._send_json(202, {"job_id": job.id, "events_url": f"/api/generate/{job.id}/events"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _find_job(self, job_id: str) -> Job | None:
        return JOBS.get(job_id) if _safe_id(job_id) else None

    def _events(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        index = 0
        try:
            while True:
                with job.condition:
                    if index >= len(job.events):
                        job.condition.wait(timeout=15)
                    pending = job.events[index:]
                    index = len(job.events)
                    terminal = job.status in {"done", "error", "cancelled"} and not pending
                for event in pending:
                    payload = json.dumps(event, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                if not pending:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                if terminal:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return

    def _artifact(self, job_id: str, action: str) -> None:
        job = self._find_job(job_id)
        if job is None or job.status != "done":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = job.output_path if action == "result.glb" else job.manifest_path
        if not path.is_file() or OUTPUT_ROOT not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = "model/gltf-binary" if action == "result.glb" else "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _cancel_job(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if job.status in {"done", "error", "cancelled"}:
            self._send_json(409, {"error": f"job is already {job.status}"})
            return
        job.cancel_requested = True
        job.status = "cancelling"
        process = job.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        job.emit({"phase": "error", "message": "Cancellation requested"})
        self._send_json(202, {"job_id": job.id, "status": "cancelling"})
