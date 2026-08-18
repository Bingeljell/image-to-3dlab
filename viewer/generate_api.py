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
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, ClassVar
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "trellis_space_generate.py"
PYTHON = REPO / "vendor" / "trellis-space-mac" / ".venv" / "bin" / "python"
OUTPUT_ROOT = REPO / "output"
BASELINE_PATH = REPO / "viewer" / "generate_baseline.json"

HUNYUAN_WRAPPER = REPO / "scripts" / "hunyuan_mlx_generate.py"
HUNYUAN_PYTHON = REPO / "vendor" / "hunyuan-mlx" / ".venv" / "bin" / "python"
HUNYUAN_PAINT_VENV = REPO / "vendor" / "hunyuan-mlx-paint" / "python" / "paint" / ".venv" / "bin" / "python"
HUNYUAN_PAINT_WEIGHTS = REPO / "vendor" / "hunyuan-mlx-paint" / "python" / "paint" / "weights" / "hunyuan3d-paintpbr-v2-1"

SF3D_REPO_DEFAULT = REPO / "vendor" / "stable-fast-3d"

# Backend-selection vars the generator must own. A stale value inherited from the server's
# environment silently selects the wrong backend (the conv_none crash); the subprocess env
# drops them and the generator's configure_environment/require_flex_gemm pin them fresh.
BACKEND_ENV_KEYS = (
    "SPARSE_CONV_BACKEND",
    "ATTN_BACKEND",
    "SPARSE_ATTN_BACKEND",
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
)

HF_HUB_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
WEIGHT_REPOS = (
    ("models--microsoft--TRELLIS.2-4B", "TRELLIS.2-4B weights"),
    ("models--facebook--dinov3-vitl16-pretrain-lvd1689m", "DINOv3 image encoder"),
)


def _job_env() -> dict[str, str]:
    """Subprocess env for the generator: inherit everything except backend-selection vars.

    The generator pins ATTN_BACKEND/SPARSE_ATTN_BACKEND/SPARSE_CONV_BACKEND itself; a stale
    value in this server's environment (e.g. SPARSE_CONV_BACKEND=none exported earlier)
    would otherwise be inherited and silently break the run.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for key in BACKEND_ENV_KEYS:
        env.pop(key, None)
    return env


def _human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def weights_on_disk(cache_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Report which model weights are already in the HF cache.

    A first run with missing weights downloads ~14 GB in the background with no progress
    bar, which reads as a hung 'load'. Surfacing this up front is what the setup card shows.
    """
    hub = (cache_dir or HF_HUB_DIR).resolve()
    out: dict[str, dict[str, Any]] = {}
    for repo, label in WEIGHT_REPOS:
        path = hub / repo
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            out[repo] = {"label": label, "present": True, "bytes": size,
                         "human": _human_bytes(size)}
        else:
            out[repo] = {"label": label, "present": False, "bytes": 0, "human": "0 B"}
    return out


@dataclass(frozen=True)
class BackendSpec:
    """Everything the job runner needs to drive one generation backend.

    ``parse_line`` is a handler, not a pure parser: it emits SSE events on ``job`` itself
    (matching the shape of the pre-existing TRELLIS tqdm/banner handling) rather than
    returning an event for the caller to emit — this keeps the well-tested TRELLIS path
    untouched, just wrapped, instead of restructured.
    """

    id: str
    label: str
    interpreter: Path
    wrapper: Path
    default_settings: dict[str, Any]
    stages: list[str]
    stage_labels: dict[str, str]
    requires_alpha: bool
    validate_settings: Callable[[Any], dict[str, Any]]
    build_args: Callable[["Job"], list[str]]
    parse_line: Callable[["Job", str], None]
    readiness: Callable[[], dict[str, Any]]
    baseline_path: Path | None = None
    finalize: Callable[["Job"], None] | None = None
    """Runs after the subprocess exits 0, before the output-file existence check — for
    backends whose wrapper doesn't write directly to job.output_path (SF3D writes
    ``<stem>_sf3d.glb`` instead)."""


BACKENDS: dict[str, BackendSpec] = {}


def clean_port_build_present() -> bool:
    """Whether the clean-port build (interpreter + wrapper) is installed."""
    return PYTHON.is_file() and WRAPPER.is_file()


def setup_status() -> dict[str, Any]:
    """Machine readiness for the clean-port generator, for the Generate > Setup card."""
    build_present = clean_port_build_present()
    weights = weights_on_disk()
    missing = [w["label"] for w in weights.values() if not w["present"]]
    return {
        "schema_version": 1,
        "build": {
            "present": build_present,
            "interpreter": str(PYTHON),
            "wrapper": str(WRAPPER),
            "hint": (
                "clean-port build missing — from the repo root run: "
                "python scripts/bootstrap_trellis_space_macos.py "
                "(requires uv, Python 3.11 and Xcode command-line tools; ~1h)"
                if not build_present else None
            ),
        },
        "weights": weights,
        "missing_weights": missing,
        "ready": build_present,
        "warning": "first run will download missing weights" if missing else None,
    }


# Setup-run state: one bootstrap subprocess at a time, mirrored to the browser over SSE.
SETUP_RUNS: dict[str, SetupRun] = {}
SETUP_ACTIVE: str | None = None
SETUP_LOCK = threading.Lock()


class SetupRun:
    """A bootstrap subprocess exposing the same SSE surface (condition/events/status) as Job."""

    def __init__(self, run_id: str):
        self.id = run_id
        self.status = "queued"
        self.process: subprocess.Popen[bytes] | None = None
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()

    def emit(self, event: dict[str, Any]) -> None:
        event = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()


def setup_available() -> tuple[bool, str | None]:
    """Whether the setup runner can start: uv on PATH and the bootstrap script present."""
    if shutil.which("uv") is None:
        return False, "uv is not installed — install it first (https://docs.astral.sh/uv/)"
    bootstrap = REPO / "scripts" / "bootstrap_trellis_space_macos.py"
    if not bootstrap.is_file():
        return False, f"bootstrap script missing: {bootstrap}"
    return True, None


def _start_setup_run(run_id: str) -> SetupRun:
    """Spawn the bootstrap and stream its output into the run's events (SSE)."""
    run = SetupRun(run_id)
    bootstrap = REPO / "scripts" / "bootstrap_trellis_space_macos.py"
    proc = subprocess.Popen(
        [sys.executable, str(bootstrap)],
        cwd=str(REPO),
        env=_job_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    run.process = proc
    run.status = "running"

    def reader() -> None:
        global SETUP_ACTIVE
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line:
                run.emit({"phase": "setup", "message": line})
        code = proc.wait()
        ok = code == 0
        run.emit({"phase": "setup_done", "status": "done" if ok else "error",
                  "message": f"bootstrap exited with code {code}"
                             + ("" if ok else " — review the output above")})
        run.status = "done" if ok else "error"
        with SETUP_LOCK:
            SETUP_ACTIVE = None

    threading.Thread(target=reader, daemon=True, name=f"setup-{run_id[:8]}").start()
    return run

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
    r"(?:<(?P<remain>\d+:\d+))?,?\s*"
    r"(?P<rate>[\d.]+)(?P<rate_unit>s/it|it/s)\]"
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
    rate = float(match.group("rate"))
    return {
        "label": match.group("label").strip(),
        "pct": int(match.group("pct")),
        "step": int(match.group("step")),
        "total": int(match.group("total")),
        "elapsed": _seconds(match.group("elapsed")),
        "remain": _seconds(match.group("remain")),
        "s_per_it": rate if match.group("rate_unit") == "s/it" else 1.0 / rate,
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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return slug[:80] or "job"


def _resolve_output_base(output_dir: str | None) -> Path:
    """Resolve the user-chosen output base dir. Defaults to <repo-root>/output; a client-
    supplied override is resolved and required to stay inside that same tree (never an
    arbitrary absolute path -- this is a local server writing files from browser input)."""
    base = (REPO / "output").resolve()
    if not output_dir or not output_dir.strip():
        return base
    candidate = (REPO / output_dir.strip()).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"output_dir must be inside {base}")
    return candidate


def _job_folder_name(image_stem: str, backend_id: str, requested: str | None) -> str:
    """Human-readable job folder name — a user-chosen slug, or ``<image>__<backend>__<time>``
    so results never end up in an unlabeled UUID directory (the exact problem hand-run
    asset sweeps kept hitting on 2026-08-18 — every result had to be moved by hand
    afterward to avoid the next run silently clobbering or burying it)."""
    if requested and requested.strip():
        return _slugify(requested)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _slugify(f"{image_stem}__{backend_id}__{stamp}")


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
                 settings: dict[str, Any], backend_id: str, debug: bool = False):
        self.id = job_id
        self.directory = directory
        self.image_path = image_path
        self.output_path = output_path
        self.manifest_path = output_path.with_suffix(".json")
        self.settings = settings
        self.backend_id = backend_id
        self.debug = debug
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

    def create(self, image_path: Path, settings: dict[str, Any], backend_id: str,
               image_stem: str, output_name: str | None, output_base: Path,
               debug: bool = False) -> Job:
        with self.lock:
            if self.active is not None:
                active = self.jobs.get(self.active)
                if active and active.status in {"queued", "running", "cancelling"}:
                    raise RuntimeError("a generation is already running")
            job_id = uuid.uuid4().hex
            name = _job_folder_name(image_stem, backend_id, output_name)
            suffix = 2
            while (output_base / name).exists():
                name = f"{_job_folder_name(image_stem, backend_id, output_name)}-{suffix}"
                suffix += 1
            directory = output_base / name
            directory.mkdir(parents=True, exist_ok=False)
            # Folder name and primary output filename match (only the extension differs) --
            # <name>/<name>.glb -- so a run is identifiable from either without cross-checking.
            output_path = directory / f"{name}.glb"
            job = Job(job_id, directory, image_path, output_path, settings, backend_id, debug)
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
    elif "decode_latent" in lower and "done in" in lower:
        event = {"phase": "decode", "stage_pct": 100, "overall_pct": _overall_pct("decode", 100),
                 "message": "Decode complete"}
    elif "to_glb + export" in lower and "done in" in lower:
        event = {"phase": "bake", "stage_pct": 100, "overall_pct": _overall_pct("bake", 100),
                 "message": "GLB export complete"}
    elif "pre-cap done" in lower:
        event = {"phase": "bake", "message": "Pre-cap complete, starting to_glb…"}
    if event:
        job.emit(event)


def _signal_hint(return_code: int) -> str:
    """Human-readable suffix for a negative (signal-killed) exit code, empty otherwise."""
    signals = {-9: "SIGKILL", -15: "SIGTERM", -6: "SIGABRT", -4: "SIGILL", -11: "SIGSEGV"}
    if return_code < 0:
        return f" — killed by {signals.get(return_code, f'signal {abs(return_code)}')}"
    return ""


def _process_rss_gb(pid: int) -> float:
    """Resident set size of a child process in GB (macOS ps; best-effort, 0.0 on failure)."""
    try:
        kb = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,  # best-effort probe
        ).stdout.strip()
        return round(int(kb) / (1024 ** 2), 2) if kb.isdigit() else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def _rss_monitor(job: Job) -> None:
    """Append the generator's RSS to the job log every 30s so a silent death leaves a
    memory trajectory behind (an OOM-style kill climbs before it vanishes)."""
    assert job.process is not None
    pid = job.process.pid
    while job.status in {"queued", "running"}:
        job.append_log(f"[rss {_process_rss_gb(pid)} GB]")
        time.sleep(30)


def _read_process(job: Job) -> None:
    assert job.process is not None and job.process.stdout is not None
    spec = BACKENDS[job.backend_id]
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
            spec.parse_line(job, line)
    if buffer:
        job.append_log(buffer)
        spec.parse_line(job, buffer)


def _trellis_parse_line(job: Job, line: str) -> None:
    """TRELLIS's original tqdm/banner handling, unchanged, just wrapped as a BackendSpec hook."""
    parsed = parse_tqdm_line(line)
    if parsed:
        job.emit(_event_from_tqdm(job, parsed))
    else:
        _emit_banner(job, line)


def _cleanup_debug_files(job: Job) -> None:
    """Debug mode off (the default): keep only the primary .glb. Deletes the manifest,
    textures, intermediate meshes, resume caches, and run.log -- everything a run writes
    that exists purely to diagnose a run, not to use the asset."""
    for path in job.directory.iterdir():
        if path == job.output_path:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _run_job(job: Job) -> None:
    spec = BACKENDS[job.backend_id]
    args = [str(spec.interpreter), str(spec.wrapper), *spec.build_args(job)]
    env = _job_env()
    try:
        job.status = "running"
        job.emit({"phase": "load", "overall_pct": 0, "message": f"Starting {spec.label} job"})
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
        threading.Thread(target=_rss_monitor, args=(job,), daemon=True,
                         name=f"rss-{job.id[:8]}").start()
        return_code = job.process.wait()
        reader.join(timeout=5)
        job.append_log(f"generator exited with code {return_code}{_signal_hint(return_code)}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.append_log("generation cancelled")
            job.emit({"phase": "error", "message": "Generation cancelled"})
        elif return_code == 0:
            if spec.finalize is not None and not job.output_path.is_file():
                spec.finalize(job)
            if job.output_path.is_file():
                job.status = "done"
                job.append_log("generation complete")
                _update_baseline(job)
                if not job.debug:
                    _cleanup_debug_files(job)
                event = {
                    "phase": "done", "overall_pct": 100, "message": "Generation complete",
                    "result_url": f"/api/generate/{job.id}/result.glb",
                }
                if job.manifest_path.is_file():
                    event["manifest_url"] = f"/api/generate/{job.id}/manifest.json"
                job.emit(event)
            else:
                tail = "\n".join(job.log_lines)[-8000:]
                job.status = "error"
                job.emit({"phase": "error",
                          "message": "generator exited 0 but produced no output file",
                          "log_tail": tail})
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


# --- TRELLIS backend spec (wraps the pre-existing, unchanged behavior above) -----------


def _trellis_build_args(job: Job) -> list[str]:
    args = [
        str(job.image_path), str(job.output_path),
        "--resolution", job.settings["resolution"],
        "--seed", str(job.settings["seed"]),
        "--decimation-target", str(job.settings["decimation_target"]),
        "--texture-size", str(job.settings["texture_size"]),
    ]
    if job.settings["allow_rembg"]:
        args.append("--allow-rembg")
    if not job.debug:
        # Skip the multi-hundred-MB resume caches entirely rather than write-then-delete.
        args += ["--no-save-latents", "--no-save-decode"]
    return args


# --- SF3D backend spec ------------------------------------------------------------------

SF3D_DEFAULT_SETTINGS: dict[str, Any] = {
    "texture_resolution": 1024,
    "foreground_ratio": 0.85,
    "remesh": "none",
    "target_vertices": -1,
}
SF3D_VALID_REMESH = {"none", "triangle", "quad"}
SF3D_STAGES = ["running"]
SF3D_STAGE_LABELS = {"running": "Running SF3D"}


def _sf3d_validate_settings(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**SF3D_DEFAULT_SETTINGS, **raw}
    try:
        settings["texture_resolution"] = int(settings["texture_resolution"])
        settings["foreground_ratio"] = float(settings["foreground_ratio"])
        settings["target_vertices"] = int(settings["target_vertices"])
    except (TypeError, ValueError) as exc:
        raise ValueError("texture_resolution/target_vertices must be integers, "
                          "foreground_ratio must be a number") from exc
    if settings["remesh"] not in SF3D_VALID_REMESH:
        raise ValueError("remesh must be one of none, triangle, or quad")
    if settings["texture_resolution"] <= 0:
        raise ValueError("texture_resolution must be positive")
    if not (0.0 < settings["foreground_ratio"] <= 1.0):
        raise ValueError("foreground_ratio must be between 0 and 1")
    return settings


def _sf3d_build_args(job: Job) -> list[str]:
    return [
        "--fast",
        "--output-dir", str(job.directory),
        "--texture-resolution", str(job.settings["texture_resolution"]),
        "--foreground-ratio", str(job.settings["foreground_ratio"]),
        "--remesh", job.settings["remesh"],
        "--target-vertices", str(job.settings["target_vertices"]),
        str(job.image_path),
    ]


def _sf3d_finalize(job: Job) -> None:
    """pipeline.py --fast writes ``<stem>_sf3d.glb`` in the output dir; move it into place."""
    produced = job.directory / f"{job.image_path.stem}_sf3d.glb"
    if produced.is_file():
        produced.replace(job.output_path)


def _sf3d_parse_line(job: Job, line: str) -> None:
    """SF3D has no structured per-step progress; relay raw lines, no false pct signal."""
    stripped = line.strip()
    if stripped:
        job.emit({"phase": "running", "message": stripped[:200]})


def _sf3d_readiness() -> dict[str, Any]:
    present = (SF3D_REPO_DEFAULT / "sf3d" / "system.py").is_file()
    return {
        "schema_version": 1,
        "build": {
            "present": present,
            "hint": None if present else (
                f"SF3D checkout not found at {SF3D_REPO_DEFAULT} — "
                "run scripts/bootstrap_macos.sh first."
            ),
        },
        "weights": {},
        "missing_weights": [],
        "ready": present,
        "warning": None,
    }


# --- Hunyuan3D-MLX backend spec ----------------------------------------------------------

HUNYUAN_DEFAULT_SETTINGS: dict[str, Any] = {
    "octree_resolution": 512,
    "seed": 42,
    "decimation_target": 300_000,
    "paint_seed": 0,
    "paint_res": 512,
    "paint_steps": 15,
    "paint_tex": 4096,
}
HUNYUAN_VALID_OCTREE = {256, 384, 512, 1024}
HUNYUAN_STAGES = ["shape", "remesh", "paint_setup", "paint_diffusion", "paint_finish"]
HUNYUAN_STAGE_LABELS = {
    "shape": "Shape generation",
    "remesh": "Remesh",
    "paint_setup": "Paint setup (mesh/UV)",
    "paint_diffusion": "Paint diffusion",
    "paint_finish": "Paint finish (super-res/bake)",
}
HUNYUAN_STEP_RE = re.compile(r"^\s*step (\d+)/(\d+) (\d+)s")


def _hunyuan_validate_settings(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**HUNYUAN_DEFAULT_SETTINGS, **raw}
    try:
        for key in ("octree_resolution", "seed", "decimation_target", "paint_seed",
                    "paint_res", "paint_steps", "paint_tex"):
            settings[key] = int(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError("all Hunyuan settings must be integers") from exc
    if settings["octree_resolution"] not in HUNYUAN_VALID_OCTREE:
        raise ValueError("octree_resolution must be one of 256, 384, 512, or 1024")
    if settings["decimation_target"] <= 0:
        raise ValueError("decimation_target must be positive")
    if settings["decimation_target"] > 600_000:
        raise ValueError(
            "decimation_target above ~500k hits a confirmed xatlas wall (500k-700k faces "
            "took 37 min in testing on 2026-08-18, 1M never finished) — keep it at or "
            "under 500,000"
        )
    return settings


def _hunyuan_build_args(job: Job) -> list[str]:
    s = job.settings
    return [
        str(job.image_path), str(job.output_path),
        "--octree-resolution", str(s["octree_resolution"]),
        "--seed", str(s["seed"]),
        "--decimation-target", str(s["decimation_target"]),
        "--paint-seed", str(s["paint_seed"]),
        "--paint-res", str(s["paint_res"]),
        "--paint-steps", str(s["paint_steps"]),
        "--paint-tex", str(s["paint_tex"]),
    ]


def _hunyuan_overall_pct(phase: str, stage_pct: int) -> int:
    try:
        index = HUNYUAN_STAGES.index(phase)
    except ValueError:
        return 0
    return round((index + max(0, min(stage_pct, 100)) / 100) / len(HUNYUAN_STAGES) * 100)


def _hunyuan_parse_line(job: Job, line: str) -> None:
    stripped = line.strip()
    step_match = HUNYUAN_STEP_RE.match(line)
    if step_match:
        step, total = int(step_match.group(1)), int(step_match.group(2))
        pct = round(step / total * 100)
        job.emit({
            "phase": "paint_diffusion", "step": step, "total": total, "stage_pct": pct,
            "overall_pct": _hunyuan_overall_pct("paint_diffusion", pct),
            "message": f"Paint diffusion — step {step}/{total}",
        })
        return
    if stripped.startswith("shape generated"):
        job.emit({"phase": "shape", "stage_pct": 100,
                  "overall_pct": _hunyuan_overall_pct("shape", 100), "message": stripped})
    elif stripped.startswith(("simplified to", "mesh at/under decimation")):
        job.emit({"phase": "remesh", "stage_pct": 100,
                  "overall_pct": _hunyuan_overall_pct("remesh", 100), "message": stripped})
    elif stripped.startswith(("mesh loaded", "xatlas parametrize done", "mesh render loaded",
                               "control renders done", "controls + dino ready")):
        job.emit({"phase": "paint_setup", "message": stripped})
    elif stripped.startswith(("views decoded", "super-res x4")):
        job.emit({"phase": "paint_finish", "message": stripped})
    elif stripped.startswith(("DONE", "paint stage done")):
        job.emit({"phase": "paint_finish", "stage_pct": 100,
                  "overall_pct": _hunyuan_overall_pct("paint_finish", 100), "message": stripped})


def _hunyuan_readiness() -> dict[str, Any]:
    shape_ok = HUNYUAN_PYTHON.is_file() and HUNYUAN_WRAPPER.is_file()
    paint_venv_ok = HUNYUAN_PAINT_VENV.is_file()
    weights_ok = HUNYUAN_PAINT_WEIGHTS.is_dir()
    ready = shape_ok and paint_venv_ok and weights_ok
    missing = []
    if not shape_ok:
        missing.append("shape venv/wrapper (vendor/hunyuan-mlx, scripts/hunyuan_mlx_generate.py)")
    if not paint_venv_ok:
        missing.append("paint venv (vendor/hunyuan-mlx-paint/python/paint/.venv)")
    if not weights_ok:
        missing.append("paint weights (.../paint/weights/hunyuan3d-paintpbr-v2-1)")
    return {
        "schema_version": 1,
        "build": {
            "present": ready,
            "hint": None if ready else (
                "Hunyuan3D-MLX setup is incomplete — missing: " + "; ".join(missing) + ". "
                "No automated bootstrap exists yet; see docs/STATE-OF-REPO-2026-08-17.md "
                "for the manual setup notes."
            ),
        },
        "weights": {},
        "missing_weights": missing,
        "ready": ready,
        "warning": None,
    }


BACKENDS.update({
    "trellis": BackendSpec(
        id="trellis", label="TRELLIS.2 (clean port)",
        interpreter=PYTHON, wrapper=WRAPPER,
        default_settings=DEFAULT_SETTINGS, stages=STAGES, stage_labels=PHASE_LABELS,
        requires_alpha=True,
        validate_settings=validate_settings, build_args=_trellis_build_args,
        parse_line=_trellis_parse_line, readiness=setup_status,
        baseline_path=BASELINE_PATH,
    ),
    "sf3d": BackendSpec(
        id="sf3d", label="Stable Fast 3D",
        interpreter=Path(sys.executable), wrapper=REPO / "pipeline.py",
        default_settings=SF3D_DEFAULT_SETTINGS, stages=SF3D_STAGES,
        stage_labels=SF3D_STAGE_LABELS, requires_alpha=False,
        validate_settings=_sf3d_validate_settings, build_args=_sf3d_build_args,
        parse_line=_sf3d_parse_line, readiness=_sf3d_readiness, finalize=_sf3d_finalize,
    ),
    "hunyuan-mlx": BackendSpec(
        id="hunyuan-mlx", label="Hunyuan3D-MLX",
        interpreter=HUNYUAN_PYTHON, wrapper=HUNYUAN_WRAPPER,
        default_settings=HUNYUAN_DEFAULT_SETTINGS, stages=HUNYUAN_STAGES,
        stage_labels=HUNYUAN_STAGE_LABELS, requires_alpha=False,
        validate_settings=_hunyuan_validate_settings, build_args=_hunyuan_build_args,
        parse_line=_hunyuan_parse_line, readiness=_hunyuan_readiness,
    ),
})


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
        if parts == ["api", "setup", "run"]:
            self._start_setup()
            return
        if parts == ["api", "generate"]:
            self._create_job()
            return
        if len(parts) == 4 and parts[:2] == ["api", "generate"] and parts[3] == "cancel":
            self._cancel_job(parts[2])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parts = self._path_parts()
        if parts == ["api", "setup"]:
            query = parse_qs(urlparse(self.path).query)
            backend_id = query.get("backend", ["trellis"])[0]
            spec = BACKENDS.get(backend_id)
            if spec is None:
                self._send_json(422, {"error": f"unknown backend {backend_id!r}"})
                return
            self._send_json(200, {**spec.readiness(), "backend": spec.id})
            return
        if parts == ["api", "backends"]:
            self._send_json(200, {
                "backends": [
                    {"id": spec.id, "label": spec.label, "requires_alpha": spec.requires_alpha,
                     "default_settings": spec.default_settings, "stages": spec.stages,
                     "stage_labels": spec.stage_labels}
                    for spec in BACKENDS.values()
                ]
            })
            return
        if len(parts) == 4 and parts[:3] == ["api", "setup", "run"] and parts[3] == "events":
            run = SETUP_RUNS.get(parts[2]) if _safe_id(parts[2]) else None
            if run is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._stream_events(run)
            return
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
                raw_settings = json.loads(settings_value)
            except json.JSONDecodeError as exc:
                self._send_json(422, {"error": f"invalid settings JSON: {exc}"})
                return
            if not isinstance(raw_settings, dict):
                self._send_json(422, {"error": "settings must be a JSON object"})
                return
            backend_id = raw_settings.pop("backend", "trellis")
            output_name = raw_settings.pop("output_name", None)
            debug = bool(raw_settings.pop("debug", False))
            try:
                output_base = _resolve_output_base(raw_settings.pop("output_dir", None))
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            spec = BACKENDS.get(backend_id)
            if spec is None:
                self._send_json(422, {"error": f"unknown backend {backend_id!r}"})
                return
            if not spec.readiness()["ready"]:
                self._send_json(503, {"error": f"{spec.label} is not installed/ready"})
                return
            if SETUP_ACTIVE is not None:
                self._send_json(409, {"error": "setup is running; wait for it to finish"})
                return
            try:
                settings = spec.validate_settings(raw_settings)
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            suffix = Path(str(image_field["filename"])).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self._send_json(422, {"error": "unsupported image type; use PNG, JPG, WebP, or BMP"})
                return
            image_stem = _slugify(Path(str(image_field["filename"])).stem)
            # Reserve a provisional directory only after validation, then persist the upload.
            provisional = OUTPUT_ROOT / uuid.uuid4().hex
            provisional.mkdir(parents=True, exist_ok=False)
            image_path = provisional / f"input{suffix}"
            with image_path.open("wb") as handle:
                handle.write(image_field["data"])
            if (spec.requires_alpha and not image_has_transparent_alpha(image_path)
                    and not settings.get("allow_rembg")):
                for child in provisional.iterdir():
                    child.unlink()
                provisional.rmdir()
                self._send_json(422, {
                    "error": "This image has no transparent alpha foreground. Enable 'allow rembg' "
                             "to use BRIA background removal, or upload a pre-masked PNG."
                })
                return
            # JobManager builds the real, human-readable job directory. Move the upload into it
            # so the id and artifact URLs are stable, without ever accepting a client-provided path.
            provisional_image = image_path
            try:
                job = JOBS.create(provisional_image, settings, backend_id, image_stem,
                                  output_name, output_base, debug)
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
            self._send_json(202, {"job_id": job.id, "events_url": f"/api/generate/{job.id}/events",
                                  "output_dir": str(job.directory.relative_to(REPO))})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _find_job(self, job_id: str) -> Job | None:
        return JOBS.get(job_id) if _safe_id(job_id) else None

    def _events(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._stream_events(job)

    def _stream_events(self, run: Job | SetupRun) -> None:
        """SSE pump shared by generation jobs and setup runs."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        index = 0
        try:
            while True:
                with run.condition:
                    if index >= len(run.events):
                        run.condition.wait(timeout=15)
                    pending = run.events[index:]
                    index = len(run.events)
                    terminal = run.status in {"done", "error", "cancelled"} and not pending
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

    def _start_setup(self) -> None:
        global SETUP_ACTIVE
        with SETUP_LOCK:
            active = JOBS.get(JOBS.active)
            if active is not None and active.status in {"queued", "running", "cancelling"}:
                self._send_json(409, {"error": "a generation is running; wait for it to finish"})
                return
            if SETUP_ACTIVE is not None:
                self._send_json(409, {"error": "setup is already running"})
                return
            if clean_port_build_present():
                self._send_json(409, {"error": "the clean-port build is already installed — nothing to set up"})
                return
            ok, reason = setup_available()
            if not ok:
                self._send_json(503, {"error": reason})
                return
            setup_id = uuid.uuid4().hex
            SETUP_ACTIVE = setup_id
            SETUP_RUNS[setup_id] = _start_setup_run(setup_id)
        self._send_json(202, {
            "setup_run_id": setup_id,
            "events_url": f"/api/setup/run/{setup_id}/events",
        })

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
