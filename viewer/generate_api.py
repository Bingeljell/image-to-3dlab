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
import subprocess
import sys
import tempfile
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

# Sibling import must also work when tests load this file directly via importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_api
from image_to_3dlab import processes
from image_to_3dlab.blender import find_blender, missing_help as blender_missing_help
from image_to_3dlab.host import executable  # image_api put the repo on sys.path
from rig_api import (
    ARTIFACTS as RIG_ARTIFACTS,
    RIG_JOBS,
    cancel_job as cancel_rig_job,
    run_job as run_rig_job,
    status_payload as rig_status_payload,
)
import backend_catalog  # noqa: E402
import hf_api  # noqa: E402
from backend_catalog import (
    APPLE,
    NVIDIA,
    _dir_state,
    catalog_status,
    host_platform,
    readiness as catalog_readiness,
)
from welcome_api import payload as welcome_payload
from update_api import check as update_check
from download_api import (
    DOWNLOADS,
    active as download_active,
    running_payload,
    cancel as cancel_download,
    remove as remove_weights,
    rebuild_reason,
    start as start_download,
    status_payload as download_status_payload,
)
from finish_api import (
    ARTIFACTS as FINISH_ARTIFACTS,
    FINISH_JOBS,
    cancel_job as cancel_finish_job,
    capabilities as finish_capabilities,
    list_runs as list_finish_runs,
    run_job as run_finish_job,
    status_payload as finish_status_payload,
)
from autorig_api import (
    ANIMATE_JOBS,
    cancel_job as cancel_animate_job,
    installed as autorig_installed,
    pickable_models as animate_models,
    presets as animate_presets,
    resolve_model as resolve_animate_model,
    run_job as run_animate_job,
    source_record_for as animate_source_record,
    status_payload as animate_status_payload,
)
from assets_api import assets_payload
from props_api import (
    PROPS_JOBS,
    cancel_job as cancel_props_job,
    generated_model,
    generated_models,
    list_runs as list_props_runs,
    run_job as run_props_job,
    source_record_for,
    status_payload as props_status_payload,
    gltfpack_install_command,
    tools_payload as props_tools_payload,
)

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "trellis_space_generate.py"
PYTHON = REPO / "vendor" / "trellis-space-mac" / ".venv" / "bin" / "python"
TRELLIS_VENDOR = REPO / "vendor" / "trellis-space-mac"
# The dispatch branch scripts/patch_trellis_mlx_attention.py injects. Its presence is how
# we know the vendored checkout can actually serve SPARSE_ATTN_BACKEND=mlx.
MLX_DISPATCH_FILE = TRELLIS_VENDOR / "TRELLIS.2" / "trellis2" / "modules" / "sparse" / "attention" / "full_attn.py"
MLX_DISPATCH_MARKER = "config.ATTN == 'mlx'"
# TRELLIS.2 on Linux + NVIDIA: Microsoft's own checkout, built by
# scripts/bootstrap_trellis_cuda.py. The Mac constants above are untouched.
TRELLIS_CUDA_VENDOR = REPO / "vendor" / "trellis-cuda"
TRELLIS_CUDA_WRAPPER = REPO / "scripts" / "trellis_cuda_generate.py"
TRELLIS_CUDA_PYTHON = TRELLIS_CUDA_VENDOR / ".venv" / "bin" / "python"
TRELLIS_CUDA_MARKER = TRELLIS_CUDA_VENDOR / ".i2l-build-complete"
# Hunyuan3D-2.1 on Linux + NVIDIA: Tencent's own checkout, built by
# scripts/bootstrap_hunyuan_cuda.py. NVIDIA only; the Mac keeps the MLX routes.
HUNYUAN_CUDA_VENDOR = REPO / "vendor" / "hunyuan-cuda"
HUNYUAN_CUDA_WRAPPER = REPO / "scripts" / "hunyuan_cuda_generate.py"
HUNYUAN_CUDA_PYTHON = HUNYUAN_CUDA_VENDOR / ".venv" / "bin" / "python"
HUNYUAN_CUDA_MARKER = HUNYUAN_CUDA_VENDOR / ".i2l-build-complete"
OUTPUT_ROOT = REPO / "output"
BASELINE_PATH = REPO / "viewer" / "generate_baseline.json"  # shipped default, never written
# Timings learned on this machine. Under output/ (git-ignored): writing the tracked file
# above left every install with a local edit, and the installer then refused to update.
LEARNED_BASELINE_PATH = REPO / "output" / ".generate_baseline.json"
TINYCLIP_ADVISOR = REPO / "scripts" / "classify_trellis_input.py"
TINYCLIP_TIMEOUT_SECONDS = 300

# dgrauet's shape stage stays vendored (Tencent-licensed code, not just weights — see
# docs/info_and_credits.md). Its shape quality is genuinely the best we've tested
# (2026-08-19 A/B against Xiong 2.0: no dents/dimples, 10/10), which is why the hybrid
# stays around despite not being part of the clone-and-go simplification below.
HUNYUAN_WRAPPER = REPO / "scripts" / "hunyuan_mlx_generate.py"
HUNYUAN_PYTHON = REPO / "vendor" / "hunyuan-mlx" / ".venv" / "bin" / "python"
PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
PIXAL3D_CLI = executable(PIXAL3D_ROOT / "build", "trellis-cli")
PIXAL3D_MODELS = PIXAL3D_ROOT / "models" / "pixal3d-sv"
PIXAL3D_WRAPPER = REPO / "scripts" / "pixal3d_generate.py"

# Xiong's paint+shape code is MIT and tracked in-repo at hunyuan_mlx/ (moved out of
# vendor/hunyuan-mlx-paint 2026-08-19) — a fresh clone of this repo alone has the code;
# only weights/ (git-ignored, downloaded) and .venv/ (uv sync) are still local-only.
HUNYUAN_PAINT_VENV = REPO / "hunyuan_mlx" / "paint" / ".venv" / "bin" / "python"
HUNYUAN_PAINT_WEIGHTS = REPO / "hunyuan_mlx" / "paint" / "weights" / "hunyuan3d-paintpbr-v2-1"

# Single-repo variant: ZimengXiong's own shape stage (not dgrauet's) chained into their own
# paint stage above — same paint venv/weights, different shape venv/weights entirely.
# Model choice benchmarked 2026-08-19 — see docs/hunyuan-mlx-recipes.md; 2.0 is the
# recommended default (Xiong's own pick too), 2.1 is not.
HUNYUAN_XIONG_WRAPPER = REPO / "scripts" / "hunyuan_mlx_xiong_generate.py"
HUNYUAN_XIONG_SHAPE_VENV = REPO / "hunyuan_mlx" / "shape" / ".venv" / "bin" / "python"
HUNYUAN_XIONG_SHAPE_ROOT = REPO / "hunyuan_mlx" / "shape" / "weights"
HUNYUAN_XIONG_SHAPE_MODELS = {
    "2.1": HUNYUAN_XIONG_SHAPE_ROOT / "Hunyuan3D-2.1" / "hunyuan3d-dit-v2-1",
    "2.0": HUNYUAN_XIONG_SHAPE_ROOT / "Hunyuan3D-2" / "hunyuan3d-dit-v2-0",
    "2.0-turbo": HUNYUAN_XIONG_SHAPE_ROOT / "Hunyuan3D-2" / "hunyuan3d-dit-v2-0-turbo",
}

SF3D_REPO_DEFAULT = REPO / "vendor" / "stable-fast-3d"

# Backend-selection vars the generator must own. A stale value inherited from the server's
# environment silently selects the wrong backend (the conv_none crash); the subprocess env
# drops them and the generator's configure_environment/require_flex_gemm pin them fresh.
BACKEND_ENV_KEYS = (
    "SPARSE_CONV_BACKEND",
    "ATTN_BACKEND",
    "SPARSE_ATTN_BACKEND",
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    # Owned per job: the attention precision is part of the chosen backend, so a value
    # exported into the server's shell must not silently change what a run computes.
    "I2L_MLX_ATTN_DTYPE",
)

HF_HUB_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
WEIGHT_REPOS = (
    ("models--microsoft--TRELLIS.2-4B", "TRELLIS.2-4B weights"),
    ("models--facebook--dinov3-vitl16-pretrain-lvd1689m", "DINOv3 image encoder"),
    (
        "models--wkcn--TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "TinyCLIP input advisor (~94 MB, advisory only)",
    ),
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


def attention_backend_spec(choice: str) -> tuple[str, dict[str, str]]:
    """Split a UI attention choice into the wrapper's flag and any env it needs.

    Returns (value for --sparse-attn-backend, environment overrides). Pure, so the mapping
    can be tested without launching anything.
    """
    if choice == "mlx-fp16":
        return "mlx", {"I2L_MLX_ATTN_DTYPE": "fp16"}
    if choice == "mlx":
        return "mlx", {"I2L_MLX_ATTN_DTYPE": "fp32"}
    return choice, {}


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
            size = _dir_state(path)[1]
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
    hidden_fields: tuple[str, ...] = ()
    """Element ids of Generate-tab controls this spec ignores, so the page hides them
    (the Mac-only attention and rembg options mean nothing on the NVIDIA route)."""


BACKENDS: dict[str, BackendSpec] = {}


def clean_port_build_present() -> bool:
    """Whether the clean-port build (interpreter + wrapper) is installed."""
    return PYTHON.is_file() and WRAPPER.is_file()


def mlx_attention_status(vendor: Path | None = None, dispatch: Path | None = None) -> dict[str, Any]:
    """Whether the mlx attention backend can actually run on this machine.

    Two independent prerequisites, and the UI needs to distinguish them because the
    remedies differ: the vendored checkout must carry the dispatch branch, and mlx must be
    installed in that checkout's venv. Offering the backend without both produces a crash
    partway into a run that has already cost real time.
    """
    vendor = vendor or TRELLIS_VENDOR
    dispatch = dispatch or MLX_DISPATCH_FILE
    try:
        patched = dispatch.is_file() and MLX_DISPATCH_MARKER in dispatch.read_text()
    except OSError:
        patched = False
    package = any((vendor / ".venv" / "lib").glob("python*/site-packages/mlx"))

    hints = []
    if not package:
        hints.append(
            "install mlx into the backend venv: uv pip install --python "
            f"{vendor / '.venv' / 'bin' / 'python'} mlx"
        )
    if not patched:
        hints.append("apply the dispatch branch: python scripts/patch_trellis_mlx_attention.py")
    return {
        "patched": patched,
        "package": package,
        "ready": patched and package,
        "hint": "; ".join(hints) or None,
    }


def backends_payload() -> dict:
    """The Generate tab's routes. `runs_here` lets the page drop a route this machine cannot
    run: an NVIDIA pod was offered the two Mac-only Hunyuan-MLX routes."""
    def runs_here(spec_id: str) -> bool:
        entry = backend_catalog.resolve(spec_id)
        return True if entry is None else entry.runs_here(backend_catalog.host_platform())

    return {"backends": [
        {"id": spec.id, "label": spec.label, "requires_alpha": spec.requires_alpha,
         "default_settings": spec.default_settings, "stages": spec.stages,
         "stage_labels": spec.stage_labels, "hidden_fields": list(spec.hidden_fields),
         "runs_here": runs_here(spec.id)}
        for spec in BACKENDS.values()
    ]}


def hf_sign_in_response(payload: Any) -> tuple[int, dict]:
    """POST /api/hf/sign-in: check and save a Hugging Face token, answer with the status.
    The token never comes back in the answer."""
    if not isinstance(payload, dict):
        return 400, {"error": "expected {\"token\": ...}"}
    result = hf_api.sign_in(str(payload.get("token", "")))
    return (422 if "error" in result else 200), result


def catalog_payload() -> dict:
    """The Setup page's catalogue, plus any setup already running so it can reattach."""
    payload = with_rebuild_reasons(catalog_status())
    payload["running_setup"] = running_payload()
    return payload


def with_rebuild_reasons(catalog: dict[str, Any],
                         reason: Callable[[str], str | None] = rebuild_reason) -> dict[str, Any]:
    """Say which installed backends want recompiling, so the page can offer a Rebuild
    button instead of a Terminal command."""
    for backend in catalog.get("backends", []):
        backend["rebuild_reason"] = reason(backend["id"]) if backend.get("supported_here") else None
    return catalog


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
                "(requires uv, Python 3.11 and Xcode command-line tools)"
                if not build_present else None
            ),
        },
        "weights": weights,
        "missing_weights": missing,
        # Advisory only: the default sdpa path works without it, so an unready mlx backend
        # must never make the machine look unready for generation.
        "mlx_attention": mlx_attention_status(),
        "ready": build_present,
        "warning": "first use will download missing weights" if missing else None,
    }


def trellis_cuda_bria_patched(checkout: Path | None = None) -> bool:
    """Whether the CUDA checkout has the BRIA guardrail. Stdlib only, like this module."""
    sys.path.insert(0, str(REPO / "scripts"))
    import patch_trellis_cuda_no_bria

    return patch_trellis_cuda_no_bria.is_patched(checkout or TRELLIS_CUDA_VENDOR / "TRELLIS.2")


def cuda_setup_status() -> dict[str, Any]:
    """Readiness of the NVIDIA TRELLIS.2 route, for the Generate tab's readiness strip.

    Not ready without the BRIA patch, even when everything else is built: the generator
    would refuse anyway, and the button should not promise a run that cannot start.
    """
    built = (TRELLIS_CUDA_MARKER.is_file() and TRELLIS_CUDA_PYTHON.is_file()
             and TRELLIS_CUDA_WRAPPER.is_file())
    patched = trellis_cuda_bria_patched() if built else False
    weights = weights_on_disk()
    missing = [w["label"] for w in weights.values() if not w["present"]]
    hint = None
    if not built:
        hint = ("TRELLIS.2 for NVIDIA is not installed. Set it up from Setup & Status, or run "
                "python scripts/bootstrap_trellis_cuda.py (Linux only, ~15 GB of weights).")
    elif not patched:
        hint = ("The TRELLIS.2 checkout still loads BRIA RMBG-2.0. Run "
                "python scripts/patch_trellis_cuda_no_bria.py first.")
    return {
        "schema_version": 1,
        "build": {"present": built and patched, "interpreter": str(TRELLIS_CUDA_PYTHON),
                  "wrapper": str(TRELLIS_CUDA_WRAPPER), "hint": hint},
        "weights": weights,
        "missing_weights": missing,
        "bria_patched": patched,
        "ready": built and patched,
        "warning": "first use will download missing weights" if missing else None,
    }


def run_trellis_input_advisor(image_path: Path) -> dict[str, Any]:
    """Run TinyCLIP out-of-process so the lightweight viewer never imports torch."""
    python = trellis_spec().interpreter
    if not python.is_file():
        raise RuntimeError("TRELLIS environment is not installed")
    if not TINYCLIP_ADVISOR.is_file():
        raise RuntimeError(f"TinyCLIP advisor is missing: {TINYCLIP_ADVISOR}")
    try:
        result = subprocess.run(
            [str(python), str(TINYCLIP_ADVISOR), str(image_path)],
            cwd=REPO,
            env=_job_env(),
            check=True,
            capture_output=True,
            text=True,
            timeout=TINYCLIP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("TinyCLIP input check timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"TinyCLIP input check failed{suffix}") from exc
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TinyCLIP input check returned invalid JSON") from exc
    if payload.get("verdict") not in {"likely_flat", "likely_dimensional", "uncertain"}:
        raise RuntimeError("TinyCLIP input check returned an invalid verdict")
    if not isinstance(payload.get("flat_risk"), (int, float)):
        raise RuntimeError("TinyCLIP input check returned an invalid score")
    return payload


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


def setup_available(host: str | None = None) -> tuple[bool, str | None]:
    """Whether the setup runner can start: uv on PATH and the bootstrap script present.

    This older runner only knows the Mac bootstrap; anywhere else, Setup & Status runs the
    right installer for the machine.
    """
    if (host or host_platform()) != APPLE:
        return False, ("this setup runs the Mac port; on this machine use Setup & Status "
                       "> TRELLIS.2 > Set up")
    if shutil.which("uv") is None:
        return False, "uv is not installed — install it first (https://docs.astral.sh/uv/)"
    bootstrap = REPO / "scripts" / "bootstrap_trellis_space_macos.py"
    if not bootstrap.is_file():
        return False, f"bootstrap script missing: {bootstrap}"
    return True, None


def blender_install_command() -> list[str]:
    return [sys.executable, str(REPO / "scripts" / "bootstrap_blender.py"), "--yes"]


def blender_install_refusal(caps: dict[str, Any], generating: bool,
                            setting_up: bool) -> tuple[int, str] | None:
    """Why Setup's Install Blender cannot start now, or None when it can."""
    if generating:
        return 409, "a generation is running; wait for it to finish"
    if setting_up:
        return 409, "another setup is running; wait for it to finish"
    if not caps.get("blender_installable"):
        return 409, ("Blender is already found, or this machine needs the app from "
                     "https://www.blender.org/download/")
    return None


def gltfpack_install_refusal(tools: dict[str, Any], generating: bool,
                             setting_up: bool) -> tuple[int, str] | None:
    """Why Setup's Install gltfpack cannot start now, or None when it can."""
    if generating:
        return 409, "a generation is running; wait for it to finish"
    if setting_up:
        return 409, "another setup is running; wait for it to finish"
    if not tools.get("gltfpack_installable"):
        return 409, ("gltfpack is already found, or there is no build for this machine at "
                     "https://github.com/zeux/meshoptimizer/releases")
    return None


def _start_setup_run(run_id: str, command: list[str] | None = None) -> SetupRun:
    """Spawn a setup script and stream its output into the run's events (SSE)."""
    run = SetupRun(run_id)
    bootstrap = REPO / "scripts" / "bootstrap_trellis_space_macos.py"
    proc = subprocess.Popen(
        command or [sys.executable, str(bootstrap)],
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
    "sparse_attn_backend": "sdpa",
}
VALID_RESOLUTIONS = {"512", "1024", "1536"}
# sdpa is the default because mlx needs scripts/patch_trellis_mlx_attention.py applied to
# the vendored checkout and mlx installed in its venv; a fresh clone has neither, and a
# default that fails on a clean machine is worse than a default that is merely slower.
# One control, not two: precision is part of the choice a user makes, and splitting it
# into a second setting invites the combination nobody wants (sdpa with fp16, which is
# meaningless -- fp16 is a property of the fused MLX kernel, and on torch MPS SDPA it is
# measurably slower than fp32).
VALID_SPARSE_ATTN = {"sdpa", "mlx", "mlx-fp16"}
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
    if settings["sparse_attn_backend"] not in VALID_SPARSE_ATTN:
        raise ValueError("sparse_attn_backend must be one of sdpa, mlx, or mlx-fp16")
    settings["resolution"] = str(settings["resolution"])
    return settings


# Mirrors the wrapper's BORDER_OPAQUE_LIMIT; see image_border_opaque_fraction.
UNCUT_BORDER_LIMIT = 0.05


def image_has_transparent_alpha(path: Path) -> bool:
    """Return whether an image has an actual (not merely opaque) alpha channel.

    Raises RuntimeError if Pillow itself isn't importable in this interpreter -- that's a
    broken server environment, not a fact about the image, and must not be reported as one.
    (2026-08-20: a bare `except Exception` here caught a missing Pillow install the same as
    a genuinely opaque image, so every upload silently failed the alpha check on a
    Pillow-less interpreter and users were told to enable rembg for images that already had
    real transparency.)"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed in the interpreter running viewer/serve.py, so the "
            "alpha-transparency check cannot run. Install it (`pip install Pillow`, or "
            "`pip install -r requirements.txt`) into that same interpreter and restart "
            "the server."
        ) from exc
    try:
        with Image.open(path) as image:
            if image.mode != "RGBA":
                return False
            return image.getextrema()[3][0] < 255
    except Exception:
        # A missing/undecodable alpha is deliberately conservative: the wrapper will refuse it
        # unless the user explicitly opts into BRIA rembg.
        return False


def image_border_opaque_fraction(path: Path) -> float | None:
    """Fraction of an image's outer border that is still opaque, or None if unmeasurable.

    Delegates to the TRELLIS wrapper's own helper rather than restating the rule here: the
    2026-09-03 slab bug happened because "has alpha" was defined in two places, and the UI's
    copy and the wrapper's copy were each separately wrong about what a cut-out image is.
    One definition, imported.

    Returns None -- rather than raising or guessing -- when the check simply cannot run in
    this interpreter (no numpy). The wrapper still enforces the same rule at run start, so a
    missing preflight costs a late refusal, never a silent bad run.
    """
    import importlib.util

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        script = REPO / "scripts" / "trellis_space_generate.py"
        spec = importlib.util.spec_from_file_location("trellis_space_generate", script)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        with Image.open(path) as image:
            if image.mode != "RGBA":
                return None
            return wrapper.border_opaque_fraction(np.array(image)[..., 3])
    except Exception:
        return None


def uncut_image_error(border_fraction: float) -> str:
    """Browser-facing text for an image whose alpha never cut the subject out."""
    return (
        f"This image has an alpha channel, but {border_fraction:.0%} of its outer border is "
        "still opaque, so the subject was never cut out of its background. Generating from it "
        "would rebuild the background as 3D geometry, "
        "ending in a slab behind the subject. Re-export it with a transparent background."
    )


def _read_seconds(path: Path) -> dict[str, float]:
    try:
        data = json.loads(path.read_text())
        seconds = data.get("seconds", data.get("stages", {}))
        return {str(k): float(v) for k, v in seconds.items() if v is not None}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _baseline() -> dict[str, float]:
    """Stage durations for the ETA: the shipped defaults, overlaid by this machine's own."""
    return {**_read_seconds(BASELINE_PATH), **_read_seconds(LEARNED_BASELINE_PATH)}


def _update_baseline(job: Job) -> None:
    """Learn real stage durations after a successful job for the next ETA estimate."""
    if not job.stage_durations:
        return
    seconds = _read_seconds(LEARNED_BASELINE_PATH)
    seconds.update({name: round(value, 1) for name, value in job.stage_durations.items()})
    data = {"schema_version": 1, "seconds": seconds,
            "source": "learned from completed Generate jobs on this machine"}
    LEARNED_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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
    """Debug mode off (the default): keep the .glb and what travels with it. Deletes
    textures, intermediate meshes, resume caches and run.log -- everything a run writes
    that exists purely to diagnose a run, not to use the asset."""
    # The licence record travels with the file (AGENTS.md), so it is never "debug". For
    # Pixal3D that record is <name>.json, and <name>.svviews/ is the camera Finish's Pixel
    # Match needs: deleting them shipped Pixal3D models with no provenance at all.
    out = job.output_path
    keep = {out, out.with_suffix(".provenance.json"), out.with_suffix(".json"),
            out.with_suffix(".svviews")}
    for path in job.directory.iterdir():
        if path in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _pid_file_path(directory: Path) -> Path:
    return directory / "pid"


def _write_pid_file(directory: Path, pid: int) -> None:
    """Record the generator subprocess's pid on disk the moment it starts.

    This is the only trace of a running job that survives the server process itself
    dying (memory-only Job/JobManager state does not) -- see _reconcile_orphaned_jobs,
    which reads this file back on the next server startup to find and clean up jobs
    whose tracker died mid-run."""
    _pid_file_path(directory).write_text(processes.pid_record(pid))


def _remove_pid_file(directory: Path) -> None:
    _pid_file_path(directory).unlink(missing_ok=True)


def _killpg_if_alive(pid: int) -> None:
    """Best-effort stop of a process group; a pid that's already gone is not an error."""
    processes.terminate_group(pid)


def _process_group_alive(pid: int) -> bool:
    return processes.group_alive(pid)


def _process_alive(pid: int) -> bool:
    return processes.process_alive(pid)


def _terminate_active_job() -> None:
    """Kill the active job's process group, if one is running. Called on a clean server
    shutdown (SIGTERM/SIGINT) so a deliberate stop doesn't leave the same kind of orphan
    _reconcile_orphaned_jobs cleans up after a crash -- this is the graceful-exit half of
    that same problem: a signal caught here means the pid file gets left behind (the job's
    own finally block, in a background thread, may not get to run before the process
    exits), but the next startup's reconciliation finds it, sees the process is already
    dead, and annotates it correctly."""
    job = JOBS.jobs.get(JOBS.active) if JOBS.active else None
    if job is not None and job.process is not None and job.process.poll() is None:
        _killpg_if_alive(job.process.pid)
    rig_job = RIG_JOBS.get(RIG_JOBS.active) if RIG_JOBS.active else None
    if rig_job is not None and rig_job.process is not None and rig_job.process.poll() is None:
        _killpg_if_alive(rig_job.process.pid)
    props_job = PROPS_JOBS.get(PROPS_JOBS.active) if PROPS_JOBS.active else None
    if props_job is not None and props_job.process is not None and props_job.process.poll() is None:
        _killpg_if_alive(props_job.process.pid)
    finish_job = FINISH_JOBS.get(FINISH_JOBS.active) if FINISH_JOBS.active else None
    if finish_job is not None and finish_job.process is not None and finish_job.process.poll() is None:
        _killpg_if_alive(finish_job.process.pid)


def _animate_running() -> bool:
    """Whether the Animate tab is rigging (SkinTokens holds ~8 GB) or retargeting."""
    return ANIMATE_JOBS.busy()


def _props_baking() -> bool:
    """Whether a prop-sheet job is running. Its Blender bakes share unified memory with a
    generation or a finishing repaint, so neither of those starts while it runs."""
    job = PROPS_JOBS.get(PROPS_JOBS.active) if PROPS_JOBS.active else None
    return job is not None and job.status not in {"done", "error", "cancelled"}


def _reconcile_orphaned_jobs(output_root: Path) -> list[str]:
    """Resolve pid files left behind by a previous server life -- this server process died
    mid-run (crash, closed terminal, etc.) before it could record a terminal outcome, same
    failure shape as the leather_satchel/jesus_chibi incidents on 2026-08-20. Call once at
    server startup, before serving.

    For each leftover <job-dir>/pid: if the process that started the job is still alive
    (another viewer, or a script driving the Finish code), the job is not an orphan and is
    left alone. Otherwise, if that process group is still running, it's an actual ghost
    (nobody has been tracking it since the old server died) -- kill it. Either way,
    annotate that job's run.log so it stops trailing off silently, and remove the pid file
    since this server now owns (or has just closed out) that job's fate.

    Returns the touched job-folder names, for a one-line startup banner."""
    touched = []
    for pid_file in sorted(output_root.rglob("pid")):
        if not pid_file.is_file():
            continue    # a folder that happens to be called pid (a prop, say), not a job's
        directory = pid_file.parent
        try:
            pid, owner = processes.parse_pid_record(pid_file.read_text())
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
            continue
        if owner is not None and owner != os.getpid() and _process_alive(owner):
            continue
        alive = _process_group_alive(pid)
        if alive:
            _killpg_if_alive(pid)
            note = "orphaned generation from a previous server session, terminated on restart"
        else:
            note = "server died mid-run (previous session) -- job status unknown, treat as failed"
        # Finish runs from 0.3.5 keep their log in steps/.
        log_path = next((path for path in (directory / "run.log", directory / "steps" / "run.log")
                         if path.is_file()), None)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(note + "\n")
        pid_file.unlink(missing_ok=True)
        touched.append(directory.name)
    return touched


# Progress chatter the generator emits constantly; never the reason it stopped.
_LOG_NOISE = re.compile(r"^(\[rss |Sampling |Loading |Pipeline loaded|\s*$)|\|\s*\d+/\d+ \[")


def failure_reason(log_lines, limit: int = 6) -> str | None:
    """The generator's own last words, or None if it said nothing but progress.

    A failed run used to surface as "generator exited with code 1" while the actual
    explanation -- an alpha refusal, a missing weight, a traceback -- sat unread in the log
    tail. Walk back from the end, skip the progress chatter, and return the trailing block
    of real output so the browser can show what the process actually said.
    """
    reason: list[str] = []
    for line in reversed(list(log_lines)):
        text = line.rstrip()
        if not text or _LOG_NOISE.match(text) or text.startswith("generator exited"):
            if reason:
                break
            continue
        reason.append(text)
        if len(reason) >= limit:
            break
    return "\n".join(reversed(reason)) or None


def _run_job(job: Job) -> None:
    spec = BACKENDS[job.backend_id]
    args = [str(spec.interpreter), str(spec.wrapper), *spec.build_args(job)]
    env = _job_env()
    if job.backend_id == "trellis" and spec.wrapper == WRAPPER:
        env.update(attention_backend_spec(job.settings["sparse_attn_backend"])[1])
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
            **processes.group_popen_kwargs(),
        )
        _write_pid_file(job.directory, job.process.pid)
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
            reason = failure_reason(job.log_lines)
            job.emit({"phase": "error",
                      "message": reason or f"generator exited with code {return_code}",
                      "exit_code": return_code,
                      "log_tail": tail})
    except Exception as exc:  # process launch errors must reach the browser, not kill the server
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        _remove_pid_file(job.directory)
        JOBS.finish(job)


# --- TRELLIS backend spec (wraps the pre-existing, unchanged behavior above) -----------


def _trellis_build_args(job: Job) -> list[str]:
    args = [
        str(job.image_path), str(job.output_path),
        "--resolution", job.settings["resolution"],
        "--seed", str(job.settings["seed"]),
        "--decimation-target", str(job.settings["decimation_target"]),
        "--texture-size", str(job.settings["texture_size"]),
        "--sparse-attn-backend", attention_backend_spec(job.settings["sparse_attn_backend"])[0],
    ]
    if job.settings["allow_rembg"]:
        args.append("--allow-rembg")
    if not job.debug:
        # Skip the multi-hundred-MB resume caches entirely rather than write-then-delete.
        args += ["--no-save-latents", "--no-save-decode"]
    return args


def _trellis_cuda_build_args(job: Job) -> list[str]:
    """The CUDA generator takes the same settings, minus the Mac-only ones: attention is
    flash-attn, and the cut-out is always our own remover, never BRIA."""
    args = [
        str(job.image_path), str(job.output_path),
        "--resolution", job.settings["resolution"],
        "--seed", str(job.settings["seed"]),
        "--decimation-target", str(job.settings["decimation_target"]),
        "--texture-size", str(job.settings["texture_size"]),
    ]
    if not job.debug:
        args.append("--no-save-latents")
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
    """pipeline.py's provenance system (image_to_3dlab.provenance.finalize_output) sorts
    the real output into a license-class subfolder with a randomized run id --
    <job-dir>/<license-folder>/<image-stem>__sf3d__<classification>__<run-id>.glb -- not
    the flat <stem>_sf3d.glb this used to assume. That wrong assumption meant every SF3D
    run through the web UI reported "generator exited 0 but produced no output file" even
    on a full, successful generation (confirmed 2026-08-20 with a real run: a genuine .glb
    existed on disk the whole time, the web UI just never found it -- the bug the user
    reported as "the web view never updated" was this, not a dropped connection).

    The exact filename can't be predicted (random run id), so glob for it instead."""
    matches = sorted(job.directory.glob(f"*/{job.image_path.stem}__sf3d__*.glb"))
    if not matches:
        return
    produced = matches[-1]
    produced.replace(job.output_path)
    sidecar = produced.with_suffix(".provenance.json")
    if sidecar.is_file():
        sidecar.replace(job.manifest_path)


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
# The shape stage's own denoise loop prints `\r[denoise] i/n` (no trailing text) per step,
# distinct from the final `[denoise] N steps (...) in Xs` summary line the regex below
# doesn't match (it has non-digit trailing content).
HUNYUAN_SHAPE_DENOISE_RE = re.compile(r"^\[denoise\] (\d+)/(\d+)$")


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
            "crawled in testing, 1M never finished) — keep it at or "
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
    shape_denoise_match = HUNYUAN_SHAPE_DENOISE_RE.match(stripped)
    if shape_denoise_match:
        step, total = int(shape_denoise_match.group(1)), int(shape_denoise_match.group(2))
        # Denoise is roughly the first 60% of the shape stage's own progress; VAE grid
        # decode + mesh extraction take the rest. Approximate — the exact split varies by
        # model/quantize/octree-decode, but "roughly right and moving" beats frozen at 0%.
        pct = round(step / total * 60)
        job.emit({
            "phase": "shape", "step": step, "total": total, "stage_pct": pct,
            "overall_pct": _hunyuan_overall_pct("shape", pct),
            "message": f"Shape denoise — step {step}/{total}",
        })
        return
    if stripped.startswith("loaded "):
        job.emit({"phase": "shape", "stage_pct": 5,
                  "overall_pct": _hunyuan_overall_pct("shape", 5), "message": stripped})
    elif stripped.startswith("[vae] grid"):
        job.emit({"phase": "shape", "stage_pct": 80,
                  "overall_pct": _hunyuan_overall_pct("shape", 80), "message": stripped})
    elif stripped.startswith("[mesh]"):
        job.emit({"phase": "shape", "stage_pct": 95,
                  "overall_pct": _hunyuan_overall_pct("shape", 95), "message": stripped})
    elif stripped.startswith("shape generated"):
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
        missing.append("paint venv (hunyuan_mlx/paint/.venv)")
    if not weights_ok:
        missing.append("paint weights (hunyuan_mlx/paint/weights/hunyuan3d-paintpbr-v2-1)")
    return {
        "schema_version": 1,
        "build": {
            "present": ready,
            "hint": None if ready else (
                "Hunyuan3D-MLX setup is incomplete — missing: " + "; ".join(missing) + ". "
                "dgrauet's shape stage (vendor/hunyuan-mlx) is Tencent-licensed and stays "
                "manually vendor-cloned. Xiong's paint stage (hunyuan_mlx/paint) is tracked "
                "in-repo — run `uv sync` there, then download weights per "
                "docs/hunyuan-mlx-recipes.md."
            ),
        },
        "weights": {},
        "missing_weights": missing,
        "ready": ready,
        "warning": None,
    }


# --- Hunyuan3D-MLX (Xiong, full single-repo pipeline) spec --------------------------------
# Same job protocol as hunyuan-mlx above (identical print-line vocabulary — both wrapper
# scripts share the same stage prints), so stages/labels/parse_line are reused as-is. Only
# settings/build_args/readiness differ: this variant exposes shape-stage quantization (the
# one real lever for a usable interactive speed, since ZimengXiong's own shape stage runs
# full-precision 3.3B-MoE by default — see the wrapper script's docstring caveat) and points
# at a completely separate shape venv/weights.

HUNYUAN_XIONG_DEFAULT_SETTINGS: dict[str, Any] = {
    **HUNYUAN_DEFAULT_SETTINGS,
    "model": "2.0",
    "quantize": 8,
}
HUNYUAN_XIONG_VALID_QUANTIZE = {0, 4, 8}


def _hunyuan_xiong_validate_settings(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**HUNYUAN_XIONG_DEFAULT_SETTINGS, **raw}
    if settings["model"] not in HUNYUAN_XIONG_SHAPE_MODELS:
        raise ValueError(
            f"model must be one of {sorted(HUNYUAN_XIONG_SHAPE_MODELS)}"
        )
    try:
        for key in ("octree_resolution", "seed", "quantize", "decimation_target",
                    "paint_seed", "paint_res", "paint_steps", "paint_tex"):
            settings[key] = int(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError("all Hunyuan settings must be integers") from exc
    if settings["octree_resolution"] not in HUNYUAN_VALID_OCTREE:
        raise ValueError("octree_resolution must be one of 256, 384, 512, or 1024")
    if settings["quantize"] not in HUNYUAN_XIONG_VALID_QUANTIZE:
        raise ValueError("quantize must be 0 (off), 4, or 8")
    if settings["decimation_target"] <= 0:
        raise ValueError("decimation_target must be positive")
    if settings["decimation_target"] > 600_000:
        raise ValueError(
            "decimation_target above ~500k hits a confirmed xatlas wall (500k-700k faces "
            "crawled in testing, 1M never finished) — keep it at or "
            "under 500,000"
        )
    return settings


def _hunyuan_xiong_build_args(job: Job) -> list[str]:
    s = job.settings
    return [
        str(job.image_path), str(job.output_path),
        "--model", str(s["model"]),
        "--octree-resolution", str(s["octree_resolution"]),
        "--seed", str(s["seed"]),
        "--quantize", str(s["quantize"]),
        "--decimation-target", str(s["decimation_target"]),
        "--paint-seed", str(s["paint_seed"]),
        "--paint-res", str(s["paint_res"]),
        "--paint-steps", str(s["paint_steps"]),
        "--paint-tex", str(s["paint_tex"]),
    ]


def _hunyuan_xiong_shape_weights_status() -> dict[str, dict[str, Any]]:
    """Per-model shape-weights presence, in the {label, present, human} shape the Setup
    panel's frontend expects for every backend's `weights` field (see weights_on_disk).

    Previously this was a flat {name: bool} dict. Object.values(...) on that yields plain
    JS booleans, and `repo.present` / `repo.label` on a boolean are just undefined -- so the
    panel rendered all three models as "undefined ... not on disk" regardless of what was
    actually downloaded (found 2026-08-20 via a real screenshot: three "undefined" rows on
    a machine that in fact had weights present)."""
    out: dict[str, dict[str, Any]] = {}
    for name, path in HUNYUAN_XIONG_SHAPE_MODELS.items():
        present = path.is_dir()
        size = _dir_state(path)[1]
        out[name] = {"label": f"Hunyuan3D-MLX shape weights ({name})", "present": present,
                     "bytes": size, "human": _human_bytes(size)}
    return out


def _hunyuan_xiong_readiness() -> dict[str, Any]:
    shape_ok = HUNYUAN_XIONG_SHAPE_VENV.is_file() and HUNYUAN_XIONG_WRAPPER.is_file()
    weights_status = _hunyuan_xiong_shape_weights_status()
    model_availability = {name: info["present"] for name, info in weights_status.items()}
    any_model_ok = any(model_availability.values())
    default_model_ok = model_availability.get(HUNYUAN_XIONG_DEFAULT_SETTINGS["model"], False)
    paint_venv_ok = HUNYUAN_PAINT_VENV.is_file()
    paint_weights_ok = HUNYUAN_PAINT_WEIGHTS.is_dir()
    ready = shape_ok and any_model_ok and paint_venv_ok and paint_weights_ok
    missing = []
    if not shape_ok:
        missing.append(
            "shape venv/wrapper (hunyuan_mlx/shape/.venv, "
            "scripts/hunyuan_mlx_xiong_generate.py)"
        )
    if not any_model_ok:
        missing.append(
            "shape weights — none of 2.1/2.0/2.0-turbo downloaded "
            "(hunyuan_mlx/shape/weights/...); see docs/hunyuan-mlx-recipes.md"
        )
    elif not default_model_ok:
        missing.append(
            f"default model ({HUNYUAN_XIONG_DEFAULT_SETTINGS['model']}) not downloaded — "
            f"other models are, but requests without an explicit model will fail"
        )
    if not paint_venv_ok:
        missing.append("paint venv (hunyuan_mlx/paint/.venv)")
    if not paint_weights_ok:
        missing.append("paint weights (hunyuan_mlx/paint/weights/hunyuan3d-paintpbr-v2-1)")
    return {
        "schema_version": 1,
        "build": {
            "present": ready,
            "hint": None if ready else (
                "Hunyuan3D-MLX (Xiong, full) setup is incomplete — missing: "
                + "; ".join(missing) + ". Code is tracked in-repo (hunyuan_mlx/); run "
                "`uv sync` in hunyuan_mlx/shape and hunyuan_mlx/paint, then download "
                "weights per docs/hunyuan-mlx-recipes.md."
            ),
        },
        "weights": weights_status,
        "missing_weights": missing,
        # Per model, so the Shape model dropdown can disable what is not on disk instead
        # of letting a run fail minutes in. Downloading only the default route is now the
        # norm, so absent models are the expected case rather than a broken install.
        "model_availability": model_availability,
        "ready": ready,
        "warning": (
            "2.0 is the default and the cleanest. 2.0-turbo is about twice as fast but "
            "can leave small dents. 2.1 is slower and not recommended."
        ),
    }


PIXAL3D_DEFAULT_SETTINGS: dict[str, Any] = {
    "res": 1024,
    "seed": 42,
    "fov": 0.3490658503988659,
    # "auto": the wrapper runs 8 on a patched trellis-cli and 12 otherwise, so an install
    # without scripts/patch_pixal3d_steps.py keeps working. 12 forces the full count.
    "steps": "auto",
}
PIXAL3D_STEP_CHOICES = {"auto", 12}
# trellis-cli --sv-image refuses anything but 1024 ("supports --res 1024 only in this
# release"), and the single-view weights ship no res-512 texture flow.
PIXAL3D_VALID_RES = {1024}
PIXAL3D_STAGES = ["views", "ss", "shape", "decode", "texture", "write"]
PIXAL3D_STAGE_LABELS = {
    "views": "Preparing view",
    "ss": "Sparse structure",
    "shape": "Shape SLAT (512 to 1024 cascade)",
    "decode": "Shape decode",
    "texture": "Texture SLAT + PBR decode",
    "write": "Writing GLB",
}
# `trellis-cli` announces `[n/6] ...`; n maps to a stage, and the bar follows n.
PIXAL3D_BANNERS = {1: "views", 2: "ss", 3: "shape", 4: "decode", 5: "texture", 6: "write"}


def _pixal3d_validate_settings(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**PIXAL3D_DEFAULT_SETTINGS, **raw}
    try:
        settings["res"] = int(settings["res"])
        settings["seed"] = int(settings["seed"])
        settings["fov"] = float(settings["fov"])
    except (TypeError, ValueError) as exc:
        raise ValueError("res and seed must be integers, fov a number") from exc
    if settings["res"] not in PIXAL3D_VALID_RES:
        raise ValueError("Pixal3D single-image runs at res 1024 only")
    if not 0.05 <= settings["fov"] <= 2.0:
        raise ValueError("fov is in radians; 0.349 is 20 degrees")
    if settings["steps"] != "auto":
        try:
            settings["steps"] = int(settings["steps"])
        except (TypeError, ValueError) as exc:
            raise ValueError('steps must be "auto" or 12') from exc
    if settings["steps"] not in PIXAL3D_STEP_CHOICES:
        raise ValueError('steps must be "auto" or 12')
    return settings


def _pixal3d_build_args(job: Job) -> list[str]:
    s = job.settings
    return [
        str(job.image_path), str(job.output_path),
        "--res", str(s["res"]),
        "--seed", str(s["seed"]),
        "--fov", str(s["fov"]),
    ] + ([] if s.get("steps", "auto") == "auto" else ["--steps", str(s["steps"])])


def _pixal3d_parse_line(job: Job, line: str) -> None:
    """Emit a stage event per `[n/6]` banner, and let the rest through as log."""
    if line.startswith("[") and "]" in line and "/6" in line[: line.index("]")]:
        marker = line[1 : line.index("]")]
        try:
            index = int(marker.split("/")[0])
        except ValueError:
            return
        stage = PIXAL3D_BANNERS.get(index)
        if stage is None:
            return
        # `phase` must BE the stage id: JobProgressPanel.apply looks the row up by it.
        # Emitting {"phase": "stage", "stage": ...} leaves the panel frozen on its
        # placeholder, which is exactly what it did.
        job.emit({
            "phase": stage,
            "stage_pct": 0,
            "overall_pct": min(99, round(index / 6 * 100)),
            "message": PIXAL3D_STAGE_LABELS[stage],
        })


def _pixal3d_readiness() -> dict[str, Any]:
    weights = sorted(PIXAL3D_MODELS.glob("*.gguf")) if PIXAL3D_MODELS.is_dir() else []
    built = PIXAL3D_CLI.is_file() and PIXAL3D_WRAPPER.is_file()
    # Nine files: four flow DiTs, three decoders, the image encoder and NAF.
    weights_ok = len(weights) >= 9
    ready = built and weights_ok
    missing = []
    if not built:
        missing.append("trellis-cli build (vendor/pixal3d-cpp/build)")
    if not weights_ok:
        missing.append(f"Q8_0 weight set ({len(weights)}/9 in {PIXAL3D_MODELS})")
    return {
        "schema_version": 1,
        "build": {
            "present": ready,
            "hint": None if ready else (
                "Pixal3D is not set up — missing: " + "; ".join(missing)
                + ". Set it up from Setup & Status, or run "
                "python scripts/bootstrap_pixal3d.py (8.4 GB of weights)."
            ),
        },
        "weights": {
            "pixal3d-sv-q8_0": {
                "label": "Pixal3D single-view Q8_0",
                "present": weights_ok,
                "human": f"{len(weights)}/9 GGUF files",
            }
        },
        "missing_weights": missing,
        "ready": ready,
        "warning": (
            "Single-view only, and these weights have no res 512 option."
        ),
    }


def trellis_spec(host: str | None = None) -> BackendSpec:
    """TRELLIS.2 for this machine: the Metal port on a Mac, Microsoft's code on NVIDIA.

    One id, two installs. The NVIDIA spec mattes images itself with our remover, so it
    does not demand a transparent upload the way the Mac port does.
    """
    if (host or host_platform()) == NVIDIA:
        return BackendSpec(
            id="trellis", label="TRELLIS.2 (NVIDIA)",
            interpreter=TRELLIS_CUDA_PYTHON, wrapper=TRELLIS_CUDA_WRAPPER,
            default_settings=DEFAULT_SETTINGS, stages=STAGES, stage_labels=PHASE_LABELS,
            requires_alpha=False,
            validate_settings=validate_settings, build_args=_trellis_cuda_build_args,
            parse_line=_trellis_parse_line, readiness=cuda_setup_status,
            baseline_path=BASELINE_PATH,
            hidden_fields=("generate-attention", "generate-rembg"),
        )
    return BackendSpec(
        id="trellis", label="TRELLIS.2 (clean port)",
        interpreter=PYTHON, wrapper=WRAPPER,
        default_settings=DEFAULT_SETTINGS, stages=STAGES, stage_labels=PHASE_LABELS,
        requires_alpha=True,
        validate_settings=validate_settings, build_args=_trellis_build_args,
        parse_line=_trellis_parse_line, readiness=setup_status,
        baseline_path=BASELINE_PATH,
    )


# --- Hunyuan3D-2.1 on NVIDIA (Tencent's own code) ---------------------------------------
# Upstream's demo.py defaults. The generator validates the same ranges again.
HUNYUAN_CUDA_DEFAULT_SETTINGS: dict[str, Any] = {
    "seed": 1234,
    "steps": 50,
    "octree_resolution": 384,
    "max_num_view": 6,
    "paint_resolution": 512,
}


def _hunyuan_cuda_validate_settings(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**HUNYUAN_CUDA_DEFAULT_SETTINGS,
                **{k: v for k, v in raw.items() if k in HUNYUAN_CUDA_DEFAULT_SETTINGS}}
    try:
        settings = {key: int(value) for key, value in settings.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("all Hunyuan3D-2.1 settings must be integers") from exc
    if settings["octree_resolution"] not in {256, 384, 512}:
        raise ValueError("octree_resolution must be 256, 384 or 512")
    if not 6 <= settings["max_num_view"] <= 9:
        raise ValueError("max_num_view must be 6 to 9")
    if settings["paint_resolution"] not in {512, 768}:
        raise ValueError("paint_resolution must be 512 or 768")
    if settings["steps"] < 1:
        raise ValueError("steps must be at least 1")
    return settings


def _hunyuan_cuda_build_args(job: Job) -> list[str]:
    s = job.settings
    return [
        str(job.image_path), str(job.output_path),
        "--seed", str(s["seed"]),
        "--steps", str(s["steps"]),
        "--octree-resolution", str(s["octree_resolution"]),
        "--max-num-view", str(s["max_num_view"]),
        "--paint-resolution", str(s["paint_resolution"]),
    ]


def _hunyuan_cuda_readiness() -> dict[str, Any]:
    """Ready only when built *and* every weight is on disk.

    Upstream fetches missing weights on first use, unannounced. The installer names and
    fetches them all up front, so a missing one means setup has not finished, and the run
    must not start and download it behind the user's back.
    """
    import backend_catalog

    built = (HUNYUAN_CUDA_MARKER.is_file() and HUNYUAN_CUDA_PYTHON.is_file()
             and HUNYUAN_CUDA_WRAPPER.is_file())
    weights = {w["label"]: {"label": w["label"], "present": w["present"],
                            "human": w["human_present"]}
               for w in backend_catalog.BY_ID["hunyuan-cuda"].describe()["weights"]}
    missing = [label for label, w in weights.items() if not w["present"]]
    ready = built and not missing
    hint = None
    if not built:
        hint = ("Hunyuan3D-2.1 for NVIDIA is not installed. Set it up from Setup & Status, "
                "or run python scripts/bootstrap_hunyuan_cuda.py (Linux only, ~19.5 GB).")
    elif missing:
        hint = ("Hunyuan3D-2.1 weights are missing: " + "; ".join(missing) + ". Run "
                "python scripts/bootstrap_hunyuan_cuda.py --weights-only.")
    return {
        "schema_version": 1,
        "build": {"present": built, "interpreter": str(HUNYUAN_CUDA_PYTHON),
                  "wrapper": str(HUNYUAN_CUDA_WRAPPER), "hint": hint},
        "weights": weights,
        "missing_weights": missing,
        "ready": ready,
        "warning": "Not licensed in the EU, the UK or South Korea.",
    }


BACKENDS.update({
    "trellis": trellis_spec(),
    "sf3d": BackendSpec(
        id="sf3d", label="Stable Fast 3D",
        interpreter=Path(sys.executable), wrapper=REPO / "pipeline.py",
        default_settings=SF3D_DEFAULT_SETTINGS, stages=SF3D_STAGES,
        stage_labels=SF3D_STAGE_LABELS, requires_alpha=False,
        validate_settings=_sf3d_validate_settings, build_args=_sf3d_build_args,
        parse_line=_sf3d_parse_line, readiness=_sf3d_readiness, finalize=_sf3d_finalize,
    ),
    "hunyuan-mlx": BackendSpec(
        id="hunyuan-mlx", label="Hunyuan3D-MLX (dgrauet shape + Xiong paint)",
        interpreter=HUNYUAN_PYTHON, wrapper=HUNYUAN_WRAPPER,
        default_settings=HUNYUAN_DEFAULT_SETTINGS, stages=HUNYUAN_STAGES,
        stage_labels=HUNYUAN_STAGE_LABELS, requires_alpha=False,
        validate_settings=_hunyuan_validate_settings, build_args=_hunyuan_build_args,
        parse_line=_hunyuan_parse_line, readiness=_hunyuan_readiness,
    ),
    "hunyuan-mlx-xiong": BackendSpec(
        id="hunyuan-mlx-xiong", label="Hunyuan3D-MLX (Xiong, full pipeline)",
        interpreter=HUNYUAN_XIONG_SHAPE_VENV, wrapper=HUNYUAN_XIONG_WRAPPER,
        default_settings=HUNYUAN_XIONG_DEFAULT_SETTINGS, stages=HUNYUAN_STAGES,
        stage_labels=HUNYUAN_STAGE_LABELS, requires_alpha=False,
        validate_settings=_hunyuan_xiong_validate_settings, build_args=_hunyuan_xiong_build_args,
        parse_line=_hunyuan_parse_line, readiness=_hunyuan_xiong_readiness,
    ),
    "pixal3d": BackendSpec(
        id="pixal3d", label="Pixal3D (C++/GGML)",
        interpreter=Path(sys.executable), wrapper=PIXAL3D_WRAPPER,
        default_settings=PIXAL3D_DEFAULT_SETTINGS, stages=PIXAL3D_STAGES,
        stage_labels=PIXAL3D_STAGE_LABELS, requires_alpha=False,
        validate_settings=_pixal3d_validate_settings, build_args=_pixal3d_build_args,
        parse_line=_pixal3d_parse_line, readiness=_pixal3d_readiness,
    ),
    # Same stage names and progress lines as the MLX Hunyuan routes, so it shares their parser.
    "hunyuan-cuda": BackendSpec(
        id="hunyuan-cuda", label="Hunyuan3D-2.1 (NVIDIA)",
        interpreter=HUNYUAN_CUDA_PYTHON, wrapper=HUNYUAN_CUDA_WRAPPER,
        default_settings=HUNYUAN_CUDA_DEFAULT_SETTINGS, stages=HUNYUAN_STAGES,
        stage_labels=HUNYUAN_STAGE_LABELS, requires_alpha=False,
        validate_settings=_hunyuan_cuda_validate_settings,
        build_args=_hunyuan_cuda_build_args,
        parse_line=_hunyuan_parse_line, readiness=_hunyuan_cuda_readiness,
    ),
})


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


def _job_status_payload(job: Job) -> dict[str, Any]:
    """A single-shot snapshot of a job's current state -- the polling fallback for when
    the SSE stream (_stream_events) drops and doesn't reconnect (found 2026-08-20: a fast
    SF3D run finished server-side but the browser never learned it had, because nothing
    covers a dead/never-recovered EventSource connection).

    Deliberately the same event shape _run_job's "done"/"error" events already carry, so
    the frontend can feed this straight into its existing applyGenerateProgress() renderer
    instead of a separate code path."""
    return {"status": job.status, "last_event": job.events[-1] if job.events else None}


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
        if parts == ["api", "blender", "install"]:
            self._start_blender_install()
            return
        if parts == ["api", "gltfpack", "install"]:
            self._start_gltfpack_install()
            return
        if parts == ["api", "hf", "sign-in"]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length)) if 0 < length <= 4096 else None
            except (ValueError, json.JSONDecodeError):
                payload = None
            self._send_json(*hf_sign_in_response(payload))
            return
        if parts == ["api", "setup", "run"]:
            self._start_setup()
            return
        if len(parts) == 4 and parts[:2] == ["api", "setup"] and parts[3] in {"download", "rebuild", "cancel", "remove"}:
            self._backend_download(parts[2], parts[3])
            return
        if parts == ["api", "generate"]:
            self._create_job()
            return
        if parts == ["api", "trellis", "input-advice"]:
            self._trellis_input_advice()
            return
        if parts == ["api", "rig", "rebind"]:
            self._create_rig_job()
            return
        if len(parts) == 4 and parts[:2] == ["api", "generate"] and parts[3] == "cancel":
            self._cancel_job(parts[2])
            return
        if len(parts) == 5 and parts[:3] == ["api", "rig", "rebind"] and parts[4] == "cancel":
            self._cancel_rig_job(parts[3])
            return
        if parts == ["api", "finish"]:
            self._create_finish_job()
            return
        if len(parts) == 4 and parts[:2] == ["api", "finish"] and parts[3] == "cancel":
            self._cancel_finish_job(parts[2])
            return
        if len(parts) == 5 and parts[:3] == ["api", "finish", "runs"] and parts[4] == "resume":
            self._resume_finish_job(parts[3])
            return
        if parts == ["api", "props"]:
            self._create_props_job()
            return
        if parts == ["api", "animate", "rig"]:
            self._create_animate_rig()
            return
        if parts == ["api", "animate", "play"]:
            self._create_animate_play()
            return
        if len(parts) == 4 and parts[:2] == ["api", "animate"] and parts[3] == "cancel":
            job = ANIMATE_JOBS.get(parts[2])
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                cancel_animate_job(job)
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._send_json(202, {"job_id": job.id, "status": job.status})
            return
        if len(parts) == 4 and parts[:2] == ["api", "props"] and parts[3] == "cancel":
            self._cancel_props_job(parts[2])
            return
        if len(parts) == 5 and parts[:3] == ["api", "props", "runs"] and parts[4] == "turn":
            self._turn_prop(parts[3])
            return
        if parts == ["api", "image"]:
            self._create_image_job()
            return
        if len(parts) == 4 and parts[:2] == ["api", "image"] and parts[3] == "cancel":
            self._cancel_image_job(parts[2])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _create_image_job(self) -> None:
        """Start one text-to-image run. The prompt is the only required field."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                self._send_json(400, {"error": "expected a small JSON body"})
                return
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"could not read the request: {exc}"})
            return
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json(422, {"error": "a prompt is required"})
            return
        if len(prompt) > image_api.MAX_PROMPT:
            self._send_json(422, {
                "error": f"prompt is longer than {image_api.MAX_PROMPT} characters"
            })
            return
        if not image_api.BINARY.exists():
            self._send_json(503, {
                "error": "stable-diffusion.cpp is not installed (vendor/sdcpp/sd-cli).",
                "needs_setup": True,
            })
            return
        try:
            image_api.resolve_weights()
        except image_api.MissingWeights as exc:
            self._send_json(503, {"error": str(exc), "needs_setup": True,
                                  "missing": exc.missing})
            return
        try:
            job = image_api.start(prompt, payload.get("settings") or {})
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(202, {
            "job_id": job.id,
            "settings": job.settings,
            "events_url": f"/api/image/{job.id}/events",
            "status_url": f"/api/image/{job.id}/status",
        })

    def _cancel_image_job(self, job_id: str) -> None:
        if not _safe_id(job_id) or job_id not in image_api.MANAGER.jobs:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not image_api.MANAGER.cancel(job_id):
            self._send_json(409, {"error": "that image job is not running"})
            return
        self._send_json(202, {"job_id": job_id, "status": "cancelling"})

    def _image_events(self, job_id: str) -> None:
        job = image_api.MANAGER.jobs.get(job_id) if _safe_id(job_id) else None
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._start_event_stream()
        index = 0
        try:
            while True:
                with job.condition:
                    if index >= len(job.events):
                        job.condition.wait(timeout=15)
                    pending = job.events[index:]
                    index = len(job.events)
                    terminal = (job.status in {"done", "error", "cancelled"}
                                and not pending)
                for event in pending:
                    body = json.dumps(event, separators=(",", ":"))
                    self.wfile.write(f"data: {body}\n\n".encode())
                if not pending:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                if terminal:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return

    def _trellis_input_advice(self) -> None:
        """Classify one upload without creating a generation job or retaining the image."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send_json(400, {"error": "image upload is missing or larger than 50 MiB"})
                return
            form = parse_multipart(
                self.headers.get("Content-Type", ""), self.rfile.read(length)
            )
            image_field = form.get("image")
            filename = str(image_field.get("filename")) if image_field else ""
            suffix = Path(filename).suffix.lower()
            if not image_field or suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self._send_json(422, {
                    "error": "multipart field 'image' must be PNG, JPG, WebP, or BMP"
                })
                return
            with tempfile.TemporaryDirectory(prefix="i2l-tinyclip-") as directory:
                image_path = Path(directory) / f"input{suffix}"
                image_path.write_bytes(image_field["data"])
                payload = run_trellis_input_advisor(image_path)
            self._send_json(200, payload)
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc), "advisory_only": True})
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "advisory_only": True})

    def do_GET(self) -> None:
        parts = self._path_parts()
        if parts == ["api", "hf", "status"]:
            self._send_json(200, hf_api.status())
            return
        if parts == ["api", "catalog"]:
            self._send_json(200, catalog_payload())
            return
        if parts == ["api", "assets"]:
            try:
                self._send_json(200, assets_payload(OUTPUT_ROOT))
            except Exception as exc:  # a broken run folder must not take the whole Library down
                self._send_json(500, {"error": str(exc)})
            return
        if parts == ["api", "update-check"]:
            self._send_json(200, update_check())
            return
        if parts == ["api", "welcome"]:
            since = parse_qs(urlparse(self.path).query).get("since", [None])[0]
            self._send_json(200, welcome_payload(since))
            return
        if parts == ["api", "setup"]:
            query = parse_qs(urlparse(self.path).query)
            backend_id = query.get("backend", ["trellis"])[0]
            # The catalogue decides what a route *is*; a backend spec, where one exists,
            # adds the live detail only it can probe (which Hunyuan checkpoint is on disk,
            # how many of the nine GGUF files arrived). Asking the specs first is what made
            # the image route, which has no spec because it makes pictures rather than
            # meshes, answer "unknown backend" for something the catalogue lists.
            payload = catalog_readiness(backend_id)
            if payload is None:
                self._send_json(422, {"error": f"unknown backend {backend_id!r}"})
                return
            spec = BACKENDS.get(backend_id)
            if spec is not None:
                live = spec.readiness()
                # A spec that reports no weights is not saying "this backend has none",
                # it is saying it does not track them -- SF3D and the dgrauet Hunyuan
                # route both return an empty dict. Letting that overwrite the catalogue
                # would hide the weight rows the Setup page shows for the same backend.
                for key in ("weights", "missing_weights"):
                    if not live.get(key):
                        live.pop(key, None)
                payload = {**payload, **live, "backend": spec.id}
            self._send_json(200, payload)
            return
        if len(parts) == 4 and parts[:2] == ["api", "image"]:
            job = image_api.MANAGER.jobs.get(parts[2]) if _safe_id(parts[2]) else None
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if parts[3] == "events":
                self._image_events(parts[2])
                return
            if parts[3] == "status":
                self._send_json(200, job.describe())
                return
            if parts[3] == "result.png":
                if job.status != "done" or not job.output_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                data = job.output_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parts == ["api", "image", "defaults"]:
            self._send_json(200, {
                "defaults": image_api.DEFAULTS,
                "samplers": list(image_api.SAMPLERS),
                "installed": image_api.is_installed(),
                "license": {"name": image_api.LICENSE_NAME, "url": image_api.LICENSE_URL,
                            "attribution": image_api.ATTRIBUTION},
            })
            return
        if parts == ["api", "backends"]:
            self._send_json(200, backends_payload())
            return
        if len(parts) == 4 and parts[:2] == ["api", "setup"] and parts[3] in {"events", "status"}:
            run = DOWNLOADS.get(parts[2])
            if run is not None:
                if parts[3] == "events":
                    self._stream_events(run)
                else:
                    self._send_json(200, download_status_payload(run))
                return
        # /api/setup/run/<id>/events: five parts, as _start_setup and the Blender
        # install hand it out.
        if len(parts) == 5 and parts[:3] == ["api", "setup", "run"] and parts[4] == "events":
            run = SETUP_RUNS.get(parts[3]) if _safe_id(parts[3]) else None
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
            if action == "status":
                self._status(job_id)
                return
            if action in {"result.glb", "manifest.json"}:
                self._artifact(job_id, action)
                return
        if parts == ["api", "finish", "capabilities"]:
            self._send_json(200, finish_capabilities())
            return
        if parts == ["api", "finish", "runs"]:
            self._send_json(200, {"runs": list_finish_runs(FINISH_JOBS.output_root)})
            return
        if len(parts) == 4 and parts[:2] == ["api", "finish"]:
            job_id, action = parts[2], parts[3]
            if action == "events":
                self._finish_events(job_id)
                return
            if action == "status":
                self._finish_status(job_id)
                return
            if action in FINISH_ARTIFACTS:
                self._finish_artifact(job_id, action)
                return
        if parts == ["api", "animate", "models"]:
            try:
                self._send_json(200, {"models": animate_models(OUTPUT_ROOT),
                                      "presets": animate_presets(),
                                      "installed": autorig_installed()})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return
        if len(parts) == 4 and parts[:2] == ["api", "animate"]:
            job = ANIMATE_JOBS.get(parts[2])
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if parts[3] == "events":
                self._stream_events(job)
                return
            if parts[3] == "status":
                self._send_json(200, animate_status_payload(job))
                return
        if parts == ["api", "props", "tools"]:
            self._send_json(200, props_tools_payload())
            return
        if parts == ["api", "props", "runs"]:
            try:
                payload = {
                    "runs": list_props_runs(PROPS_JOBS.output_root),
                    "generated": generated_models(OUTPUT_ROOT),
                    "tools": props_tools_payload(),
                }
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
                return
            self._send_json(200, payload)
            return
        if len(parts) == 4 and parts[:2] == ["api", "props"]:
            job = PROPS_JOBS.get(parts[2])
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if parts[3] == "events":
                self._stream_events(job)
                return
            if parts[3] == "status":
                self._send_json(200, props_status_payload(job))
                return
        if len(parts) == 5 and parts[:3] == ["api", "rig", "rebind"]:
            job_id, action = parts[3], parts[4]
            if action == "events":
                self._rig_events(job_id)
                return
            if action == "status":
                self._rig_status(job_id)
                return
            if action in RIG_ARTIFACTS:
                self._rig_artifact(job_id, action)
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
            rig_active = RIG_JOBS.get(RIG_JOBS.active) if RIG_JOBS.active else None
            if rig_active is not None and rig_active.status not in {"done", "error", "cancelled"}:
                self._send_json(409, {"error": "a rig rebind is running; wait for it to finish"})
                return
            if _props_baking():
                self._send_json(409, {"error": "a prop sheet is baking; wait for it to finish"})
                return
            if _animate_running():
                self._send_json(409, {"error": "an auto-rig or animation is running; wait for it to finish"})
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
            try:
                lacks_alpha = spec.requires_alpha and not image_has_transparent_alpha(image_path)
            except RuntimeError as exc:
                for child in provisional.iterdir():
                    child.unlink()
                provisional.rmdir()
                self._send_json(500, {"error": str(exc)})
                return
            if lacks_alpha and not settings.get("allow_rembg"):
                for child in provisional.iterdir():
                    child.unlink()
                provisional.rmdir()
                self._send_json(422, {
                    "error": "This image has no transparent alpha foreground. Enable 'allow rembg' "
                             "to use BRIA background removal, or upload a pre-masked PNG."
                })
                return
            if spec.requires_alpha and not lacks_alpha:
                border = image_border_opaque_fraction(image_path)
                if border is not None and border > UNCUT_BORDER_LIMIT:
                    for child in provisional.iterdir():
                        child.unlink()
                    provisional.rmdir()
                    self._send_json(422, {"error": uncut_image_error(border)})
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

    def _create_rig_job(self) -> None:
        try:
            active = JOBS.get(JOBS.active) if JOBS.active else None
            if active is not None and active.status in {"queued", "running", "cancelling"}:
                self._send_json(409, {"error": "a generation is running; wait for it to finish"})
                return
            if _props_baking():
                self._send_json(409, {"error": "a prop sheet is baking; wait for it to finish"})
                return
            if _animate_running():
                self._send_json(409, {"error": "an auto-rig or animation is running; wait for it to finish"})
                return
            if SETUP_ACTIVE is not None:
                self._send_json(409, {"error": "setup is running; wait for it to finish"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 768 * 1024 * 1024:
                self._send_json(400, {"error": "rebind bundle is missing or larger than 768 MiB"})
                return
            form = parse_multipart(
                self.headers.get("Content-Type", ""), self.rfile.read(length)
            )
            required = {"asset": ".glb", "scene": ".blend", "sidecar": ".rig.json"}
            for field, suffix in required.items():
                value = form.get(field)
                filename = str(value.get("filename")) if value else ""
                if not value or not filename.lower().endswith(suffix):
                    self._send_json(422, {
                        "error": f"multipart field {field!r} must be a {suffix} file"
                    })
                    return
            try:
                job = RIG_JOBS.create(
                    str(form["asset"]["filename"]), form["asset"]["data"],
                    form["scene"]["data"], form["sidecar"]["data"],
                )
            except (ValueError, RuntimeError) as exc:
                self._send_json(422 if isinstance(exc, ValueError) else 409, {"error": str(exc)})
                return
            threading.Thread(
                target=run_rig_job, args=(job,), daemon=True, name=f"rig-{job.id[:8]}"
            ).start()
            self._send_json(202, {
                "job_id": job.id,
                "events_url": f"/api/rig/rebind/{job.id}/events",
                "status_url": f"/api/rig/rebind/{job.id}/status",
            })
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _animate_blocked(self) -> str | None:
        """Why an Animate job cannot start now. The auto-rig loads an ~8 GB model, so it
        waits for anything else that holds the GPU or unified memory."""
        active = JOBS.get(JOBS.active) if JOBS.active else None
        if active is not None and active.status in {"queued", "running", "cancelling"}:
            return "a generation is running; wait for it to finish"
        if SETUP_ACTIVE is not None:
            return "setup is running; wait for it to finish"
        if _props_baking():
            return "a prop sheet is baking; wait for it to finish"
        finishing = FINISH_JOBS.get(FINISH_JOBS.active) if FINISH_JOBS.active else None
        if finishing is not None and finishing.status not in {"done", "error", "cancelled"}:
            return "a model is being finished; wait for it to finish"
        if not autorig_installed():
            return "Auto-rig is not installed. Set it up on the Setup & Status page."
        return None

    def _start_animate(self, job) -> None:
        threading.Thread(target=run_animate_job, args=(job,), daemon=True,
                         name=f"animate-{job.id[:8]}").start()
        self._send_json(202, {"job_id": job.id, "kind": job.kind,
                              "events_url": f"/api/animate/{job.id}/events",
                              "status_url": f"/api/animate/{job.id}/status"})

    def _create_animate_rig(self) -> None:
        """Rig a model: a GLB uploaded in field 'asset', or one the lab made, named in
        field 'model' by the path /api/animate/models listed it under."""
        try:
            blocked = self._animate_blocked()
            if blocked:
                self._send_json(409, {"error": blocked})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 512 * 1024 * 1024:
                self._send_json(400, {"error": "request is missing or larger than 512 MiB"})
                return
            form = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
            asset = form.get("asset")
            chosen = str((form.get("model") or {}).get("value", "")).strip()
            if asset and str(asset.get("filename", "")).lower().endswith(".glb"):
                name, data, record, origin = str(asset["filename"]), asset["data"], None, "uploaded"
            elif chosen:
                try:
                    path = resolve_animate_model(chosen, OUTPUT_ROOT)
                except ValueError as exc:
                    self._send_json(422, {"error": str(exc)})
                    return
                name, data = path.name, path.read_bytes()
                record, origin = animate_source_record(path), "generated"
            else:
                self._send_json(422, {"error": "send a .glb in 'asset' or a listed 'model'"})
                return
            try:
                job = ANIMATE_JOBS.create_rig(name, data, record, origin)
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._start_animate(job)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _create_animate_play(self) -> None:
        """Play a preset on a rigged model: JSON {model, preset, arm_spread, speed}."""
        try:
            blocked = self._animate_blocked()
            if blocked:
                self._send_json(409, {"error": blocked})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length)) if 0 < length <= 4096 else None
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                self._send_json(422, {"error": "expected a JSON object"})
                return
            try:
                rigged = resolve_animate_model(str(body.get("model", "")), OUTPUT_ROOT)
                job = ANIMATE_JOBS.create_animation(rigged, str(body.get("preset", "")),
                                                    body.get("arm_spread", 0), body.get("speed"))
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._start_animate(job)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _create_finish_job(self) -> None:
        """Retopologise, repaint and compress a GLB the viewer already has.

        Deliberately refuses while a generation is running: the repaint stage loads its own
        multi-gigabyte model, and two of those at once is how this machine runs out of
        unified memory.
        """
        try:
            active = JOBS.get(JOBS.active) if JOBS.active else None
            if active is not None and active.status in {"queued", "running", "cancelling"}:
                self._send_json(409, {"error": "a generation is running; wait for it to finish"})
                return
            if SETUP_ACTIVE is not None:
                self._send_json(409, {"error": "setup is running; wait for it to finish"})
                return
            if _props_baking():
                self._send_json(409, {"error": "a prop sheet is baking; wait for it to finish"})
                return
            if _animate_running():
                self._send_json(409, {"error": "an auto-rig or animation is running; wait for it to finish"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 768 * 1024 * 1024:
                self._send_json(400, {"error": "finish bundle is missing or larger than 768 MiB"})
                return
            form = parse_multipart(
                self.headers.get("Content-Type", ""), self.rfile.read(length)
            )
            asset = form.get("asset")
            asset_name = str(asset.get("filename")) if asset else ""
            if not asset or not asset_name.lower().endswith(".glb"):
                self._send_json(422, {"error": "multipart field 'asset' must be a .glb file"})
                return
            image = form.get("image")
            image_name = str(image.get("filename")) if image else ""
            if not image or Path(image_name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                self._send_json(422, {
                    "error": "multipart field 'image' must be the source art (PNG/JPG/WebP)"
                })
                return
            try:
                raw_settings = json.loads(form.get("settings", {}).get("value", "{}"))
            except json.JSONDecodeError as exc:
                self._send_json(422, {"error": f"invalid settings JSON: {exc}"})
                return
            if not isinstance(raw_settings, dict):
                self._send_json(422, {"error": "settings must be a JSON object"})
                return
            # Checked here rather than a minute into the run, where a missing Blender used
            # to surface as a bare worker exit code.
            if find_blender() is None:
                self._send_json(409, {"error": blender_missing_help()})
                return
            try:
                job = FINISH_JOBS.create(
                    asset_name, asset["data"], image["data"], raw_settings,
                )
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            threading.Thread(
                target=run_finish_job, args=(job,), daemon=True, name=f"finish-{job.id[:8]}"
            ).start()
            self._send_json(202, {
                "job_id": job.id,
                "settings": job.settings,
                "events_url": f"/api/finish/{job.id}/events",
                "status_url": f"/api/finish/{job.id}/status",
            })
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _finish_events(self, job_id: str) -> None:
        job = FINISH_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._stream_events(job)

    def _finish_status(self, job_id: str) -> None:
        job = FINISH_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(200, finish_status_payload(job))

    def _finish_artifact(self, job_id: str, action: str) -> None:
        job = FINISH_JOBS.get(job_id)
        if job is None or job.status != "done":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        attribute, content_type = FINISH_ARTIFACTS[action]
        path = getattr(job, attribute)
        if not path.is_file() or job.directory not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        disposition = "inline" if action == "result.glb" else "attachment"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _resume_finish_job(self, name: str) -> None:
        """Re-run one existing run directory, skipping the stages it already completed.

        No upload and no settings: both come from the directory, which is what makes the
        resume faithful. A run that still needs its repaint costs six minutes; one that
        only lost its compression costs a second.
        """
        with SETUP_LOCK:
            active = JOBS.get(JOBS.active)
            if active is not None and active.status in {"queued", "running", "cancelling"}:
                self._send_json(409, {"error": "a generation is running; wait for it to finish"})
                return
            if SETUP_ACTIVE is not None:
                self._send_json(409, {"error": "setup is running; wait for it to finish"})
                return
            if _props_baking():
                self._send_json(409, {"error": "a prop sheet is baking; wait for it to finish"})
                return
            if _animate_running():
                self._send_json(409, {"error": "an auto-rig or animation is running; wait for it to finish"})
                return
        try:
            if find_blender() is None:
                raise RuntimeError(blender_missing_help())
            job = FINISH_JOBS.adopt(name)
        except (RuntimeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc)})
            return
        threading.Thread(
            target=run_finish_job, args=(job,), daemon=True, name=f"finish-{job.id[:8]}"
        ).start()
        self._send_json(202, {
            "job_id": job.id,
            "directory": job.directory.name,
            "settings": job.settings,
            "events_url": f"/api/finish/{job.id}/events",
            "status_url": f"/api/finish/{job.id}/status",
        })

    def _cancel_finish_job(self, job_id: str) -> None:
        job = FINISH_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            cancel_finish_job(job)
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(202, {"job_id": job.id, "status": job.status})

    def _props_busy(self) -> str | None:
        """Why a prop-sheet job cannot start now, or None.

        Its bakes are Blender on the CPU, but a generation or a finishing repaint holds
        gigabytes of model in unified memory, and a Blender bake on top of that is how
        this machine starts swapping.
        """
        active = JOBS.get(JOBS.active) if JOBS.active else None
        if active is not None and active.status in {"queued", "running", "cancelling"}:
            return "a generation is running; wait for it to finish"
        if SETUP_ACTIVE is not None:
            return "setup is running; wait for it to finish"
        finishing = FINISH_JOBS.jobs.get(FINISH_JOBS.active) if FINISH_JOBS.active else None
        if finishing is not None and finishing.status in {"queued", "running", "cancelling"}:
            return "a finishing job is running; wait for it to finish"
        rebind = RIG_JOBS.get(RIG_JOBS.active) if RIG_JOBS.active else None
        if rebind is not None and rebind.status not in {"done", "error", "cancelled"}:
            return "a rig rebind is running; wait for it to finish"
        if _animate_running():
            return "an auto-rig or animation is running; wait for it to finish"
        return None

    def _start_props_job(self, job) -> None:
        threading.Thread(
            target=run_props_job, args=(job,), daemon=True, name=f"props-{job.id[:8]}"
        ).start()
        self._send_json(202, {
            "job_id": job.id,
            "directory": job.directory.name,
            "settings": job.settings,
            "events_url": f"/api/props/{job.id}/events",
            "status_url": f"/api/props/{job.id}/status",
        })

    def _create_props_job(self) -> None:
        """Split a prop-sheet GLB into props and bake each one's LODs.

        The GLB is an upload, or one the Generate tab already wrote, named by the path
        `/api/props/runs` listed it under; nothing else on disk can be named.
        """
        try:
            busy = self._props_busy()
            if busy:
                self._send_json(409, {"error": busy})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 768 * 1024 * 1024:
                self._send_json(400, {"error": "prop sheet is missing or larger than 768 MiB"})
                return
            form = parse_multipart(
                self.headers.get("Content-Type", ""), self.rfile.read(length)
            )
            asset = form.get("asset")
            generated = str(form.get("generated", {}).get("value", "")).strip()
            source_record, from_generated = None, False
            if asset and str(asset.get("filename", "")).lower().endswith(".glb"):
                asset_name, data = str(asset["filename"]), asset["data"]
            elif generated:
                try:
                    path = generated_model(OUTPUT_ROOT, generated)
                except RuntimeError as exc:
                    self._send_json(422, {"error": str(exc)})
                    return
                asset_name, data = path.name, path.read_bytes()
                source_record, from_generated = source_record_for(path), True
            else:
                self._send_json(422, {
                    "error": "send a .glb as 'asset', or a generated model as 'generated'"
                })
                return
            try:
                raw_settings = json.loads(form.get("settings", {}).get("value", "{}"))
            except json.JSONDecodeError as exc:
                self._send_json(422, {"error": f"invalid settings JSON: {exc}"})
                return
            if not isinstance(raw_settings, dict):
                self._send_json(422, {"error": "settings must be a JSON object"})
                return
            try:
                job = PROPS_JOBS.create(asset_name, data, raw_settings, source_record,
                                        generated=from_generated)
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._start_props_job(job)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _turn_prop(self, name: str) -> None:
        """Turn one prop of a finished run about the vertical and re-bake only it."""
        try:
            busy = self._props_busy()
            if busy:
                self._send_json(409, {"error": busy})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                self._send_json(400, {"error": "expected a small JSON body"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"could not read the request: {exc}"})
            return
        try:
            job = PROPS_JOBS.turn(name, str(payload.get("prop", "")), payload.get("degrees", 90))
        except ValueError as exc:
            self._send_json(422, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._start_props_job(job)

    def _cancel_props_job(self, job_id: str) -> None:
        job = PROPS_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            cancel_props_job(job)
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(202, {"job_id": job.id, "status": job.status})

    def _find_job(self, job_id: str) -> Job | None:
        return JOBS.get(job_id) if _safe_id(job_id) else None

    def _events(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._stream_events(job)

    def _rig_events(self, job_id: str) -> None:
        job = RIG_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._stream_events(job)

    def _rig_status(self, job_id: str) -> None:
        job = RIG_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(200, rig_status_payload(job))

    def _rig_artifact(self, job_id: str, action: str) -> None:
        job = RIG_JOBS.get(job_id)
        if job is None or job.status != "done":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        attribute, content_type = RIG_ARTIFACTS[action]
        path = getattr(job, attribute)
        if not path.is_file() or job.directory not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        disposition = "inline" if action == "result.glb" else "attachment"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _cancel_rig_job(self, job_id: str) -> None:
        job = RIG_JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            cancel_rig_job(job)
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        job.emit({"phase": "error", "message": "Cancellation requested"})
        self._send_json(202, {"job_id": job.id, "status": "cancelling"})

    def _start_event_stream(self) -> None:
        """Headers for a server-sent event stream. X-Accel-Buffering stops proxies (nginx,
        Cloudflare, RunPod's) from holding progress back and delivering it minutes late."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _stream_events(self, run: Job | SetupRun) -> None:
        """SSE pump shared by generation jobs and setup runs."""
        self._start_event_stream()
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

    def _backend_download(self, backend_id: str, action: str) -> None:
        """Start or stop one backend's weight download.

        Deliberately POST-only and per backend: this is the one call in the viewer that
        spends the user's disk and bandwidth, and AGENTS.md requires it to name what it
        is fetching before it runs. The page does the naming; this refuses to start a
        second download while one is in flight.
        """
        try:
            if action == "cancel":
                cancel_download(backend_id)
                self._send_json(202, {"backend": backend_id, "status": "cancelling"})
                return
            if action == "remove":
                self._send_json(200, remove_weights(backend_id))
                return
            start_download(backend_id, rebuild=action == "rebuild")
        except KeyError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(202, {
            "backend": backend_id,
            "events_url": f"/api/setup/{backend_id}/events",
            "status_url": f"/api/setup/{backend_id}/status",
        })

    def _start_blender_install(self) -> None:
        global SETUP_ACTIVE
        with SETUP_LOCK:
            active = JOBS.get(JOBS.active)
            refusal = blender_install_refusal(
                finish_capabilities(),
                generating=active is not None
                and active.status in {"queued", "running", "cancelling"},
                setting_up=SETUP_ACTIVE is not None or download_active() is not None)
            if refusal:
                self._send_json(refusal[0], {"error": refusal[1]})
                return
            setup_id = uuid.uuid4().hex
            SETUP_ACTIVE = setup_id
            SETUP_RUNS[setup_id] = _start_setup_run(setup_id, blender_install_command())
        self._send_json(202, {"setup_run_id": setup_id,
                              "events_url": f"/api/setup/run/{setup_id}/events"})

    def _start_gltfpack_install(self) -> None:
        global SETUP_ACTIVE
        with SETUP_LOCK:
            active = JOBS.get(JOBS.active)
            refusal = gltfpack_install_refusal(
                props_tools_payload(),
                generating=active is not None
                and active.status in {"queued", "running", "cancelling"},
                setting_up=SETUP_ACTIVE is not None or download_active() is not None)
            if refusal:
                self._send_json(refusal[0], {"error": refusal[1]})
                return
            setup_id = uuid.uuid4().hex
            SETUP_ACTIVE = setup_id
            SETUP_RUNS[setup_id] = _start_setup_run(setup_id, gltfpack_install_command())
        self._send_json(202, {"setup_run_id": setup_id,
                              "events_url": f"/api/setup/run/{setup_id}/events"})

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

    def _status(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(200, _job_status_payload(job))

    def _status(self, job_id: str) -> None:
        job = self._find_job(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(200, _job_status_payload(job))

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
            _killpg_if_alive(process.pid)
        job.emit({"phase": "error", "message": "Cancellation requested"})
        self._send_json(202, {"job_id": job.id, "status": "cancelling"})
