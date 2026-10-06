"""Queued finishing jobs for the browser: retopologise, repaint, compress an existing GLB.

Deliberately a sibling of `rig_api.py` rather than part of `generate_api.py`: this operates
on an asset that already exists, so it has no image-to-3D backend, no model readiness and no
setup. What it shares is the job shape — one at a time, SSE progress, artifacts fetched by
URL — so the browser can drive it with the same code it already uses.

Finishing an asset the viewer generated earlier is the point: the surface response
(`metallic`/`roughness`/`ior`) is tuned per asset by eye, and re-finishing with different
values must not mean regenerating the geometry underneath it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_to_3dlab import processes  # noqa: E402
from image_to_3dlab import blender as blender_lookup  # noqa: E402
from image_to_3dlab.host import APPLE, host_platform  # noqa: E402

OUTPUT_ROOT = REPO / "output" / "finish"
# Where generation runs (and their records) live, for matching an uploaded model to the run
# that made it.
GENERATED_ROOT = REPO / "output"
WORKER = REPO / "scripts" / "retopo_repaint.py"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
# The exact shape `create` builds — a resume takes its directory name from a URL, so
# the name is matched against the generator rather than merely sanitised.
RUN_DIRECTORY = re.compile(r"^[A-Za-z0-9_-]{1,80}__finish__\d{8}-\d{6}(?:-\d+)?$")
TERMINAL = {"done", "error", "cancelled"}

ARTIFACTS = {
    "result.glb": ("result_glb", "model/gltf-binary"),
    "record.json": ("record_path", "application/json"),
}

# Bounds, not preferences. Everything here reaches a subprocess argument, so each value is
# clamped to a range the underlying scripts actually accept — `blender_retopo_bake.py`
# rejects a face target outside 1k..200k, and a voxel fraction outside its own range melts
# the subject or exhausts memory.
SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "faces": (1000, 200000),
    "atlas": (1024, 4096),
    "angle": (1.0, 89.9),
    "voxel": (0.0, 0.05),
    "metallic": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "ior": (1.0, 3.0),
    "paint_seed": (0, 2**31 - 1),
    "paint_res": (256, 1024),
    "paint_steps": (1, 50),
    "paint_tex": (1024, 4096),
    "texture_size": (256, 4096),
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "faces": 40000, "atlas": 2048, "angle": 89.0, "voxel": 0.004,
    "metallic": 0.25, "roughness": 0.65, "ior": 1.45,
    "paint_seed": 0, "paint_res": 512, "paint_steps": 15, "paint_tex": 4096,
    "texture_size": 2048, "skip_paint": True, "skip_compress": False, "skip_bake": False,
    "skip_photo": False,
}

INTEGER_SETTINGS = {
    "faces", "atlas", "paint_seed", "paint_res", "paint_steps", "paint_tex", "texture_size",
}

# Which artifact proves a stage already ran, for `--resume` and for describing a run that
# the browser lost track of: a key into the worker's `step_paths`, or None for the final
# model. Ordered as the worker runs them.
STAGE_ARTIFACTS: tuple[tuple[str, str | None], ...] = (
    ("retopologise", "retopo"),
    ("repaint", "painted"),
    ("photo", "photo"),
    ("bake", "baked"),
    ("compress", None),
)

# How the overall bar divides between stages. Grossly uneven on purpose: a measured run on
# 2026-09-21 spent 7.8s retopologising, 324.5s repainting and 0.6s compressing, so equal
# thirds would park the bar at 33% for five of its six minutes. The bake-detail stage took
# 4.3s on the orc (2026-09-27), about half a retopology.
STAGE_WEIGHTS: dict[str, float] = {
    "retopologise": 5.0, "repaint": 90.0, "photo": 1.0, "bake": 2.0, "compress": 3.0,
}

# Where the repaint's own log lines fall inside that stage. Same run: ~30s of setup, 165s
# of diffusion steps, 9s decoding views and 72s of super-res. The diffusion steps are
# barely half the stage, so finishing them must not park the bar at 100% -- that silent
# half-minute is exactly what reads as a hang.
PAINT_SETUP_FRACTION = 0.08
PAINT_STEPS_END_FRACTION = 0.65
PAINT_MARKS: tuple[tuple[str, float], ...] = (
    ("controls + dino ready", PAINT_SETUP_FRACTION),
    ("views decoded", 0.70),
    ("super-res", 0.78),
)
PAINT_STEP = re.compile(r"^\s*step (\d+)/(\d+)\b")


def normalise_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge client settings over the defaults, rejecting anything out of range.

    Raises ValueError with the offending key. Unknown keys are ignored rather than passed
    through — a typo must not silently reach a subprocess argument.
    """
    settings = dict(DEFAULT_SETTINGS)
    for key, value in raw.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key in {"skip_paint", "skip_compress", "skip_bake", "skip_photo"}:
            settings[key] = bool(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
        low, high = SETTING_BOUNDS[key]
        if not low <= number <= high:
            raise ValueError(f"{key} must be within {low}..{high}, got {number}")
        settings[key] = int(number) if key in INTEGER_SETTINGS else number
    return settings


REPAINT_NOTE = ("Repaint runs on Apple Silicon only for now. Here Pixel Match still keeps "
                "the front exact; the sides and back keep the generator's own paint.")


def capabilities(platform: str | None = None, find=blender_lookup.find_blender,
                 version=blender_lookup.blender_version,
                 installable=blender_lookup.can_install) -> dict[str, Any]:
    """What Finish can do on this machine, for the page to say before anyone clicks."""
    repaint = (platform or host_platform()) == APPLE
    found = find()
    found_version = version(found) if found else None
    return {
        "repaint": repaint,
        "repaint_note": None if repaint else REPAINT_NOTE,
        "blender": str(found) if found else None,
        "blender_version": ".".join(map(str, found_version)) if found_version else None,
        "blender_problem": (blender_lookup.missing_help() if found is None
                            else blender_lookup.version_problem(found_version)),
        "ready": found is not None,
        # Setup's Blender card offers Install Blender (scripts/bootstrap_blender.py).
        "blender_installable": found is None and installable(),
    }


def fit_to_machine(settings: dict[str, Any], platform: str | None = None) -> dict[str, Any]:
    """Settings this machine can actually run: no repaint off Apple Silicon (it is MLX)."""
    if (platform or host_platform()) != APPLE:
        return {**settings, "skip_paint": True}
    return settings


def worker_module():
    """The worker, imported, so the stage plan has one definition rather than two.

    Its module level is constants and an argparse builder; the expensive imports live
    inside `main`, so this costs nothing and cannot drift from what actually runs.
    """
    spec = importlib.util.spec_from_file_location("retopo_repaint", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_command(
    job: FinishJob, settings: dict[str, Any], *, resume: bool = False,
) -> list[str]:
    """The worker invocation. Flags, not positions — the worker owns its own ordering."""
    command = [
        sys.executable, "-u", str(WORKER),
        str(job.asset_path), str(job.image_path), str(job.result_glb),
        *(["--steps-dir", str(job.steps_dir)] if job.steps_dir is not None else []),
        *(["--views", str(job.views_dir)] if uses_photo(job, settings) else []),
        "--faces", str(settings["faces"]),
        "--atlas", str(settings["atlas"]),
        "--angle", str(settings["angle"]),
        "--voxel", str(settings["voxel"]),
        "--metallic", str(settings["metallic"]),
        "--roughness", str(settings["roughness"]),
        "--ior", str(settings["ior"]),
        "--paint-seed", str(settings["paint_seed"]),
        "--paint-res", str(settings["paint_res"]),
        "--paint-steps", str(settings["paint_steps"]),
        "--paint-tex", str(settings["paint_tex"]),
        "--texture-size", str(settings["texture_size"]),
    ]
    if settings["skip_paint"]:
        command.append("--skip-paint")
    if settings["skip_compress"]:
        command.append("--skip-compress")
    if settings["skip_bake"]:
        command.append("--skip-bake")
    if resume:
        command.append("--resume")
    return command


def uses_photo(job: "FinishJob", settings: dict[str, Any]) -> bool:
    """Whether this run gets the photo stage: a camera was found and it was not turned off."""
    return not settings.get("skip_photo", False) and (job.views_dir / "transforms.json").is_file()


def photo_note(job: "FinishJob", settings: dict[str, Any]) -> str | None:
    """Why Pixel Match did not run, when it was asked for and could not: None otherwise.

    Said at the start and again on the finished result, because a ticked box with no stage
    and a one-second start-up line read as Pixel Match having run."""
    if settings.get("skip_photo", False) or uses_photo(job, settings):
        return None
    return ("Pixel Match skipped: no saved camera for this model. It works on Pixal3D "
            "models generated in this lab, which keep their camera beside the .glb.")


def find_source_views(asset: bytes, root: Path = GENERATED_ROOT) -> Path | None:
    """The saved camera + view directory of the Pixal3D run that produced `asset`.

    The browser uploads bytes, not a path, so the model is matched by content: every
    Pixal3D run writes `<output>.json` with the GLB's sha256, and its staged views sit
    beside the GLB as `<output>.svviews/`. Anything else (another generator, a model from
    elsewhere, an edited file) has no match and simply gets no photo stage.
    """
    digest = hashlib.sha256(asset).hexdigest()
    if not root.is_dir():
        return None
    for record_path in root.rglob("*.json"):
        if record_path.stat().st_size > 20000:
            continue
        record = _read_json(record_path)
        if not isinstance(record, dict) or record.get("backend") != "pixal3d":
            continue
        output = record.get("output") or {}
        if output.get("sha256") != digest:
            continue
        views = Path(output.get("path", "")).with_suffix(".svviews")
        if not (views / "transforms.json").is_file():
            views = record_path.with_suffix(".svviews")
        if (views / "transforms.json").is_file():
            return views
    return None


def stage_bands(stages: list[str]) -> dict[str, tuple[float, float]]:
    """Slice 0..100 between the stages this run will actually execute."""
    active = [stage for stage in stages if stage in STAGE_WEIGHTS]
    total = sum(STAGE_WEIGHTS[stage] for stage in active) or 1.0
    bands: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for stage in active:
        cursor_end = cursor + STAGE_WEIGHTS[stage] / total * 100.0
        bands[stage] = (cursor, cursor_end)
        cursor = cursor_end
    return bands


def remaining_seconds(elapsed: float, fraction: float) -> float | None:
    """Seconds still to go, extrapolated from how long a fraction took.

    Linear and self-correcting rather than a table of per-stage constants: the repaint
    scales with paint steps, paint resolution and face count, so any constant would be
    wrong for most runs. None until there is enough evidence to be worth showing.
    """
    if not 0.02 < fraction < 1.0 or elapsed <= 1.0:
        return None
    return round(elapsed * (1.0 - fraction) / fraction, 1)


class FinishProgress:
    """Turn one line of worker stdout into a progress event, or None.

    Two line shapes carry progress. `I2L_STAGE::phase::message` marks a stage boundary,
    and the repaint prints its own `step 7/15 107s` as it denoises. The second is the one
    that matters: the repaint is ~95% of the wall clock, and without it the panel shows a
    single row reading "estimating…" for six minutes, with nothing to separate a working
    run from a wedged one.

    The worker's final `I2L_STAGE::done` marker is deliberately swallowed. Only `run_job`
    knows the artifact URLs, so only `run_job` may emit the job's terminal event; letting
    the worker's marker through made the browser close its event stream one event early
    and render a download link pointing at `undefined` (found 2026-09-21, on a run whose
    finished GLB had been sitting complete on disk the whole time).
    """

    def __init__(self, stages: list[str], clock=time.monotonic):
        self.bands = stage_bands(stages)
        self.clock = clock
        self.started = clock()
        self.phase: str | None = None
        self.stage_started = self.started
        self.stage_fraction = 0.0

    def feed(self, line: str) -> dict[str, Any] | None:
        if line.startswith("I2L_STAGE::"):
            _, phase, message = line.split("::", 2)
            if phase == "done":
                return None
            self.phase = phase
            self.stage_started = self.clock()
            self.stage_fraction = 0.0
            return self._event(message)
        if self.phase != "repaint":
            return None
        match = PAINT_STEP.match(line)
        if match:
            step, total = int(match.group(1)), int(match.group(2))
            span = PAINT_STEPS_END_FRACTION - PAINT_SETUP_FRACTION
            fraction = PAINT_SETUP_FRACTION + span * step / max(total, 1)
            return self._advance(
                fraction, f"Denoising step {step}/{total}", step=step, total=total,
            )
        for needle, fraction in PAINT_MARKS:
            if needle in line:
                return self._advance(fraction, line.strip())
        return None

    def _advance(self, fraction: float, message: str, **extra: Any) -> dict[str, Any]:
        # Never backwards. A bar that retreats reads as a restart, and the paint stage's
        # marks do not arrive in a guaranteed order across resolutions.
        self.stage_fraction = max(self.stage_fraction, min(fraction, 1.0))
        return self._event(message, measured=True, **extra)

    def _event(self, message: str, *, measured: bool = False, **extra: Any) -> dict[str, Any]:
        low, high = self.bands.get(self.phase or "", (0.0, 100.0))
        overall = low + (high - low) * self.stage_fraction
        event: dict[str, Any] = {
            "phase": self.phase,
            "message": message,
            "overall_pct": round(overall, 1),
            "total_eta_seconds": remaining_seconds(
                self.clock() - self.started, overall / 100.0,
            ),
            **extra,
        }
        # A stage with no sub-progress of its own (retopology, compress) reports none:
        # a row frozen at "0%" is a worse lie than a row that simply says it is running.
        if measured:
            event["stage_pct"] = round(self.stage_fraction * 100)
            event["stage_eta_seconds"] = remaining_seconds(
                self.clock() - self.stage_started, self.stage_fraction,
            )
        return event


# Files only a pre-0.3.5 (flat) run has at its top level.
LEGACY_MARKERS = ("source.glb", "source.png", "settings.json", "result.glb")


def faces_label(faces: int) -> str:
    """A face count the way people say it: 5000 -> "5k", 12500 -> "12.5k", 800 -> "800"."""
    if faces < 1000:
        return str(faces)
    return f"{faces / 1000:g}k"


def final_name(directory_name: str, faces: int) -> str:
    """The finished model's file name: the asset, then its face count (`vanguard_5k.glb`).

    The asset is the run folder's name up to its first `__`, which drops the generator
    and timestamp tags a generated file carries (`vanguard__pixal3d` -> `vanguard`).
    """
    return f"{directory_name.split('__')[0]}_{faces_label(faces)}.glb"


class FinishJob:
    """One run's folder. Laid out so the finished model is the only GLB at the top:

        vanguard_5k.glb                  the result
        vanguard_5k.retopo-repaint.json  what made it
        input/   source.glb, source.png, source.views/, settings.json
        steps/   1_retopo.glb, 2_painted.glb, 3_photo.glb, 4_baked.glb, logs, run.log

    Runs from before 0.3.5 kept everything flat (`result.glb` among `result_*.glb` and a
    `source.glb`); they are recognised by any of those top-level files and still read.
    """

    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.legacy = any((directory / name).exists() for name in LEGACY_MARKERS)
        self.input_dir = directory if self.legacy else directory / "input"
        self.steps_dir = None if self.legacy else directory / "steps"
        self.asset_path = self.input_dir / "source.glb"
        self.image_path = self.input_dir / "source.png"
        # The source photo(s) and cameras for the photo stage, copied in from the Pixal3D
        # run so a resume or a re-run does not depend on that run still being on disk.
        self.views_dir = self.input_dir / "source.views"
        self.settings_path = self.input_dir / "settings.json"
        self.log_path = (self.steps_dir or directory) / "run.log"
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.resume = False
        self.log_lines: deque[str] = deque(maxlen=300)

    @property
    def result_glb(self) -> Path:
        if self.legacy:
            return self.directory / "result.glb"
        return self.directory / final_name(self.directory.name, int(self.settings["faces"]))

    @property
    def record_path(self) -> Path:
        return self.result_glb.with_suffix(".retopo-repaint.json")

    def stage_artifact(self, key: str | None) -> Path:
        """The file that proves a stage ran (see STAGE_ARTIFACTS)."""
        if key is None:
            return self.result_glb
        return worker_module().step_paths(self.result_glb, self.steps_dir)[key]

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


class FinishJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT, generated_root: Path = GENERATED_ROOT):
        self.output_root = output_root
        self.generated_root = generated_root
        self.jobs: dict[str, FinishJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def create(
        self, asset_name: str, asset: bytes, image: bytes, settings: dict[str, Any],
    ) -> FinishJob:
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a finishing job is already running")
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__finish__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__finish__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = FinishJob(uuid.uuid4().hex, directory)
            try:
                job.settings = fit_to_machine(normalise_settings(settings))
                job.input_dir.mkdir(parents=True, exist_ok=True)
                job.asset_path.write_bytes(asset)
                job.image_path.write_bytes(image)
                views = None if job.settings["skip_photo"] else find_source_views(
                    asset, self.generated_root)
                if views is not None:
                    shutil.copytree(views, job.views_dir)
                # Written up front, not with the record at the end: a run that dies
                # halfway is exactly the one worth resuming, and resuming it means
                # knowing the settings the surviving intermediates were made with.
                job.settings_path.write_text(json.dumps(job.settings, indent=2))
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def adopt(self, name: str) -> FinishJob:
        """A job bound to an existing run directory, to be run with `--resume`.

        The settings come from the directory rather than the request. Re-tuning on resume
        would silently mix a new face target with an old retopology; a changed setting is
        a new run, and the browser offers it as one.
        """
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a finishing job is already running")
            directory = _run_directory(self.output_root, name)
            job = FinishJob(uuid.uuid4().hex, directory)
            if not job.settings_path.is_file():
                raise RuntimeError(f"{name} predates resume support and has no settings.json")
            for required in (job.asset_path, job.image_path):
                if not required.is_file():
                    raise RuntimeError(f"{name} is missing {required.name}")
            job.settings = fit_to_machine(
                normalise_settings(json.loads(job.settings_path.read_text())))
            job.resume = True
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> FinishJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: FinishJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


FINISH_JOBS = FinishJobManager()


def run_job(job: FinishJob, manager: FinishJobManager = FINISH_JOBS) -> None:
    if not WORKER.is_file():
        job.status = "error"
        job.emit({"phase": "error", "message": f"worker missing: {WORKER}"})
        manager.finish(job)
        return
    try:
        job.status = "running"
        stages = worker_module().stage_plan(
            job.settings["skip_paint"], job.settings["skip_compress"],
            job.settings.get("skip_bake", False), photo=uses_photo(job, job.settings),
        )
        if job.resume:
            message = "Resuming from what is already on disk"
        elif "photo" in stages:
            message = "Starting: found the source camera, so Pixel Match will keep the photo's pixels"
        elif photo_note(job, job.settings):
            message = photo_note(job, job.settings)
        else:
            message = "Starting"
        job.emit({"phase": "queued", "overall_pct": 0, "stages": stages, "message": message})
        progress = FinishProgress(stages)
        job.process = subprocess.Popen(
            build_command(job, job.settings, resume=job.resume), cwd=str(REPO),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            **processes.group_popen_kwargs(),
        )
        job.directory.joinpath("pid").write_text(processes.pid_record(job.process.pid))
        assert job.process.stdout is not None
        for raw in job.process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            job.append_log(line)
            event = progress.feed(line)
            if event:
                job.emit(event)
        return_code = job.process.wait()
        job.append_log(f"worker exited with code {return_code}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Finishing cancelled"})
        elif return_code != 0:
            job.status = "error"
            job.emit({
                "phase": "error", "message": f"worker exited with code {return_code}",
                "log_tail": "\n".join(job.log_lines)[-8000:],
            })
        elif job.result_glb.is_file():
            job.status = "done"
            job.emit({
                "phase": "done", "overall_pct": 100,
                "message": f"Finished at {job.result_glb.stat().st_size / 1048576:.1f} MB",
                "result_url": f"/api/finish/{job.id}/result.glb",
                "record_url": f"/api/finish/{job.id}/record.json",
                "size_bytes": job.result_glb.stat().st_size,
                "directory": job.directory.name,
                "path": library_path(job.result_glb),
                "source_url": served_url(job.asset_path),
                "pixal3d": (job.views_dir / "transforms.json").is_file(),
                "note": photo_note(job, job.settings),
            })
        else:
            job.status = "error"
            job.emit({"phase": "error", "message": "worker exited without writing a GLB"})
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        job.directory.joinpath("pid").unlink(missing_ok=True)
        manager.finish(job)


def library_path(path: Path) -> str | None:
    """A file's path inside output/, as the studio's Library names it (the next step's input)."""
    try:
        return path.relative_to(OUTPUT_ROOT.parent).as_posix()
    except ValueError:
        return None


def cancel_job(job: FinishJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: FinishJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "settings": job.settings,
        "directory": job.directory.name,
        "log_tail": "\n".join(job.log_lines)[-4000:],
        "last_event": job.events[-1] if job.events else None,
    }


def _run_directory(root: Path, name: str) -> Path:
    """One run directory under `root`, by name, with no way out of it.

    The name reaches the filesystem straight from a URL, so it is matched against what
    `create` actually produces rather than merely scrubbed of "..".
    """
    if not RUN_DIRECTORY.fullmatch(name):
        raise RuntimeError(f"not a run directory name: {name!r}")
    directory = root / name
    if not directory.is_dir():
        raise RuntimeError(f"no such run: {name}")
    return directory


def describe_run(directory: Path) -> dict[str, Any]:
    """What survives of one run on disk, and whether it is worth resuming.

    Every stage leaves its artifact behind, so the directory alone says how far the run
    got — no server memory required, which is the point: jobs live in this process and a
    finished run must still be recoverable after a restart.
    """
    layout = FinishJob("", directory)
    settings = _read_json(layout.settings_path)
    if settings is not None:
        layout.settings.update(settings)
    complete = [
        stage for stage, key in STAGE_ARTIFACTS if layout.stage_artifact(key).is_file()
    ]
    result = layout.result_glb
    record = _read_json(layout.record_path)
    return {
        "directory": directory.name,
        "modified": directory.stat().st_mtime,
        "stages_complete": complete,
        "finished": result.is_file(),
        "size_bytes": result.stat().st_size if result.is_file() else None,
        "result_url": served_url(result) if result.is_file() else None,
        "source_url": served_url(layout.asset_path) if layout.asset_path.is_file() else None,
        # A copied-in camera means the source was a Pixal3D run (it faces the other way).
        "pixal3d": (layout.views_dir / "transforms.json").is_file(),
        "record_url": served_url(layout.record_path) if record else None,
        "seconds": (record or {}).get("seconds"),
        "settings": settings,
        # A finished run has nothing left to resume; one without its sources or its
        # settings cannot be resumed faithfully, so it is not offered.
        "resumable": (
            not result.is_file()
            and settings is not None
            and layout.asset_path.is_file()
            and layout.image_path.is_file()
        ),
    }


def list_runs(root: Path = OUTPUT_ROOT, limit: int = 25) -> list[dict[str, Any]]:
    """Every finishing run on disk, newest first."""
    if not root.is_dir():
        return []
    directories = sorted(
        (d for d in root.iterdir() if d.is_dir() and RUN_DIRECTORY.fullmatch(d.name)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [describe_run(d) for d in directories[:limit]]


def served_url(path: Path) -> str | None:
    """The URL the viewer's static handler serves this file at, if it serves it at all.

    The handler is rooted at the repository, so anything outside it has no URL — say so
    rather than emitting a link that would 404 in the browser.
    """
    try:
        return "/" + path.relative_to(REPO).as_posix()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "asset"
