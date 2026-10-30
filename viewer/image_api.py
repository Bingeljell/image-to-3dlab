"""Text to image, one step upstream of everything else in this repo.

Every other tool here starts from a picture. This one makes the picture, for people who
do not have one yet. It drives `stable-diffusion.cpp` (a prebuilt Metal binary in
`vendor/sdcpp/`) against one of two picture engines in GGUF form (`ENGINES`):

- **FLUX.2 [klein] 4B**, the default. Apache-2.0, about 1 minute a picture at 1024 on the
  Mac it was measured on, and it draws a back view by reading the front directly, so a
  front + back build needs nothing more (bake-off 2026-10-28/29, journal/todo.md).
- **Qwen-Image 2.1**, under the Qwen Research License. Follows long prompts more closely;
  its edits (back views, skins) need the vision projector as well.

The module-level constants below (`DEFAULTS`, `WEIGHT_FILES`, ...) are Qwen's, as they
always were; Flux's live on its `PictureEngine`.

Two things are deliberate:

**The defaults are the measured fast ones, not the upstream ones.** The model's own recipe
says `--cfg-scale 6.0`, which makes it do two passes per step for nothing, because
Qwen-Image 2.1 does not use guidance. Measured on one Mac on 2026-09-22: cfg 6.0 at 1024px
took 17 minutes, cfg 1.0 took 11, and 768px at 10 steps took 4m22s for the same picture.
So the defaults here are cfg 1.0, 768px and 8 steps: 8 was then tested by eye against 10
and lost nothing visible (2026-10). **Those timings are one machine.** A
faster chip does better and an NVIDIA card is in a different league; nothing here should
be read as "this is how long it takes".

**The licence travels with the image.** The pictures Qwen-Image makes are yours, per Qwen's
own statement. Every run writes a sidecar saying so, with the licence, because
a PNG in a folder six months from now remembers nothing on its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import usage_api

from image_to_3dlab import back_view, matte
from image_to_3dlab.host import APPLE, executable, host_platform, with_rocm_libraries
from image_to_3dlab.provenance import FLUX_OUTPUT_RIGHTS, LICENSES, QWEN_OUTPUT_RIGHTS
from image_to_3dlab.sdcpp import BackendWatch, no_gpu_help

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
BINARY = executable(REPO / "vendor" / "sdcpp", "sd-cli")
# Every picture lands in output/images/. Its licence (Qwen's research licence, today) is
# recorded in the provenance written beside it, which is what travels with the file; a
# folder name did not, and sorting by it only made the pictures harder to find.
OUTPUT_ROOT = REPO / "output" / "images"

MODEL_ID = "qwen-image-2.1"
LICENSE_NAME = "Qwen Research License"
LICENSE_URL = "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
ATTRIBUTION = "Built with Qwen"
OUTPUT_RIGHTS = QWEN_OUTPUT_RIGHTS

# Each weight file, as (hugging face cache directory, filename glob).
WEIGHT_FILES = {
    "diffusion_model": ("models--leejet--Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q8_0.gguf"),
    "llm": ("models--Qwen--Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
    "vae": ("models--Comfy-Org--Qwen-Image-2.1", "qwen_image_2.1_vae_bf16.safetensors"),
}

# Edit mode (a reference picture, such as the front for a back view) needs the LLM's vision
# projector as well. Optional: plain pictures do not, so it is not part of WEIGHT_FILES.
VISION_FILE = ("models--Qwen--Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")

DEFAULTS: dict[str, Any] = {
    "width": 768,
    "height": 768,
    "steps": 8,
    "cfg_scale": 1.0,
    "sampler": "euler",
    "seed": 42,
    "negative_prompt": "",
}

SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpmpp2m", "lcm")
# sd.cpp requires both dimensions to be a multiple of 32; anything else fails deep in the
# run rather than at the door.
SIZE_STEP = 32
MIN_SIZE, MAX_SIZE = 256, 1536
MAX_STEPS = 50
MAX_PROMPT = 2000
PICTURE_FILES = {".png", ".jpg", ".jpeg", ".webp"}
# The largest picture anything here decodes from a request: far past what any engine draws,
# well short of the memory a crafted file could ask for (Pillow only refuses past ~179M pixels).
MAX_PICTURE_SIDE = 4096
# The brush (repaint one area of a picture): measured 2026-10-30 on Flux klein, a satchel
# recoloured cleanly at 8 steps and strength 0.9 (output/images/brush-test-2026-10-30/).
BRUSH_STEPS = 8
BRUSH_STRENGTH = 0.9
# The model draws inside the stroke grown by a few pixels, and exactly that area is pasted
# back, feathered at its edge so no seam shows. The two must match: with a paste area
# smaller than the drawn one (12 px against 4, 2026-10-30) whatever the model drew in the
# gap was cut off in a hard line.
BRUSH_GROW = 6
BRUSH_FEATHER = 3

# "  |=====>    | 7/10 - 13.38s/it"
PROGRESS = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*([\d.]+)s/it")
COMPLETED = re.compile(r"generate_image completed in ([\d.]+)s")
DECODE = re.compile(r"decode_first_stage completed, taking ([\d.]+)s")


@dataclass(frozen=True)
class PictureEngine:
    """One model sd-cli can draw with: its files, its defaults and its licence."""

    id: str                       # what Create, agents and the picture's record call it
    label: str
    setup_id: str                 # its card on Setup & Status (viewer/backend_catalog.py)
    licence_key: str              # its reviewed profile in provenance.LICENSES
    attribution: str | None
    output_rights: str
    weight_files: dict[str, tuple[str, str]]
    defaults: dict[str, Any]
    # More steps when the picture is the end product rather than a 3D build's source: Flux at
    # 8 steps took 103 s against 62 s at 4 for a near-identical picture (2026-10-28).
    picture_only_steps: int | None = None
    # What an edit (a back view, a skin) needs beyond the weights. None: the engine reads the
    # reference picture by itself, as Flux does.
    vision_file: tuple[str, str] | None = None

    @property
    def licence_name(self) -> str:
        return LICENSES[self.licence_key].license_name

    @property
    def licence_url(self) -> str:
        return LICENSES[self.licence_key].license_url


QWEN = PictureEngine(
    id="qwen", label="Qwen-Image 2.1", setup_id="qwen-image", licence_key=MODEL_ID,
    attribution=ATTRIBUTION, output_rights=OUTPUT_RIGHTS, weight_files=WEIGHT_FILES,
    defaults=DEFAULTS, vision_file=VISION_FILE,
)

FLUX = PictureEngine(
    id="flux", label="FLUX.2 [klein] 4B", setup_id="flux-image", licence_key="flux2-klein-4b",
    attribution=None, output_rights=FLUX_OUTPUT_RIGHTS,
    weight_files={
        "diffusion_model": ("models--leejet--FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
        "llm": ("models--unsloth--Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
        "vae": ("models--Comfy-Org--Flux-2-Klein-4B", "flux2-vae.safetensors"),
    },
    # 1024 and 4 steps: ~62 s a picture on the Mac it was measured on, 11 s of it decoding.
    # klein is distilled, so guidance stays at 1.0 like Qwen's.
    defaults={**DEFAULTS, "width": 1024, "height": 1024, "steps": 4},
    picture_only_steps=8,
)

# Flux first: the default when it is installed, and the one a fresh machine is pointed at.
ENGINES = {engine.id: engine for engine in (FLUX, QWEN)}


class MissingWeights(RuntimeError):
    """Raised with the names of what is absent, so the page can say what to fetch."""

    def __init__(self, missing: list[str], engine: PictureEngine | None = None):
        self.missing = missing
        super().__init__(
            f"{(engine or QWEN).label} is not set up yet. Missing: " + ", ".join(missing)
            + ". Open Setup & Status to download it."
        )


def find_weight(directory: str, filename: str) -> Path | None:
    """Locate one cached file, whatever snapshot hash it landed under."""
    root = HF_HUB / directory
    if not root.exists():
        return None
    matches = sorted(root.glob(f"snapshots/*/**/{filename}"))
    return matches[0] if matches else None


def resolve_weights(engine: PictureEngine = QWEN) -> dict[str, Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, (directory, filename) in engine.weight_files.items():
        path = find_weight(directory, filename)
        if path is None:
            missing.append(filename)
        else:
            found[key] = path
    if missing:
        raise MissingWeights(missing, engine)
    return found


def resolve_vision(engine: PictureEngine = QWEN) -> Path | None:
    """The vision projector edit mode needs, or None when it has not been downloaded."""
    return find_weight(*engine.vision_file) if engine.vision_file else None


def can_edit(engine: PictureEngine) -> bool:
    """Whether `engine` can redraw a reference picture (a back view, a skin) here and now."""
    if engine.vision_file is None:
        return True
    return resolve_vision(engine) is not None


def engine_of(picture: Path) -> PictureEngine | None:
    """The engine that drew `picture`, from its provenance sidecar; None for an upload or an
    older picture with no record. A redraw uses it by default, so one model's pictures match."""
    try:
        record = json.loads(picture.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model = ((record or {}).get("model") or {}).get("id") if isinstance(record, dict) else None
    return next((e for e in ENGINES.values() if e.licence_key == model), None)


def usable(engine: PictureEngine, *, edit: bool = False) -> bool:
    try:
        check_ready(engine, edit=edit)
    except MissingWeights:
        return False
    return True


def choose_engine(asked: str | None, picture: Path, *, edit: bool, prefer: str | None = None) -> PictureEngine:
    """Who draws a back or a brush. Asked for by name: that one, whatever its state (the run then
    says what to set up). Otherwise the preferred one, else the engine that drew `picture`, when it
    can run here; else any engine that can."""
    if asked:
        return pick_engine(asked)
    preferred = pick_engine(prefer) if prefer else engine_of(picture)
    if preferred is not None and usable(preferred, edit=edit):
        return preferred
    return pick_engine(edit=edit)


def check_ready(engine: PictureEngine, *, edit: bool = False) -> None:
    """Raise MissingWeights naming what `engine` still needs; return quietly when it can run."""
    resolve_weights(engine)
    if edit and not can_edit(engine):
        raise MissingWeights([engine.vision_file[1]], engine)


def is_installed(engine: PictureEngine = QWEN) -> bool:
    """Both halves, build and weights, the way the rest of the viewer separates them."""
    if not BINARY.exists():
        return False
    try:
        resolve_weights(engine)
    except MissingWeights:
        return False
    return True


def pick_engine(requested: str | None = None, *, edit: bool = False,
                ready: Callable[[PictureEngine], bool] | None = None) -> PictureEngine:
    """The engine to draw with: the one asked for, else the first in `ENGINES` that is set up.

    With nothing set up it answers Flux, so the "not set up" message points at the smaller,
    Apache-licensed download. `edit` asks for an engine that can redraw a reference.
    """
    if requested:
        if requested not in ENGINES:
            raise ValueError(f"no picture engine {requested!r}; choose {' or '.join(ENGINES)}")
        return ENGINES[requested]
    if ready is None:
        def ready(engine: PictureEngine) -> bool:
            try:
                check_ready(engine, edit=edit)
            except MissingWeights:
                return False
            return True
    return next((engine for engine in ENGINES.values() if ready(engine)), FLUX)


def clean_settings(raw: dict[str, Any], engine: PictureEngine = QWEN,
                   picture_only: bool = False) -> dict[str, Any]:
    """Coerce whatever the form sent into something sd-cli will accept.

    Sizes are rounded to a multiple of 32 rather than rejected: a user typing 700 means
    "about this big", and failing them for it would be pedantry. Everything else is
    clamped to a range the binary tolerates. Anything left out takes `engine`'s default;
    a `picture_only` run (no 3D build after it) takes its picture-only steps.
    """
    defaults = dict(engine.defaults)
    if picture_only and engine.picture_only_steps:
        defaults["steps"] = engine.picture_only_steps
    settings = dict(defaults)
    for key in ("width", "height"):
        value = raw.get(key, defaults[key])
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = defaults[key]
        number = max(MIN_SIZE, min(MAX_SIZE, number))
        settings[key] = max(SIZE_STEP, round(number / SIZE_STEP) * SIZE_STEP)
    try:
        settings["steps"] = max(1, min(MAX_STEPS, int(raw.get("steps", defaults["steps"]))))
    except (TypeError, ValueError):
        settings["steps"] = defaults["steps"]
    try:
        settings["cfg_scale"] = max(0.0, min(20.0, float(raw.get("cfg_scale", defaults["cfg_scale"]))))
    except (TypeError, ValueError):
        settings["cfg_scale"] = defaults["cfg_scale"]
    try:
        settings["seed"] = int(raw.get("seed", defaults["seed"]))
    except (TypeError, ValueError):
        settings["seed"] = defaults["seed"]
    sampler = str(raw.get("sampler", defaults["sampler"]))
    settings["sampler"] = sampler if sampler in SAMPLERS else defaults["sampler"]
    settings["negative_prompt"] = str(raw.get("negative_prompt", ""))[:MAX_PROMPT]
    return settings


def build_command(prompt: str, settings: dict[str, Any], output_path: Path,
                  weights: dict[str, Path], binary: Path = BINARY,
                  reference: Path | None = None, vision: Path | None = None,
                  engine: PictureEngine = QWEN, init: Path | None = None,
                  mask: Path | None = None, strength: float = BRUSH_STRENGTH) -> list[str]:
    """The full argument list, as a pure function so a test can read it without a GPU.

    With `reference`, an edit. Qwen reads that picture through the `vision` projector
    (VISION_FILE), which is then required; Flux reads it by itself.

    `--offload-to-cpu` is deliberately absent. The upstream recipe includes it, but it
    shuttles weights between RAM and VRAM to save video memory a unified-memory Mac does
    not have separately; measured, it was both slower and used MORE peak RAM.
    """
    command = [
        str(binary),
        "--diffusion-model", str(weights["diffusion_model"]),
        "--vae", str(weights["vae"]),
        "--llm", str(weights["llm"]),
        "-p", prompt,
        "--cfg-scale", str(settings["cfg_scale"]),
        "--sampling-method", settings["sampler"],
        "--steps", str(settings["steps"]),
        "-W", str(settings["width"]),
        "-H", str(settings["height"]),
        "--seed", str(settings["seed"]),
        "--diffusion-fa",
        "-v",
        "-o", str(output_path),
    ]
    if settings.get("negative_prompt"):
        command[command.index("-p") + 2:command.index("-p") + 2] = [
            "--negative-prompt", settings["negative_prompt"]
        ]
    if init is not None:
        # a brush: start from `init` and redraw only where `mask` is white
        command += ["-i", str(init), "--strength", str(strength)]
        if mask is not None:
            command += ["--mask", str(mask)]
    if reference is not None:
        command += ["-r", str(reference)]
        if engine.vision_file is not None:
            if vision is None:
                raise ValueError("an edit needs the vision projector (VISION_FILE)")
            command += ["--llm_vision", str(vision)]
    return command


def parse_progress(line: str) -> dict[str, Any] | None:
    """Turn one line of sd-cli chatter into an event, or None if it says nothing useful.

    sd-cli redraws its progress bar with carriage returns, so a single read can carry
    several updates; callers split on both \\r and \\n before calling this.
    """
    match = PROGRESS.search(line)
    if match:
        current, total, seconds = int(match[1]), int(match[2]), float(match[3])
        remaining = max(0, total - current) * seconds
        return {
            "phase": "sampling",
            "step": current,
            "total_steps": total,
            "seconds_per_step": seconds,
            "percent": round(100.0 * current / total, 1) if total else 0.0,
            "eta_seconds": round(remaining, 1),
        }
    match = DECODE.search(line)
    if match:
        return {"phase": "decoding", "decode_seconds": float(match[1]), "percent": 100.0}
    match = COMPLETED.search(line)
    if match:
        return {"phase": "finished", "generate_seconds": float(match[1]), "percent": 100.0}
    return None


def failure_reason(log_lines) -> str | None:
    """sd-cli's last meaningful words, or None if it said nothing but progress.

    A failed run used to surface as "sd-cli exited with code 2" while the actual
    explanation -- a rejected flag, a missing file -- sat unread in the log tail, and
    the image tab has no log panel to find it in. So the one line that matters goes
    into the error itself; it is shown as plain text, hence one line only.
    """
    for line in reversed(list(log_lines)):
        text = line.strip()
        if not text or PROGRESS.search(text):
            continue
        return text
    return None


def slug(prompt: str, limit: int = 40) -> str:
    """A short folder-safe name taken from the prompt's first few words."""
    words = re.sub(r"[^a-z0-9\s-]", "", prompt.lower()).split()
    name = "-".join(words)[:limit].strip("-")
    return name or "image"


def weight_manifest(engine: PictureEngine = QWEN) -> dict[str, dict[str, str]]:
    """Which repository and file each part of the model came from.

    Named fields rather than the raw WEIGHT_FILES tuples: a sidecar that records
    "('models--leejet--Qwen-Image-2.1-GGUF', 'qwen_image_2.1-Q8_0.gguf')" as a single
    string is a tuple's repr leaking into a record meant to be read by people.
    """
    return {
        key: {"cache_dir": directory, "file": filename}
        for key, (directory, filename) in engine.weight_files.items()
    }


def provenance(prompt: str, settings: dict[str, Any], seconds: float,
               output_path: Path, engine: PictureEngine = QWEN) -> dict[str, Any]:
    """What this picture is, and what the licence lets you do with it."""
    return {
        "schema_version": 1,
        "kind": "generated-image",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "id": engine.licence_key,
            "runtime": "stable-diffusion.cpp (Metal)",
            "weights": weight_manifest(engine),
        },
        "license": {
            "name": engine.licence_name,
            "url": engine.licence_url,
            "classification": LICENSES[engine.licence_key].classification,
            "attribution": engine.attribution,
            # Key kept for sidecar compatibility; it now says what derivatives do NOT inherit.
            "inherited_by_derivatives": engine.output_rights,
        },
        "prompt": prompt,
        "settings": settings,
        "elapsed_seconds": round(seconds, 1),
        "output": output_path.name,
    }


@dataclass
class ImageJob:
    id: str
    directory: Path
    output_path: Path
    prompt: str
    settings: dict[str, Any]
    reference: Path | None = None   # an edit: the engine reads this picture (a back view's front)
    mode: str = "edit (back view)"  # what kind of edit, for the provenance record
    engine: PictureEngine = QWEN
    # A redraw: the picture lands at output_path first and takes this place only once it is
    # done, so a failed or cancelled try leaves the old one where it was.
    replaces: Path | None = None
    instruction: str | None = None  # what the person asked to change, for the record
    init: Path | None = None        # a brush: the picture it starts from
    mask: Path | None = None        # ...the area it may redraw (white), as the model sees it
    paste_mask: Path | None = None  # ...and the area pasted back onto `init` when it is done

    def __post_init__(self) -> None:
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.log_lines: deque[str] = deque(maxlen=400)
        self.condition = threading.Condition()
        self.process: subprocess.Popen[bytes] | None = None
        self.cancel_requested = False
        self.error: str | None = None

    def emit(self, event: dict[str, Any]) -> None:
        event = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def set_status(self, status: str, **extra: Any) -> None:
        self.status = status
        self.emit({"status": status, **extra})

    def describe(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "prompt": self.prompt,
            "settings": self.settings,
            "engine": self.engine.id,
            "error": self.error,
            "elapsed_seconds": round(time.monotonic() - self.started, 1),
            "result_url": f"/api/image/{self.id}/result.png"
            if self.status == "done" else None,
            "last_event": self.events[-1] if self.events else None,
            "log_tail": "\n".join(self.log_lines)[-4000:],
            "picture": self.picture_path() if self.status == "done" else None,
        }

    def picture_path(self) -> str:
        """Where the picture sits relative to output/, as the studio's Library names it.
        Usually images/<run>/, but a skin's tries live in skins/<id>/."""
        try:
            return self.output_path.relative_to(OUTPUT_ROOT.parent).as_posix()
        except ValueError:
            return f"images/{self.directory.name}/{self.output_path.name}"


class ImageJobManager:
    """One picture at a time. Two at once would only fight over the same memory."""

    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, ImageJob] = {}
        self.lock = threading.Lock()
        self.active: str | None = None

    def create(self, prompt: str, settings: dict[str, Any], output_path: Path | None = None,
               reference: Path | None = None, mode: str = "edit (back view)",
               engine: PictureEngine = QWEN) -> ImageJob:
        with self.lock:
            if self.active is not None:
                running = self.jobs.get(self.active)
                if running and running.status in {"queued", "running", "cancelling"}:
                    raise RuntimeError("an image is already being generated")
            if output_path is None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                name = f"{slug(prompt)}__{stamp}"
                output_path = self.output_root / name / f"{name}.png"
            directory = output_path.parent
            directory.mkdir(parents=True, exist_ok=True)
            job = ImageJob(uuid.uuid4().hex, directory, output_path, prompt, settings, reference, mode,
                           engine)
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def finish(self, job: ImageJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return False
        job.cancel_requested = True
        job.set_status("cancelling")
        if job.process and job.process.poll() is None:
            job.process.terminate()
        return True


def job_command(job: ImageJob, weights: dict[str, Path], vision: Path | None,
                binary: Path = BINARY) -> list[str]:
    """`build_command` for a job: an edit when the job has a reference picture."""
    return build_command(job.prompt, job.settings, job.output_path, weights, binary=binary,
                         reference=job.reference, vision=vision if job.reference else None,
                         engine=job.engine, init=job.init, mask=job.mask)


MAX_INSTRUCTION = 500


def back_prompt(instruction: str | None = None) -> str:
    """The back prompt, with what the person wants changed added as its own sentence.

    Their words go in as written: they name what they want, the way every prompt here should.
    """
    extra = " ".join(str(instruction or "").split())[:MAX_INSTRUCTION].strip()
    return f"{back_view.BACK_PROMPT} {extra}" if extra else back_view.BACK_PROMPT


def archive_name(path: Path, stamp: str) -> Path:
    """Where a replaced back is kept: `<front>__back_prev_<stamp>.png`, beside it."""
    return path.with_name(f"{path.stem}_prev_{stamp}{path.suffix}")


def take_place(new: Path, final: Path, stamp: str | None = None) -> Path | None:
    """Move `new` (and its sidecar) to `final`, keeping any picture already there as an
    archive. Returns where the old one went, or None when there was none."""
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    kept = None
    if final.exists():
        kept = archive_name(final, stamp)
        final.rename(kept)
        old_sidecar = final.with_suffix(".provenance.json")
        if old_sidecar.exists():
            old_sidecar.rename(kept.with_suffix(".provenance.json"))
    new.rename(final)
    return kept


def brush_masks(stroke, size: tuple[int, int]):
    """From the painted stroke (white = repaint, any size), the two masks a brush needs at
    the picture's own size: the model's (grown, hard) and the paste-back (feathered)."""
    from PIL import ImageChops, ImageFilter

    painted = stroke.convert("L").resize(size).point(lambda v: 255 if v > 127 else 0)
    grown = painted.filter(ImageFilter.MaxFilter(BRUSH_GROW * 2 + 1))
    # the same area, softened at its edge: full strength everywhere inside it, fading out just
    # past it, where the model left the picture as it was, so the fade blends like with like
    feathered = ImageChops.lighter(grown, grown.filter(ImageFilter.GaussianBlur(BRUSH_FEATHER)))
    return grown, feathered


def check_picture_size(picture) -> None:
    """Refuse a picture too large to decode safely; call it after open, before load."""
    width, height = picture.size
    if max(width, height) > MAX_PICTURE_SIDE:
        raise ValueError(f"the picture is {width}x{height}; the largest taken is "
                         f"{MAX_PICTURE_SIDE} pixels on a side")


def opened_mode(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as opened:
        return opened.mode


def opaque(path: Path):
    """The picture as plain RGB. A see-through back (an uploaded cut-out) is laid on the grey
    a drawn back has, not converted: converting drops the alpha and shows whatever colour
    hides under it, often black. The 3D step cuts it out again as it does a drawn back."""
    from PIL import Image

    with Image.open(path) as opened:
        check_picture_size(opened)
        opened.load()
        if "A" in opened.getbands() or "transparency" in opened.info:
            return back_view.reference_on_grey(opened.convert("RGBA"))
        return opened.convert("RGB")


def paste_through(original: Path, drawn: Path, paste_mask: Path) -> None:
    """Keep the drawn picture only inside the brush: everything outside is the original's
    own pixels again. The model re-encodes the whole picture, which shifts every pixel a
    little and leaves a halo at the stroke's edge (2026-10-30); this undoes both."""
    from PIL import Image

    with opaque(original) as base, Image.open(drawn) as new, Image.open(paste_mask) as mask:
        Image.composite(new.convert("RGB").resize(base.size), base,
                        mask.convert("L").resize(base.size)).save(drawn)


def brush_prompt(instruction: str) -> str:
    """What goes in the painted area, set in the picture it belongs to."""
    words = " ".join(str(instruction or "").split())[:MAX_INSTRUCTION].strip().rstrip(".")
    return (f"{words}. Keep the style, colours, materials and lighting of the rest of the picture, "
            "plain light grey background.")


def create_brush_job(manager: ImageJobManager, picture: Path, stroke, instruction: str,
                     seed: int = 42, engine: PictureEngine = FLUX) -> ImageJob:
    """Repaint the stroked area of `picture` as `instruction` says; the rest stays as it is.

    The new picture is drawn beside it (`<name>_new.png`) and swapped in when it is done; the
    old one is kept, as a back drawn again is. `stroke`: a PIL image, white where painted.
    """
    with opaque(picture) as base:
        size = base.size
        # the engine draws at a size it accepts (multiples of 32, 256 to 1536): a back you
        # uploaded can be any size, so it starts from a copy at that size and the result is
        # scaled back to the picture's own size when it is pasted
        settings = clean_settings({"seed": seed, "steps": BRUSH_STEPS, "width": size[0], "height": size[1]}, engine)
        work = (settings["width"], settings["height"])
        grown, _ = brush_masks(stroke, work)
        _, feathered = brush_masks(stroke, size)
        if not grown.getbbox():
            raise ValueError("paint over the part to change first")
        # The job first: a request refused as busy must not touch the files a running brush
        # reads when it finishes (its paste area above all).
        job = manager.create(brush_prompt(instruction), settings,
                             output_path=picture.with_name(f"{picture.stem}_new.png"),
                             mode="brush", engine=engine)
        try:
            init = picture
            if work != size or opened_mode(picture) != "RGB":   # resized, or laid on grey
                init = picture.with_name(f"{picture.stem}_brush_init.png")
                base.resize(work).save(init)
            mask = picture.with_name(f"{picture.stem}_brush_mask.png")
            paste = picture.with_name(f"{picture.stem}_brush_paste.png")
            grown.save(mask)
            feathered.save(paste)
        except Exception:
            manager.finish(job)
            raise
    job.init, job.mask, job.paste_mask, job.replaces = init, mask, paste, picture
    if engine.vision_file is None:
        # Flux also SEES the picture it repaints, so the prompt can stay the change alone:
        # told only the change, it filled the paint with it (2026-10-30); told the scene in
        # words, it drew what the words named (a lantern from the front's prompt)
        job.reference = init
    job.instruction = " ".join(str(instruction or "").split())[:MAX_INSTRUCTION] or None
    return job


def start_brush(picture: Path, stroke, instruction: str, seed: int = 42,
                engine: PictureEngine = FLUX) -> ImageJob:
    job = create_brush_job(MANAGER, picture, stroke, instruction, seed, engine)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job


def create_back_job(manager: ImageJobManager, front: Path, seed: int = 3,
                    engine: PictureEngine = QWEN, instruction: str | None = None) -> ImageJob:
    """A job drawing the back of the subject in `front`, saved beside it as `<front>__back`.

    The reference is the front's cut-out on plain grey (image_to_3dlab/back_view.py), cut
    out here unless the picture already is. Seed 3 drew every back in the eight-character
    test of 2026-10-12. The back is drawn at the engine's own size and steps.

    When a back is already there, the new one is drawn beside it (`__back_new`) and swapped in
    when it is done; the old one is kept as `__back_prev_<time>`. `instruction`: what to change.
    """
    from PIL import Image

    reference = front.with_name(f"{front.stem}__back_reference.png")
    final = front.with_name(f"{front.stem}__back.png")
    # the job first: a request refused as busy must not pay for a cut-out, or rewrite the
    # reference a running back is reading
    job = manager.create(back_prompt(instruction), clean_settings({"seed": seed}, engine),
                         output_path=final.with_name(f"{front.stem}__back_new.png") if final.exists() else final,
                         reference=reference, engine=engine)
    try:
        with Image.open(front) as opened:
            check_picture_size(opened)
            cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(
                opened.convert("RGB"))[0]
        back_view.reference_on_grey(cut).save(reference)
    except Exception:
        manager.finish(job)
        raise
    job.replaces = final if job.output_path != final else None
    job.instruction = " ".join(str(instruction or "").split())[:MAX_INSTRUCTION] or None
    return job


def back_source(output_root: Path, relative: str) -> Path:
    """The front picture a back job may read: an existing file inside `output_root`."""
    root = output_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.suffix.lower() not in PICTURE_FILES:
        raise ValueError(f"not a picture in output/: {relative}")
    return path


def run_job(job: ImageJob, manager: ImageJobManager,
            weights: dict[str, Path] | None = None) -> None:
    """Drive one sd-cli run to completion, emitting progress as it goes."""
    try:
        weights = weights or resolve_weights(job.engine)
        vision = resolve_vision(job.engine) if job.reference else None
        if job.reference and job.engine.vision_file and vision is None:
            raise MissingWeights([job.engine.vision_file[1]], job.engine)
        command = job_command(job, weights, vision)
        job.set_status("running", total_steps=job.settings["steps"])
        started = time.monotonic()
        job.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), env=with_rocm_libraries(dict(os.environ)),
        )
        assert job.process.stdout is not None
        # Only where a GPU is expected from a runtime-loaded backend. The Mac build is
        # Metal and says nothing of the sort. Every other machine gets it, AMD included:
        # the ROCm and Vulkan builds fall back to the CPU just as silently as CUDA does,
        # and an AMD user watching a twenty-minute grind on an idle card deserves the
        # same one-line explanation.
        platform = host_platform()
        watch = BackendWatch() if platform != APPLE else None
        no_gpu = False
        buffer = b""
        while True:
            # read1, not read: read(256) waits for 256 bytes, so a short line sat unseen
            # until more output came, which is useless for stopping a CPU-only run.
            chunk = job.process.stdout.read1(256)
            if not chunk:
                break
            buffer += chunk
            # sd-cli redraws its bar with \r, so split on both or the whole run arrives
            # as one unreadable line at the end.
            parts = re.split(rb"[\r\n]", buffer)
            buffer = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                job.log_lines.append(line)
                if watch is not None and not no_gpu and watch.feed(line):
                    no_gpu = True
                    job.process.terminate()
                event = parse_progress(line)
                if event:
                    job.emit(event)
        code = job.process.wait()
        elapsed = time.monotonic() - started
        if job.cancel_requested:
            job.set_status("cancelled")
        elif no_gpu:
            job.error = no_gpu_help(platform)
            job.set_status("error", error=job.error,
                           log="\n".join(list(job.log_lines)[-12:]))
        elif code != 0 or not job.output_path.exists():
            reason = failure_reason(job.log_lines)
            job.error = f"sd-cli exited with code {code}"
            if reason:
                job.error += f": {reason}"
            job.set_status("error", error=job.error,
                           log="\n".join(list(job.log_lines)[-12:]))
        else:
            if job.paste_mask is not None and job.replaces is not None:
                paste_through(job.replaces, job.output_path, job.paste_mask)
            kept = None
            if job.replaces is not None:
                kept = take_place(job.output_path, job.replaces)
                job.output_path = job.replaces
            sidecar = job.output_path.with_suffix(".provenance.json")
            record = provenance(job.prompt, job.settings, elapsed, job.output_path, job.engine)
            if job.reference:
                record["reference"] = {"picture": job.reference.name, "mode": job.mode}
                if job.instruction:
                    record["reference"]["instruction"] = job.instruction
            if job.init is not None:
                # the result is the old picture outside the paint and new paint inside it:
                # the old one's own record stays beside it, under the name it was kept as
                record["brush"] = {"picture": job.output_path.name, "mask": job.mask.name if job.mask else None,
                                   "instruction": job.instruction, "strength": BRUSH_STRENGTH,
                                   "replaced": kept.name if kept else None}
            sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
            job.set_status("done", result_url=f"/api/image/{job.id}/result.png",
                           elapsed_seconds=round(elapsed, 1))
            usage_api.count_done("picture")
    except MissingWeights as exc:
        job.error = str(exc)
        job.set_status("error", error=job.error, needs_setup=True)
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
        job.set_status("error", error=job.error)
    finally:
        manager.finish(job)


MANAGER = ImageJobManager()


def start_back(front: Path, seed: int = 3, engine: PictureEngine = QWEN,
               instruction: str | None = None) -> ImageJob:
    job = create_back_job(MANAGER, front, seed, engine, instruction)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job


def start_edit(reference: Path, output_path: Path, prompt: str, seed: int,
               mode: str = "edit (skin)") -> ImageJob:
    """Qwen redraws `reference` as `prompt` asks, into `output_path` (a skin's picture try).
    Skins stay on Qwen until Flux's skin edits have been judged."""
    job = MANAGER.create(prompt, clean_settings({"seed": seed}), output_path=output_path,
                         reference=reference, mode=mode)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job


def start(prompt: str, raw_settings: dict[str, Any], engine: PictureEngine = QWEN,
          picture_only: bool = False) -> ImageJob:
    settings = clean_settings(raw_settings, engine, picture_only)
    job = MANAGER.create(prompt, settings, engine=engine)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job
