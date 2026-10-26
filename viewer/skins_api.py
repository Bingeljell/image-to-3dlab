"""Skins: new paint on a model that already exists, with the same shape underneath.

A skin starts from a finished Pixal3D model. The person says what to change ("make the
skirt gold"), Qwen edits the model's own front picture, and they approve it or ask again
with changes; then the same for the back picture when the model was built from one. Once
both are approved, Pixal3D repaints the model's saved shape from the new pictures and
Finish runs with the model's own settings. Mesh, UVs and normal map come out identical to
the original, so the rig and moves made for it fit every skin.

Everything a skin makes lives in `output/skins/<id>/`: its picture tries, the approved
pair, the repaint and the finished skin. The Library never mistakes those for new models;
it lists them under the model they belong to (`skins_for`).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))   # standing alone, not only after another module put it there
import usage_api

from image_to_3dlab import processes

OUTPUT_ROOT = REPO / "output"
SKINS_DIR = "skins"
RECORD = "skin.json"
SKIN_ID = re.compile(r"^[0-9a-f]{12}$")
SIDES = ("front", "back")
MAX_INSTRUCTION = 600
GENERATOR = REPO / "scripts" / "pixal3d_generate.py"
TERMINAL = {"done", "error", "cancelled"}
# Each try gets its own seed, so asking again draws a different picture. The back's are
# offset so a front try and a back try never share one by accident.
FRONT_SEED, BACK_SEED = 42, 142
# The repaint's in-between files nobody needs once the skin is finished (~230 MB).
SCRATCH_SUFFIXES = (".post.bin", ".ply", "_base.png")


class SkinError(ValueError):
    """Why a skin cannot be made or moved on, in words for the person asking."""


def install_keeps_shapes() -> bool:
    """Whether this machine's Pixal3D build can save shapes (scripts/patch_pixal3d_shape_reuse.py)."""
    if str(GENERATOR.parent) not in sys.path:
        sys.path.insert(0, str(GENERATOR.parent))
    import pixal3d_generate

    return pixal3d_generate.shape_supported()


def edit_prompt(instruction: str, side: str) -> str:
    """What Qwen is asked: the change, and everything that must stay as it is.

    The keep-list matters more than the change. Pixal3D paints the picture straight onto
    the old shape, so a moved arm or a re-framed picture paints onto thin air, and a part
    Qwen invents (straps the original never had) gets painted on regardless.
    """
    view = "the back view of the same character" if side == "back" else "the same character"
    return (f"Change only this: {instruction.strip().rstrip('.')}. This is {view}. Keep "
            "everything else exactly the same: the same pose, the same face, the same "
            "proportions, the same parts and shapes, the same position and size in the frame, "
            "the same plain background and the same lighting. Do not add or remove any part.")


def skin_name(instruction: str, name: str | None = None) -> str:
    """What the skin is called in the studio and in game engines' variant lists: the name given,
    or the change asked for without its "make the" ("Make the boots white" -> "Boots white")."""
    chosen = " ".join(str(name or "").split())[:60]
    if chosen:
        return chosen
    words = re.sub(r"^(please\s+)?(make|change|turn|paint|give)\s+(the\s+|her\s+|his\s+|its\s+|their\s+)?",
                   "", instruction.strip(), flags=re.IGNORECASE).rstrip(". ")
    words = words or instruction.strip()
    return (words[:1].upper() + words[1:])[:60]


def slug(text: str, limit: int = 32) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:limit].rstrip("-") or "skin"


def _rel(output: Path, path: Path) -> str:
    return path.resolve().relative_to(output.resolve()).as_posix()


def _inside(output: Path, relative: str) -> Path:
    path = (output / relative).resolve()
    if output.resolve() not in path.parents:
        raise SkinError(f"not a file in output/: {relative}")
    return path


def base_from_asset(output: Path, asset: dict[str, Any]) -> dict[str, Any]:
    """Everything a skin needs from the model it repaints, or SkinError saying what is missing.

    `asset` is a Library entry (assets_api.list_assets). The model must be a finished
    Pixal3D model that kept its shape; its record says the seed, steps and whether it was
    built from a front and a back picture, and its Finish run kept the settings that made
    the shape every skin must match.
    """
    if not asset.get("finished") or not asset.get("model"):
        raise SkinError("Skins start from a finished model. Finish this one first.")
    model = _inside(output, asset["model"])
    try:
        record = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    if record.get("backend") != "pixal3d":
        raise SkinError("Only Pixal3D models can take new skins.")
    shape = model.with_suffix(".shape.bin")
    if not shape.is_file():
        if not install_keeps_shapes():
            raise SkinError("This Pixal3D install can't keep a model's shape, so skins aren't "
                            "available here yet (the ready-made NVIDIA and AMD builds lack it).")
        raise SkinError("This model was made before skins, so it kept no shape to repaint. "
                        "Make it again (same picture) to give it skins.")
    if not asset.get("picture"):
        raise SkinError("This model has no picture to edit.")
    params = record.get("parameters") or {}
    back = None
    if params.get("views") == "front + back":
        if not asset.get("back"):
            raise SkinError("This model was built from a front and a back picture, and the "
                            "back picture is missing.")
        back = asset["back"]
    finished = _inside(output, asset["finished"])
    if (finished.parent / "fix-spot.json").is_file():
        # The skin repaints the shape the model was made with, so the fixed spots would come back.
        raise SkinError("Skins start from the version before Fix a spot. Make the skin from that version.")
    try:
        finish = json.loads((finished.parent / "input" / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SkinError("This model's Finish settings were not kept, so a skin could not "
                        "match its shape. Finish it again first.") from None
    # Pixal3D's own clean-up (a kept dump) and the classic route make different meshes: a skin
    # must be finished the way its model was, or the rig made for the model would not fit it.
    native = (finished.parent / "input" / "source.post.bin").is_file()
    return {
        "asset": asset.get("id"), "name": asset.get("name"),
        "finished": asset["finished"], "model": asset["model"], "shape": _rel(output, shape),
        "front": asset["picture"], "back": back,
        "seed": params.get("seed"), "steps": params.get("steps"), "finish": finish, "native": native,
    }


@dataclass
class Skin:
    """One skin in the making, saved as output/skins/<id>/skin.json after every change."""

    name: str
    instruction: str
    base: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tries: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"front": [], "back": []})
    approved: dict[str, str | None] = field(default_factory=lambda: {"front": None, "back": None})
    status: str = "pictures"          # pictures -> building -> done | error
    result: str | None = None         # the finished skin, relative to output/
    error: str | None = None
    created: float = field(default_factory=time.time)

    def needs_back(self) -> bool:
        return bool(self.base.get("back"))

    def sides(self) -> tuple[str, ...]:
        return SIDES if self.needs_back() else ("front",)

    def ready(self) -> bool:
        return all(self.approved.get(side) for side in self.sides())

    def next_side(self) -> str | None:
        """The side still waiting for an approved picture, front first."""
        return next((side for side in self.sides() if not self.approved.get(side)), None)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "instruction": self.instruction, "base": self.base,
                "tries": self.tries, "approved": self.approved, "status": self.status,
                "result": self.result, "error": self.error, "created": self.created,
                "needs_back": self.needs_back(), "ready": self.ready(), "next_side": self.next_side()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skin:
        return cls(name=data["name"], instruction=data["instruction"], base=data["base"], id=data["id"],
                   tries=data.get("tries") or {"front": [], "back": []},
                   approved=data.get("approved") or {"front": None, "back": None},
                   status=data.get("status", "pictures"), result=data.get("result"),
                   error=data.get("error"), created=data.get("created", time.time()))


class SkinStore:
    """Skins on disk. The folder is the truth; nothing is held only in memory."""

    def __init__(self, output: Path = OUTPUT_ROOT):
        self.output = output
        self.lock = threading.Lock()

    def folder(self, skin_id: str) -> Path:
        if not SKIN_ID.fullmatch(skin_id):
            raise SkinError("not a skin id")
        return self.output / SKINS_DIR / skin_id

    def save(self, skin: Skin) -> None:
        folder = self.folder(skin.id)
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / (RECORD + ".part")
        partial.write_text(json.dumps(skin.to_dict(), indent=2) + "\n", encoding="utf-8")
        partial.replace(folder / RECORD)

    def get(self, skin_id: str) -> Skin:
        try:
            return Skin.from_dict(json.loads((self.folder(skin_id) / RECORD).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            raise SkinError("no such skin") from None

    def all(self) -> list[Skin]:
        found = []
        for record in sorted((self.output / SKINS_DIR).glob(f"*/{RECORD}")):
            try:
                found.append(Skin.from_dict(json.loads(record.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(found, key=lambda s: s.created)

    def create(self, asset: dict[str, Any], instruction: str, name: str | None = None) -> Skin:
        instruction = (instruction or "").strip()
        if not instruction:
            raise SkinError("Say what to change, for example \"make the skirt gold\".")
        if len(instruction) > MAX_INSTRUCTION:
            raise SkinError(f"Keep the change under {MAX_INSTRUCTION} characters.")
        base = base_from_asset(self.output, asset)
        skin = Skin(name=skin_name(instruction, name), instruction=instruction, base=base)
        self.save(skin)
        return skin

    def picture_try(self, skin: Skin, side: str, instruction: str | None = None) -> tuple[Path, Path, str, int]:
        """Record a new try for `side` and say what to draw: (reference, output, prompt, seed)."""
        if side not in skin.sides():
            raise SkinError("This model has no back picture.")
        if skin.status == "done":
            raise SkinError("This skin is finished. Start a new skin to change it.")
        skin.status, skin.error = "pictures", None
        if side == "back" and not skin.approved.get("front"):
            raise SkinError("Approve the front picture first.")
        wanted = (instruction or "").strip() or skin.instruction
        if len(wanted) > MAX_INSTRUCTION:
            raise SkinError(f"Keep the change under {MAX_INSTRUCTION} characters.")
        number = len(skin.tries[side]) + 1
        seed = (FRONT_SEED if side == "front" else BACK_SEED) + number - 1
        reference = _inside(self.output, skin.base[side])
        target = self.folder(skin.id) / f"{side}-{number}.png"
        skin.tries[side].append({"number": number, "instruction": wanted, "seed": seed,
                                 "picture": _rel(self.output, target)})
        skin.approved[side] = None
        if side == "front" and skin.needs_back():
            skin.approved["back"] = None   # a new front can change what the back must match
        self.save(skin)
        return reference, target, edit_prompt(wanted, side), seed

    def approve(self, skin: Skin, side: str, number: int | None = None) -> Skin:
        """Approve one try (the latest by default). Its picture must exist: a try still drawing
        or one that failed cannot be approved."""
        if side not in skin.sides():
            raise SkinError("This model has no back picture.")
        if side == "back" and not skin.approved.get("front"):
            raise SkinError("Approve the front picture first.")
        tries = skin.tries[side]
        chosen = next((t for t in tries if t["number"] == number), None) if number else (tries[-1] if tries else None)
        if chosen is None:
            raise SkinError(f"There is no {side} picture to approve yet.")
        if not (self.output / chosen["picture"]).is_file():
            raise SkinError(f"That {side} picture is not ready yet.")
        skin.approved[side] = chosen["picture"]
        self.save(skin)
        return skin

    def discard(self, skin: Skin) -> None:
        """Delete a skin and everything it made. The caller checks nothing is drawing or building it."""
        shutil.rmtree(self.folder(skin.id))

    def skins_for(self, finished: str, skins: list[Skin] | None = None) -> list[dict[str, Any]]:
        """The finished skins of one finished model, oldest first: what the viewer lists. Pass
        `skins` (from `all`) when asking for many models, so the folder is read once."""
        return [{"id": s.id, "name": s.name, "file": s.result, "front": s.approved.get("front")}
                for s in (self.all() if skins is None else skins)
                if s.base.get("finished") == finished and s.status == "done" and s.result]

    def state(self, skin: Skin, build_running: bool) -> dict[str, Any]:
        """The skin as the studio and agents read it: whether each try's picture exists yet, and
        a build the studio lost (it restarted mid-build) reported as one to start again."""
        state = skin.to_dict()
        for side in SIDES:
            for attempt in state["tries"][side]:
                attempt["ready"] = (self.output / attempt["picture"]).is_file()
        if skin.status == "building" and not build_running:
            state["status"] = "error"
            state["error"] = "The build was interrupted (the studio restarted). Try building again."
        return state


# ---------------------------------------------------------------------------- the build

# How far a new picture's cut-out may sit from the original's, as a share of the picture.
# Pixal3D frames each picture by its cut-out, so a part lost to the background (white boots
# on light grey) or a shifted pose moves every bit of paint, not just that part.
FRAMING_TOLERANCE = 0.02


def cut_out_box(path: Path) -> tuple[float, float, float, float] | None:
    """The cut-out's bounding box as fractions of the picture (left, top, right, bottom), made
    the way the repaint makes it (pixal3d_generate's matte), or None when nothing is left."""
    if str(GENERATOR.parent) not in sys.path:
        sys.path.insert(0, str(GENERATOR.parent))
    from PIL import Image
    from pixal3d_generate import _rembg_remove

    with Image.open(path) as opened:
        cut, _ = _rembg_remove(opened.convert("RGB"))
    box = cut.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()
    if box is None:
        return None
    w, h = cut.size
    return box[0] / w, box[1] / h, box[2] / w, box[3] / h


def framing_problem(output: Path, skin: Skin, box=cut_out_box) -> str | None:
    """Why the approved pictures would paint off target, or None. Checked before the repaint."""
    for side in skin.sides():
        before, after = box(_inside(output, skin.base[side])), box(_inside(output, skin.approved[side]))
        if before is None or after is None:
            return f"The cut-out of the {side} picture came out empty. Ask again for that picture."
        drift = max(abs(a - b) for a, b in zip(before, after))
        if drift > FRAMING_TOLERANCE:
            words = "lower" if after[3] < before[3] - FRAMING_TOLERANCE else "differently"
            return (f"The new {side} picture's cut-out is framed {words} than the original ({drift:.0%} off), "
                    "so the paint would slide out of place. A light part may have been lost to the background, "
                    f"or the pose moved. Ask again for the {side} picture, for example with a darker outline "
                    "or the same pose and size.")
    return None


def finished_name(skin: Skin) -> str:
    from finish_api import faces_label

    return f"{slug(skin.name)}_{faces_label(int(skin.base['finish'].get('faces', 20000)))}.glb"


def repaint_command(output: Path, skin: Skin, python: str = sys.executable) -> list[str]:
    """pixal3d_generate.py painting the model's saved shape from the approved pictures."""
    folder = output / SKINS_DIR / skin.id
    command = [python, "-u", str(GENERATOR), str(_inside(output, skin.approved["front"])),
               str(folder / "repaint" / "repaint.glb")]
    if skin.needs_back():
        command += ["--back", str(_inside(output, skin.approved["back"]))]
    if skin.base.get("seed") is not None:
        command += ["--seed", str(int(skin.base["seed"]))]
    if isinstance(skin.base.get("steps"), int):
        command += ["--steps", str(skin.base["steps"])]
    return command + ["--reuse-shape", str(_inside(output, skin.base["shape"]))]


def finish_command(output: Path, skin: Skin) -> list[str]:
    """Finish with the model's own settings: same face count, so the same mesh and UVs."""
    import finish_api

    folder = output / SKINS_DIR / skin.id
    repaint = folder / "repaint" / "repaint.glb"
    # the model's own route: no dump handed over when the model was finished without one
    dump = repaint.with_suffix(".post.bin") if skin.base.get("native", True) else folder / "no-dump"
    job = SimpleNamespace(asset_path=repaint, image_path=_inside(output, skin.approved["front"]),
                          result_glb=folder / finished_name(skin), steps_dir=folder / "finish-steps",
                          views_dir=repaint.with_suffix(".svviews"), dump_path=dump)
    settings = finish_api.fit_to_machine(finish_api.normalise_settings(skin.base["finish"]))
    return finish_api.build_command(job, settings)


@dataclass
class SkinBuild:
    """Repaint, then Finish: one job, so nothing else can slip in between the two."""

    id: str
    skin_id: str
    status: str = "queued"
    error: str | None = None
    process: subprocess.Popen | None = None
    cancel_requested: bool = False
    started: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=400))

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append({"elapsed_seconds": round(time.monotonic() - self.started, 1), **event})

    def describe(self) -> dict[str, Any]:
        return {"job_id": self.id, "skin_id": self.skin_id, "status": self.status, "error": self.error,
                "elapsed_seconds": round(time.monotonic() - self.started, 1),
                "last_event": self.events[-1] if self.events else None,
                "log_tail": "\n".join(self.log_lines)[-4000:]}


PIXAL3D_BANNER = re.compile(r"^\[(\d)/6\]")


def progress(stage: str, line: str) -> float | None:
    """Overall percent from one log line: the repaint is ~80% of the time, Finish the rest."""
    if stage == "repaint":
        match = PIXAL3D_BANNER.match(line)
        return round(int(match.group(1)) / 6 * 80, 1) if match else None
    if line.startswith("I2L_STAGE::"):
        return {"retopologise": 82.0, "photo": 88.0, "bake": 93.0, "compress": 97.0}.get(line.split("::")[1])
    return None


class SkinBuilds:
    """One build at a time; it also counts as a heavy job (generate_api.HEAVY_JOBS)."""

    def __init__(self, store: SkinStore, stop=None):
        self.store = store
        # Stops a step's whole process group: the engine and Blender it started, not just the step.
        self.stop = stop or processes.terminate_group
        self.jobs: dict[str, SkinBuild] = {}
        self.lock = threading.Lock()
        self.active: str | None = None

    def running(self) -> bool:
        job = self.jobs.get(self.active) if self.active else None
        return job is not None and job.status not in TERMINAL

    def start(self, skin: Skin, *, background: bool = True, runner=subprocess.Popen, box=cut_out_box) -> SkinBuild:
        if not skin.ready():
            raise SkinError(f"Approve the {skin.next_side()} picture first.")
        if skin.status == "done":
            raise SkinError("This skin is already finished.")
        with self.lock:
            if self.running():
                raise RuntimeError("a skin is already being made; wait for it to finish")
            job = SkinBuild(uuid.uuid4().hex, skin.id)
            self.jobs[job.id] = job
            self.active = job.id
        skin.status, skin.error = "building", None
        self.store.save(skin)
        if background:
            threading.Thread(target=self.run, args=(job, skin, runner, box), daemon=True,
                             name=f"skin-{skin.id}").start()
        else:
            self.run(job, skin, runner, box)
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in TERMINAL:
            return False
        job.cancel_requested = True
        if job.process and job.process.poll() is None:
            self.stop(job.process.pid)
        return True

    def run(self, job: SkinBuild, skin: Skin, runner=subprocess.Popen, box=cut_out_box) -> None:
        output = self.store.output
        folder = self.store.folder(skin.id)
        job.status = "running"
        try:
            job.emit({"phase": "check", "message": "Checking the new pictures line up with the originals", "overall_pct": 1})
            problem = framing_problem(output, skin, box)
            if job.cancel_requested:
                raise InterruptedError
            if problem:
                raise RuntimeError(problem)
            # Each command is built when its step starts: Finish only finds the repaint's clean-up
            # dump and cameras once the repaint has written them (2026-10-19: built up front, the
            # first skin took the classic route without Pixel Match, a different mesh).
            for stage, make in (("repaint", repaint_command), ("finish", finish_command)):
                if job.cancel_requested:
                    raise InterruptedError
                command = make(output, skin)
                job.emit({"phase": stage, "message": "Repainting the model" if stage == "repaint"
                          else "Finishing, with the model's own settings", "overall_pct": 0 if stage == "repaint" else 80})
                job.process = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     bufsize=1, cwd=str(REPO), env=processes.child_env(), **processes.TEXT_PIPE,
                                     **processes.group_popen_kwargs())
                if job.cancel_requested:            # Cancel landed while it started: it found no process to stop
                    self.stop(job.process.pid)
                for raw in job.process.stdout:
                    line = raw.rstrip()
                    if not line:
                        continue
                    job.log_lines.append(line)
                    percent = progress(stage, line)
                    if percent is not None:
                        job.emit({"phase": stage, "message": line[:160], "overall_pct": percent})
                code = job.process.wait()
                if job.cancel_requested:
                    raise InterruptedError
                if code != 0:
                    raise RuntimeError(f"the {stage} step stopped (exit {code})")
            result = folder / finished_name(skin)
            if not result.is_file():
                raise RuntimeError("Finish ended without writing the skin")
            for leftover in (folder / "repaint").glob("*"):
                if leftover.name.endswith(SCRATCH_SUFFIXES):
                    leftover.unlink(missing_ok=True)
            shutil.rmtree(folder / "finish-steps", ignore_errors=True)
            skin.status, skin.result = "done", _rel(output, result)
            job.status = "done"
            usage_api.count_done("skin")
            job.emit({"phase": "done", "message": "Skin ready", "overall_pct": 100, "path": skin.result})
        except InterruptedError:
            skin.status, job.status = "pictures", "cancelled"
            job.emit({"phase": "cancelled", "message": "Cancelled. The approved pictures are kept."})
        except Exception as exc:
            skin.status, skin.error = "error", str(exc)
            job.status, job.error = "error", str(exc)
            job.emit({"phase": "error", "message": str(exc)})
        finally:
            self.store.save(skin)
            self._history(job, skin)
            with self.lock:
                if self.active == job.id:
                    self.active = None

    def _history(self, job: SkinBuild, skin: Skin) -> None:
        """One line in output/.activity.jsonl, so Activity lists the build and Report a bug can
        send its log like any other step's."""
        from activity_api import append_history

        append_history(self.store.output, {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "chain": f"skin-{skin.id}",
            "title": f"{skin.base.get('name') or 'model'} · {skin.name}", "prompt": skin.instruction,
            "want": "skin", "step": "skin", "result": job.status,
            "seconds": round(time.monotonic() - job.started, 1),
            "note": job.error or (job.events[-1]["message"] if job.events else ""),
            "made": skin.result, "log": "\n".join(job.log_lines)[-3000:]})


# ---------------------------------------------------------------------------- wearing a skin


def _repo_on_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def _fresh(target: Path, *sources: Path) -> bool:
    return target.is_file() and all(target.stat().st_mtime >= s.stat().st_mtime for s in sources)


def dressed(output: Path, skin: Skin, file: str) -> str:
    """`file` (the model's rigged character or one of its moves, or the finished model) wearing
    `skin`, made once and kept in the skin's folder. Returns its path relative to output/."""
    _repo_on_path()
    from image_to_3dlab.glb_skins import dress

    if skin.status != "done" or not skin.result:
        raise SkinError("This skin is not finished yet.")
    source = _inside(output, file)
    if not source.is_file() or source.suffix != ".glb":
        raise SkinError(f"not a model in output/: {file}")
    if file == skin.base.get("finished"):
        return skin.result                       # the skin itself is the finished model in that paint
    paint = output / skin.result
    # keyed by the whole path: two rigs of one model can share a file name and most of a folder name
    import hashlib

    key = hashlib.sha1(file.encode()).hexdigest()[:10]
    target = output / SKINS_DIR / skin.id / "dressed" / f"{key}__{source.name}"
    if not _fresh(target, source, paint):
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(dress(source.read_bytes(), paint.read_bytes()))
        partial.replace(target)
    return _rel(output, target)


def all_skins(output: Path, file: str, skins: list[Skin], original: str = "Original paint") -> str:
    """One GLB of `file` with its own paint and every one of `skins` as a named variant, kept in
    output/skins/all/. Remade when the model or any skin changes."""
    _repo_on_path()
    from image_to_3dlab.glb_skins import with_variants

    done = [s for s in skins if s.status == "done" and s.result]
    if not done:
        raise SkinError("This model has no finished skins yet.")
    source = _inside(output, file)
    if not source.is_file():
        raise SkinError(f"not a model in output/: {file}")
    import hashlib

    key = hashlib.sha1("|".join([file, *sorted(s.id for s in done)]).encode()).hexdigest()[:10]
    target = output / SKINS_DIR / "all" / f"{source.stem[:60]}__skins-{key}.glb"
    paints = [output / s.result for s in done]
    if not _fresh(target, source, *paints):
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(with_variants(source.read_bytes(),
                                          [(s.name, p.read_bytes()) for s, p in zip(done, paints, strict=True)],
                                          original=original))
        partial.replace(target)
    return _rel(output, target)


STORE = SkinStore()
BUILDS = SkinBuilds(STORE)
