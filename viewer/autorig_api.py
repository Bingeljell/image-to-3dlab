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
import re
import socket
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

import usage_api

from image_to_3dlab import cuda_arch, host, processes
from image_to_3dlab.provenance import set_glb_generator, write_provenance

OUTPUT_ROOT = REPO / "output" / "animate"
# Where the activity list and Report a bug read failures from: the output folder itself, not
# the animate one, or a refused Rig would be nowhere either of them looks.
ACTIVITY_ROOT = REPO / "output"
SKINTOKENS = REPO / "vendor" / "SkinTokens"
RETARGET = REPO / "scripts" / "kimodo_retarget.py"
CLEAN_WEIGHTS = REPO / "scripts" / "clean_skin_weights.py"
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
SPEED_RANGE = (0.5, 2.0)
# Clips that return to their first pose: walk, run, idle, sneak and strafe, including their
# takes and variants (walk_b, walk_woman, strafe_left). One place, shared by the tab.
LOOPING_PRESETS = ("idle", "walk", "run", "sneak", "strafe")
# Old names for moves that were renamed or whose second take became the move. A saved plan or an
# agent asking by the old name still gets the move that replaced it; the list lives in presets.json
# so the studio, this API and the MCP all read one answer.

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
    return host.venv_python(root)


def installed(root: Path = SKINTOKENS) -> bool:
    return venv_python(root).exists() and (root / "demo.py").is_file()


# ---------------------------------------------------------------------------- presets


def aliases(root: Path = MOTIONS) -> dict[str, str]:
    """Old move name -> the id it means now."""
    index = _read_json(root / "presets.json") or {}
    return {str(k): str(v) for k, v in (index.get("aliases") or {}).items()}


def resolve_move(preset_id: str, root: Path = MOTIONS) -> str:
    """The preset a request means: an old name maps to the move that replaced it."""
    return aliases(root).get(preset_id, preset_id)


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


def free_ports(count: int = 2) -> list[int]:
    """Ports the system hands out right now, all different. A fixed port can sit inside a
    range Windows keeps for Hyper-V/WSL (#193); one the system just handed out never does."""
    probes = [socket.socket() for _ in range(count)]
    try:
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        return [probe.getsockname()[1] for probe in probes]
    finally:
        for probe in probes:
            probe.close()


def rig_ports() -> dict[str, str]:
    """The environment that hands SkinTokens two free ports (read by the patched spec.py)."""
    port, bpy_port = free_ports(2)
    return {"SKINTOKENS_PORT": str(port), "SKINTOKENS_BPY_PORT": str(bpy_port)}


def port_edits() -> list:
    """The SkinTokens patch's edits that let the ports come from the environment."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import patch_skintokens_portable as portable
    return [edit for edit in portable.EDITS if "SKINTOKENS_" in edit.new]


def ensure_port_patch(root: Path = SKINTOKENS) -> str | None:
    """Give an install made before the port edit that edit, or say why not (None = fine).

    Installs only get the patch when SkinTokens is set up, so older ones need it here. If
    it no longer fits, SkinTokens keeps its fixed ports, which is how it ran before."""
    try:
        import patch_skintokens_portable as portable
        edits = port_edits()
        spec = root / edits[0].path
        text = spec.read_text(encoding="utf-8")
        new = text
        for edit in edits:
            new, _ = portable.apply_edit(new, edit)
        if new != text:
            spec.write_text(new, encoding="utf-8")
        return None
    except Exception as exc:
        return f"note: SkinTokens keeps its fixed ports ({exc})"


def rig_commands(source: Path, result: Path) -> list[tuple[list[str], Path]]:
    """SkinTokens, then a skin-weight clean-up: it sometimes weights fingertips to a toe,
    which stretches them into spikes once the hand moves (scripts/clean_skin_weights.py)."""
    return [
        ([str(venv_python()), "demo.py", "--input", str(source), "--output", str(result),
          "--use_transfer"], SKINTOKENS),
        ([str(venv_python()), str(CLEAN_WEIGHTS), "--in", str(result), "--out", str(result)], REPO),
    ]


def repeats(preset_id: str) -> bool:
    """Whether a preset is a cycle, so the retargeter should trim and blend its loop."""
    return preset_id.startswith(LOOPING_PRESETS)


def loops(preset: dict[str, Any]) -> bool:
    """Whether to trim this move to its best cycle. `presets.json` can say no with `"loop": false`
    for a move made of distinct beats (an idle that looks left, then right) or a clip that is
    already a clean loop; trimming those would cut the beats out. Otherwise the name rule decides."""
    return bool(preset.get("loop", repeats(str(preset.get("id", "")))))


def keeps_whole(preset: dict[str, Any]) -> bool:
    """A move that says `"loop": false` keeps every frame and only has its seam blended.

    It still gets `--loop` so the seam is smoothed at the wrap; without `--keep-whole` the picker
    would trim it back to its longest matching pair and cut the beats out.
    """
    return preset.get("loop") is False


def default_in_place(preset: dict[str, Any]) -> bool:
    """Whether this move stays in place unless the person says otherwise.

    Loops walk, run and strafe on the spot, as a game's locomotion does; one-offs such as jump,
    roll and the kicks keep their travel. `presets.json` carries the answer per move, so the tab,
    the API and agents all read it from there; a preset without it falls back to the loop rule.
    """
    return bool(preset.get("in_place", repeats(str(preset.get("id", "")))))



def animate_command(rigged: Path, motion: Path, result: Path, arm_spread: float,
                    speed: float = 1.0, repeat: bool = False, in_place: bool = False,
                    keep_whole: bool = False) -> list[str]:
    command = [str(venv_python()), str(RETARGET), "--rig", str(rigged), "--motion", str(motion),
               "--out", str(result), "--arm-spread", f"{arm_spread:g}", "--speed", f"{speed:g}"]
    if repeat or keep_whole:
        command.append("--loop")
    if keep_whole:
        command.append("--keep-whole")
    if in_place:
        command.append("--in-place")
    return command


def clamp_speed(value: Any, preset: dict[str, Any]) -> float:
    """The playback speed: the caller's, else the preset's own default, else 1.

    Kimodo's strikes are slow next to real ones, so kick and punch presets ship faster.
    """
    if value is None or value == "":
        value = preset.get("speed", 1.0)
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed must be a number") from exc
    low, high = SPEED_RANGE
    if not low <= speed <= high:
        raise ValueError(f"speed must be between {low:g} and {high:g}")
    return speed


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
                 commands: list[tuple[list[str], Path]], expected_seconds: float,
                 record: dict[str, Any]):
        self.id = job_id
        self.kind = kind
        self.directory = directory
        self.result = result
        self.commands = commands  # run in order, each as (argv, working directory)
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
    def __init__(self, output_root: Path = OUTPUT_ROOT, motions: Path = MOTIONS,
                 activity_root: Path | None = None):
        self.output_root = output_root
        self.motions = motions
        # A test builds a manager over a temp folder and keeps its history there too.
        self.activity_root = activity_root or output_root
        self.jobs: dict[str, AnimateJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def busy(self) -> bool:
        # Work, work.
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
                             rig_commands(source_glb, result), RIG_SECONDS,
                             rig_record(name, data, source, origin))
            return self._claim(job)

    def create_animation(self, rigged: Path, preset_id: str, arm_spread: Any,
                         speed: Any = None, in_place: Any = None) -> AnimateJob:
        with self.lock:
            if self.busy():
                raise RuntimeError("an Animate job is already running")
            run = rig_run_of(rigged)
            if run is None:
                raise ValueError("pick a rigged model first")
            preset_id = resolve_move(preset_id, self.motions)
            preset = next((p for p in presets(self.motions) if p["id"] == preset_id), None)
            if preset is None:
                offered = ", ".join(p["id"] for p in presets(self.motions))
                raise ValueError(f"no such preset: {preset_id!r}. The moves are: {offered}")
            spread = clamp_spread(arm_spread)
            pace = clamp_speed(speed, preset)
            if in_place is None:
                keep = default_in_place(preset)
            elif isinstance(in_place, bool):
                keep = in_place
            else:
                raise ValueError("in place is true or false")
            stem = rigged.name.removesuffix("_rigged.glb")
            result = run / f"{stem}_{preset_id}.glb"
            record = animation_record(_read_json(rigged.with_suffix(".provenance.json")),
                                      preset, spread)
            record["motion"]["speed"] = pace
            whole = keeps_whole(preset)
            record["motion"]["loop"] = loops(preset)
            record["motion"]["keep_whole"] = whole
            record["motion"]["in_place"] = keep
            record["motion"]["grounded"] = True
            job = AnimateJob(uuid.uuid4().hex, "animate", run, result,
                             [(animate_command(rigged, self.motions / f"{preset_id}.npz",
                                               result, spread, pace, loops(preset),
                                               keep, whole), REPO)],
                             ANIMATE_SECONDS, record)
            return self._claim(job)

    def get(self, job_id: str) -> AnimateJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: AnimateJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None  # Job's done!


ANIMATE_JOBS = AnimateJobManager(activity_root=ACTIVITY_ROOT)


def _tick(job: AnimateJob, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        elapsed = time.monotonic() - job.started
        job.emit({"phase": "running", "overall_pct": estimated_pct(elapsed, job.expected_seconds),
                  "message": "Fitting a skeleton" if job.kind == "rig" else "Playing the clip"})


def rig_fit(glb: Path) -> dict[str, Any]:
    """Whether the preset moves fit this rig (scripts/rig_check.py); never fails the rig itself."""
    try:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        import rig_check
        return rig_check.check(glb)
    except Exception:
        return {}


def card_problem(python: Path | None = None) -> str | None:
    """Why the rig cannot run on this card, in one plain sentence; None when it can.

    SkinTokens runs on the torch in its own venv, so that is the torch that matters: this
    asks it what it was built for, and the card what it is. No model is loaded either way.
    """
    return cuda_arch.card_problem(python or venv_python())


def write_history(job: AnimateJob, manager: AnimateJobManager, note: str) -> None:
    """One line in output/.activity.jsonl for a failed Rig or Animate (what, why, and the log).

    Successes are not listed: the studio shows the result, and the file says what it is.
    A failure has to reach Report a bug, which reads the newest error record (`note` and
    `log`), or the sentence we wrote for the user never leaves the browser.
    """
    from activity_api import append_history

    append_history(manager.activity_root, {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chain": f"{job.kind}-{job.id}",
        "title": job.record.get("name") or job.directory.name,
        "want": "character",
        "step": "rigged" if job.kind == "rig" else "animated",
        "result": job.status,
        "seconds": round(time.monotonic() - job.started, 1),
        "note": note,
        "made": None,
        "log": "\n".join(job.log_lines)[-3000:],
    })


def run_job(job: AnimateJob, manager: AnimateJobManager = ANIMATE_JOBS) -> None:
    stop = threading.Event()
    try:
        job.status = "running"
        if job.kind == "rig":
            # Before the ~8 GB model is loaded: a card this torch has no code for would die
            # on its first kernel, and die with a CUDA traceback (Quadro P5000, 2026-11-13).
            unsupported = card_problem()
            if unsupported:
                job.append_log(unsupported)
                job.status = "error"
                job.emit({"phase": "error", "message": unsupported})
                write_history(job, manager, unsupported)
                return
        job.emit({"phase": "queued", "overall_pct": 0,
                  "message": "Starting the auto-rig" if job.kind == "rig" else "Starting"})
        threading.Thread(target=_tick, args=(job, stop), daemon=True).start()
        code = 0
        ports = rig_ports() if job.kind == "rig" else {}
        for command, cwd in job.commands:
            if job.kind == "rig" and Path(cwd) == SKINTOKENS:
                note = ensure_port_patch(SKINTOKENS)
                if note:
                    job.append_log(note)
            job.process = subprocess.Popen(
                command, cwd=str(cwd),
                env=processes.child_env(HF_HUB_OFFLINE="1", **ports),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, **processes.TEXT_PIPE,
                **processes.group_popen_kwargs(),
            )
            job.directory.joinpath("pid").write_text(
                processes.pid_record(job.process.pid), encoding="utf-8")
            assert job.process.stdout is not None
            for raw in job.process.stdout:
                line = raw.rstrip("\n")
                if line:
                    job.append_log(line)
            code = job.process.wait()
            if code != 0 or job.cancel_requested:
                break
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
            write_history(job, manager, job.events[-1]["message"])
        else:
            set_glb_generator(job.result)   # the rigger's or Blender's own value is not ours
            write_provenance(job.result.with_suffix(".provenance.json"), job.record)
            # The done event (with the file's path) goes out BEFORE the status flips: a plan polls
            # the status and reads the last event, and the fit check takes seconds. The other order
            # lost a robot's rig mid-plan (2026-10-25).
            fit = rig_fit(job.result) if job.kind == "rig" else {}
            job.emit({"phase": "done", "overall_pct": 100,
                      "message": fit.get("message") or ("Rigged" if job.kind == "rig" else "Animated"),
                      **({"humanoid": fit["humanoid"]} if "humanoid" in fit else {}),
                      "kind": job.kind,
                      "path": job.result.relative_to(REPO / "output").as_posix()
                      if REPO / "output" in job.result.parents else None,
                      "result_url": served_url(job.result),
                      "size_bytes": job.result.stat().st_size})
            job.status = "done"
            usage_api.count_done("rig" if job.kind == "rig" else "animation")
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
        write_history(job, manager, str(exc))
    finally:
        stop.set()
        job.directory.joinpath("pid").unlink(missing_ok=True)
        if job.process is not None and job.process.poll() is None:
            try:
                processes.terminate_group(job.process.pid)
                job.process.wait(timeout=30)
            except Exception as cleanup_exc:
                job.append_log(f"cleanup warning: {cleanup_exc}")
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
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
