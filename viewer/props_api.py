"""Queued prop-sheet jobs for the browser: split one GLB into props, then give each LODs.

A sibling of `finish_api.py`, with the same job shape (one at a time, SSE progress, runs
kept on disk), driving the two scripts `docs/prop-sheets.md` walks through:
`scripts/blender_split_props.py` takes the sheet apart, and `scripts/finish_props.py`
bakes each prop's LODs and compresses them. Nothing here touches the GPU: the split is
headless Blender, and the LOD bakes are Blender's CPU bake.

A run is re-openable from its directory alone, so the result list is built from what is
on disk rather than from this process's memory. That is also how a prop that came back
facing sideways is fixed: "turn" re-splits the sheet with the extra turn and re-bakes
that one prop, in the same directory, in a few seconds plus its LODs.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finish_api import _read_json, remaining_seconds, served_url  # noqa: E402
from rig_api import blender_executable  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_to_3dlab import processes  # noqa: E402
from image_to_3dlab.blender import missing_help  # noqa: E402

OUTPUT_ROOT = REPO / "output" / "props"
SPLITTER = REPO / "scripts" / "blender_split_props.py"
FINISHER = REPO / "scripts" / "finish_props.py"
GLTFPACK_BOOTSTRAP = REPO / "scripts" / "bootstrap_gltfpack.py"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
# The exact shape `create` builds. A turn takes its directory name from a URL, so the
# name is matched against the generator rather than merely sanitised.
RUN_DIRECTORY = re.compile(r"^[A-Za-z0-9_-]{1,80}__props__\d{8}-\d{6}(?:-\d+)?$")
# A prop name becomes a file name and a command-line argument, so it cannot start with
# "-": after `--names` a name like `--no-straighten` is read as that flag.
PROP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
TERMINAL = {"done", "error", "cancelled"}
MAX_PROPS = 64
MAX_LODS = 4

# `scripts/blender_split_props.py`'s YAW_THRESHOLD, repeated because that script imports
# numpy at module level and this server must not. A test keeps the two equal.
YAW_THRESHOLD = 0.10

DEFAULT_SETTINGS: dict[str, Any] = {
    "names": [], "lods": [5000, 2500, 1000], "atlas": 1024,
    "metallic": 0.25, "roughness": 0.65, "ior": 1.45,
    "yaw_threshold": YAW_THRESHOLD, "compress": True, "turns": {},
}
SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "metallic": (0.0, 1.0), "roughness": (0.0, 1.0), "ior": (1.0, 3.0),
    "yaw_threshold": (0.0, 0.99),
}
ATLAS_SIZES = (1024, 2048, 4096)
FACE_RANGE = (1000, 200000)

# How the bar divides, from a measured nine-prop run: the split takes about 4 seconds
# (8 before its tilt search was batched, 2026-10-01) and each prop's three LODs about 14.
# A fixed slice for the split was right for nine props and wrong for one: turning a single
# prop showed "~2 min left" for a 20-second job.
SPLIT_SECONDS = 4.0
PROP_SECONDS = 14.0


def split_share(props: int) -> float:
    """The fraction of the bar the split takes, for a run baking this many props."""
    return SPLIT_SECONDS / (SPLIT_SECONDS + PROP_SECONDS * max(props, 1))

# Top-level directories under output/ that hold something other than generated models.
NOT_GENERATED = {"finish", "props", "images", "rig-rebind"}


def normalise_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge client settings over the defaults, rejecting anything the scripts would.

    Raises ValueError naming the offending key. Unknown keys are dropped, not forwarded:
    a typo must not reach a command line as a stray flag.
    """
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    for key, value in raw.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key == "names":
            settings[key] = normalise_names(value)
        elif key == "lods":
            settings[key] = normalise_lods(value)
        elif key == "turns":
            settings[key] = normalise_turns(value)
        elif key == "compress":
            settings[key] = bool(value)
        elif key == "atlas":
            atlas = _number(key, value)
            if atlas not in ATLAS_SIZES:
                raise ValueError(f"atlas must be one of {ATLAS_SIZES}, got {value!r}")
            settings[key] = int(atlas)
        else:
            number = _number(key, value)
            low, high = SETTING_BOUNDS[key]
            if not low <= number <= high:
                raise ValueError(f"{key} must be within {low}..{high}, got {number}")
            settings[key] = number
    return settings


def normalise_names(value: Any) -> list[str]:
    """Names in reading order. A string is one name per line, or comma-separated; blank
    entries are skipped."""
    lines = None
    if isinstance(value, str):
        lines = value.splitlines()
        value = [part for line in lines for part in line.split(",")]
    if not isinstance(value, list):
        raise ValueError("names must be a list or one name per line")
    names = [str(name).strip() for name in value if str(name).strip()]
    if len(names) > MAX_PROPS:
        raise ValueError(f"names: at most {MAX_PROPS}, got {len(names)}")
    for name in names:
        if not PROP_NAME.fullmatch(name):
            if " " in name:
                # Two props typed on one line read as one bad name; say where and how.
                where = next((f"line {n} " for n, line in enumerate(lines or [], 1)
                              if name in line), "")
                raise ValueError(f"names: {where}has {name!r}. Put one name per line, and "
                                 f"use - instead of a space ({name.replace(' ', '-')!r})")
            raise ValueError(f"names: {name!r} must be letters, digits, - or _ (up to 40), "
                             "starting with a letter or digit")
    # Compared without case: macOS disks are case-insensitive, so `Barrel.glb` and
    # `barrel.glb` are one file and one of the two props would silently vanish.
    folded = [name.casefold() for name in names]
    duplicates = sorted({name for name in names if folded.count(name.casefold()) > 1})
    if duplicates:
        raise ValueError(f"names: each name once, {', '.join(duplicates)} repeated")
    return names


def normalise_lods(value: Any) -> list[int]:
    """Face counts, most detailed first, the same rule `finish_props.parse_lods` applies."""
    if isinstance(value, str):
        value = [part for part in value.split(",") if part.strip()]
    if not isinstance(value, list) or not value:
        raise ValueError("lods must be a list of face counts")
    if len(value) > MAX_LODS:
        raise ValueError(f"lods: at most {MAX_LODS}, got {len(value)}")
    lods = [int(_number("lods", faces)) for faces in value]
    low, high = FACE_RANGE
    for faces in lods:
        if not low <= faces <= high:
            raise ValueError(f"lods: each must be {low}..{high} faces, got {faces}")
    if any(later >= earlier for earlier, later in pairwise(lods)):
        raise ValueError(f"lods go from most detailed to least, got {lods}")
    return lods


def normalise_turns(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("turns must map a prop name to degrees")
    turns = {}
    for name, degrees in value.items():
        if not PROP_NAME.fullmatch(str(name)):
            raise ValueError(f"turns: {name!r} is not a prop name")
        turns[str(name)] = wrap_degrees(_number("turns", degrees))
    return turns


def wrap_degrees(degrees: float) -> float:
    """Into (-180, 180], so four quarter turns come back to none rather than 360."""
    wrapped = (degrees + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _number(key: str, value: Any) -> float:
    """A finite number. JSON from the browser can carry NaN and Infinity, which would
    reach Blender as `--turn chest=nan`."""
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return number


@functools.cache
def finisher_module():
    """`finish_props.py`, imported once, for its gltfpack lookup and file layout.

    Its module level is constants and small functions; importing it runs nothing. Once,
    because each import runs its `sys.path.insert`, and the server would otherwise grow
    `sys.path` by an entry on every visit to the tab.
    """
    spec = importlib.util.spec_from_file_location("finish_props", FINISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_gltfpack() -> Path | None:
    return finisher_module().find_gltfpack()


@functools.cache
def gltfpack_bootstrap():
    """`bootstrap_gltfpack.py`, imported once, to ask whether this machine has a build."""
    spec = importlib.util.spec_from_file_location("bootstrap_gltfpack", GLTFPACK_BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    # Registered first: its @dataclass looks itself up in sys.modules while it is built.
    sys.modules.setdefault("bootstrap_gltfpack", module)
    spec.loader.exec_module(module)
    return module


def gltfpack_install_command() -> list[str]:
    return [sys.executable, str(GLTFPACK_BOOTSTRAP), "--yes"]


def lod_triangles(path: Path) -> int | None:
    """The triangles a LOD holds, read from its JSON chunk alone.

    The JSON comes first in a GLB, so the megabytes of texture after it stay unread.
    None when the file is not a GLB this can read; the page then shows the target.
    Remembered per file version: the run list asks again for every LOD on each visit.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return _triangles(path, stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=1024)
def _triangles(path: Path, mtime_ns: int, size: int) -> int | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(20)
            (length,) = struct.unpack_from("<I", header, 12)
            return finisher_module().glb_triangles(header + handle.read(length))
    except (OSError, ValueError, struct.error, KeyError, IndexError, TypeError):
        return None


def bake_order(split_dir: Path, names: list[str]) -> list[str]:
    """The props in the order `finish_props.py` will bake them, which is its own file
    order rather than reading order.

    The panel ticks every row above the running one, so rows in reading order would
    show the barrel and the crate as done while the anvil, alphabetically first, was
    still baking (seen on the first real run, 2026-09-25).
    """
    wanted = set(names)
    found = [p.stem for p in finisher_module().collect_props(split_dir) if p.stem in wanted]
    return found + [name for name in names if name not in found]


def split_command(job: PropsJob, blender: Path) -> list[str]:
    # --python-exit-code: without it Blender exits 0 when the script raises, and a split
    # that crashed would look like one that worked.
    command = [
        str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
        "-P", str(SPLITTER), "--",
        str(job.source_glb), str(job.split_dir),
        "--yaw-threshold", str(job.settings["yaw_threshold"]),
    ]
    if job.settings["names"]:
        command += ["--names", *job.settings["names"]]
    for name, degrees in job.settings["turns"].items():
        command += ["--turn", f"{name}={degrees:g}"]
    return command


def finish_command(job: PropsJob, blender: Path, gltfpack: Path | None) -> list[str]:
    """Bake every prop the split wrote, or just the one a turn re-does."""
    source = job.split_dir / f"{job.only}.glb" if job.only else job.split_dir
    settings = job.settings
    command = [
        sys.executable, "-u", str(FINISHER), str(source), str(job.finished_dir),
        "--lods", ",".join(map(str, settings["lods"])),
        "--atlas", str(settings["atlas"]),
        "--metallic", str(settings["metallic"]),
        "--roughness", str(settings["roughness"]),
        "--ior", str(settings["ior"]),
        "--blender", str(blender),
    ]
    if settings["compress"] and gltfpack is not None:
        command += ["--gltfpack", str(gltfpack)]
    else:
        command.append("--no-compress")
    return command


def stage_meta(names: list[str]) -> dict[str, Any]:
    """The panel's rows: the split, then one row per prop. Prefixed so a prop called
    `split` or `done` cannot collide with a phase the panel treats specially."""
    stages = ["split", *(f"prop:{name}" for name in names)]
    labels = {"split": "Split the sheet", **{f"prop:{name}": name for name in names}}
    return {"stages": stages, "stage_labels": labels}


class PropsProgress:
    """Turn the two scripts' output into progress events.

    The split has no sub-progress worth showing (it takes seconds). Once it has written
    its props, the rest of the bar is shared equally between them, and each prop's row
    counts its LODs: `finish_props.py` announces every bake as it starts, on a
    `FINISH::` line of its own, so a prop on its second of three LODs has one done.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.lods = 1
        self.done: dict[str, int] = {}     # LODs baked, per prop, in bake order
        self.stage_started = self.started

    def begin_split(self) -> dict[str, Any]:
        return self._event("split", "Splitting the sheet", 0.0)

    def split_line(self, line: str) -> dict[str, Any] | None:
        if not line.startswith("SPLIT::"):
            return None
        return self._event("split", line[len("SPLIT::"):].strip(), 0.0)

    def begin_props(self, names: list[str], lods: int) -> dict[str, Any]:
        self.lods = max(lods, 1)
        self.done = {name: 0 for name in names}
        event = self._event(f"prop:{names[0]}" if names else "split",
                            f"{len(names)} props to bake", self.fraction())
        event.update(stage_meta(names))
        return event

    def finish_line(self, line: str) -> dict[str, Any] | None:
        marker = finisher_module().PROGRESS_MARKER
        if not line.startswith(marker):
            return None
        try:
            fields = json.loads(line[len(marker):])
        except ValueError:
            return None
        name = fields.get("prop") if isinstance(fields, dict) else None
        if name not in self.done:
            return None
        if isinstance(fields.get("lod"), int):
            index = fields["lod"]
            self.done[name] = max(self.done[name], min(index, self.lods))
            if index == 0:
                self.stage_started = self.clock()
            return self._prop_event(name, f"{name}: baking LOD{index}")
        if "done" in fields:
            self.done[name] = self.lods
            return self._prop_event(name, str(fields["done"]))
        return None

    def fraction(self) -> float:
        if not self.done:
            return 0.0
        split = split_share(len(self.done))
        baked = sum(self.done.values()) / (len(self.done) * self.lods)
        return split + (1.0 - split) * baked

    def _prop_event(self, name: str, message: str) -> dict[str, Any]:
        step = self.done[name]
        event = self._event(f"prop:{name}", message, self.fraction(),
                            step=step, total=self.lods)
        event["stage_pct"] = round(100 * step / self.lods)
        event["stage_eta_seconds"] = remaining_seconds(
            self.clock() - self.stage_started, step / self.lods,
        )
        return event

    def _event(self, phase: str, message: str, fraction: float, **extra: Any) -> dict[str, Any]:
        return {
            "phase": phase,
            "message": message,
            "overall_pct": round(100.0 * fraction, 1),
            "total_eta_seconds": remaining_seconds(self.clock() - self.started, fraction),
            **extra,
        }


class PropsJob:
    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.source_glb = directory / "source.glb"
        self.settings_path = directory / "settings.json"
        self.provenance_path = directory / "props.provenance.json"
        self.split_dir = directory / "split"
        self.finished_dir = directory / "finished"
        # A turn works here and is swapped in only once it has baked, so a turn that is
        # cancelled or fails leaves the run exactly as it was.
        self.staging = directory / "turn"
        self.settings: dict[str, Any] = normalise_settings({})
        self.only: str | None = None   # set for a turn: re-bake just this prop
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


class PropsJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, PropsJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def _refuse_if_busy(self) -> None:
        active = self.jobs.get(self.active) if self.active else None
        if active is not None and active.status not in TERMINAL:
            raise RuntimeError("a prop-sheet job is already running")

    def create(
        self, asset_name: str, asset: bytes, settings: dict[str, Any],
        source_record: dict[str, Any] | None = None, generated: bool = False,
    ) -> PropsJob:
        """A new run. `source_record` is the provenance record of a generated model,
        so its licence travels with the props made from it; `generated` says the model
        was picked from what this machine made, whether or not a record was kept."""
        with self.lock:
            self._refuse_if_busy()
            clean = normalise_settings(settings)
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__props__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__props__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = PropsJob(uuid.uuid4().hex, directory)
            try:
                job.settings = clean
                job.source_glb.write_bytes(asset)
                job.settings_path.write_text(json.dumps(job.settings, indent=2))
                job.provenance_path.write_text(json.dumps(
                    provenance_record(asset_name, asset, source_record, generated),
                    indent=2))
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def turn(self, name: str, prop: str, degrees: float) -> PropsJob:
        """A job that turns one prop of an existing run and re-bakes only that prop.

        The turn is added to the run's recorded turns, so turning the chest twice by 90
        faces it backwards, as clicking the button twice should. It is recorded when the
        job finishes, so one that is cancelled or fails leaves no turn behind.
        """
        with self.lock:
            self._refuse_if_busy()
            directory = run_directory(self.output_root, name)
            job = PropsJob(uuid.uuid4().hex, directory)
            if not job.source_glb.is_file() or not job.settings_path.is_file():
                raise RuntimeError(f"{name} is missing its source.glb or settings.json")
            settings = normalise_settings(json.loads(job.settings_path.read_text()))
            if prop not in prop_names(job.split_dir):
                raise RuntimeError(f"{name} has no prop called {prop!r}")
            step = _number("degrees", degrees)
            if not -180.0 <= step <= 180.0:
                raise ValueError(f"degrees must be within -180..180, got {step}")
            if settings["compress"] and find_gltfpack() is None and any(
                    (directory / "finished" / prop).glob("*.web.glb")):
                raise RuntimeError(
                    f"gltfpack is missing now, so turning {prop} would lose its smaller "
                    "web files. Install it in Setup & Status, then turn it again.")
            turns = settings["turns"]
            turns[prop] = wrap_degrees(turns.get(prop, 0.0) + step)
            if turns[prop] == 0.0:
                del turns[prop]
            job.settings = settings
            job.only = prop
            job.split_dir = job.staging / "split"
            job.finished_dir = job.staging / "finished"
            # Recorded by run_job once the prop is re-baked, not here: a cancelled turn
            # that stayed recorded made the next click turn the prop twice (seen
            # 2026-09-25, a barrel that came out facing backwards).
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> PropsJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: PropsJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


PROPS_JOBS = PropsJobManager()


def run_job(job: PropsJob, manager: PropsJobManager = PROPS_JOBS) -> None:
    try:
        blender = blender_executable()
        if blender is None:
            raise RuntimeError(missing_help())
        for script in (SPLITTER, FINISHER):
            if not script.is_file():
                raise RuntimeError(f"script missing: {script}")
        if job.cancel_requested:     # cancelled while still queued
            _cancelled(job)
            return
        gltfpack = find_gltfpack() if job.settings["compress"] else None
        job.status = "running"
        if job.only:
            shutil.rmtree(job.staging, ignore_errors=True)   # a crashed turn's leftovers
        progress = PropsProgress()
        rows = [job.only] if job.only else []
        job.emit({**progress.begin_split(), **stage_meta(rows)})

        if not _run_step(job, "split", split_command(job, blender), progress.split_line):
            return
        names = prop_names(job.split_dir)
        if not names:
            raise RuntimeError("the split found no props in this GLB")
        if job.only:
            if job.only not in names:
                raise RuntimeError(f"the split no longer has a prop called {job.only!r}")
            names = [job.only]

        event = progress.begin_props(bake_order(job.split_dir, names), len(job.settings["lods"]))
        if job.settings["compress"] and gltfpack is None:
            event["message"] += "; gltfpack not found, so no .web.glb files"
        job.emit(event)
        if not _run_step(job, "LODs", finish_command(job, blender, gltfpack),
                         progress.finish_line):
            return

        if job.only:
            swap_in_turn(job)
            job.settings_path.write_text(json.dumps(job.settings, indent=2))
        run = describe_run(job.directory)
        job.status = "done"
        job.emit({
            "phase": "done", "overall_pct": 100,
            "message": (f"Turned and re-baked {job.only}" if job.only
                        else f"{len(run['props'])} props finished"),
            "directory": job.directory.name,
            "run": run,
        })
    # SystemExit too: the finish_props helpers used here end that way, as a script would,
    # and without it the job stayed "running" with no word to the page.
    except (Exception, SystemExit) as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc),
                  "log_tail": "\n".join(job.log_lines)[-8000:]})
    finally:
        if job.only:
            shutil.rmtree(job.staging, ignore_errors=True)
        job.directory.joinpath("pid").unlink(missing_ok=True)
        manager.finish(job)


def swap_in_turn(job: PropsJob) -> None:
    """Move a finished turn into the run: the new split, the turned prop's LODs, and its
    entry in `finished/finish_props.json`.

    Up to four renames, undone in reverse if one fails, so the run ends with all the old
    files or all the new ones. A listing read between two of them can miss the split or
    the prop for that moment; the next refresh sees the run whole.
    """
    live_split = job.directory / "split"
    live_finished = job.directory / "finished"
    live_prop = live_finished / job.only
    staged_prop = job.finished_dir / job.only
    if not (job.split_dir / "props.json").is_file() or not staged_prop.is_dir():
        raise RuntimeError(f"the turn of {job.only} did not produce its files")
    record = turned_record(job)
    old = job.staging / "old"
    # The old prop gets a folder of its own: a prop may well be called split.
    (old / "finished").mkdir(parents=True)
    live_finished.mkdir(exist_ok=True)
    moves = [(live_split, old / "split"), (job.split_dir, live_split)]
    if live_prop.exists():
        moves.append((live_prop, old / "finished" / job.only))
    moves.append((staged_prop, live_prop))
    done: list[tuple[Path, Path]] = []
    try:
        for source, target in moves:
            source.replace(target)
            done.append((source, target))
    except OSError:
        for source, target in reversed(done):
            target.replace(source)
        raise
    if record is not None:
        (live_finished / "finish_props.json").write_text(json.dumps(record, indent=2))


def turned_record(job: PropsJob) -> dict[str, Any] | None:
    """The run's finish record with the turned prop's entry swapped for the turn's.

    The turn bakes one prop into its own record, which goes when staging does; without
    this the run's record kept the old triangles and sizes of files that were replaced.
    The entry moves as it is: its paths are relative to the folder it was baked into,
    and `turn/finished/` sits beside `turn/split/` as `finished/` does beside `split/`.
    """
    record = _read_json(job.directory / "finished" / "finish_props.json")
    staged = _read_json(job.finished_dir / "finish_props.json")
    if not isinstance(record, dict) or not isinstance(staged, dict):
        return None
    entry = next((e for e in staged.get("props", [])
                  if isinstance(e, dict) and e.get("name") == job.only), None)
    if entry is None:
        return None
    entries = [e for e in record.get("props", []) if isinstance(e, dict)]
    names = [e.get("name") for e in entries]
    if job.only in names:
        entries[names.index(job.only)] = entry
    else:
        entries.append(entry)
    return {**record, "props": entries}


def _cancelled(job: PropsJob) -> None:
    job.status = "cancelled"
    job.emit({"phase": "cancelled", "message": "Cancelled"})


def _run_step(job: PropsJob, label: str, command: list[str], feed) -> bool:
    """One subprocess, its output logged and fed to the progress parser.

    False, with the cancel already reported, when the job was cancelled; a step that
    fails raises, for run_job to report. A cancel that lands between two steps is
    honoured here, before the next one starts: cancel_job only kills a running process.
    """
    if job.cancel_requested:
        _cancelled(job)
        return False
    job.process = subprocess.Popen(
        command, cwd=str(REPO), env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
        event = feed(line)
        if event:
            job.emit(event)
    code = job.process.wait()
    job.append_log(f"{label} exited with code {code}")
    if job.cancel_requested:
        _cancelled(job)
        return False
    if code != 0:
        raise RuntimeError(f"{label} exited with code {code}")
    return True


def cancel_job(job: PropsJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: PropsJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "settings": job.settings,
        "directory": job.directory.name,
        "log_tail": "\n".join(job.log_lines)[-4000:],
        "last_event": job.events[-1] if job.events else None,
    }


def run_directory(root: Path, name: str) -> Path:
    """One run directory under `root`, by name, with no way out of it."""
    if not RUN_DIRECTORY.fullmatch(name):
        raise RuntimeError(f"not a run directory name: {name!r}")
    directory = root / name
    if not directory.is_dir():
        raise RuntimeError(f"no such run: {name}")
    return directory


def split_entries(split_dir: Path) -> list[dict[str, Any]]:
    """The split's record of each prop, in reading order, skipping any without a usable
    name: the name becomes a path and a command-line argument."""
    record = _read_json(split_dir / "props.json")
    entries = record.get("props") if isinstance(record, dict) else None
    return [entry for entry in entries or []
            if isinstance(entry, dict) and PROP_NAME.fullmatch(str(entry.get("name", "")))]


def prop_names(split_dir: Path) -> list[str]:
    """The props the split wrote, in reading order, from its own record."""
    return [entry["name"] for entry in split_entries(split_dir)]


def describe_run(directory: Path) -> dict[str, Any]:
    """Everything the page shows about one run, read from disk.

    The LODs come from the files in `finished/`, not from `finish_props.json`, so a
    prop whose bake stopped part-way shows the LODs it has.
    """
    settings = _read_json(directory / "settings.json") or {}
    lod_path = finisher_module().lod_path
    props = []
    for entry in split_entries(directory / "split"):
        name = entry["name"]
        lods = []
        for index in range(MAX_LODS):
            plain = lod_path(directory / "finished", name, index)
            size = _size(plain)
            if size is None:
                break
            web = lod_path(directory / "finished", name, index, web=True)
            web_size = _size(web)
            lods.append({
                "index": index,
                "url": served_url(plain),
                "bytes": size,
                "triangles": lod_triangles(plain),
                "web_url": served_url(web) if web_size is not None else None,
                "web_bytes": web_size,
            })
        # The plain LOD0, not the .web one: the viewer has no meshopt decoder, and the two
        # look the same. Before its LODs exist, the split prop itself.
        preview = lods[0]["url"] if lods else _url_if_file(directory / "split" / f"{name}.glb")
        props.append({
            "name": name,
            "faces": entry.get("faces"),
            "size": entry.get("size"),
            "tilt_degrees": entry.get("tilt_degrees"),
            "yaw_degrees": entry.get("yaw_degrees", 0.0),
            "yaw_tie": bool(entry.get("yaw_tie")),
            "extra_turn_degrees": entry.get("extra_turn_degrees", 0.0),
            "lods": lods,
            "preview_url": preview,
        })
    finished = bool(props) and all(len(p["lods"]) == len(settings.get("lods", [1])) for p in props)
    return {
        "directory": directory.name,
        "modified": directory.stat().st_mtime,
        "settings": settings,
        "props": props,
        "finished": finished,
        "licence": licence_of(_read_json(directory / "props.provenance.json")),
        "blend_url": _url_if_file(directory / "split" / "props.blend"),
    }


def list_runs(root: Path = OUTPUT_ROOT, limit: int = 15) -> list[dict[str, Any]]:
    """Every prop-sheet run on disk, newest first."""
    if not root.is_dir():
        return []
    directories = sorted(
        (d for d in root.iterdir() if d.is_dir() and RUN_DIRECTORY.fullmatch(d.name)),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    runs = []
    for directory in directories[:limit]:
        try:
            runs.append(describe_run(directory))
        except OSError:
            continue     # removed or half-written while listing; the next refresh has it
    return runs


def provenance_record(
    asset_name: str, asset: bytes, source_record: dict[str, Any] | None,
    generated: bool = False,
) -> dict[str, Any]:
    """The run's own provenance: what went in, and the licence it inherits.

    The props are derived from the source model, so its licence is theirs; an upload has
    no record, and says so rather than implying it is clear to ship. A generated model
    can have lost its record too (the viewer deletes it with Debug off), and is still
    called generated, not uploaded.
    """
    source = source_record if isinstance(source_record, dict) else None
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "viewer Props tab: scripts/blender_split_props.py, scripts/finish_props.py",
        "input": {
            "name": asset_name,
            "sha256": hashlib.sha256(asset).hexdigest(),
            "source": "generated" if source or generated else "uploaded",
        },
        "license": (source or {}).get("license"),
        "classification": ((source or {}).get("output") or {}).get("classification"),
        "source_record": source,
    }


def licence_of(record: dict[str, Any] | None) -> dict[str, Any]:
    """What the page says about a run's licence."""
    record = record or {}
    licence = record.get("license") if isinstance(record.get("license"), dict) else {}
    return {
        "source": (record.get("input") or {}).get("source", "unknown"),
        "name": licence.get("name"),
        "url": licence.get("url"),
        "classification": record.get("classification"),
    }


def source_record_for(glb: Path) -> dict[str, Any] | None:
    """The licence record kept beside a generated GLB, or None.

    `<name>.provenance.json` is the sidecar that travels with the file; `<name>.json` is
    the run manifest some backends write, which the viewer deletes with Debug off.
    """
    for sidecar in (glb.with_suffix(".provenance.json"), glb.with_suffix(".json")):
        record = _read_json(sidecar)
        if isinstance(record, dict) and "license" in record:
            return record
    return None


def generated_models(root: Path, limit: int = 25) -> list[dict[str, Any]]:
    """GLBs the Generate tab made, newest first: `<name>/<name>.glb` one or two levels
    under output/ (a licence-class folder may sit in between)."""
    if not root.is_dir():
        return []
    found = []
    for top in root.iterdir():
        if not top.is_dir() or top.name in NOT_GENERATED or top.name.startswith("."):
            continue
        for directory in (top, *(d for d in top.iterdir() if d.is_dir())):
            glb = directory / f"{directory.name}.glb"
            if glb.is_file():
                found.append(glb)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.stem, "path": p.relative_to(root).as_posix(),
             "bytes": p.stat().st_size, "url": served_url(p)} for p in found[:limit]]


def generated_model(root: Path, relative: str) -> Path:
    """One of `generated_models`, by the path the page was given, and nothing else."""
    for model in generated_models(root, limit=10_000):
        if model["path"] == relative:
            return root / relative
    raise RuntimeError(f"not a generated model: {relative!r}")


def tools_payload() -> dict[str, Any]:
    """What the page must know before offering a run: can it split, and can it compress."""
    blender = blender_executable()
    gltfpack = find_gltfpack()
    return {"blender": str(blender) if blender else None,
            "blender_problem": None if blender else missing_help(),
            "gltfpack": str(gltfpack) if gltfpack else None,
            # Setup's gltfpack card offers Install (scripts/bootstrap_gltfpack.py).
            "gltfpack_installable": gltfpack is None and gltfpack_bootstrap().can_install()}


def _size(path: Path) -> int | None:
    """A file's size, or None if it is missing, including one removed a moment ago."""
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _url_if_file(path: Path) -> str | None:
    return served_url(path) if path.is_file() else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "sheet"
