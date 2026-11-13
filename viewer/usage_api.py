"""Anonymous usage counts: how many people use AssetFurnace, how, and how much they make.

On unless the person unticks the box on the About page (or sets `I3D_NO_USAGE=1` or
`DO_NOT_TRACK=1`). Nothing is sent until the studio has shown its one-time notice and the
person has dismissed it or made a choice on About; counts wait on this computer till then.
At most once a day, one small message goes to assetfurnace.com, and it is exactly what
`payload` returns: a random id (made on this copy, not derived from the machine or the
person), the day, the version, the kind of machine and graphics card, and since the last
message how many jobs were started from the studio and from an agent (the MCP server), how
many finished, which failed and how, and which of a short list of choices were used. Every
value comes from a closed list written here, never from an error message or the person. No
pictures, prompts, file names or paths, ever.

Turning it off deletes the id and every unsent count and remembers only "off", so turning it
on again starts a new id. A state file that exists but cannot be read counts as off: a fault
must never turn sharing back on. Offline, the counts wait, with one try per half hour.

Counting is a nicety: every entry point the studio calls — including the ones called from
inside a job's own error handling — swallows its own errors, so a usage fault can never fail a
job or a request, and can never hide the error a person needs to see.
"""

from __future__ import annotations

import datetime
import functools
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_to_3dlab
from image_to_3dlab.host import chip_name, host_platform, os_family

USAGE_URL = os.environ.get("I3D_USAGE_URL", "https://assetfurnace.com/api/usage")
POLICY_URL = "https://assetfurnace.com/privacy#usage-counts"
STATE = REPO / "output" / ".usage.json"
SCHEMA = 3
CARD_MAX = 64              # a graphics card name is a product name, short by nature
COUNT_MAX = 10000          # the site refuses more of one kind in one message; the rest waits
RETRY_SECONDS = 30 * 60    # offline or refused: one more try per half hour, not one per job
LOCK = threading.RLock()   # every read and write of the state file; a job, a page and an agent can land together
_SENDING = threading.Event()

# The header the MCP server puts on every call, so a job an agent started counts as an agent's.
CLIENT_HEADER = "X-AssetFurnace-Client"
AGENT = "agent"
SOURCES = ("studio", AGENT)

# POST routes that start work, by what they make. Matched on the path; "*" is any one part.
STARTS = {
    ("api", "chains"): "chain",
    ("api", "generate"): "model",
    ("api", "image"): "picture",
    ("api", "image", "back"): "picture",
    ("api", "finish"): "finish",
    ("api", "finish", "runs", "*", "resume"): "finish",
    ("api", "props"): "prop",
    ("api", "props", "runs", "*", "turn"): "prop",
    ("api", "skins", "*", "build"): "skin",
    ("api", "animate", "rig"): "rig",
    ("api", "animate", "play"): "animation",
    ("api", "rig", "rebind"): "rig",
}
KINDS = frozenset(STARTS.values())

# Why a job failed, as far as we can tell from our own error path. A closed list: the error
# text never leaves the machine, only which of these it looked like. In order of asking, so
# "out of memory" is not filed as a crash.
FAILURE_KINDS = ("card_too_old", "missing_package", "out_of_memory", "download", "crashed", "other")
CARD_TOO_OLD = re.compile(r"older than rig supports|no kernel image|unsupported gpu architecture")
MISSING_PACKAGE = re.compile(r"modulenotfounderror|importerror|no module named|cannot import name")
OUT_OF_MEMORY = re.compile(r"out of memory|outofmemory|memoryerror|\boom\b|\bkilled\b"
                           r"|(?:exited with|exit) (?:code )?137")
DOWNLOAD = re.compile(r"download|urlopen|urllib|huggingface|checksum|http error \d|connection (?:refused|reset)")
CRASHED = re.compile(r"exited with code|exit code|exit \d|non-zero exit|stopped \(exit|returned non-zero"
                     r"|without writing|wrote no glb")

#: The height Finish gives a character when nobody asked for one (mcp_server and the studio both
#: fill it in). Choosing it is not a choice, so it is not counted as one.
DEFAULT_HEIGHT_M = 1.8

# Choices worth knowing about, from a closed list. `x` means never: a typo in our own code is
# a bug, not a new statistic.
FEATURES = ("hero", "pixel_match_off", "skin", "cleanup_fast", "cleanup_rebuild", "height_set",
            "upright_off", "surface_maps_drop", "texture_512", "front_back")


def _quiet(default: Any = None) -> Callable:
    """The studio-facing calls never raise: a usage fault must not fail a job or a request."""
    def wrap(function: Callable) -> Callable:
        @functools.wraps(function)
        def safe(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                return default() if callable(default) else default
        return safe
    return wrap


def _path(state: Path | None) -> Path:
    """Looked up at call time, so tests point it at a temporary folder, never a real install."""
    return STATE if state is None else state


def _read(path: Path | None) -> dict[str, Any] | None:
    """The state, {} when there is none yet, or None when a file is there but cannot be read."""
    path = _path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return None
    try:
        state = json.loads(text)
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def _write(path: Path | None, state: dict[str, Any]) -> None:
    path = _path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())   # a crash must not leave an empty file behind
        for attempt in range(3):
            try:
                temporary.replace(path)
                return
            except PermissionError:   # Windows: something else has the file open for a moment
                time.sleep(0.05 * (attempt + 1))
    except OSError:
        pass   # counts are a nicety: a read-only folder must never break the studio


def _today() -> str:
    """The UTC day, so every copy agrees where a day starts when the counts are added up."""
    return datetime.datetime.now(datetime.UTC).date().isoformat()


def blocked(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    truthy = {"1", "true", "yes"}
    return (env.get("I3D_NO_USAGE", "").strip().lower() in truthy
            or env.get("DO_NOT_TRACK", "").strip().lower() in truthy)


def _sharing(current: dict[str, Any] | None, env: dict[str, str] | None) -> bool:
    return current is not None and not blocked(env) and not current.get("opted_out")


def sharing(state: Path | None = None, env: dict[str, str] | None = None) -> bool:
    """On by default: only an unticked box, the environment, or an unreadable file turn it off."""
    with LOCK:
        return _sharing(_read(state), env)


def _with_id(current: dict[str, Any]) -> dict[str, Any]:
    """The state with an id, made the first time anything is counted or sent."""
    if not current.get("id"):
        current = {"noticed": bool(current.get("noticed")),
                   "schema_version": SCHEMA, "id": uuid.uuid4().hex, "sent_day": None}
    return current


def set_sharing(on: bool, state: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Tick or untick the box. Off forgets the id and the counts; on starts afresh.
    Either way the person has seen the choice, so the first-start notice stays away."""
    if blocked(env):
        return status(state, env)   # the environment decides; the box is greyed out
    with LOCK:
        if on:
            _write(state, {"schema_version": SCHEMA, "noticed": True})
        else:
            _write(state, {"schema_version": SCHEMA, "noticed": True, "opted_out": True})
    return status(state, env)


def set_noticed(state: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """The first-start notice was dismissed: it does not come back, and sending may begin."""
    with LOCK:
        current = _read(state)
        if current is not None:
            current["noticed"] = True
            _write(state, current)
    send_soon(state, env)
    return status(state, env)


def _bump(current: dict[str, Any], group: str, kind: str, source: str | None = None) -> None:
    counts = current.get(group)
    if not isinstance(counts, dict):
        counts = current[group] = {}
    if source is not None:
        if not isinstance(counts.get(source), dict):
            counts[source] = {}
        counts = counts[source]
    counts[kind] = _number(counts.get(kind)) + 1


def _number(value: Any) -> int:
    """A count from the file, or 0 for anything that is not one (a hand-edited file, say)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def classify_failure(reason: object) -> str:
    """Which kind of failure this was, from FAILURE_KINDS. The text itself is never kept.

    `reason` is whatever our own error path already has: the worker's log tail, the exception
    that ended a job, or the exit code it left. Called in order of specificity, so a crash
    that says "out of memory" is filed as memory rather than as a generic exit.
    """
    if isinstance(reason, bool):          # True is not an exit code, whatever Python thinks
        return "other"
    if isinstance(reason, int):
        if reason in (137, -9):      # SIGKILL: the kernel's way of saying out of memory
            return "out_of_memory"
        return "crashed" if reason != 0 else "other"
    if isinstance(reason, BaseException):
        if isinstance(reason, ImportError):   # ModuleNotFoundError is an ImportError
            return "missing_package"
        text = f"{type(reason).__name__}: {reason}"
    else:
        text = str(reason or "")
    low = text.lower()
    for kind, pattern in (("card_too_old", CARD_TOO_OLD), ("missing_package", MISSING_PACKAGE),
                          ("out_of_memory", OUT_OF_MEMORY), ("download", DOWNLOAD),
                          ("crashed", CRASHED)):
        if pattern.search(low):
            return kind
    return "other"


def feature_keys(settings: dict[str, Any] | None = None, *, skin: bool = False,
                 front_and_back: bool = False) -> list[str]:
    """Which of FEATURES this job used, from the settings the job already carries.

    Pure and closed: a key is named here or it is not counted, and a value nobody chose
    (None, missing) counts as nothing. A skin is a job of its own; the Finish choices it
    inherits from its model were counted when the model was finished.
    """
    settings = settings if isinstance(settings, dict) else {}
    used = []
    if settings.get("hero"):
        used.append("hero")
    if settings.get("skip_photo"):
        used.append("pixel_match_off")
    if skin:
        used.append("skin")
    if settings.get("cleanup") == "fast":
        used.append("cleanup_fast")
    elif settings.get("cleanup") == "rebuild":
        used.append("cleanup_rebuild")
    height = settings.get("height_m")
    if (isinstance(height, (int, float)) and not isinstance(height, bool)
            and abs(float(height) - DEFAULT_HEIGHT_M) > 1e-6):
        used.append("height_set")   # 1.8 m is what every character gets; only another height is a choice
    if settings.get("upright") is False:
        used.append("upright_off")
    if settings.get("drop_surface_maps"):
        used.append("surface_maps_drop")
    if settings.get("texture_size") == 512:
        used.append("texture_512")
    if front_and_back:
        used.append("front_back")
    return used


@_quiet()
def count_features(settings: dict[str, Any] | None = None, *, skin: bool = False,
                   front_and_back: bool = False, state: Path | None = None,
                   env: dict[str, str] | None = None) -> None:
    """Count the choices a job was started with. Nothing when sharing is off.

    A skin passes `skin=True` and no settings: its Finish choices are the model's, already
    counted once, and counting them again per skin would make one choice look like several.
    """
    used = feature_keys(settings, skin=skin, front_and_back=front_and_back)
    if not used:
        return
    with LOCK:
        current = _read(state)
        if not _sharing(current, env):
            return
        current = _with_id(current)
        for feature in used:
            _bump(current, "features", feature)
        _write(state, current)
    send_soon(state, env)


@_quiet()
def count_failed(step: str, reason: object, state: Path | None = None,
                 env: dict[str, str] | None = None) -> None:
    """One more job of `step` failed, filed by kind. The reason is classified, never sent.

    An unknown step is dropped, not raised: this is called from our own error paths, and the
    rule there is that counting can never fail a job. A typo would simply stop counting, so
    `test_every_counted_failure_names_a_step_we_count` checks the names at the call sites.
    """
    if step not in KINDS:
        return
    _count_failed(step, classify_failure(reason), state, env)


@_quiet()
def _count_failed(step: str, kind: str, state: Path | None, env: dict[str, str] | None) -> None:
    with LOCK:
        current = _read(state)
        if not _sharing(current, env):
            return
        current = _with_id(current)
        failed = current.get("failed")
        if not isinstance(failed, dict):
            failed = current["failed"] = {}
        counts = failed.get(step)
        if not isinstance(counts, dict):
            counts = failed[step] = {}
        counts[kind] = _number(counts.get(kind)) + 1
        _write(state, current)
    send_soon(state, env)


def card_name() -> str | None:
    """This machine's graphics card (or Apple chip), as a product name, or None.

    Cut to what the site accepts (printable ASCII, no path separators, 64 characters): an
    odd name from a vendor must never cost the whole message. A product name, never a
    host name, a path or anything that could pick out the person.
    """
    try:
        name = chip_name()
    except Exception:
        return None
    text = str(name or "").encode("ascii", "ignore").decode("ascii")
    text = " ".join(text.replace("/", " ").replace("\\", " ").split())
    return text[:CARD_MAX] or None


def source_of(header: str | None) -> str:
    return AGENT if (header or "").strip().lower() == AGENT else "studio"


def started_kind(parts: list[str]) -> str | None:
    """What a POST to `parts` starts, or None for a route that starts no work."""
    for route, kind in STARTS.items():
        if len(route) == len(parts) and all(r in ("*", p) for r, p in zip(route, parts)):
            return kind
    return None


@_quiet()
def note_request(parts: list[str], header: str | None, method: str, status_code: int | None,
                 state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Called after every studio request. Counts an agent being used, and work started."""
    source = source_of(header)
    kind = started_kind(parts) if method == "POST" and status_code and 200 <= status_code < 300 else None
    if source != AGENT and kind is None:
        return
    with LOCK:
        current = _read(state)
        if not _sharing(current, env):
            return
        if kind is None and current.get("agent_at") and not _SENDING.is_set():
            return   # already marked since the last message: an agent polling must not rewrite the file
        current = _with_id(current)
        if source == AGENT:
            current["agent_at"] = time.time()
        if kind:
            _bump(current, "started", kind, source)
        _write(state, current)
    if kind:
        send_soon(state, env)


def count_done(kind: str, state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """One more job of `kind` finished, sent with the next daily message. Nothing when off."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")   # a typo in our own code, caught by the tests
    _count_done(kind, state, env)


@_quiet()
def _count_done(kind: str, state: Path | None, env: dict[str, str] | None) -> None:
    with LOCK:
        current = _read(state)
        if not _sharing(current, env):
            return
        current = _with_id(current)
        _bump(current, "finished", kind)
        _write(state, current)
    send_soon(state, env)


def _counts(value: Any) -> dict[str, int]:
    """Clean counts for the message: real numbers only, each at most COUNT_MAX."""
    if not isinstance(value, dict):
        return {}
    return {k: min(_number(n), COUNT_MAX) for k, n in value.items()
            if isinstance(k, str) and _number(n) > 0}


def _failures(value: Any) -> dict[str, dict[str, int]]:
    """Clean failed counts: known steps, known kinds, real numbers. Junk in the file is dropped."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for step, counts in value.items():
        if step not in KINDS or not isinstance(counts, dict):
            continue
        kept = {kind: min(_number(n), COUNT_MAX) for kind, n in counts.items()
                if kind in FAILURE_KINDS and _number(n) > 0}
        if kept:
            out[step] = kept
    return out


def _features(value: Any) -> dict[str, int]:
    """Clean feature counts: only the keys we named, so a hand-edited file sends no strangers."""
    if not isinstance(value, dict):
        return {}
    return {k: min(_number(n), COUNT_MAX) for k, n in value.items()
            if k in FEATURES and _number(n) > 0}


def payload(current: dict[str, Any], day: str) -> dict[str, Any]:
    """Everything that is sent. The About page shows this too, so what it says is what goes."""
    started = current.get("started") if isinstance(current.get("started"), dict) else {}
    return {"schema_version": SCHEMA, "id": current.get("id"), "day": day,
            "version": image_to_3dlab.__version__, "os": os_family(), "machine": host_platform(),
            "card": card_name(),
            "started": {source: _counts(started.get(source)) for source in SOURCES},
            "finished": _counts(current.get("finished")),
            "failed": _failures(current.get("failed")),
            "features": _features(current.get("features")),
            "agent": bool(current.get("agent_at"))}


def _post(body: dict[str, Any]) -> None:
    request = urllib.request.Request(USAGE_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": "assetfurnace-usage"})
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def _take_off(latest: dict[str, Any], sent: dict[str, Any], started_at: float) -> None:
    """Subtract what was sent, so a job counted during the send waits for the next one."""
    def minus(counts: Any, gone: dict[str, int]) -> dict[str, int]:
        counts = counts if isinstance(counts, dict) else {}
        left = {k: _number(n) - gone.get(k, 0) for k, n in counts.items()}
        return {k: n for k, n in left.items() if n > 0}

    started = latest.get("started") if isinstance(latest.get("started"), dict) else {}
    latest["started"] = {s: minus(started.get(s), sent["started"].get(s, {})) for s in started}
    latest["finished"] = minus(latest.get("finished"), sent["finished"])
    latest["features"] = minus(latest.get("features"), sent["features"])
    failed = latest.get("failed") if isinstance(latest.get("failed"), dict) else {}
    left: dict[str, dict[str, int]] = {}
    for step, counts in failed.items():
        remaining = minus(counts, sent["failed"].get(step, {}))
        if remaining:
            left[step] = remaining
    latest["failed"] = left
    if sent["agent"] and float(latest.get("agent_at") or 0) <= started_at:
        latest.pop("agent_at", None)   # an agent call during the send stays for the next message


def _due(current: dict[str, Any], today: str, now: float) -> bool:
    return (bool(current.get("noticed")) and current.get("sent_day") != today
            and now - float(current.get("tried_at") or 0) >= RETRY_SECONDS)


def send_daily(state: Path | None = None, env: dict[str, str] | None = None, today: str | None = None,
               post: Callable[[dict[str, Any]], None] | None = None,
               clock: Callable[[], float] = time.time) -> bool:
    """Send today's message if sharing, the notice was seen, and it is due. True when one went out."""
    today = today or _today()
    started_at = clock()
    with LOCK:
        current = _read(state)
        if _SENDING.is_set() or not _sharing(current, env) or not _due(current, today, started_at):
            return False   # one send at a time, or the same counts would go twice
        current = _with_id(current)
        current["tried_at"] = started_at
        _write(state, current)   # keep the id the message carries, and the time of this try
        snapshot = json.loads(json.dumps(current))
        _SENDING.set()
    try:
        body = payload(snapshot, today)   # outside the lock: the first machine check can take seconds
        refused = False
        try:
            (post or _post)(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                return False   # busy or down: try again later
            refused = True     # the site will never take these counts: drop them, do not resend forever
        except (OSError, ValueError, urllib.error.URLError, TimeoutError, http.client.HTTPException):
            return False       # offline: the counts wait
        with LOCK:
            latest = _read(state)
            if latest is None or latest.get("id") != body["id"]:
                return False   # turned off (or off and on) while it was sending: keep the new state
            _take_off(latest, body, started_at)
            latest["sent_day"] = today
            _write(state, latest)
        return not refused
    finally:
        _SENDING.clear()


@_quiet()
def send_soon(state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """send_daily on a thread, never two at once, so a request never waits on the network."""
    with LOCK:
        current = _read(state)
        if _SENDING.is_set() or not _sharing(current, env) or not _due(current, _today(), time.time()):
            return
    threading.Thread(target=_quiet()(send_daily), args=(state, env), daemon=True, name="usage-send").start()


def _off_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    return {"enabled": False, "allowed": not blocked(env), "sends": None, "noticed": True,
            "url": USAGE_URL, "policy": POLICY_URL}


@_quiet(lambda: _off_status())
def status(state: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    with LOCK:
        current = _read(state)
        on = _sharing(current, env)
    if not on:
        return _off_status(env)   # nothing to tell someone who is not sharing
    return {"enabled": True, "allowed": True, "sends": payload(current, _today()),
            "noticed": bool(current.get("noticed")), "url": USAGE_URL, "policy": POLICY_URL}
