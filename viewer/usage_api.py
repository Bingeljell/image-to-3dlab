"""Anonymous usage counts: how many people use AssetFurnace, how, and how much they make.

On unless the person unticks the box on the About page (or sets `I3D_NO_USAGE=1` or
`DO_NOT_TRACK=1`). At most once a day, one small message goes to assetfurnace.com, and it is
exactly what `payload` returns: a random id (made on this copy, not derived from the machine
or the person), the day, the version, the kind of machine, and since the last message how
many jobs were started from the studio and from an agent (the MCP server), how many finished,
and whether an agent was used at all. No pictures, prompts, file names or paths, ever.

Turning it off deletes the id and every unsent count and remembers only "off", so turning it
on again starts a new id. Offline, the counts wait for the next send.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import threading
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
from image_to_3dlab.host import host_platform, os_family

USAGE_URL = os.environ.get("I3D_USAGE_URL", "https://assetfurnace.com/api/usage")
POLICY_URL = "https://assetfurnace.com/privacy#usage-counts"
STATE = REPO / "output" / ".usage.json"
SCHEMA = 2
LOCK = threading.Lock()   # a finished job, a page load and an agent call can land together
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
    ("api", "props"): "prop",
    ("api", "skins", "*", "build"): "skin",
    ("api", "animate", "rig"): "rig",
    ("api", "animate", "play"): "animation",
    ("api", "rig", "rebind"): "rig",
}
KINDS = frozenset(STARTS.values())


def _path(state: Path | None) -> Path:
    """Looked up at call time, so tests point it at a temporary folder, never a real install."""
    return STATE if state is None else state


def _read(path: Path | None) -> dict[str, Any]:
    path = _path(path)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path | None, state: dict[str, Any]) -> None:
    path = _path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass   # counts are a nicety: a read-only folder must never break the studio


def _today() -> str:
    """The UTC day, so every copy agrees where a day starts when the counts are added up."""
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def blocked(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("I3D_NO_USAGE") == "1" or env.get("DO_NOT_TRACK") == "1"


def sharing(state: Path | None = None, env: dict[str, str] | None = None) -> bool:
    """On by default: only an unticked box or the environment turns it off."""
    return not blocked(env) and not _read(state).get("opted_out")


def _current(state: Path | None) -> dict[str, Any]:
    """The state with an id, made the first time anything is counted or sent."""
    current = _read(state)
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
    """The first-start notice was dismissed: it does not come back."""
    with LOCK:
        current = _read(state)
        current["noticed"] = True
        _write(state, current)
    return status(state, env)


def _bump(current: dict[str, Any], group: str, kind: str, source: str | None = None) -> None:
    counts = current.setdefault(group, {})
    if source is not None:
        counts = counts.setdefault(source, {})
    counts[kind] = int(counts.get(kind) or 0) + 1


def source_of(header: str | None) -> str:
    return AGENT if (header or "").strip().lower() == AGENT else "studio"


def started_kind(parts: list[str]) -> str | None:
    """What a POST to `parts` starts, or None for a route that starts no work."""
    for route, kind in STARTS.items():
        if len(route) == len(parts) and all(r in ("*", p) for r, p in zip(route, parts)):
            return kind
    return None


def note_request(parts: list[str], header: str | None, method: str, status_code: int | None,
                 state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Called after every studio request. Counts an agent being used, and work started."""
    source = source_of(header)
    kind = started_kind(parts) if method == "POST" and status_code and 200 <= status_code < 300 else None
    if source != AGENT and kind is None:
        return
    with LOCK:
        if not sharing(state, env):
            return
        current = _current(state)
        if source == AGENT:
            current["agent_seen"] = True
        if kind:
            _bump(current, "started", kind, source)
        _write(state, current)
    if kind:
        send_soon(state, env)


def count_done(kind: str, state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """One more job of `kind` finished, sent with the next daily message. Nothing when off."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    with LOCK:
        if not sharing(state, env):
            return
        current = _current(state)
        _bump(current, "finished", kind)
        _write(state, current)
    send_soon(state, env)


def payload(current: dict[str, Any], day: str) -> dict[str, Any]:
    """Everything that is sent. The About page shows this too, so what it says is what goes."""
    started = current.get("started") or {}
    return {"schema_version": SCHEMA, "id": current.get("id"), "day": day,
            "version": image_to_3dlab.__version__, "os": os_family(), "machine": host_platform(),
            "started": {source: dict(started.get(source) or {}) for source in SOURCES},
            "finished": dict(current.get("finished") or {}),
            "agent": bool(current.get("agent_seen"))}


def _post(body: dict[str, Any]) -> None:
    request = urllib.request.Request(USAGE_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": "assetfurnace-usage"})
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def _take_off(latest: dict[str, Any], sent: dict[str, Any]) -> None:
    """Subtract what was sent, so a job counted during the send waits for the next one."""
    def minus(counts: dict[str, Any], gone: dict[str, Any]) -> dict[str, int]:
        left = {k: int(counts.get(k) or 0) - int(gone.get(k) or 0) for k in counts}
        return {k: n for k, n in left.items() if n > 0}

    started = latest.get("started") or {}
    latest["started"] = {s: minus(started.get(s) or {}, sent["started"].get(s) or {}) for s in started}
    latest["finished"] = minus(latest.get("finished") or {}, sent["finished"])
    if sent["agent"]:
        latest["agent_seen"] = False


def send_daily(state: Path | None = None, env: dict[str, str] | None = None, today: str | None = None,
               post: Callable[[dict[str, Any]], None] | None = None) -> bool:
    """Send today's message if sharing and not yet sent today. True when one went out."""
    today = today or _today()
    with LOCK:
        if _SENDING.is_set() or not sharing(state, env):
            return False   # one send at a time, or the same counts would go twice
        current = _current(state)
        if current.get("sent_day") == today:
            return False
        _write(state, current)   # keep the id the message carries
        body = payload(current, today)
        _SENDING.set()
    try:
        (post or _post)(body)
    except (OSError, ValueError, urllib.error.URLError, TimeoutError):
        return False   # offline, or the site is down: the counts wait for another day
    else:
        with LOCK:
            latest = _read(state)
            if latest.get("id") != body["id"]:
                return False   # turned off (or off and on) while it was sending: keep the new state
            _take_off(latest, body)
            latest["sent_day"] = today
            _write(state, latest)
        return True
    finally:
        _SENDING.clear()


def send_soon(state: Path | None = None, env: dict[str, str] | None = None) -> None:
    """send_daily on a thread, never two at once, so a request never waits on the network."""
    if not sharing(state, env) or _read(state).get("sent_day") == _today() or _SENDING.is_set():
        return
    threading.Thread(target=send_daily, args=(state, env), daemon=True, name="usage-send").start()


def status(state: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    current = _read(state)
    on = sharing(state, env)
    return {"enabled": on, "allowed": not blocked(env),
            "sends": payload(current, _today()) if on else None,
            "noticed": bool(current.get("noticed")) or not on,   # nothing to tell someone who is not sharing
            "url": USAGE_URL, "policy": POLICY_URL}
