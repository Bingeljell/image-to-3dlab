"""Runs a plan of steps on the server, and keeps the history of every job.

The studio's Create (and each Steps-panel button, as a one-step plan) sends a plan here
instead of driving the jobs from the browser, so closing the tab no longer stops a run.
The runner calls the same job endpoints the pages use (image, generate, finish, rig,
animate, props) on this server, waits for each, and hands the file it made to the next.
Every step that ends, however it ends, is appended to `output/.activity.jsonl`.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HISTORY_NAME = ".activity.jsonl"
TERMINAL = {"done", "error", "cancelled"}
STALL_SECONDS = 180
STEPS = ("picture", "model", "finished", "split", "rigged", "animated")
LABELS = {"picture": "Picture", "model": "3D model", "finished": "Finish", "split": "Split into props",
          "rigged": "Rig", "animated": "Animate"}
KIND = {"picture": "image", "model": "generate", "finished": "finish", "split": "props",
        "rigged": "animate", "animated": "animate"}
NOISE = re.compile(r"\[(VERBOSE|DEBUG)\]|ggml_|kernel_|compile_pipeline|th_max|^\s*[|\-=]*\s*$|^\s*0x[0-9a-f]+", re.IGNORECASE)

# request(method, path, body, content_type) -> (http status, parsed JSON)
Request = Callable[[str, str, bytes | None, str | None], tuple[int, dict[str, Any]]]


def readable_log(text: str, count: int = 4) -> str:
    """The last `count` log lines worth showing a person (engine chatter dropped)."""
    lines = [line.rstrip() for line in str(text or "").split("\n")]
    return "\n".join([line for line in lines if line and not NOISE.search(line)][-count:])


def plain_error(message: str) -> str:
    """A failure in words a person can act on (the studio's jobs.js says the same)."""
    text = str(message or "")
    rules = [
        (r"memory|MemoryError|out of memory|Killed", "This Mac ran out of memory. Close other big apps and try again, or pick a lower detail or texture size."),
        (r"not installed|is not installed/ready|needs_setup", "This engine is not installed yet. Install it from Setup, then try again."),
        (r"is running; wait|wait for it to finish|already being", "Another job is running. This one can start when it finishes."),
        (r"humanoid|map.*SOMA|could not map", "This doesn't look like a humanoid, so the preset moves can't fit it. You can still download the rig and animate it in Blender."),
        (r"Blender", 'Blender stopped with an error. The log has the details; "Copy details" makes it easy to report.'),
    ]
    for pattern, words in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return words
    return text or "The step stopped without saying why. The log has the details."


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    """A multipart/form-data body, as a browser's FormData would send it."""
    boundary = f"----assetfurnace{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data) in files.items():
        safe = filename.replace('"', "")
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{safe}"\r\n'
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@dataclass
class Chain:
    """One plan being run: its steps, what each has made, and where it is now."""

    title: str
    steps: list[str]
    want: str = "character"
    prompt: str | None = None
    engine: str | None = None
    first_move: str | None = None
    picture_settings: dict[str, Any] = field(default_factory=dict)  # size, steps, seed (the image job checks them)
    model_seed: int | None = None                                    # a fresh 3D try each time; None = engine default
    keep_everything: bool = False                                    # the 3D step keeps its textures, meshes, caches
    made: dict[str, str | None] = field(default_factory=dict)       # picture/model/finished/rigged paths
    upload: tuple[str, bytes] | None = None                          # (file name, bytes) when the user gave a file
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    current: str | None = None
    done: list[str] = field(default_factory=list)
    percent: float | None = None
    message: str = ""
    log: str = ""
    full_log: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    step_started: float = field(default_factory=time.time)
    last_change: float = field(default_factory=time.time)
    cancel_requested: bool = False
    job: tuple[str, str] | None = None                               # (kind, job id) of the running step
    note: str = ""                                                   # why a run ended early, calmly

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        return {
            "id": self.id, "title": self.title, "want": self.want, "steps": self.steps, "status": self.status,
            "current": self.current, "done": self.done, "percent": self.percent, "message": self.message,
            "log": self.log, "full_log": self.full_log[-4000:], "error": self.error,
            "made": {k: v for k, v in self.made.items() if v}, "first_move": self.first_move, "note": self.note,
            "elapsed": round(now - self.started, 1), "step_elapsed": round(now - self.step_started, 1),
            "stalled": self.status == "running" and now - self.last_change > STALL_SECONDS,
        }


class ChainRunner:
    """One plan at a time (the jobs underneath refuse to overlap anyway)."""

    def __init__(self, output: Path, request: Request | None = None, *, poll: float = 1.2,
                 now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.output = output
        self.request = request
        self.poll, self.now, self.sleep = poll, now, sleep
        self.chains: dict[str, Chain] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def running(self) -> Chain | None:
        return next((c for c in self.chains.values() if c.status == "running"), None)

    def submit(self, chain: Chain, *, background: bool = True) -> Chain:
        with self.lock:
            if self.running() is not None:
                raise RuntimeError("something is already being made; wait for it to finish")
            bad = [s for s in chain.steps if s not in STEPS]
            if bad or not chain.steps:
                raise ValueError(f"unknown steps: {bad or 'none'}")
            self.chains[chain.id] = chain
        if background:
            threading.Thread(target=self.run, args=(chain,), daemon=True, name=f"chain-{chain.id[:8]}").start()
        else:
            self.run(chain)
        return chain

    def cancel(self, chain_id: str) -> bool:
        chain = self.chains.get(chain_id)
        if chain is None or chain.status != "running":
            return False
        chain.cancel_requested = True
        if chain.job:
            kind, job_id = chain.job
            try:
                self.request("POST", f"/api/{kind}/{job_id}/cancel", None, None)
            except OSError:
                pass  # the job may already have ended; the run stops at its next step either way
        return True

    def recent(self, limit: int = 10) -> list[Chain]:
        return sorted(self.chains.values(), key=lambda c: c.started, reverse=True)[:limit]

    # ------------------------------------------------------------------ the run
    def run(self, chain: Chain) -> None:
        for step in chain.steps:
            if chain.cancel_requested:
                chain.status = "cancelled"
                break
            if step == "animated" and chain.note:
                chain.status, chain.current = "done", None  # the rig does not fit the moves: a calm stop
                break
            chain.current, chain.percent, chain.message, chain.log = step, None, "Starting…", ""
            chain.step_started = chain.last_change = self.now()
            ok = self._run_step(chain, step)
            if not ok:
                break
            chain.done.append(step)
        else:
            chain.status = "done"
            chain.current = None
        chain.job = None

    def _run_step(self, chain: Chain, step: str) -> bool:
        started = self.now()
        try:
            kind, start = self._start(chain, step)
        except Exception as exc:  # noqa: BLE001 - any refused start becomes a plain-words failure
            return self._fail(chain, step, plain_error(str(exc)), "", started)
        chain.job = (kind, start["job_id"])
        status_url = start.get("status_url") or f"/api/{kind}/{start['job_id']}/status"
        last_key = None
        while True:
            try:
                _, payload = self.request("GET", status_url, None, None)
            except (OSError, ValueError):  # a missed poll is not a failure; the next one catches up
                self.sleep(self.poll)
                continue
            event = payload.get("last_event") or {}
            key = json.dumps(event, sort_keys=True) + str(len(payload.get("log_tail") or ""))
            if key != last_key:
                last_key, chain.last_change = key, self.now()
            chain.percent = _percent(event)
            chain.message = event.get("message") or event.get("phase") or "Working…"
            chain.full_log = payload.get("log_tail") or ""
            chain.log = readable_log(chain.full_log)
            if payload.get("status") in TERMINAL:
                break
            self.sleep(self.poll)
        status = payload.get("status")
        if status != "done":
            why = ("Cancelled. Everything finished before this step is kept." if status == "cancelled"
                   else plain_error(payload.get("error") or event.get("message") or payload.get("log_tail") or ""))
            if status == "cancelled":
                chain.cancel_requested = True
            return self._fail(chain, step, why, payload.get("log_tail") or "", started, status)
        self._record(chain, step, payload)
        self._history(chain, step, "done", started, chain.message, chain.full_log)
        return True

    def _fail(self, chain: Chain, step: str, why: str, log: str, started: float, status: str = "error") -> bool:
        chain.status = "cancelled" if status == "cancelled" else "error"
        chain.error, chain.full_log, chain.log = why, log, readable_log(log, 12)
        self._history(chain, step, chain.status, started, why, log)
        return False

    def _record(self, chain: Chain, step: str, final: dict[str, Any]) -> None:
        event = final.get("last_event") or {}
        if step == "picture":
            chain.made["picture"] = final.get("picture")
        elif step == "model":
            chain.made["model"] = final.get("model")
        elif step == "finished":
            chain.made["finished"] = event.get("path")
        elif step == "rigged":
            chain.made["rigged"] = event.get("path")
            if event.get("humanoid") is False:
                chain.note = event.get("message") or "This rig does not fit the preset moves."

    # ------------------------------------------------------------------ starting each step
    def _file(self, relative: str) -> tuple[str, bytes]:
        path = (self.output / relative).resolve()
        if self.output.resolve() not in path.parents:
            raise ValueError(f"not a file in output/: {relative}")
        return path.name, path.read_bytes()

    def _picture(self, chain: Chain) -> tuple[str, bytes]:
        if chain.made.get("picture"):
            return self._file(chain.made["picture"])
        if chain.upload:
            return chain.upload
        raise ValueError("this step needs a picture, and there is none yet")

    def _start(self, chain: Chain, step: str) -> tuple[str, dict[str, Any]]:
        kind = KIND[step]
        if step == "picture":
            body = json.dumps({"prompt": chain.prompt or chain.title, "settings": chain.picture_settings}).encode()
            return kind, self._post("/api/image", body, "application/json")
        if step == "model":
            settings = {"backend": chain.engine} if chain.engine else {}
            if chain.model_seed is not None:
                settings["seed"] = chain.model_seed
            if chain.keep_everything:
                settings["debug"] = True
            body, ctype = multipart({"settings": json.dumps(settings)},
                                    {"image": self._picture(chain)})
            return kind, self._post("/api/generate", body, ctype)
        if step == "finished":
            if not chain.made.get("model"):
                raise ValueError("Finish needs a 3D model, and there is none yet")
            body, ctype = multipart({"settings": "{}"},
                                    {"asset": self._file(chain.made["model"]), "image": self._picture(chain)})
            return kind, self._post("/api/finish", body, ctype)
        if step == "rigged":
            source = chain.made.get("finished") or chain.made.get("model")
            if source:
                body, ctype = multipart({"model": source}, {})
            elif chain.upload:
                body, ctype = multipart({}, {"asset": chain.upload})
            else:
                raise ValueError("Rig needs a 3D model, and there is none yet")
            return kind, self._post("/api/animate/rig", body, ctype)
        if step == "animated":
            if not chain.made.get("rigged") or not chain.first_move:
                raise ValueError("Animate needs a rigged model and a move")
            body = json.dumps({"model": chain.made["rigged"], "preset": chain.first_move}).encode()
            return kind, self._post("/api/animate/play", body, "application/json")
        if step == "split":
            if not chain.made.get("model"):
                raise ValueError("Splitting needs the prop sheet's 3D model")
            body, ctype = multipart({"generated": chain.made["model"], "settings": "{}"}, {})
            return kind, self._post("/api/props", body, ctype)
        raise ValueError(f"unknown step {step}")

    def _post(self, path: str, body: bytes, content_type: str) -> dict[str, Any]:
        status, payload = self.request("POST", path, body, content_type)
        if status >= 400 or "job_id" not in payload:
            raise RuntimeError(payload.get("error") or f"{path} answered {status}")
        return payload

    # ------------------------------------------------------------------ history
    def _history(self, chain: Chain, step: str, result: str, started: float, note: str, log: str) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.now())), "chain": chain.id,
                  "title": chain.title, "step": step, "result": result,
                  "seconds": round(self.now() - started, 1), "note": note,
                  "made": chain.made.get({"picture": "picture", "model": "model", "finished": "finished",
                                          "rigged": "rigged"}.get(step, ""), None),
                  "log": str(log or "")[-3000:]}
        append_history(self.output, record)


def _percent(event: dict[str, Any]) -> float | None:
    for key in ("overall_pct", "pct", "percent", "progress"):
        try:
            value = float(event[key])
        except (KeyError, TypeError, ValueError):
            continue
        return max(0.0, min(100.0, value * 100 if key == "progress" and value <= 1 else value))
    return None


def append_history(output: Path, record: dict[str, Any]) -> None:
    try:
        output.mkdir(parents=True, exist_ok=True)
        with (output / HISTORY_NAME).open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass  # a history that cannot be written must never stop a run


def read_history(output: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Newest first. A damaged line is skipped, not fatal."""
    try:
        lines = (output / HISTORY_NAME).read_text().splitlines()
    except OSError:
        return []
    records = []
    for line in reversed(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
        if len(records) >= limit:
            break
    return records


def local_request(base: str) -> Request:
    """Calls this same server, as the browser would. Long-running jobs answer their start at once."""
    import urllib.error
    import urllib.request

    def request(method: str, path: str, body: bytes | None, content_type: str | None):
        req = urllib.request.Request(base + path, data=body, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, {"error": f"{path} answered {exc.code}"}
    return request


def chain_from_form(form: dict[str, dict[str, Any]]) -> Chain:
    """A plan as the studio sends it (multipart: text fields plus an optional uploaded file)."""
    def text(name: str) -> str | None:
        value = (form.get(name) or {}).get("value")
        return str(value).strip() if value not in (None, "") else None
    try:
        steps = json.loads(text("steps") or "[]")
        made = json.loads(text("made") or "{}")
        picture = json.loads(text("picture_settings") or "{}")
    except ValueError as exc:
        raise ValueError(f"the plan is not readable: {exc}") from exc
    upload = form.get("file")
    return Chain(
        title=text("title") or "Untitled", steps=[str(s) for s in steps], want=text("want") or "character",
        prompt=text("prompt"), engine=text("engine"), first_move=text("first_move"),
        picture_settings={k: picture[k] for k in ("width", "height", "steps", "seed") if k in picture}
        if isinstance(picture, dict) else {},
        model_seed=int(text("model_seed")) if (text("model_seed") or "").isdigit() else None,
        keep_everything=text("keep_everything") in ("1", "true"),
        made={k: str(v) for k, v in made.items() if k in ("picture", "model", "finished", "rigged") and v},
        upload=(upload.get("filename") or "upload", upload["data"]) if upload and upload.get("filename") and upload.get("data") else None,
    )
