"""AssetFurnace for agents: an MCP server that drives the studio.

    ./lab mcp                     # what an agent's MCP config runs (stdio)

A thin remote control, on purpose. Every tool calls the studio server's own API (the one
the Create sheet and Steps panel use), so a feature added to the studio reaches agents
without being built twice: engines, moves and the prompt wording are read from the studio
each call, and `tests/test_mcp_server.py` fails when the server grows an action that is
neither offered here nor listed in `STUDIO_ONLY` with a reason.

If the studio is not running, the first tool call starts it in the background (it is
never restarted: a run in progress belongs to whoever started it).
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_to_3dlab import __version__
from image_to_3dlab.agent_plan import compose_prompt, plan_for, refusal, stop_after

REPO = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8777"
START_TIMEOUT = 30
WAIT_LIMIT = 50           # an MCP call should come back well inside the clients' own timeouts
STEP_IDS = ("picture", "back", "model", "finished", "split", "rigged", "animated")

# Server actions agents do not get, and why. Everything else the studio can do must be a
# tool (the test reads the server's routes and checks this list against them).
STUDIO_ONLY = {
    "api/assets/*/delete": "deleting is the user's call, made in the studio",
    "api/assets/*/hidden": "Library tidying, a person's choice",
    "api/assets/*/main": "marking the main version is the user's verdict",
    "api/assets/*/name": "renaming is the user's choice",
    "api/blender/install": "installs an app; the Setup page asks first",
    "api/gltfpack/install": "installs a tool; the Setup page asks first",
    "api/hf/sign-in": "takes the user's Hugging Face token; only they should paste it",
    "api/setup/run": "first-run setup is a guided page for a person",
    "api/setup/*/rebuild": "rebuilds a backend from source; Setup page only",
    "api/setup/*/remove": "deletes downloaded weights; the user's call",
    "api/setup/*/cancel": "cancels a download the user started in Setup",
    "api/trellis/input-advice": "a hint shown inside the TRELLIS form",
    "api/finish/runs/*/resume": "resumes a Finish the Steps panel paused",
    "api/props/runs/*/turn": "turns one prop by hand in the Steps panel",
    # the single-step job endpoints: agents reach every one of them through `create` and
    # `continue_asset`, which run the same steps through the server's plan runner
    "api/generate": "via create / continue_asset", "api/finish": "via create / continue_asset",
    "api/props": "via create / continue_asset", "api/image": "via create / continue_asset",
    "api/image/back": "via create / continue_asset", "api/animate/rig": "via create / continue_asset",
    "api/animate/play": "via create / continue_asset", "api/rig/rebind": "re-rigs from the Steps panel",
    "api/generate/*/cancel": "via cancel_run", "api/finish/*/cancel": "via cancel_run",
    "api/props/*/cancel": "via cancel_run", "api/image/*/cancel": "via cancel_run",
    "api/animate/*/cancel": "via cancel_run", "api/rig/rebind/*/cancel": "via cancel_run",
}
# Server actions the tools below call.
USED = {"api/chains", "api/chains/*/cancel", "api/setup/*/download"}


class StudioError(RuntimeError):
    """Something the agent should read and pass on in plain words."""


# --------------------------------------------------------------------------- the studio
class Studio:
    """HTTP to the studio server, starting it once if nobody has."""

    def __init__(self, base: str | None = None, *, http=None, start: Callable[[], None] | None = None):
        self.base = (base or os.environ.get("ASSETFURNACE_URL") or DEFAULT_URL).rstrip("/")
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self._start = start or self._start_server
        self._checked = False

    # -- plumbing
    def _url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"

    def _ensure(self) -> None:
        if self._checked:
            return
        if not self._alive():
            self._start()
            deadline = time.time() + START_TIMEOUT
            while not self._alive():
                if time.time() > deadline:
                    raise StudioError(f"The studio did not start at {self.base}. Start it with ./lab and try again.")
                time.sleep(0.5)
        self._checked = True

    def _alive(self) -> bool:
        try:
            return self.http.get(self._url("api/catalog"), timeout=5).status_code == 200
        except Exception:  # noqa: BLE001 - any failure to connect means "not running"
            return False

    def _start_server(self) -> None:
        if not self.base.startswith(("http://127.0.0.1", "http://localhost")):
            raise StudioError(f"No studio answers at {self.base}, and it is not this machine, so it can't be started from here.")
        log = REPO / "output" / ".studio-mcp.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        port = self.base.rsplit(":", 1)[-1]
        with log.open("ab") as out:
            subprocess.Popen([sys.executable, str(REPO / "viewer" / "serve.py"), "--no-browser", "--port", port],
                             cwd=REPO, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)

    def _check(self, response) -> Any:
        if response.status_code >= 400:
            try:
                message = response.json().get("error")
            except ValueError:
                message = None
            raise StudioError(message or f"the studio answered {response.status_code}")
        return response

    def get(self, path: str, **params) -> Any:
        self._ensure()
        return self._check(self.http.get(self._url(path), params=params or None, timeout=60)).json()

    def get_bytes(self, path: str, timeout: float = 320, **params):
        self._ensure()
        return self._check(self.http.get(self._url(path), params=params or None, timeout=timeout))

    def post(self, path: str, *, data=None, files=None) -> Any:
        """POST as multipart (the only form the studio's plan endpoint reads)."""
        self._ensure()
        parts = {k: (None, v) for k, v in (data or {}).items()} | (files or {})
        return self._check(self.http.post(self._url(path), files=parts or None, timeout=60)).json()

    def studio_link(self, asset_id: str | None = None) -> str:
        return f"{self.base}/viewer/studio.html" + (f"#asset={asset_id}" if asset_id else "")


# --------------------------------------------------------------------------- helpers
def ready_engines(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """3D engines that can run now, best first (the studio's `readyEngines`)."""
    ready = [b for b in catalog.get("backends", []) if b.get("kind") == "3d" and b.get("state") == "ready"]
    return sorted(ready, key=lambda b: b.get("rank", 99))


def first_move(presets: list[dict[str, Any]]) -> str | None:
    """The default move: the first one that is not an alternative take (as Create picks)."""
    return next((p["id"] for p in presets if not p.get("take")), None)


def asset_summary(asset: dict[str, Any]) -> dict[str, Any]:
    """The parts of a Library entry an agent needs, without the noise."""
    files = {k: asset[k] for k in ("picture", "back", "model", "finished", "rigged") if asset.get(k)}
    return {
        "id": asset["id"], "name": asset.get("name"), "kind": asset.get("kind"), "stage": asset.get("stage"),
        "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(asset.get("updated", 0))),
        "files": {k: f"output/{v}" for k, v in files.items()},
        "moves": [c["name"] for c in asset.get("clips", [])],
        "props": [p.get("name") for p in asset.get("props", [])] or None,
        "prompt": asset.get("prompt"),
    }


def run_summary(run: dict[str, Any], studio: Studio) -> dict[str, Any]:
    """A plan's state in short: what's done, what's running, what it made, what went wrong."""
    out = {k: run.get(k) for k in ("id", "title", "status", "steps", "done", "current", "percent", "message", "error", "note")}
    out["elapsed_seconds"] = run.get("elapsed")
    out["made"] = {k: f"output/{v}" for k, v in (run.get("made") or {}).items()}
    out["stalled"] = run.get("stalled") or None
    if run.get("status") == "running":
        out["log"] = plain_log(run.get("log")) or None
    if run.get("status") == "error":
        out["log"] = plain_log(run.get("full_log"))[-1500:] or None
    asset_id = None
    if run.get("status") == "done" and run.get("made"):
        asset_id = asset_for_run(studio.get("api/assets")["assets"], run["made"])
    if asset_id:
        out["asset_id"] = asset_id
        out["next"] = "preview this asset_id and show the user; open_in_studio if they want a closer look."
    elif run.get("status") == "done":
        out["next"] = "Find the new asset with list_assets, then preview it."
    out["studio"] = studio.studio_link(asset_id)
    return {k: v for k, v in out.items() if v not in (None, [], {})}


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain_log(text: str | None) -> str:
    """A log as words: terminal colour and line-erase codes stripped."""
    return ANSI.sub("", str(text or "")).strip()


def asset_for_run(assets: list[dict[str, Any]], made: dict[str, str]) -> str | None:
    """The Library entry a run produced: the one holding the furthest file it made."""
    for step in ("animated", "rigged", "finished", "model", "back", "picture"):
        file = made.get(step)
        if not file:
            continue
        for asset in assets:
            if step == "animated" and any(c.get("file") == file for c in asset.get("clips", [])):
                return asset["id"]
            if asset.get(step) == file:
                return asset["id"]
    return None


def made_from(asset: dict[str, Any]) -> dict[str, str]:
    """What an existing asset already has, as a plan's `made` (the files the next step starts from)."""
    return {k: asset[k] for k in ("picture", "back", "model", "finished", "rigged") if asset.get(k)}


def continue_steps(asset: dict[str, Any], want: str, up_to: str | None, front_back: bool) -> list[str]:
    """The steps still to run for this asset, up to `up_to`.

    An animated character asked to go to "animated" again gets one more move: that is the
    only step that can sensibly run twice without making a new version of the model.
    """
    made = made_from(asset)
    full = plan_for("idea", want, front_back=front_back and not made.get("model"))
    have = set(made) | ({"animated"} if asset.get("clips") else set()) | ({"split"} if asset.get("props") else set())
    todo = [s for s in stop_after(full, up_to) if s not in have]
    if not todo and up_to == "animated" and made.get("rigged"):
        return ["animated"]
    return todo


def again_steps(asset: dict[str, Any], step: str | None) -> list[str]:
    """One finished step made again, as a new version (the Steps panel's redo)."""
    needs = {"model": "picture", "finished": "model", "rigged": "finished", "animated": "rigged"}
    if step not in needs:
        raise StudioError(f"again=True works with up_to = {', '.join(needs)}.")
    source = asset.get(needs[step]) or (asset.get("model") if step == "rigged" else None)
    if not source:
        raise StudioError(f"{asset.get('name')} has no {needs[step]} to make the {step} step from.")
    return [step]


def want_for(asset: dict[str, Any]) -> str:
    return {"prop": "prop", "prop set": "set"}.get(asset.get("kind", ""), "character")


# Finish's Detail choices, as the Steps panel offers them (tests check the two lists agree).
FINISH_DETAILS: dict[str, dict[str, Any]] = {
    "light": {"faces": 10000}, "game": {"faces": 20000}, "more": {"faces": 40000}, "full": {"keep_mesh": True},
}
STEPS_AFTER_FINISH = ("rigged", "animated")


def finish_choices(detail: str | int | None, pixel_match: bool = True) -> dict[str, Any]:
    """What a plan tells Finish: a detail level (a name above, or a triangle count) and Pixel Match."""
    out: dict[str, Any] = {}
    if detail is not None and str(detail).strip():
        text = str(detail).strip().lower()
        if text in FINISH_DETAILS:
            out.update(FINISH_DETAILS[text])
        elif text.isdigit() and 1000 <= int(text) <= 200000:
            out["faces"] = int(text)
        else:
            raise StudioError(f"finish_detail is one of {', '.join(FINISH_DETAILS)} or a triangle count from "
                              "1000 to 200000.")
    if not pixel_match:
        out["skip_photo"] = True
    return out


def full_detail_refusal(steps: list[str], choices: dict[str, Any], asset: dict[str, Any] | None = None) -> str | None:
    """Rigging a full-detail model (often a million triangles) is slow and too heavy for a game."""
    full = (choices.get("keep_mesh") and "finished" in steps) or (
        "finished" not in steps and str((asset or {}).get("finished") or "").endswith("_full.glb"))
    if full and any(s in steps for s in STEPS_AFTER_FINISH):
        return ("Full detail is for renders and stills, not rigging. Stop after 'finished', then make a "
                "game-ready version (continue_asset with up_to='finished', again=True, finish_detail='game') "
                "and rig that one.")
    return None


def find_asset(studio: Studio, asset_id: str) -> dict[str, Any]:
    assets = studio.get("api/assets")["assets"]
    asset = next((a for a in assets if a["id"] == asset_id), None)
    if asset is None:
        raise StudioError(f"No asset with id {asset_id!r}. list_assets shows what there is.")
    return asset


def _upload(path: str | None, kinds: tuple[str, ...]) -> tuple[str, bytes, str] | None:
    if not path:
        return None
    file = Path(path).expanduser()
    if not file.is_file():
        raise StudioError(f"No file at {file}.")
    if file.suffix.lower() not in kinds:
        raise StudioError(f"{file.name} is not one of {', '.join(kinds)}.")
    return file.name, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream"


# --------------------------------------------------------------------------- the tools
def status(studio: Studio) -> dict[str, Any]:
    catalog = studio.get("api/catalog")
    moves = studio.get("api/animate/models")
    chains = studio.get("api/chains")
    return {
        "version": __version__,
        "studio": studio.studio_link(),
        "machine": (catalog.get("host") or {}).get("label"),
        "ready_engines": [b["id"] for b in ready_engines(catalog)],
        "pictures": any(b.get("kind") == "image" and b.get("state") == "ready" for b in catalog.get("backends", [])),
        "front_and_back": bool((catalog.get("front_back") or {}).get("ready")),
        "animation": bool(moves.get("installed")),
        "running": run_summary(chains["running"], studio) if chains.get("running") else None,
        "needs_setup": catalog.get("needs_onboarding") or None,
    }


def list_engines(studio: Studio) -> list[dict[str, Any]]:
    catalog = studio.get("api/catalog")
    out = []
    for b in sorted(catalog.get("backends", []), key=lambda b: (b.get("kind") != "3d", b.get("rank", 99))):
        out.append({k: v for k, v in {
            "id": b["id"], "label": b.get("label"), "makes": "3D models" if b.get("kind") == "3d" else "pictures",
            "state": b.get("state"), "recommended": b.get("recommended") or None, "best_for": b.get("best_for"),
            "licence": (b.get("license") or {}).get("name"), "caveat": b.get("caveat"),
            "download_size": b.get("human_expected"), "runs_here": b.get("supported_here"),
        }.items() if v is not None})
    return out


def list_moves(studio: Studio) -> dict[str, Any]:
    payload = studio.get("api/animate/models")
    if not payload.get("installed"):
        return {"installed": False, "note": "Animation is not set up. The studio's Setup page installs it."}
    return {"installed": True, "default": first_move(payload["presets"]),
            "moves": [{"id": p["id"], "label": p["label"], "seconds": p.get("seconds")} for p in payload["presets"]]}


def create(studio: Studio, *, want: str, description: str = "", image_path: str | None = None,
           model_path: str | None = None, stop_after_step: str | None = None, engine: str | None = None,
           front_and_back: bool | None = None, move: str | None = None, add_wording: bool = True,
           finish_upload: bool = False, keep_everything: bool = False, finish_detail: str | int | None = None,
           pixel_match: bool = True) -> dict[str, Any]:
    have = "model" if model_path else "picture" if image_path else "idea"
    if model_path and image_path:
        raise StudioError("Give an image or a model, not both.")
    why = refusal(have, want)
    if why:
        raise StudioError(why)
    if have == "idea" and not description.strip():
        raise StudioError("Describe what you want, or give image_path / model_path.")
    catalog = studio.get("api/catalog")
    engines = ready_engines(catalog)
    if engine and engine not in {e["id"] for e in engines}:
        raise StudioError(f"Engine {engine!r} isn't ready. Ready: {', '.join(e['id'] for e in engines) or 'none'}.")
    fb_ready = bool((catalog.get("front_back") or {}).get("ready"))
    front_back = fb_ready if front_and_back is None else front_and_back
    if front_back and not fb_ready:
        raise StudioError("Front and back isn't installed. The studio's Setup page adds it.")
    steps = stop_after(plan_for(have, want, finish_upload=finish_upload, front_back=front_back), stop_after_step)
    engine = engine or (engines[0]["id"] if engines else None)
    if "back" in steps and engine != "pixal3d":
        steps = [s for s in steps if s != "back"]   # only Pixal3D builds from front and back
    if "model" in steps and not engine:
        raise StudioError("No 3D engine is installed yet. list_engines shows them; install_backend sets one up.")
    if "animated" in steps:
        moves = studio.get("api/animate/models")
        presets = moves.get("presets", []) if moves.get("installed") else []
        if move and move not in {p["id"] for p in presets}:
            raise StudioError(f"No move {move!r}. list_moves shows them.")
        move = move or first_move(presets)
        if not move:
            steps = stop_after(steps, "rigged")
    recipes = studio.get("api/recipes") if have == "idea" else None
    form = {
        "title": description.strip() or Path(image_path or model_path or "").name or "New asset",
        "steps": json.dumps(steps), "want": want, "made": "{}",
    }
    if have == "idea":
        form["prompt"] = compose_prompt(description, want, recipes, add_wording)
    if engine and "model" in steps:
        form["engine"] = engine
        form["model_seed"] = str(int.from_bytes(os.urandom(3), "big"))  # a new try each time, like Create
    if move and "animated" in steps:
        form["first_move"] = move
    if keep_everything and "model" in steps:
        form["keep_everything"] = "1"
    choices = finish_choices(finish_detail, pixel_match)
    refused = full_detail_refusal(steps, choices)
    if refused:
        raise StudioError(refused)
    if choices and "finished" in steps:
        form["finish_settings"] = json.dumps(choices)
    upload = _upload(image_path, (".png", ".jpg", ".jpeg", ".webp")) or _upload(model_path, (".glb",))
    run = studio.post("api/chains", data=form, files={"file": upload} if upload else None)
    return {"run_id": run["id"], "steps": steps, "engine": form.get("engine"), "move": form.get("first_move"),
            "prompt_sent": form.get("prompt"), "watch": "run_status with this run_id (wait_seconds up to 50)",
            "studio": studio.studio_link()}


def continue_asset(studio: Studio, *, asset_id: str, up_to: str | None = None, want: str | None = None,
                   move: str | None = None, engine: str | None = None, again: bool = False,
                   finish_detail: str | int | None = None, pixel_match: bool = True) -> dict[str, Any]:
    asset = find_asset(studio, asset_id)
    want = want or want_for(asset)
    if up_to and up_to not in STEP_IDS:
        raise StudioError(f"up_to must be one of {', '.join(STEP_IDS)}.")
    catalog = studio.get("api/catalog")
    fb_ready = bool((catalog.get("front_back") or {}).get("ready"))
    if again:
        steps = again_steps(asset, up_to)
    else:
        steps = continue_steps(asset, want, up_to, fb_ready)
    if not steps:
        return {"nothing_to_do": f"{asset['name']} has already reached {up_to or 'its last step'}. "
                                 "Pass again=True to make that step again as a new version."}
    choices = finish_choices(finish_detail, pixel_match)
    refused = full_detail_refusal(steps, choices, asset)
    if refused:
        raise StudioError(refused)
    engines = ready_engines(catalog)
    engine = engine or (engines[0]["id"] if engines else None)
    if "back" in steps and engine != "pixal3d":
        steps = [s for s in steps if s != "back"]
    form = {"title": asset.get("name") or "Asset", "steps": json.dumps(steps), "want": want,
            "made": json.dumps(made_from(asset))}
    if choices and "finished" in steps:
        form["finish_settings"] = json.dumps(choices)
    if "picture" in steps and asset.get("prompt"):
        form["prompt"] = asset["prompt"]
    if "model" in steps:
        if not engine:
            raise StudioError("No 3D engine is installed yet.")
        form["engine"] = engine
        form["model_seed"] = str(int.from_bytes(os.urandom(3), "big"))
    if "animated" in steps:
        moves = studio.get("api/animate/models")
        presets = moves.get("presets", []) if moves.get("installed") else []
        if move and move not in {p["id"] for p in presets}:
            raise StudioError(f"No move {move!r}. list_moves shows them.")
        form["first_move"] = move or first_move(presets) or ""
        if not form["first_move"]:
            raise StudioError("Animation is not set up. The studio's Setup page installs it.")
    run = studio.post("api/chains", data=form)
    out = {"run_id": run["id"], "steps": steps, "watch": "run_status with this run_id",
           "studio": studio.studio_link(asset_id)}
    if form.get("first_move") in {c.get("name") for c in asset.get("clips", [])}:
        out["replaces"] = (f"{asset['name']} already had {form['first_move']!r}; this makes it again and the "
                           "new one replaces the old. Tell the user.")
    return out


def run_status(studio: Studio, *, run_id: str | None = None, wait_seconds: int = 0,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    if not run_id:
        chains = studio.get("api/chains")
        current = chains.get("running") or (chains.get("recent") or [None])[0]
        if current is None:
            return {"status": "idle", "note": "Nothing has run since the studio started."}
        run_id = current["id"]
    run = studio.get(f"api/chains/{run_id}")
    deadline = time.time() + max(0, min(int(wait_seconds), WAIT_LIMIT))
    start_done = len(run.get("done", []))
    while run.get("status") == "running" and time.time() < deadline and len(run.get("done", [])) == start_done:
        sleep(2)
        run = studio.get(f"api/chains/{run_id}")
    return run_summary(run, studio)


def cancel_run(studio: Studio, *, run_id: str) -> dict[str, Any]:
    result = studio.post(f"api/chains/{run_id}/cancel")
    return {"cancelled": bool(result.get("cancelled")),
            "note": None if result.get("cancelled") else "It was not running (already finished, failed or cancelled)."}


def list_assets(studio: Studio, *, search: str = "", kind: str | None = None, limit: int = 20) -> dict[str, Any]:
    assets = [a for a in studio.get("api/assets")["assets"] if not a.get("hidden")]
    if kind:
        assets = [a for a in assets if a.get("kind") == kind]
    if search:
        words = search.lower().split()
        assets = [a for a in assets if all(w in f"{a.get('name', '')} {a.get('prompt') or ''}".lower() for w in words)]
    return {"total": len(assets), "assets": [asset_summary(a) for a in assets[: max(1, min(limit, 100))]]}


def preview(studio: Studio, *, asset_id: str, stage: str = "latest", look: str = "paint",
            move: str | None = None) -> tuple[bytes, dict[str, Any]]:
    params = {"asset": asset_id, "stage": stage, "look": look} | ({"move": move} if move else {})
    response = studio.get_bytes("api/preview", **params)
    path = REPO / "output" / response.headers.get("X-Preview-Path", "")
    return response.content, {
        "shown": response.headers.get("X-Preview-Stage", stage),
        "file": str(path) if path.is_file() else None,
        "studio": studio.studio_link(asset_id),
        "tip": "Show the user this picture, or open_in_studio so they can spin it round themselves.",
    }


def open_in_studio(studio: Studio, *, asset_id: str | None = None,
                   opener: Callable[[str], bool] = webbrowser.open) -> dict[str, Any]:
    if asset_id:
        find_asset(studio, asset_id)
    else:
        studio.get("api/catalog")  # starts the studio if it isn't running
    link = studio.studio_link(asset_id)
    opened = opener(link)
    return {"opened": bool(opened), "link": link}


def install_backend(studio: Studio, *, backend_id: str, confirm: bool = False) -> dict[str, Any]:
    """Name the download, its size and licence; only `confirm=True` starts it (AGENTS.md)."""
    catalog = studio.get("api/catalog")
    backend = next((b for b in catalog.get("backends", []) if b["id"] == backend_id), None)
    if backend is None:
        raise StudioError(f"No backend {backend_id!r}. list_engines shows them.")
    if backend.get("state") == "ready":
        return {"already_installed": backend_id}
    if not backend.get("supported_here", True):
        raise StudioError(f"{backend.get('label')} needs {backend.get('requires')}; it can't run on this machine.")
    plan = {
        "backend": backend_id, "label": backend.get("label"),
        "route": backend.get("best_for"),
        "downloads": [{"what": w.get("label"), "from": w.get("source_url") or w.get("source"),
                       "size": w.get("human_expected")} for w in backend.get("weights", []) if not w.get("present")],
        "total_size": backend.get("human_expected"),
        "licence": (backend.get("license") or {}).get("name"), "licence_url": (backend.get("license") or {}).get("url"),
        "caveat": backend.get("caveat"),
        "setup_minutes": backend.get("setup_minutes"),
    }
    if not confirm:
        plan["to_start"] = ("Nothing has been downloaded. Tell the user what this fetches, its size and licence, "
                            "and call again with confirm=true only after they say yes.")
        return {k: v for k, v in plan.items() if v not in (None, [])}
    started = studio.post(f"api/setup/{backend_id}/download")
    return {"started": True, **{k: v for k, v in plan.items() if v not in (None, [])},
            "progress": "list_engines shows it as ready when done; the studio's Setup page shows progress.",
            "status_url": started.get("status_url")}


# --------------------------------------------------------------------------- the server
INSTRUCTIONS = """AssetFurnace makes game-ready 3D assets on this machine: a picture or an idea becomes a
3D model, then a finished game-ready model, then a rigged and animated character. Runs take minutes
(a 3D model is ~5-15 min), one at a time. Typical flow: status -> create -> run_status (wait) ->
list_assets -> preview (show the user the picture) -> open_in_studio if they want to look closer.
Never download model weights without the user's explicit yes (install_backend asks first)."""


def build_server(studio: Studio | None = None):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.utilities.types import Image

    studio = studio or Studio()
    server = MCPServer("assetfurnace", title="AssetFurnace", version=__version__, instructions=INSTRUCTIONS,
                       website_url="https://assetfurnace.com")

    def safe(fn):
        try:
            return fn()
        except StudioError as exc:
            return {"error": str(exc)}

    @server.tool(name="status", structured_output=False)
    def tool_status():
        """Is the studio up, what is installed, and what is running now. Call this first."""
        return safe(lambda: status(studio))

    @server.tool(name="list_engines", structured_output=False)
    def tool_list_engines():
        """Every engine (3D and picture), whether it is ready, its licence and download size."""
        return safe(lambda: list_engines(studio))

    @server.tool(name="list_moves", structured_output=False)
    def tool_list_moves():
        """The animation moves a rigged character can be given (idle, walk, run, ...)."""
        return safe(lambda: list_moves(studio))

    @server.tool(name="create", structured_output=False)
    def tool_create(want: str, description: str = "", image_path: str | None = None, model_path: str | None = None,
                  stop_after: str | None = None, engine: str | None = None, front_and_back: bool | None = None,
                  move: str | None = None, add_wording: bool = True, finish_upload: bool = False,
                  finish_detail: str | None = None, pixel_match: bool = True):
        """Make something new, as the studio's Create does. Returns a run_id at once; the run takes minutes.

        want: "picture" | "prop" (one game-ready object) | "set" (a 3x3 sheet cut into nine props) |
              "character" (rigged and animated; humanoids).
        Start from ONE of: description (an idea), image_path (a PNG/JPG on this machine) or model_path (a GLB).
        stop_after: end early after a step: picture, back, model, finished, split, rigged or animated.
        engine: a ready 3D engine id (default: the best ready one). front_and_back: also draw a back
        picture for a fuller model (default: on when installed). move: first animation (see list_moves).
        add_wording: append the studio's proven prompt wording (pose, background, ...) to a description.
        finish_upload: also run Finish on an uploaded GLB.
        finish_detail: "light" (10k triangles), "game" (20k, the default), "more" (40k), a count from
        1000 to 200000, or "full" (keeps every triangle, often ~1M: the sharpest look, for renders and
        stills, not games or rigging). pixel_match: paint the picture's real pixels on (default on).
        """
        return safe(lambda: create(studio, want=want, description=description, image_path=image_path,
                                   model_path=model_path, stop_after_step=stop_after, engine=engine,
                                   front_and_back=front_and_back, move=move, add_wording=add_wording,
                                   finish_upload=finish_upload, finish_detail=finish_detail,
                                   pixel_match=pixel_match))

    @server.tool(name="continue_asset", structured_output=False)
    def tool_continue_asset(asset_id: str, up_to: str | None = None, want: str | None = None,
                          move: str | None = None, engine: str | None = None, again: bool = False,
                          finish_detail: str | None = None, pixel_match: bool = True):
        """Take an existing asset further: e.g. a picture to 3D, a model to rigged, or give an animated
        character another move (up_to="animated", move=...). up_to: the last step to run (default: all).
        again=True makes the up_to step again as a new version (another 3D model, another Finish at a
        different detail, a re-rig); the current one is kept. finish_detail and pixel_match: as in create."""
        return safe(lambda: continue_asset(studio, asset_id=asset_id, up_to=up_to, want=want, move=move,
                                           engine=engine, again=again, finish_detail=finish_detail,
                                           pixel_match=pixel_match))

    @server.tool(name="run_status", structured_output=False)
    def tool_run_status(run_id: str | None = None, wait_seconds: int = 30):
        """How a run is going. Waits up to wait_seconds (max 50) for the next step to finish, so call it
        in a loop until status is done, error or cancelled. No run_id = the current or latest run."""
        return safe(lambda: run_status(studio, run_id=run_id, wait_seconds=wait_seconds))

    @server.tool(name="cancel_run", structured_output=False)
    def tool_cancel_run(run_id: str):
        """Stop a run. Steps already finished are kept."""
        return safe(lambda: cancel_run(studio, run_id=run_id))

    @server.tool(name="list_assets", structured_output=False)
    def tool_list_assets(search: str = "", kind: str | None = None, limit: int = 20):
        """The Library, newest first: ids, names, how far each got, file paths, moves.
        kind: picture | character | prop | "prop set" | model. search: words in the name or prompt."""
        return safe(lambda: list_assets(studio, search=search, kind=kind, limit=limit))

    @server.tool(name="preview", structured_output=False)
    def tool_preview(asset_id: str, stage: str = "latest", look: str = "paint", move: str | None = None):
        """A picture of an asset from the front, three-quarter, side and back, so you can SEE the result
        and show it to the user. stage: latest | picture | model | finished | rigged. look: paint | clay
        (clay = shape only, no colours). move: one of the asset's moves (see list_assets) to show four
        moments of it instead. A few seconds with Blender installed, then cached."""
        try:
            data, info = preview(studio, asset_id=asset_id, stage=stage, look=look, move=move)
        except StudioError as exc:
            return {"error": str(exc)}
        return [Image(data=data, format="png"), json.dumps(info)]

    @server.tool(name="open_in_studio", structured_output=False)
    def tool_open_in_studio(asset_id: str | None = None):
        """Open the studio in the user's browser, on this asset, so they can spin it, play its moves and
        download it. Without an asset_id it opens the studio's front page."""
        return safe(lambda: open_in_studio(studio, asset_id=asset_id))

    @server.tool(name="install_backend", structured_output=False)
    def tool_install_backend(backend_id: str, confirm: bool = False):
        """Set up an engine. Without confirm it only reports what would be downloaded, the size and the
        licence; tell the user and get a clear yes before calling again with confirm=true."""
        return safe(lambda: install_backend(studio, backend_id=backend_id, confirm=confirm))

    return server


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="./lab mcp", description="AssetFurnace's MCP server (stdio by default).")
    parser.add_argument("--http", type=int, nargs="?", const=8779, metavar="PORT",
                        help="serve over HTTP instead, for clients that take a URL (default port 8779)")
    args = parser.parse_args(argv)
    server = build_server()
    if args.http:
        print(f"MCP over HTTP: http://127.0.0.1:{args.http}/mcp", file=sys.stderr, flush=True)
        server.run("streamable-http", host="127.0.0.1", port=args.http)
    else:
        server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
