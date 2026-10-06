"""The studio Library: every run under output/ grouped into assets.

An asset is one thing the user is making, followed through its steps:
picture -> 3D model -> finished model -> rigged model -> clips (or, for a prop sheet,
the props cut from it). The steps were separate tabs with separate run folders, so the
links have to be rebuilt from what each run left behind:

* picture -> 3D model: the 3D run's folder name starts with the picture's
  (`<picture>__pixal3d__...` or `<picture>-cutout__...`).
* 3D model -> finished, and 3D model -> prop sheet: the finish or props run keeps a copy
  of its input (`input/source.glb`, `source.glb`); same bytes, same sha256.
* picture -> finished: a finish run also keeps the source picture (`input/source.png`),
  which links it even when its 3D model was made outside the Generate tab.
* finished (or raw) model -> rig: the rig's provenance records its input's sha256.

Hashes are cached in `output/.assets-index.json`, keyed by path and invalidated by size
and modification time, so a Library refresh hashes only new or changed files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))  # rig_check, for whether a rig fits the moves

FINISH_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__finish__\d{8}-\d{6}(?:-\d+)?$")
RIG_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__rig__\d{8}-\d{6}(?:-\d+)?$")
PROPS_RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__props__\d{8}-\d{6}(?:-\d+)?$")
# Folders under output/ that hold other steps' runs, not 3D generations.
NOT_GENERATED = {"finish", "props", "images", "rig-rebind", "animate"}
# A second take of a preset (`_b`) is a choice between versions, not another move.
TAKE_SUFFIX = re.compile(r"_[b-z]$")
STAGES = ("picture", "model", "finished", "rigged", "animated")
INDEX_NAME = ".assets-index.json"
META_NAME = ".assets-meta.json"   # the user's own choices about assets, e.g. hidden ones
ASSET_ID = re.compile(r"^[0-9a-f]{12}$")


def friendly_name(stem: str) -> str:
    """'a-chunky-knight__20261005-205524-cutout__pixal3d__...' -> 'a chunky knight'."""
    words = stem.split("__")[0].removesuffix("-cutout").replace("-", " ").replace("_", " ").strip()
    words = re.sub(r"\s+\d+k$", "", words)  # '... 40k' is a face count, not a name
    if len(words) > 40:
        words = words[:39].rstrip() + "…"
    return words or stem


class _Hashes:
    """sha256 per file, cached by (size, mtime)."""

    def __init__(self, output: Path):
        self.output = output
        self.path = output / INDEX_NAME
        try:
            self.index = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.index = {}
        self.changed = False

    def of(self, file: Path) -> str:
        rel = file.relative_to(self.output).as_posix()
        stat = file.stat()
        cached = self.index.get(rel)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
            return cached["sha256"]
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self.index[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest.hexdigest()}
        self.changed = True
        return self.index[rel]["sha256"]

    def fits_moves(self, file: Path) -> bool | None:
        """Whether a rig fits the preset moves (scripts/rig_check.py), cached like the hashes.
        None when it cannot be told (the move list stays, as before)."""
        key = "fits:" + file.relative_to(self.output).as_posix()
        stat = file.stat()
        cached = self.index.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
            return cached["humanoid"]
        try:
            import rig_check  # scripts/ is on the path (see the top of this module)
            humanoid = bool(rig_check.check(file)["humanoid"])
        except Exception:  # noqa: BLE001 - an unreadable rig must not take the Library down
            return None
        self.index[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "humanoid": humanoid}
        self.changed = True
        return humanoid

    def save(self) -> None:
        if not self.changed:
            return
        try:
            self.path.write_text(json.dumps(self.index, indent=0, sort_keys=True))
        except OSError:
            pass  # a read-only output folder still gets a Library, just a slower one


def _dirs(path: Path):
    return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")) if path.is_dir() else []


def _pictures(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "images"):
        png = run / f"{run.name}.png"
        if png.is_file():
            found.append({"stem": run.name, "file": png})
    return found


def _models(output: Path) -> list[dict[str, Any]]:
    """GLBs the Generate tab made: `<run>/<run>.glb`, maybe one licence-class folder down."""
    found = []
    for top in _dirs(output):
        if top.name in NOT_GENERATED:
            continue
        for run in (top, *_dirs(top)):
            glb = run / f"{run.name}.glb"
            if glb.is_file():
                found.append({"stem": run.name, "file": glb})
    return found


def _finishes(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "finish"):
        if not FINISH_RUN.fullmatch(run.name):
            continue
        results = sorted(run.glob("*.glb"))
        if results:
            source, image = run / "input" / "source.glb", run / "input" / "source.png"
            found.append({"stem": run.name, "file": results[0], "source": source if source.is_file() else None,
                          "image": image if image.is_file() else None})
    return found


def _rigs(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "animate"):
        if not RIG_RUN.fullmatch(run.name):
            continue
        for rigged in sorted(run.glob("*_rigged.glb")):
            base = rigged.name.removesuffix("_rigged.glb")
            try:
                record = json.loads(rigged.with_suffix(".provenance.json").read_text())
                input_sha = (record.get("input") or {}).get("sha256")
            except (OSError, ValueError, AttributeError):
                input_sha = None
            clips = []
            for clip in sorted(run.glob(f"{base}_*.glb")):
                name = clip.stem.removeprefix(f"{base}_")
                if clip == rigged or name == "rigged" or TAKE_SUFFIX.search(name):
                    continue
                clips.append({"name": name, "file": clip})
            found.append({"stem": base, "file": rigged, "input_sha": input_sha, "clips": clips})
    return found


def _prop_sets(output: Path) -> list[dict[str, Any]]:
    found = []
    for run in _dirs(output / "props"):
        if not PROPS_RUN.fullmatch(run.name):
            continue
        props = []
        for prop in _dirs(run / "finished"):
            lod0 = prop / f"{prop.name}_LOD0.glb"
            if lod0.is_file():
                props.append({"name": prop.name, "file": lod0})
        source = run / "source.glb"
        found.append({"stem": run.name, "file": source if source.is_file() else None, "props": props,
                      "dir": run})
    return found


def list_assets(output: Path) -> list[dict[str, Any]]:
    """Every asset on disk, the one touched most recently first."""
    if not output.is_dir():
        return []
    hashes = _Hashes(output)
    pictures, models, finishes = _pictures(output), _models(output), _finishes(output)
    rigs, sets = _rigs(output), _prop_sets(output)

    # union-find over every run; each connected group is one asset
    nodes: list[tuple[str, dict[str, Any]]] = (
        [("picture", n) for n in pictures] + [("model", n) for n in models]
        + [("finished", n) for n in finishes] + [("rigged", n) for n in rigs] + [("set", n) for n in sets])
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    index_of = {id(n): i for i, (_, n) in enumerate(nodes)}
    model_by_sha: dict[str, int] = {}
    for model in models:
        model_by_sha[hashes.of(model["file"])] = index_of[id(model)]
    for picture in pictures:
        for model in models:
            if model["stem"].startswith(picture["stem"] + "__") or model["stem"].startswith(picture["stem"] + "-cutout__"):
                join(index_of[id(picture)], index_of[id(model)])
    picture_by_sha = {hashes.of(p["file"]): index_of[id(p)] for p in pictures}
    finished_by_sha: dict[str, int] = {}
    for fin in finishes:
        if fin["image"] is not None:
            match = picture_by_sha.get(hashes.of(fin["image"]))
            if match is not None:
                join(index_of[id(fin)], match)
        finished_by_sha[hashes.of(fin["file"])] = index_of[id(fin)]
        if fin["source"] is not None:
            match = model_by_sha.get(hashes.of(fin["source"]))
            if match is not None:
                join(index_of[id(fin)], match)
    for rig in rigs:
        sha = rig["input_sha"]
        match = finished_by_sha.get(sha) if sha else None
        if match is None and sha:
            match = model_by_sha.get(sha)
        if match is not None:
            join(index_of[id(rig)], match)
    for prop_set in sets:
        if prop_set["file"] is not None:
            match = model_by_sha.get(hashes.of(prop_set["file"]))
            if match is not None:
                join(index_of[id(prop_set)], match)
    hashes.save()

    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)

    def rel(path: Path | None) -> str | None:
        return path.relative_to(output).as_posix() if path is not None else None

    assets = []
    for members in groups.values():
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for i in members:
            by_kind.setdefault(nodes[i][0], []).append(nodes[i][1])

        def latest(kind: str) -> dict[str, Any] | None:
            items = [n for n in by_kind.get(kind, []) if n.get("file") is not None]
            return max(items, key=lambda n: n["file"].stat().st_mtime) if items else None

        picture, model, fin, rig = latest("picture"), latest("model"), latest("finished"), latest("rigged")
        prop_set = max(by_kind.get("set", []), key=lambda n: n["dir"].stat().st_mtime, default=None)
        files = [n["file"] for i in members for n in [nodes[i][1]] if n.get("file") is not None]
        files += [c["file"] for r in by_kind.get("rigged", []) for c in r["clips"]]
        files += [p["file"] for s in by_kind.get("set", []) for p in s["props"]]
        clips = rig["clips"] if rig else []

        if prop_set is not None:
            kind, stage = "prop set", "finished" if prop_set["props"] else "model"
        elif rig is not None:
            kind, stage = "character", "animated" if clips else "rigged"
        else:
            kind = "picture" if model is None and fin is None else "model"
            stage = "finished" if fin else "model" if model else "picture"
        first = picture or model or (prop_set if prop_set and prop_set.get("file") else None) or fin or rig
        # A person-given name ("rune_knight" at the finish or rig step) beats the prompt the
        # picture was named after; it is the shortest name in the chain.
        stems = [n["stem"] for n in (rig, fin, prop_set, model, picture) if n is not None]
        name_from = min(stems, key=lambda stem: len(friendly_name(stem)))
        assets.append({
            "id": hashlib.sha1(rel(first["file"]).encode()).hexdigest()[:12] if first and first.get("file") else name_from,
            "name": friendly_name(name_from),
            "kind": kind,
            "stage": stage,
            "picture": rel(picture["file"]) if picture else None,
            "model": rel(model["file"]) if model else None,
            "finished": rel(fin["file"]) if fin else None,
            "rigged": rel(rig["file"]) if rig else None,
            "fits_moves": hashes.fits_moves(rig["file"]) if rig else None,
            "clips": [{"name": c["name"], "file": rel(c["file"])} for c in clips],
            "props": [{"name": p["name"], "file": rel(p["file"])} for p in (prop_set["props"] if prop_set else [])],
            "updated": max((f.stat().st_mtime for f in files), default=0),
        })
    hashes.save()
    hidden = set(_read_meta(output).get("hidden", []))
    for asset in assets:
        asset["hidden"] = asset["id"] in hidden
    assets.sort(key=lambda a: a["updated"], reverse=True)
    return assets


def _read_meta(output: Path) -> dict[str, Any]:
    try:
        meta = json.loads((output / META_NAME).read_text())
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def set_hidden(output: Path, asset_id: str, hidden: bool) -> None:
    """Hide an asset from the Library, or bring it back. Never touches its files."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    if asset_id not in {a["id"] for a in list_assets(output)}:
        raise ValueError(f"no such asset: {asset_id}")
    meta = _read_meta(output)
    ids = set(meta.get("hidden", []))
    ids.add(asset_id) if hidden else ids.discard(asset_id)
    meta["hidden"] = sorted(ids)
    (output / META_NAME).write_text(json.dumps(meta, indent=1))


def assets_payload(output: Path) -> dict[str, Any]:
    """What `GET /api/assets` sends: the list, plus where its relative paths are served."""
    return {"base": "/output/", "stages": list(STAGES), "assets": list_assets(output)}
