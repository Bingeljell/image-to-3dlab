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
from itertools import pairwise
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
RECIPES = Path(__file__).resolve().parents[1] / "image_to_3dlab" / "prompt_recipes.json"


def _history(output: Path) -> list[dict[str, Any]]:
    from activity_api import (
        read_history,  # same folder; imported late, it is only needed here
    )

    return read_history(output, limit=100_000)


def _asked_for_props(history: list[dict[str, Any]]) -> set[str]:
    """Files (relative to output/) of chains that were asked to make one prop."""
    return {str(r["made"]) for r in history if r.get("want") == "prop" and r.get("made")}


def _step_times(history: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    """How long the step that made each file took: {file relative to output/: (step, seconds)}.
    History is newest first, so the first record for a file wins."""
    times: dict[str, tuple[str, float]] = {}
    for record in history:
        made, seconds = record.get("made"), record.get("seconds")
        if record.get("result") == "done" and made and isinstance(seconds, (int, float)):
            times.setdefault(str(made), (str(record.get("step")), float(seconds)))
    return times


def _prop_wording() -> str | None:
    try:
        return json.loads(RECIPES.read_text())["goals"]["prop"]["wording"] or None
    except (OSError, ValueError, KeyError):
        return None


def _prompted_as_prop(picture: Path, wording: str | None) -> bool:
    """Before the history said what a chain wanted: Create added the prop wording to the
    prompt, and the picture's provenance record kept the prompt."""
    if not wording:
        return False
    try:
        record = json.loads(picture.with_suffix(".provenance.json").read_text())
    except (OSError, ValueError):
        return False
    return wording in str(record.get("prompt", ""))


def _picture_prompt(picture: Path | None) -> str | None:
    """What the picture was drawn from, kept in its provenance record, so it can be drawn again."""
    if picture is None:
        return None
    try:
        text = json.loads(picture.with_suffix(".provenance.json").read_text()).get("prompt")
    except (OSError, ValueError, AttributeError):
        return None
    return text if isinstance(text, str) and text.strip() else None


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

    def rig_facts(self, file: Path) -> dict[str, Any]:
        """{"fits_moves": bool | None, "facing": [x, y, z] | None} for a rig (scripts/rig_check.py),
        cached like the hashes. None when it cannot be told (the move list stays, as before)."""
        key = "fits:" + file.relative_to(self.output).as_posix()
        stat = file.stat()
        cached = self.index.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime and "facing" in cached:
            return {"fits_moves": cached["humanoid"], "facing": cached["facing"]}
        try:
            import rig_check  # scripts/ is on the path (see the top of this module)
            result = rig_check.check(file)
        except Exception:
            return {"fits_moves": None, "facing": None}
        humanoid, facing = bool(result["humanoid"]), result.get("forward")
        self.index[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "humanoid": humanoid, "facing": facing}
        self.changed = True
        return {"fits_moves": humanoid, "facing": facing}

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


def list_assets(output: Path, with_runs: bool = False) -> list[dict[str, Any]]:
    """Every asset on disk, the one touched most recently first. `with_runs` adds the run folders
    that belong to this version alone (what deleting it would remove); the Library list never needs them."""
    if not output.is_dir():
        return []
    hashes = _Hashes(output)
    history = _history(output)
    asked_props, step_times, prop_wording = _asked_for_props(history), _step_times(history), _prop_wording()
    pictures, models, finishes = _pictures(output), _models(output), _finishes(output)
    rigs, sets = _rigs(output), _prop_sets(output)

    # One family tree per picture: picture -> 3D models -> finishes -> rigs (a prop sheet hangs off a
    # model). Each branch end is its own asset, so a re-rig or a second 3D model is a new version
    # of the same family, not a silent replacement of the one before.
    nodes: list[tuple[str, dict[str, Any]]] = (
        [("picture", n) for n in pictures] + [("model", n) for n in models]
        + [("finished", n) for n in finishes] + [("rigged", n) for n in rigs] + [("set", n) for n in sets])
    index_of = {id(n): i for i, (_, n) in enumerate(nodes)}
    parent: dict[int, int | None] = {i: None for i in range(len(nodes))}
    model_by_sha = {hashes.of(m["file"]): index_of[id(m)] for m in models}
    picture_by_sha = {hashes.of(p["file"]): index_of[id(p)] for p in pictures}
    finished_by_sha: dict[str, int] = {}
    picture_hint: dict[int, int] = {}     # a finish kept its picture: where a model with no picture of its own looks
    for picture in pictures:
        for model in models:
            if model["stem"].startswith(picture["stem"] + "__") or model["stem"].startswith(picture["stem"] + "-cutout__"):
                parent[index_of[id(model)]] = index_of[id(picture)]
    for fin in finishes:
        me = index_of[id(fin)]
        finished_by_sha[hashes.of(fin["file"])] = me
        picture_of = picture_by_sha.get(hashes.of(fin["image"])) if fin["image"] is not None else None
        model_of = model_by_sha.get(hashes.of(fin["source"])) if fin["source"] is not None else None
        if picture_of is not None:
            picture_hint[me] = picture_of
        if model_of is not None:
            parent[me] = model_of
            if picture_of is not None and parent[model_of] is None:
                parent[model_of] = picture_of    # a model made outside the app, found through its finish
        else:
            parent[me] = picture_of
    for rig in rigs:
        sha = rig["input_sha"]
        match = finished_by_sha.get(sha) if sha else None
        if match is None and sha:
            match = model_by_sha.get(sha)
        parent[index_of[id(rig)]] = match
    for prop_set in sets:
        if prop_set["file"] is not None:
            parent[index_of[id(prop_set)]] = model_by_sha.get(hashes.of(prop_set["file"]))
    hashes.save()

    def born(i: int) -> tuple[float, str]:
        node = nodes[i][1]
        target = node["file"] if node.get("file") is not None else node["dir"]
        return (target.stat().st_mtime, rel(target) or "")

    def rel(path: Path | None) -> str | None:
        return path.relative_to(output).as_posix() if path is not None else None

    children: dict[int, list[int]] = {}
    for i, up in parent.items():
        if up is not None:
            children.setdefault(up, []).append(i)
    # The first child made at each fork carries on the original; later ones are new versions.
    first_child = {up: min(kids, key=born) for up, kids in children.items()}
    # a picture a finish was made from belongs to that family, even when the model came from another copy of it
    claimed = set(picture_hint.values())
    leaves = [i for i in range(len(nodes)) if i not in children and not (nodes[i][0] == "picture" and i in claimed)]

    def path_to(leaf: int) -> list[int]:
        path = [leaf]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    def branch_start(path: list[int]) -> int | None:
        """The node where this chain split off the original, or None for the original itself."""
        for up, node in pairwise(path):
            if first_child[up] != node:
                return node
        return None

    chains = [path_to(leaf) for leaf in leaves]
    shared = {i: sum(1 for c in chains if i in c) > 1 for i in range(len(nodes))}

    def run_dir(i: int) -> Path | None:
        node = nodes[i][1]
        return node["dir"] if node.get("dir") is not None else node["file"].parent if node.get("file") is not None else None
    # family = the root of the tree; version 1 is the original, the rest count by when they split off
    family_chains: dict[int, list[list[int]]] = {}
    for chain in chains:
        family_chains.setdefault(chain[0], []).append(chain)
    version_of: dict[int, int] = {}
    for members in family_chains.values():
        members.sort(key=lambda c: (branch_start(c) is not None,
                                    born(branch_start(c)) if branch_start(c) is not None else (0.0, "")))
        for number, chain in enumerate(members, start=1):
            version_of[chain[-1]] = number

    assets = []
    for chain in chains:
        by_kind: dict[str, dict[str, Any]] = {nodes[i][0]: nodes[i][1] for i in chain}
        picture, model, fin, rig = by_kind.get("picture"), by_kind.get("model"), by_kind.get("finished"), by_kind.get("rigged")
        if picture is None and fin is not None and index_of[id(fin)] in picture_hint:
            picture = nodes[picture_hint[index_of[id(fin)]]][1]
        prop_set = by_kind.get("set")
        files = [nodes[i][1]["file"] for i in chain if nodes[i][1].get("file") is not None]
        if picture is not None:
            files.append(picture["file"])
        if rig is not None:
            files += [c["file"] for c in rig["clips"]]
        if prop_set is not None:
            files += [p["file"] for p in prop_set["props"]]
        clips = rig["clips"] if rig else []

        if prop_set is not None:
            kind, stage = "prop set", "finished" if prop_set["props"] else "model"
        elif rig is not None:
            kind, stage = "character", "animated" if clips else "rigged"
        else:
            kind = "picture" if model is None and fin is None else "model"
            stage = "finished" if fin else "model" if model else "picture"
            if kind == "model" and (any(rel(f) in asked_props for f in files)
                                    or (picture and _prompted_as_prop(picture["file"], prop_wording))):
                kind = "prop"
        root_node = nodes[chain[0]][1]
        first = picture or model or (prop_set if prop_set and prop_set.get("file") else None) or fin or rig
        # A person-given name ("rune_knight" at the finish or rig step) beats the prompt the
        # picture was named after; it is the shortest name in the chain.
        stems = [n["stem"] for n in (rig, fin, prop_set, model, picture) if n is not None]
        name_from = min(stems, key=lambda stem: len(friendly_name(stem)))
        root_file = root_node.get("file") or (first or {}).get("file")
        family_id = hashlib.sha1(rel(root_file).encode()).hexdigest()[:12] if root_file else name_from
        split = branch_start(chain)
        if split is None:
            asset_id = family_id            # the original keeps the id it always had
        else:
            split_node = nodes[split][1]
            split_file = split_node.get("file") or split_node.get("dir")
            asset_id = hashlib.sha1(rel(split_file).encode()).hexdigest()[:12]
        version = version_of[chain[-1]]
        assets.append({
            "id": asset_id,
            "family": family_id,
            "version": version,
            "name": friendly_name(name_from) + (f" · v{version}" if version > 1 else ""),
            "kind": kind,
            "stage": stage,
            "picture": rel(picture["file"]) if picture else None,
            "prompt": _picture_prompt(picture["file"] if picture else None),
            # Front + back: the drawn back sits beside its front as <front>__back.png.
            "back": rel(back) if picture and (back := picture["file"].with_name(
                f"{picture['file'].stem}__back.png")).is_file() else None,
            "model": rel(model["file"]) if model else None,
            "finished": rel(fin["file"]) if fin else None,
            "rigged": rel(rig["file"]) if rig else None,
            **(hashes.rig_facts(rig["file"]) if rig else {"fits_moves": None, "facing": None}),
            "clips": [{"name": c["name"], "file": rel(c["file"])} for c in clips],
            "props": [{"name": p["name"], "file": rel(p["file"])} for p in (prop_set["props"] if prop_set else [])],
            "timings": {step: seconds for f in files if (rel(f) in step_times)
                        for step, seconds in [step_times[rel(f)]]},
            "updated": max((f.stat().st_mtime for f in files), default=0),
            **({"runs": [rel(run_dir(i)) for i in chain if not shared[i] and run_dir(i) is not None]} if with_runs else {}),
        })
    counts: dict[str, int] = {}
    for asset in assets:
        counts[asset["family"]] = counts.get(asset["family"], 0) + 1
    for asset in assets:
        asset["versions"] = counts[asset["family"]]
    hashes.save()   # the rig facts read above are cached too
    meta = _read_meta(output)
    hidden = set(meta.get("hidden", []))
    names = meta.get("names") if isinstance(meta.get("names"), dict) else {}
    mains = meta.get("main") if isinstance(meta.get("main"), dict) else {}
    for asset in assets:
        asset["hidden"] = asset["id"] in hidden
        asset["main"] = mains.get(asset["family"]) == asset["id"]
        if isinstance(names.get(asset["id"]), str):
            asset["name"] = names[asset["id"]]
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


def set_main(output: Path, asset_id: str, main: bool) -> None:
    """Mark one version as the family's main one (the one export and game use should take), or
    clear it. One per family: starring a version un-stars the one before. Never touches files."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    found = {a["id"]: a for a in list_assets(output)}
    if asset_id not in found:
        raise ValueError(f"no such asset: {asset_id}")
    family = found[asset_id]["family"]
    meta = _read_meta(output)
    mains = meta.get("main") if isinstance(meta.get("main"), dict) else {}
    if main:
        mains[family] = asset_id
    elif mains.get(family) == asset_id:
        del mains[family]
    meta["main"] = mains
    (output / META_NAME).write_text(json.dumps(meta, indent=1))


def _folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def plan_delete(output: Path, asset_id: str) -> dict[str, Any]:
    """What deleting one version would remove, and whether it may be deleted at all.

    Only the run folders that belong to this version alone go: the picture, model and finish it
    shares with other versions stay. The original (v1) is kept while later versions exist,
    because they were built from it."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    found = {a["id"]: a for a in list_assets(output, with_runs=True)}
    if asset_id not in found:
        raise ValueError(f"no such asset: {asset_id}")
    asset = found[asset_id]
    root = output.resolve()
    folders = []
    for run in asset["runs"]:
        path = (output / run).resolve()
        if root not in path.parents or not path.is_dir():
            raise ValueError(f"refusing to touch {run}: not a run folder inside output/")
        folders.append({"path": run, "bytes": _folder_bytes(path)})
    refused = None
    if asset["version"] == 1 and asset["versions"] > 1:
        refused = "This is the original: the other versions were built from it. Delete those first, or hide this one."
    return {"id": asset_id, "name": asset["name"], "stage": asset["stage"], "folders": folders,
            "bytes": sum(f["bytes"] for f in folders), "refused": refused}


def delete_version(output: Path, asset_id: str) -> dict[str, Any]:
    """Delete the run folders only this version owns. Cannot be undone."""
    import shutil

    plan = plan_delete(output, asset_id)
    if plan["refused"]:
        raise ValueError(plan["refused"])
    for folder in plan["folders"]:
        shutil.rmtree(output / folder["path"])
    meta = _read_meta(output)
    meta["hidden"] = [i for i in meta.get("hidden", []) if i != asset_id]
    for key in ("names",):
        if isinstance(meta.get(key), dict):
            meta[key].pop(asset_id, None)
    if isinstance(meta.get("main"), dict):
        meta["main"] = {family: i for family, i in meta["main"].items() if i != asset_id}
    (output / META_NAME).write_text(json.dumps(meta, indent=1))
    return plan


NAME_LIMIT = 80


def set_name(output: Path, asset_id: str, name: str) -> str:
    """Give an asset the name a person chose; an empty name goes back to the one made from its
    prompt. Only the Library's label changes: no file is renamed or moved. Returns the name."""
    if not ASSET_ID.fullmatch(asset_id or ""):
        raise ValueError(f"not an asset id: {asset_id!r}")
    found = {a["id"]: a for a in list_assets(output)}
    if asset_id not in found:
        raise ValueError(f"no such asset: {asset_id}")
    clean = " ".join(str(name or "").split())[:NAME_LIMIT]
    meta = _read_meta(output)
    names = meta.get("names") if isinstance(meta.get("names"), dict) else {}
    if clean:
        names[asset_id] = clean
    else:
        names.pop(asset_id, None)
    meta["names"] = names
    (output / META_NAME).write_text(json.dumps(meta, indent=1))
    return clean or next(a["name"] for a in list_assets(output) if a["id"] == asset_id)


def assets_payload(output: Path) -> dict[str, Any]:
    """What `GET /api/assets` sends: the list, plus where its relative paths are served."""
    return {"base": "/output/", "stages": list(STAGES), "assets": list_assets(output)}


THUMBS_DIR = ".thumbs"


def thumbnail(output: Path, relative: str, size: int = 128) -> Path:
    """A small copy of a picture in output/, for the Library list, made once and kept in
    output/.thumbs (a new one when the picture changes). Raises ValueError for paths outside output/."""
    from PIL import Image

    source = (output / relative).resolve()
    if output.resolve() not in source.parents or not source.is_file():
        raise ValueError(f"not a picture in output/: {relative}")
    stat = source.stat()
    key = hashlib.sha1(f"{relative}|{stat.st_size}|{stat.st_mtime}|{size}".encode()).hexdigest()[:20]
    target = output / THUMBS_DIR / f"{key}.png"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.thumbnail((size, size))
            partial = target.with_suffix(".part")
            image.save(partial, format="PNG", optimize=True)
            partial.replace(target)  # never leave half a thumbnail behind for the next request
    return target
