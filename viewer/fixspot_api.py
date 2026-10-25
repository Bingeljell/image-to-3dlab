"""The studio's "Fix a spot" brush, as one editing session.

The user marks spots on a finished model, presses Patch to see a preview, flips Before /
After, then Keeps it (one new version) or Discards it. Marks are applied by
`image_to_3dlab.spot_fix`: "fill" cuts and caps a pocket, "smooth" irons the surface out
(by `strength`), "repaint" repaints the paint only.

    POST /api/fix-spot/preview  {from, base?, spots: [{point, radius, mode, strength?}]} -> {preview, reports}
    POST /api/fix-spot/keep     {from, preview} -> {file}   (a new sibling Finish run = new version)
    POST /api/fix-spot/discard  {preview}

Previews live in output/.fix-previews (hidden from the Library, served like any output
file) and are cleared after a day. The finished model they came from is never touched.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assets_api import FINISH_RUN

from image_to_3dlab import spot_fix

# The studio's brush slider range, a share of the model's height (viewer/studio/viewer.js).
BRUSH_MIN, BRUSH_MAX = 0.005, 0.08
PREVIEWS = ".fix-previews"
MAX_SPOTS = 64
PREVIEW_DAYS = 1


def finished_model(output: Path, relative: Any) -> Path | None:
    """The finished GLB `relative` names, or None unless it sits in a Finish run of `output`."""
    if not isinstance(relative, str) or not relative.endswith(".glb"):
        return None
    path = (output / relative).resolve()
    finish = (output / "finish").resolve()
    if path.parent.parent != finish or not FINISH_RUN.fullmatch(path.parent.name) or not path.is_file():
        return None
    return path


def preview_file(output: Path, relative: Any) -> Path | None:
    """A preview this module wrote, or None."""
    if not isinstance(relative, str) or not relative.endswith(".glb"):
        return None
    path = (output / relative).resolve()
    if path.parent != (output / PREVIEWS).resolve() or not path.is_file():
        return None
    return path


def new_run_dir(run: Path, now: float | None = None) -> Path:
    """A sibling Finish run named like `run`, stamped with the time (made unique if it must be)."""
    base = run.name.split("__finish__")[0]
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    candidate, n = run.parent / f"{base}__finish__{stamp}", 1
    while candidate.exists():
        n += 1
        candidate = run.parent / f"{base}__finish__{stamp}-{n}"
    return candidate


def _spots(raw: Any, height: float) -> list[dict] | str:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SPOTS:
        return f"Mark between 1 and {MAX_SPOTS} spots, then Patch."
    spots = []
    for spot in raw:
        if not isinstance(spot, dict):
            return "Each spot needs a point, a size and a mode."
        point, share, mode = spot.get("point"), spot.get("radius"), spot.get("mode", "fill")
        if (not isinstance(point, list) or len(point) != 3
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)):
            return "Each point must be three numbers."
        if not isinstance(share, (int, float)) or not BRUSH_MIN <= share <= BRUSH_MAX:
            return f"The brush is {BRUSH_MIN} to {BRUSH_MAX} of the model's height."
        if mode not in spot_fix.MODES:
            return f"The brush mode is one of {', '.join(spot_fix.MODES)}."
        strength = spot.get("strength", spot_fix.STRENGTH_DEFAULT)
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not spot_fix.STRENGTH_MIN <= strength <= spot_fix.STRENGTH_MAX):
            return f"Smooth's strength is {spot_fix.STRENGTH_MIN} to {spot_fix.STRENGTH_MAX}."
        spots.append({"point": point, "radius": float(share) * height, "mode": mode, "share": float(share),
                      "strength": float(strength)})
    return spots


def _clear_old_previews(folder: Path, now: float) -> None:
    for old in folder.glob("*.glb"):
        if now - old.stat().st_mtime > PREVIEW_DAYS * 86400:
            old.unlink(missing_ok=True)


def answer(handler, output: Path, payload: Any) -> tuple[int, dict[str, Any]]:
    """The handler's answer, or a plain error: an exception here used to drop the connection,
    which the browser could only call "Failed to fetch"."""
    try:
        return handler(output, payload)
    except Exception as exc:   # anything a model can throw at the patch code: say what, keep serving
        return 500, {"error": f"Fix a spot could not patch this model: {exc}"}


def preview_response(output: Path, payload: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        return 400, {"error": "Send the model and the marked spots."}
    origin = finished_model(output, payload.get("from"))
    if origin is None:
        return 400, {"error": "Fix a spot works on finished models only."}
    base = origin if payload.get("base") is None else preview_file(output, payload.get("base"))
    if base is None:
        return 400, {"error": "That preview is gone. Patch again from the model."}
    spots = _spots(payload.get("spots"), spot_fix.model_height(base))
    if isinstance(spots, str):
        return 400, {"error": spots}
    views = origin.parent / "input" / "source.views"
    views = views if (views / "transforms.json").is_file() else None
    if views is None and any(s["mode"] == "repaint" for s in spots):
        return 422, {"error": "Repaint needs the picture this model was made from, and this one has none."}
    folder = output / PREVIEWS
    folder.mkdir(parents=True, exist_ok=True)
    _clear_old_previews(folder, time.time())
    result = folder / f"{uuid.uuid4().hex}.glb"
    reports = spot_fix.apply_spots(base, result, spots, views)
    if not any(r.get("cut") or r.get("smoothed") or r.get("repainted_faces") for r in reports):
        result.unlink(missing_ok=True)
        return 422, {"error": "Nothing changed: click right on the model."}
    if base != origin:
        base.unlink(missing_ok=True)   # the previous preview in this session is superseded
    return 200, {"preview": result.relative_to(output).as_posix(), "reports": reports}


def keep_response(output: Path, payload: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        return 400, {"error": "Send the model and the preview to keep."}
    origin = finished_model(output, payload.get("from"))
    preview = preview_file(output, payload.get("preview"))
    if origin is None or preview is None:
        return 400, {"error": "That preview is gone. Patch again from the model."}
    run = new_run_dir(origin.parent)
    run.mkdir(parents=True)
    if (origin.parent / "input").is_dir():
        shutil.copytree(origin.parent / "input", run / "input")
    result = run / f"{origin.stem.removesuffix('_fixed')}_fixed.glb"
    shutil.move(str(preview), result)
    (run / "fix-spot.json").write_text(json.dumps(
        {"from": origin.relative_to(output).as_posix(), "spots": payload.get("spots")}, indent=2), encoding="utf-8")
    return 200, {"file": result.relative_to(output).as_posix()}


def discard_response(output: Path, payload: Any) -> tuple[int, dict[str, Any]]:
    preview = preview_file(output, (payload or {}).get("preview") if isinstance(payload, dict) else None)
    if preview is not None:
        preview.unlink(missing_ok=True)
    return 200, {"discarded": preview is not None}
