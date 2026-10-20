"""The studio's "Fix a spot" brush: POST /api/fix-spot.

The user clicks a hole or a messy patch on a finished model; `image_to_3dlab.spot_fix` cuts
the triangles around the point and caps the opening. The result is saved as a new sibling
Finish run (its own folder under output/finish, with the same inputs), so the Library shows
it as a new version and the one it came from is never touched.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assets_api import FINISH_RUN

from image_to_3dlab import spot_fix

# The studio's brush slider range (viewer/studio/viewer.js BRUSH_MIN / BRUSH_MAX).
BRUSH_MIN, BRUSH_MAX = 0.005, 0.08


def finished_model(output: Path, relative: Any) -> Path | None:
    """The finished GLB `relative` names, or None unless it sits in a Finish run of `output`."""
    if not isinstance(relative, str) or not relative.endswith(".glb"):
        return None
    path = (output / relative).resolve()
    finish = (output / "finish").resolve()
    if path.parent.parent != finish or not FINISH_RUN.fullmatch(path.parent.name) or not path.is_file():
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


def fix_spot_response(output: Path, payload: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        return 400, {"error": "Send the model, the point and a brush size."}
    source = finished_model(output, payload.get("file"))
    if source is None:
        return 400, {"error": "Fix a spot works on finished models only."}
    point = payload.get("point")
    if (not isinstance(point, list) or len(point) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)):
        return 400, {"error": "The point must be three numbers."}
    # brush: a share of the model's height (the studio's slider), or a named size (agents, scripts)
    share = payload.get("radius")
    if share is not None:
        if not isinstance(share, (int, float)) or not BRUSH_MIN <= share <= BRUSH_MAX:
            return 400, {"error": f"The brush is {BRUSH_MIN} to {BRUSH_MAX} of the model's height."}
        size = round(float(share), 4)
    else:
        size = payload.get("size", "medium")
        if size not in spot_fix.RADII:
            return 400, {"error": f"Brush size is one of {', '.join(spot_fix.RADII)}."}
        share = spot_fix.RADII[size]
    radius = share * spot_fix.model_height(source)
    run = new_run_dir(source.parent)
    run.mkdir(parents=True)
    try:
        if (source.parent / "input").is_dir():
            shutil.copytree(source.parent / "input", run / "input")
        result = run / f"{source.stem.removesuffix('_fixed')}_fixed.glb"
        views = run / "input" / "source.views"
        report = spot_fix.fix_spot(source, result, point, radius, views if (views / "transforms.json").is_file() else None)
        if not report["cut"]:
            shutil.rmtree(run)
            return 422, {"error": "Nothing to fix there: click right on the model."}
        (run / "fix-spot.json").write_text(json.dumps(
            {"from": source.relative_to(output).as_posix(), "point": point, "size": size, **report}, indent=2))
    except Exception:
        shutil.rmtree(run, ignore_errors=True)
        raise
    return 200, {"file": result.relative_to(output).as_posix(), **report}
