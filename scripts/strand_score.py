#!/usr/bin/env python3
"""Count the hair strands a Pixal3D figure has in front of its face, to keep the cleanest of a few tries.

    python scripts/strand_score.py output/<run>/ [output/<run2>/ ...]
    python scripts/strand_score.py model.glb --views output/<run>/<run>.svviews

Pixal3D often builds a strand of hair a little in front of the forehead or the eyes where the front picture shows
bare skin; Pixel Match paints it with the skin, and from the side it reads as a crack across the face. A new seed
of the same picture often has fewer. Give it the run folders of a few tries of one character (each holds the raw
GLB and its stored views); it prints each try's strand triangles and names the cleanest. How a strand is told
from the face is in `image_to_3dlab/strands.py`. Reports only; nothing is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import strands


def model_and_views(path: Path, views: Path | None) -> tuple[Path, Path]:
    """(the raw GLB, its stored views) from a run folder or a GLB."""
    if path.is_dir():
        glbs = sorted(p for p in path.glob("*.glb") if not p.name.endswith((".post.glb", "_rigged.glb")))
        if not glbs:
            raise SystemExit(f"{path}: no GLB in this run folder")
        path = glbs[0]
    found = views or path.with_suffix(".svviews")
    if not found.exists():
        raise SystemExit(f"{path}: no stored views beside it ({found.name}); pass --views")
    return path, found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Pixal3D run folders (or GLBs with --views)")
    parser.add_argument("--views", type=Path, help="the stored views when a single GLB is given")
    parser.add_argument("--colour-gap", type=float, default=30.0,
                        help="how far the generator's paint must differ from the picture to count (0-441)")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)
    if args.views and len(args.runs) > 1:
        raise SystemExit("--views goes with a single GLB")
    results = []
    for run in args.runs:
        glb, views_dir = model_and_views(run, args.views)
        positions, uvs, faces, texture = pp.read_glb(glb)
        views, mesh_scale = pp.load_views(views_dir)
        report = strands.count_strands(positions, uvs, faces, texture, views, mesh_scale, colour_gap=args.colour_gap)
        results.append({"model": str(glb), "strand_triangles": report.count, "pieces": report.pieces,
                        "faces": report.faces, "face_found": report.face_box is not None})
        if not args.json:
            note = "" if report.face_box else "  (no face found in the front picture)"
            print(f"{report.count:7d} strand triangles in {report.pieces} pieces  {glb.name}{note}", flush=True)
    scored = [r for r in results if r["face_found"]]
    best = min(scored, key=lambda r: r["strand_triangles"]) if scored else None
    if args.json:
        print(json.dumps({"tries": results, "cleanest": best["model"] if best else None}, indent=1))
    elif best and len(results) > 1:
        print(f"cleanest: {Path(best['model']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
