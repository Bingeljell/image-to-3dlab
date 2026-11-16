#!/usr/bin/env python3
"""Make a copy of a GLB that Roblox Studio imports at the right size, with its pictures loose.

    python scripts/export_roblox.py MODEL.glb [--out DIR] [--kind prop|character|accessory]
                                              [--height-studs 4] [--max-texture 1024]

Writes DIR (default: MODEL_roblox/ beside the model) holding:

* `model.glb`: plain mesh, PNG pictures, scaled to studs so the importer's default
  "Scale Unit: Studs" gives the right size
* `color.png`, `normal.png`, `roughness.png`, `metalness.png`: for a SurfaceAppearance
* `check.json`: pass / warn / fail against Roblox's triangle and texture limits
* `HOW-TO-IMPORT.txt`: four steps in Roblox Studio

Rigged characters made by the lab are already person-sized; give props a height with
`--height-studs` (a Roblox character is about 5-6 studs tall). Too many triangles is
reported, never fixed here. The logic is `image_to_3dlab/roblox_export.py`.
Exits 1 when a check fails, 2 when the file can't be exported at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.roblox_export import (
    KINDS,
    MARKETPLACE_TEXTURE,
    RobloxExportError,
    export,
    write_folder,
)

MARK = {"pass": "ok  ", "warn": "warn", "fail": "FAIL", "note": "note"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, help="folder to write (default: MODEL_roblox beside it)")
    parser.add_argument("--kind", choices=KINDS,
                        help="default: character if it has a skeleton, else prop. "
                             "accessory = a wearable item: one mesh, at most 4,000 triangles")
    parser.add_argument("--height-studs", type=float, help="make the model this tall, in studs")
    parser.add_argument("--max-texture", type=int, default=MARKETPLACE_TEXTURE,
                        help=f"longest picture side, 64..{MARKETPLACE_TEXTURE}")
    args = parser.parse_args(argv)

    try:
        result = export(args.model.read_bytes(), kind=args.kind, max_texture=args.max_texture,
                        height_studs=args.height_studs)
    except (RobloxExportError, ValueError) as exc:
        print(f"Can't export {args.model.name}: {exc}", file=sys.stderr)
        return 2
    out = write_folder(result, args.out or args.model.with_name(f"{args.model.stem}_roblox"))
    for line in result.checks:
        print(f"{MARK[line['level']]}  {line['text']}")
    for note in result.notes:
        print(f"      {note}")
    if result.size_studs:
        print(f"size  {result.size_studs[1]:g} studs tall ("
              + " x ".join(f"{v:g}" for v in result.size_studs) + ")")
    print(f"wrote {out}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
