#!/usr/bin/env python3
"""Say which parts of a model stick out past the pictures it was made from.

Where a generator builds a part longer than its picture shows (a tail past its tip, a second
sole under a boot), the shape there is its guess. Laid over the pictures with the generator's
own cameras, such a part reaches past the figure's outline; this finds it, says how far it
reaches, which way and where on the model, and can draw it in red over the pictures. It only
reports: nothing in the model changes.

    python scripts/outline_check.py output/finish/<run>                 # a Finish run
    python scripts/outline_check.py output/<run>.glb                    # a Pixal3D model
    python scripts/outline_check.py model.glb --views <dir>             # any model + its views
    python scripts/outline_check.py output/finish/<run> --height-m 1.8 --picture marked.png

Exits 1 when something sticks out, 0 when nothing does. The logic lives in
`image_to_3dlab/outline_check.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_to_3dlab import outline_check as oc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", type=Path, help="a Finish run folder, or a model (.glb)")
    parser.add_argument("--views", type=Path, help="the folder with transforms.json and the pictures "
                        "(found on its own for a Finish run or a Pixal3D model)")
    parser.add_argument("--height-m", type=float, help="the model's real height, to give lengths in cm")
    parser.add_argument("--tolerance", type=float, default=oc.TOLERANCE,
                        help="how far past the cut-out counts, as a share of the picture's width")
    parser.add_argument("--picture", type=Path, help="write the pictures with what sticks out in red")
    parser.add_argument("--json", action="store_true", help="print the whole report as JSON")
    args = parser.parse_args(argv)

    if args.views is not None:
        model, views = args.target, args.views
    else:
        try:
            model, views = oc.find_inputs(args.target)
        except FileNotFoundError as error:
            parser.error(str(error))
    result = oc.check(model, views, args.height_m, frame=oc.PIXAL3D_FRAME, tolerance=args.tolerance,
                      picture=args.picture)
    print(json.dumps(result, indent=2) if args.json else result["message"])
    return 0 if result["clean"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
