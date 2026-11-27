#!/usr/bin/env python3
"""Paint a finished model with the real pixels of its source photos, where they can see.

    python scripts/photo_paint.py finished.glb views_dir out.glb [--weights weights.png]

`views_dir` holds a `transforms.json` and the view images, as pixal3d.cpp stages next to
every run (`<run>.svviews/`). Every surface a view can see, facing it and away from the
silhouette edge, takes that photo's pixel; everything else keeps the model's own paint.
That is what brings back text, logos and faces that a repaint redraws as lookalikes.

The GLB must be one mesh with one base-colour texture and no node transforms, which is
what Finish writes. Only the base-colour image changes; every other byte stays.

`--weights` writes the per-texel trust map (white = photo, black = own paint), the first
thing to look at when a result looks wrong. A four-view build's side pictures paint as the
build staged them (`pixal3d_generate.py --sides-paint`); `--sides` overrides it: `safe` for a
figure's drawn sides (ears, side hair, arms and legs only, never the face or the body between
the neck and the hands; `image_to_3dlab/side_paint.py`), `all` or `none`. Each step can be switched off to compare:
`--no-align`, `--no-local-colour`, `--cutout-edge 0`, `--inner-edge 0`.

`--keep-top-paint` leaves upward-facing surfaces that no picture sees in the model's own
paint. Use it when Pixel Match shifts the colour of a hat brim, shoulders or hair.

The logic lives in `image_to_3dlab/photo_paint.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from image_to_3dlab import photo_paint as pp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="finished GLB (Finish's output)")
    parser.add_argument("views", type=Path, help="directory with transforms.json + images")
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, help="also write the trust map as a PNG")
    defaults = pp.Settings()
    parser.add_argument("--facing-low", type=float, default=defaults.facing_low)
    parser.add_argument("--facing-high", type=float, default=defaults.facing_high)
    parser.add_argument("--edge-px", type=int, default=defaults.edge_px)
    parser.add_argument("--cutout-edge", type=float, default=defaults.cutout_edge,
                        help="fade from the picture's cut-out edge, share of its width (0 = off)")
    parser.add_argument("--inner-edge", type=float, default=defaults.inner_edge,
                        help="fade beside inner depth edges, share of the view's width (0 = off)")
    parser.add_argument("--no-local-colour", action="store_true",
                        help="match colour with one shift for the whole model instead")
    parser.add_argument("--no-align", action="store_true",
                        help="paint from the pictures as they are, without nudging them onto the model")
    parser.add_argument("--keep-top-paint", action="store_true",
                        help="surfaces facing up that no picture sees (a brim, shoulders, hair tops) keep the "
                             "model's own paint instead of the colour shift of their neighbours")
    parser.add_argument("--sides", choices=("staged", "all", "safe", "none"), default="staged",
                        help="how the side views paint: as the build staged them (default), everywhere, only "
                             "where a drawn side cannot misplace a hand or an eye (safe), or not at all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in (args.model, args.views / "transforms.json"):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    started = time.time()
    positions, uvs, faces, texture = pp.read_glb(args.model)
    views, mesh_scale = pp.load_views(args.views)
    if args.sides != "staged":
        views = [pp.View(v.image, v.cam_to_world, v.fov_x, v.name, args.sides) if pp.is_side(v) else v
                 for v in views]
    settings = pp.Settings(facing_low=args.facing_low, facing_high=args.facing_high,
                           edge_px=args.edge_px, cutout_edge=args.cutout_edge,
                           inner_edge=args.inner_edge, local_colour=not args.no_local_colour,
                           align=not args.no_align, keep_top_paint=args.keep_top_paint)
    reports: list = []
    sides: list = []
    painted, weight = pp.paint_texture(texture, positions, uvs, faces, views, mesh_scale,
                                       settings=settings, align_reports=reports, side_reports=sides)
    for view, report in zip(views, reports):
        print(f"[photo-paint] align {view.name}: outline {report.iou_before:.3f} -> "
              f"{report.iou_after:.3f} (scale {report.scale:.3f}, shift {report.dx:+d},{report.dy:+d} px), "
              f"{report.matches} feature matches", flush=True)
    for side in sides:
        print(f"[photo-paint] safe side {side['view']}: paints {side['texels_painted']} of the "
              f"{side['texels_seen']} texels it sees (neck at row {side['neck_row']}, hands down to row "
              f"{side['hands_row']} of the front picture)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pp.replace_base_colour(args.model.read_bytes(),
                                                   pp.encode_png(painted)))
    if args.weights:
        Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(args.weights)
    share = float((weight > 0.5).mean())
    print(f"[photo-paint] {len(views)} view(s), {share:.0%} of the texture from the photo, "
          f"{time.time() - started:.0f}s -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
