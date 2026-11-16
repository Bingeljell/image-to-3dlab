#!/usr/bin/env python3
"""Draw a subject's right and left side from its front picture, for a four-view 3D build.

    python scripts/make_side_views.py front.png right.png left.png [--seed 3] [--engine flux]
    python scripts/pixal3d_generate.py front.png out.glb --back back.png --right right.png --left left.png --sides-shape-only

A front and a back picture say nothing about how deep a subject is from front to back, so an
animal or a vehicle built from them comes out short, or doubles what it cannot place (a second
pair of ears behind the first, a third pair of legs). The studio's own picture engine draws the
right side from the front (in edit mode, the front as its reference), and the left side is that
picture mirrored, so both sides agree on the length. `--sides-shape-only` lets the drawn sides
shape the model but not paint it: a drawn side never puts an eye or a nose exactly where the
front has them, and painting from it doubles them. For a subject whose two sides differ, draw
the left one yourself (or with another seed) and pass it to `pixal3d_generate.py` instead.

`--four-legged` checks the drawn side before it is kept (`image_to_3dlab/leg_check.py`): where
the front and the hind legs touch in the picture, the generator fuses them into one lump, so a
side without open space under the body is drawn again with the next seed, up to `--tries`
pictures, and the first one whose legs stand apart is kept (else the one with the most open
space, named in the output). The prompt stays the same: naming the legs would summon more.

The prompt and the reference rules live in `image_to_3dlab/back_view.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "viewer")]

import image_api
from PIL import Image

from image_to_3dlab import back_view, leg_check, matte
from image_to_3dlab.provenance import write_provenance

DEFAULT_SEED = 3
DEFAULT_TRIES = 4


def reference_for(front: Path, destination: Path) -> Path:
    """Write the front's cut-out on plain grey, cutting it out first unless it already is."""
    with Image.open(front) as opened:
        cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(
            opened.convert("RGB"))[0]
    back_view.reference_on_grey(cut).save(destination)
    return destination


def command_for(reference: Path, right: Path, seed: int, weights: dict[str, Path], vision: Path | None,
                engine, binary: Path = image_api.BINARY) -> list[str]:
    settings = image_api.clean_settings({"seed": seed}, engine=engine)
    return image_api.build_command(back_view.SIDE_PROMPT, settings, right, weights, binary=binary,
                                   reference=reference, vision=vision, engine=engine)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("front", type=Path)
    parser.add_argument("right", type=Path, help="the right side to write (the subject faces right in it)")
    parser.add_argument("left", type=Path, help="the left side to write: the right one, mirrored")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--engine", choices=sorted(image_api.ENGINES), default=None,
                        help="picture engine (default: the first one set up)")
    parser.add_argument("--four-legged", action="store_true",
                        help="check that the legs stand apart in the drawn side, and draw again if they touch")
    parser.add_argument("--tries", type=int, default=DEFAULT_TRIES,
                        help=f"with --four-legged: at most this many pictures (default {DEFAULT_TRIES})")
    args = parser.parse_args(argv)
    if not args.front.is_file():
        raise SystemExit(f"not found: {args.front}")
    engine = image_api.pick_engine(args.engine, edit=True)
    vision = image_api.resolve_vision(engine)
    if engine.vision_file and vision is None:
        print(f"{engine.label} needs its vision file to redraw a picture; the studio's Setup page fetches it.")
        return 1
    weights = image_api.resolve_weights(engine)
    args.right.parent.mkdir(parents=True, exist_ok=True)
    reference = reference_for(args.front, args.right.with_name(f"{args.right.stem}__reference.png"))
    tries = max(1, args.tries) if args.four_legged else 1
    started = time.time()
    best = None   # (measure, problems, seed, picture)
    for n in range(tries):
        seed = args.seed + n
        out = args.right if tries == 1 else args.right.with_name(f"{args.right.stem}__try{n}{args.right.suffix}")
        command = command_for(reference, out, seed, weights, vision, engine)
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO), check=False)
        if result.returncode or not out.exists():
            print((result.stdout + result.stderr)[-1500:])
            return 1
        if not args.four_legged:
            best = (None, [], seed, out)
            break
        measure, found = leg_check.check_picture(out, "side")
        print(f"[side-views] seed {seed}: {measure['open_under_body']:.0%} open under the body"
              + (f", {found[0]}" if found else ", legs apart"), flush=True)
        if best is None or leg_check.better(measure, best[0]):
            best = (measure, found, seed, out)
        if not found:
            break
    measure, found, seed, kept = best
    if kept != args.right:
        kept.replace(args.right)
        for extra in args.right.parent.glob(f"{args.right.stem}__try*{args.right.suffix}"):
            extra.unlink()
    if args.four_legged and found:
        print(f"[side-views] the legs still touch after {tries} pictures; kept seed {seed}: {found[0]}. "
              "Draw the side yourself or try other seeds before building.", flush=True)
    seconds = time.time() - started
    record = image_api.provenance(back_view.SIDE_PROMPT, image_api.clean_settings({"seed": seed}, engine=engine),
                                  seconds, args.right, engine=engine)
    record["reference"] = {"front": str(args.front), "mode": "edit (right side)"}
    if measure is not None:
        record["legs"] = {"open_under_body": measure["open_under_body"], "problems": found, "pictures_drawn": n + 1}
    write_provenance(args.right.with_suffix(".provenance.json"), record)
    args.left.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.right) as right:
        back_view.mirror_side(right).save(args.left)
    record["reference"] = {"front": str(args.front), "mode": f"mirror of the right side {args.right.name}"}
    write_provenance(args.left.with_suffix(".provenance.json"), record)
    print(f"[side-views] {seconds:.0f}s -> {args.right}, {args.left} (mirrored)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
