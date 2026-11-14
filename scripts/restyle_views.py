#!/usr/bin/env python3
"""Repaint a shape guide's views with the studio's own picture editor (FLUX.2 [klein] 4B).

The shape-guide kit (`shape_guide.py`) writes five views of a model, front, right, back, left
and top, with a `crops.json` that says which rig view each one is. This runs one edit per view,
the view itself as the reference picture, so the picture model paints the model realistically
and keeps its outline; `<out>/<rig view>.png` is then what Pixel Match projects onto the model.

    python scripts/restyle_views.py <views dir> <out dir> --prompts prompts.json [--size 1024] [--steps 8]

The prompts file has a `subject` ("a modern 9 m European low-floor midibus"), a `shared` part
with `{subject}` in it, an `ending`, and per side (`front`, `right`, `back`, `left`, `top`)
what that view shows. Needs FLUX.2 [klein] 4B (Setup & Status, or `bootstrap_flux_image.py`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def view_prompt(prompts: dict, side: str) -> str:
    """One view's prompt: the shared part with the subject, what this side shows, the ending."""
    parts = [prompts["shared"].format(subject=prompts["subject"]),
             prompts.get("sides", {}).get(side, ""), prompts.get("ending", "")]
    return " ".join(part.strip() for part in parts if part.strip())


def views_of(folder: Path) -> list[tuple[Path, str, str]]:
    """(picture, side, rig view name) for every picture the folder's `crops.json` lists."""
    crops = json.loads((folder / "crops.json").read_text(encoding="utf-8"))
    return [(folder / entry["file"], entry["side"], Path(entry["rig_view"]).stem) for entry in crops["pictures"]]


def edit_command(prompt: str, reference: Path, output: Path, weights: dict[str, Path], size: int, steps: int,
                 seed: int, binary: Path) -> list[str]:
    """The picture editor's edit of one view, built by the studio's own `build_command`."""
    from viewer import image_api

    settings = {**image_api.FLUX.defaults, "width": size, "height": size, "steps": steps, "seed": seed}
    return image_api.build_command(prompt, settings, output, weights, binary=binary, reference=reference,
                                   engine=image_api.FLUX)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("views", type=Path, help="the shape guide's views and their crops.json")
    parser.add_argument("out_dir", type=Path, help="folder for the repainted views, named after their rig view")
    parser.add_argument("--prompts", type=Path, required=True, help="JSON: subject, shared, ending, sides")
    parser.add_argument("--size", type=int, default=1024, help="picture size in pixels (default 1024)")
    parser.add_argument("--steps", type=int, default=8, help="sampling steps (default 8, the studio's for a final picture)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs the picture engine; its parts are tested
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(REPO))
    from viewer import image_api

    weights = image_api.resolve_weights(image_api.FLUX)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for picture, side, rig_view in views_of(args.views):
        output = args.out_dir / f"{rig_view}.png"
        started = time.time()
        command = edit_command(view_prompt(prompts, side), picture, output, weights, args.size, args.steps,
                               args.seed, image_api.BINARY)
        done = subprocess.run(command, check=False, capture_output=True, text=True)
        if done.returncode != 0 or not output.is_file():
            print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
            raise SystemExit(f"the edit of {picture.name} failed (exit {done.returncode})")
        print(f"restyled {picture.name} -> {output.name} in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
