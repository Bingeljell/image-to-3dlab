#!/usr/bin/env python3
"""Draw the back of a subject from its front picture, for a front + back 3D build.

    python scripts/make_back_view.py front.png back.png [--seed 3]

Qwen-Image in edit mode reads the front (cut out, on plain grey) and draws the same subject
from behind, at 768 x 768 in 8 steps, the size and step count every front is made at. The
prompt and the reference rules live in `image_to_3dlab/back_view.py`.

Edit mode needs Qwen's vision projector (`mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, 0.75 GB)
beside the Qwen weights. This never downloads it: with it missing, it says what it is and
how to get it, and stops. A `<back>.provenance.json` records the run like any picture.
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

from image_to_3dlab import back_view, matte
from image_to_3dlab.provenance import write_provenance

# Every back in the 2026-10-12 eight-character test was drawn with this seed.
DEFAULT_SEED = 3
VISION_GB = 0.75


def reference_for(front: Path, destination: Path) -> Path:
    """Write the front's cut-out on plain grey, cutting it out first unless it already is."""
    with Image.open(front) as opened:
        cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(
            opened.convert("RGB"))[0]
    back_view.reference_on_grey(cut).save(destination)
    return destination


def missing_vision_message() -> str:
    directory, filename = image_api.VISION_FILE
    repo = directory.removeprefix("models--").replace("--", "/")
    return (f"The back picture needs Qwen's vision projector, {filename} ({VISION_GB} GB),\n"
            f"from {repo}. Fetch it with:\n"
            f"    python scripts/bootstrap_qwen_image.py --vision")


def command_for(reference: Path, back: Path, seed: int, weights: dict[str, Path],
                vision: Path, binary: Path = image_api.BINARY) -> list[str]:
    settings = image_api.clean_settings({"seed": seed})
    return image_api.build_command(back_view.BACK_PROMPT, settings, back, weights,
                                   binary=binary, reference=reference, vision=vision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("front", type=Path)
    parser.add_argument("back", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if not args.front.is_file():
        raise SystemExit(f"not found: {args.front}")
    vision = image_api.resolve_vision()
    if vision is None:
        print(missing_vision_message())
        return 1
    weights = image_api.resolve_weights()
    args.back.parent.mkdir(parents=True, exist_ok=True)
    reference = reference_for(args.front, args.back.with_name(f"{args.back.stem}__reference.png"))
    command = command_for(reference, args.back, args.seed, weights, vision)
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO), check=False)
    if result.returncode or not args.back.exists():
        print((result.stdout + result.stderr)[-1500:])
        return 1
    seconds = time.time() - started
    record = image_api.provenance(back_view.BACK_PROMPT, image_api.clean_settings(
        {"seed": args.seed}), seconds, args.back)
    record["reference"] = {"front": str(args.front), "mode": "edit (back view)"}
    write_provenance(args.back.with_suffix(".provenance.json"), record)
    print(f"[back-view] {seconds:.0f}s -> {args.back}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
