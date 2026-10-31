#!/usr/bin/env python3
"""Install Front + back: what builds a model from a front and a back picture.

    python scripts/bootstrap_front_back.py          # says what it wants, then asks
    python scripts/bootstrap_front_back.py --yes    # for the studio's Setup page and agents

- **Pixal3D's multi-view flows**, 5.9 GB, which build the shape from both pictures
  (`bootstrap_pixal3d.py --multiview`; the encoder and decoders are shared with the
  single-view set, so they are linked rather than fetched).
- **A picture engine that can draw the back.** FLUX.2 [klein] reads the front by itself, so
  with it nothing more is fetched. Qwen-Image needs its vision projector (0.75 GB), fetched
  here only when Qwen is the engine set up and the projector is missing.

Both build on routes that must already be set up, Pixal3D and a picture engine; this says so
and stops if either is missing. `AGENTS.md`: name the backend, the route and the size, and
ask. Nothing downloads without `--yes` or a typed yes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "viewer")]

import bootstrap_pixal3d as pixal3d
import bootstrap_qwen_image as qwen
import image_api

# Exact bytes (Hugging Face, pinned revisions), so the total is not a sum of roundings.
MV_BYTES = 5_856_899_828
VISION_BYTES = 752_289_728


def needs_vision() -> bool:
    """Whether the back can only be drawn once Qwen's vision projector is fetched: no engine
    can draw backs yet, and Qwen is the one installed."""
    if any(image_api.is_installed(e) and image_api.can_edit(e) for e in image_api.ENGINES.values()):
        return False
    return image_api.is_installed(image_api.QWEN)


def announcement(vision: bool | None = None) -> str:
    vision = needs_vision() if vision is None else vision
    repo, filename, size = qwen.VISION
    total = MV_BYTES + (VISION_BYTES if vision else 0)
    lines = [
        "", "About to install Front + back:", "",
        "  backend: Pixal3D (raven38/pixal3d.cpp)" + (" + Qwen-Image 2.1 edit mode" if vision else ""),
        "  route:   front + back (multi-view shape, drawn back picture)",
        f"  weights: {pixal3d.MV_GB:.1f} GB  Pixal3D multi-view flows ({pixal3d.MV_REPO})",
        f"             -> vendor/pixal3d-cpp/models/{pixal3d.MV_MODELS.name}/"]
    if vision:
        lines.append(f"           {size:.2f} GB  {filename} ({repo}) -> Hugging Face cache")
    lines += [f"           {total / 1e9:.1f} GB in all", "",
              "  licences: " + pixal3d.LICENCE.replace("pixal3d-sv-q8_0-v1", "pixal3d-q8_0-v1")]
    if vision:
        lines.append("            " + qwen.LICENCE.splitlines()[0])
    return "\n".join(lines + [""])


def missing_routes() -> list[str]:
    """What has to be set up before Front + back can be."""
    missing = []
    if not (pixal3d.cli_path().is_file() and pixal3d.MODELS.is_dir()):
        missing.append("Pixal3D (python scripts/bootstrap_pixal3d.py)")
    if not any(image_api.is_installed(e) for e in image_api.ENGINES.values()):
        missing.append("a picture engine (python scripts/bootstrap_flux_image.py)")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the studio and for agents.")
    args = parser.parse_args(argv)
    missing = missing_routes()
    if missing:
        print("Front + back builds on routes that are not set up yet: " + "; ".join(missing)
              + ". Set those up first. Nothing downloaded.")
        return 1
    vision = needs_vision()
    print(announcement(vision))
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    pixal3d.install_multiview()
    if vision:
        qwen.install_vision()
    print("\nDone. Create now builds characters and props from a front and a back picture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
