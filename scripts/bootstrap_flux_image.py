#!/usr/bin/env python3
"""Install FLUX.2 [klein] 4B, the studio's default picture engine, Apache-2.0 throughout.

It runs on the same prebuilt stable-diffusion.cpp binary as Qwen-Image
(bootstrap_qwen_image.py installs it, and this calls that), so the only new download is
three weight files, about 8.9 GB. It also draws the back picture for front + back builds by
reading the front itself, so nothing else is needed for those.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_flux_image.py            # says what it wants, then asks
    python scripts/bootstrap_flux_image.py --yes      # for the studio and for agents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import bootstrap_qwen_image as sdcpp

# (repo, file inside it, gigabytes as Hugging Face lists them). Q8_0 for the model, as for
# Qwen; the text encoder is plain Qwen3-4B (Apache-2.0), not Qwen-Image's research-licensed one.
WEIGHTS = [
    ("leejet/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf", 4.30),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", 4.28),
    ("Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors", 0.34),
]

LICENCE = (
    "Apache-2.0 (FLUX.2 [klein] 4B, its VAE, and the Qwen3-4B text encoder).\n"
    "  The pictures you make are yours, commercial use included.\n"
    "  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def announcement(build: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is."""
    build_for = sdcpp.BUILDS.get(sdcpp.target())
    route = build_for.route if build_for else "no prebuilt build for this machine"
    lines = ["", "About to install:", "", "  backend: FLUX.2 [klein] 4B (text to image)",
             f"  route:   GGUF via stable-diffusion.cpp, {route}"]
    if build and build_for and not sdcpp.binary_present():
        lines.append(f"  build:   prebuilt sd-cli release ({build_for.size}) -> vendor/sdcpp/")
    elif build and build_for and sdcpp.binary_stale():
        # ensure_binary updates an older engine: say so before the yes, not after
        lines.append(f"  build:   update sd-cli to {sdcpp.PINNED[sdcpp.target()]} ({build_for.size}) "
                     "-> vendor/sdcpp/")
    lines.append(f"  weights: {total_gb():.1f} GB total ->  Hugging Face cache")
    for repo, filename, size in WEIGHTS:
        lines.append(f"             {size:>5.2f} GB  {Path(filename).name}  ({repo})")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def install_weights() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    for repo, filename, size in WEIGHTS:
        print(f"\nFetching {Path(filename).name} ({size:.2f} GB) from {repo}...", flush=True)
        print(f"  {hf_hub_download(repo_id=repo, filename=filename)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the studio and for scripts.")
    args = parser.parse_args(argv)
    print(announcement())
    if not args.yes:
        # Non-interactive without --yes must not silently proceed, and must not hang
        # waiting on a stdin nobody is attached to.
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    if not sdcpp.ensure_binary():
        return 1
    install_weights()
    print("\nDone. In the studio, press + Create and start from an idea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
