#!/usr/bin/env python3
"""Install Stable Fast 3D: its code and compiled extensions, then its gated weights.

Two halves, the same way the viewer tracks every backend. The **code** is Stability's
repository in `vendor/stable-fast-3d`, installed into this interpreter together with its
two compiled extensions, a texture baker and a UV unwrapper:

- **Apple Silicon only:** the baker is built with Metal. It needs Homebrew's `libomp`.
  NVIDIA is not supported: SF3D pins packages (an old huggingface-hub and rembg) that
  broke the lab's own environment there, and the newer NVIDIA routes beat it anyway.

The **weights** are SF3D itself (3.8 GB, **gated**: accept Stability's licence on Hugging
Face and log in first) and DINOv2 (1.1 GB), which SF3D would otherwise fetch unannounced
on its first run.

Install into Python 3.10 or 3.11: SF3D pins numpy 1.26 and transformers 4.42.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_sf3d.py            # says what it wants, then asks
    python scripts/bootstrap_sf3d.py --yes      # for the viewer and for agents
    python scripts/bootstrap_sf3d.py --code-only
    python scripts/bootstrap_sf3d.py --weights-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host

VENDOR = REPO / "vendor" / "stable-fast-3d"
UPSTREAM = "https://github.com/Stability-AI/stable-fast-3d.git"
# Upstream's head since January 2025, and the code every SF3D run here has used.
COMMIT = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2"

# (repo, files, approximate gigabytes). DINOv2 is here because SF3D's image tokenizer
# downloads it on first use otherwise, which is exactly the surprise AGENTS.md forbids.
WEIGHTS = [
    ("stabilityai/stable-fast-3d", ["config.yaml", "model.safetensors"], 3.75),
    ("facebook/dinov2-large", ["config.json", "model.safetensors"], 1.13),
]

LICENCE = (
    "Stability AI Community License: free under $1M annual revenue, otherwise a\n"
    "  commercial licence from Stability. Gated: accept it at\n"
    "  https://huggingface.co/stabilityai/stable-fast-3d and run `hf auth login`.\n"
    "  DINOv2 is Apache 2.0."
)

# Looked up through the module so a test can pretend to be another machine.
target = host.build_target


class GatedAccess(Exception):
    """Hugging Face refused a gated repository: the licence is not accepted yet, or the
    user is not logged in."""


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def build_env(base: dict[str, str]) -> dict[str, str]:
    """Environment for building SF3D's extensions: the Metal baker, no CUDA."""
    return {**base, "USE_CUDA": "0", "USE_METAL": "1"}


def route(key: str | None) -> str | None:
    if key == "macos-arm64":
        return "PyTorch on MPS, texture baker built with Metal"
    return None


def announcement(code: bool = True, weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: Stable Fast 3D (Stability AI)",
             f"  route:   {route(target()) or 'none for this machine'}"]
    if code:
        lines.append("  code:    Stability-AI/stable-fast-3d -> vendor/stable-fast-3d/, "
                     "plus its pinned Python packages into this interpreter")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total -> Hugging Face cache")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>5.2f} GB  {repo}")
        lines.append("           stabilityai/stable-fast-3d is gated (see licence below)")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def has_pip() -> bool:
    return subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True,
                          check=False).returncode == 0


def pip_install_command() -> list[str]:
    """How to install packages into this interpreter.

    The one-line installer builds the environment with uv, and uv environments have no
    pip, so uv does the installing when it is there.
    """
    uv = shutil.which("uv")
    if uv and not has_pip():
        return [uv, "pip", "install", "--python", sys.executable]
    if has_pip():
        return [sys.executable, "-m", "pip", "install"]
    raise SystemExit("Neither pip nor uv is available to install SF3D's packages. "
                     "Install uv (https://docs.astral.sh/uv/) and run this again.")


def fetch_commands(vendor: Path = VENDOR) -> list[list[str]]:
    """Check out exactly COMMIT, shallow: `git clone --branch` takes no commit."""
    git = ["git", "-C", str(vendor)]
    return [["git", "init", "-q", str(vendor)],
            [*git, "remote", "add", "origin", UPSTREAM],
            [*git, "fetch", "-q", "--depth", "1", "origin", COMMIT],
            [*git, "checkout", "-q", "FETCH_HEAD"]]


def install_code(key: str) -> None:
    if key == "macos-arm64" and not Path("/opt/homebrew/opt/libomp").exists():
        raise SystemExit("SF3D's Metal baker needs libomp: brew install libomp")
    if not (VENDOR / ".git").is_dir():
        print(f"Fetching {UPSTREAM} @ {COMMIT[:12]}", flush=True)
        VENDOR.mkdir(parents=True, exist_ok=True)
        for command in fetch_commands():
            subprocess.run(command, check=True)
    print("Installing SF3D's packages and building its extensions...", flush=True)
    # --no-build-isolation so the extensions compile against the torch already installed
    # here, not a fresh one pip would fetch into a throwaway build environment.
    install = pip_install_command()
    subprocess.run([*install, "setuptools", "wheel"], check=True)
    subprocess.run([*install, "--no-build-isolation", "-r", "requirements.txt"],
                   cwd=VENDOR, env=build_env(dict(os.environ)), check=True)


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements.txt"
        ) from exc
    for repo, files, size in WEIGHTS:
        print(f"\nFetching {repo} ({size:.2f} GB)...", flush=True)
        try:
            snapshot_download(repo, allow_patterns=files)
        except GatedRepoError as exc:
            raise GatedAccess(repo) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and extensions, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    args = parser.parse_args(argv)

    key = target()
    if route(key) is None:
        print("SF3D installs on an Apple Silicon Mac only. On NVIDIA, use TRELLIS.2, "
              "Hunyuan3D-2.1 or Pixal3D. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    print(announcement(code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if code:
        install_code(key)
    if weights:
        try:
            install_weights()
        except GatedAccess as exc:
            print(f"\nHugging Face refused {exc}: it is gated.\n"
                  f"  1. Accept the licence at https://huggingface.co/{exc}\n"
                  "  2. Run `hf auth login` with a token from that account\n"
                  "  3. Run this again with --weights-only")
            return 1
    print("\nDone. Pick Stable Fast 3D in the studio (Create, Advanced: 3D engine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
