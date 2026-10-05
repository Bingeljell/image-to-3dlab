#!/usr/bin/env python3
"""Install the auto-rigger: SkinTokens (VAST-AI, MIT), which gives a humanoid mesh a skeleton
and skin weights.

    python scripts/bootstrap_autorig.py          # says what it will fetch, then asks
    python scripts/bootstrap_autorig.py --yes    # non-interactive (the viewer passes this)
    python scripts/bootstrap_autorig.py --check  # exit 0 if installed, 1 if not

What it does, in order:
1. Clones SkinTokens into `vendor/SkinTokens` at a pinned commit (git-ignored, ~5 MB).
2. Applies `patch_skintokens_portable.py`, so it runs on a Mac and on NVIDIA without
   flash-attn.
3. Makes a Python 3.11 venv there (Blender's `bpy` needs 3.11) with torch and upstream's
   requirements, about 2 GB of packages.
4. Fetches the two checkpoints (~1.6 GB) at a pinned revision into the shared Hugging Face
   cache, links them where upstream looks, and fetches Qwen3-0.6B's config and tokenizer
   (~16 MB, no weights: SkinTokens trains its own).

Nothing is fetched without an explicit yes: AGENTS.md forbids weight downloads the user
has not chosen. The checkpoints are pickles, which can run code when loaded, so the
revision is pinned to the one we tested rather than whatever is newest.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import patch_skintokens_portable as portable

UPSTREAM = "https://github.com/VAST-AI-Research/SkinTokens.git"
COMMIT = "273b691d35989d71cd17ff2895fdc735097b92d1"
VENDOR = REPO / "vendor" / "SkinTokens"
PYTHON_VERSION = "3.11"

HF_REPO = "VAST-AI/SkinTokens"
HF_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
LLM_REPO = "Qwen/Qwen3-0.6B"
LLM_DIR = "models/Qwen3-0.6B"


@dataclass(frozen=True)
class Checkpoint:
    path: str  # inside the HF repo, and where upstream's demo.py looks for it
    bytes: int


CHECKPOINTS = (
    Checkpoint("experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
               1_131_603_979),
    Checkpoint("experiments/skin_vae_2_10_32768/last.ckpt", 487_311_745),
)
WEIGHT_BYTES = sum(c.bytes for c in CHECKPOINTS)
# Upstream's requirements.txt uses scipy without listing it.
EXTRA_PACKAGES = ("torch", "torchvision", "scipy")


def venv_python(root: Path = VENDOR) -> Path:
    return root / ".venv" / "bin" / "python"


def announcement(build: bool = True, weights: bool = True) -> str:
    lines = ["Auto-rig: SkinTokens by VAST-AI (MIT licence)"]
    if build:
        lines.append(f"  code     {UPSTREAM} @ {COMMIT[:7]} -> {VENDOR.relative_to(REPO)}/")
        lines.append(f"  packages a Python {PYTHON_VERSION} venv there, about 2 GB "
                     "(torch, Blender's bpy)")
    if weights:
        lines.append(f"  weights  {HF_REPO} @ {HF_REVISION[:7]}, "
                     f"{WEIGHT_BYTES / 1e9:.1f} GB, into the Hugging Face cache")
        lines.append(f"           plus {LLM_REPO} config and tokenizer only (~16 MB)")
    return "\n".join(lines)


def build_present(root: Path = VENDOR) -> bool:
    return venv_python(root).exists() and (root / "demo.py").exists()


def weights_present(root: Path = VENDOR) -> bool:
    return (all((root / c.path).exists() for c in CHECKPOINTS)
            and (root / LLM_DIR / "config.json").exists())


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_pinned(root: Path = VENDOR, runner=run) -> None:
    """Clone at COMMIT, or move an existing clone there. A clone with local edits is ours
    (the portable patch), which `git checkout` keeps."""
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        runner(["git", "clone", UPSTREAM, str(root)])
    runner(["git", "-C", str(root), "checkout", "--quiet", COMMIT])


def venv_needs_rebuild(root: Path = VENDOR) -> bool:
    cfg = root / ".venv" / "pyvenv.cfg"
    if not cfg.exists():
        return True
    return f"version_info = {PYTHON_VERSION}" not in cfg.read_text()


def install_packages(root: Path = VENDOR, runner=run) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    if venv_needs_rebuild(root):
        shutil.rmtree(root / ".venv", ignore_errors=True)
        runner([uv, "venv", str(root / ".venv"), "--python", PYTHON_VERSION])
    runner([uv, "pip", "install", "--python", str(venv_python(root)), *EXTRA_PACKAGES,
            "-r", str(root / "requirements.txt")])


def link_checkpoint(snapshot_file: Path, root: Path, checkpoint: Checkpoint) -> Path:
    """Point upstream's expected path at the cached file. A symlink, so the 1.6 GB lives
    once in the shared cache and survives a re-clone of vendor/."""
    target = root / checkpoint.path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(snapshot_file)
    return target


def install_weights(root: Path = VENDOR) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as missing:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from missing
    for checkpoint in CHECKPOINTS:
        print(f"\nFetching {checkpoint.path} ({checkpoint.bytes / 1e9:.2f} GB)...", flush=True)
        cached = Path(hf_hub_download(repo_id=HF_REPO, filename=checkpoint.path,
                                      revision=HF_REVISION))
        print(f"  linked {link_checkpoint(cached, root, checkpoint)}")
    print(f"\nFetching {LLM_REPO} config and tokenizer...", flush=True)
    snapshot_download(repo_id=LLM_REPO, local_dir=str(root / LLM_DIR),
                      ignore_patterns=["*.bin", "*.safetensors"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--build-only", action="store_true",
                        help="Clone, patch and install packages; leave the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming the code is installed.")
    parser.add_argument("--check", action="store_true",
                        help="Exit 0 if code and weights are installed, 1 if not.")
    args = parser.parse_args(argv)

    if args.check:
        ok = build_present() and weights_present()
        print("Auto-rig is installed." if ok else "Auto-rig is not installed.")
        return 0 if ok else 1

    build = not args.weights_only
    weights = not args.build_only
    print(announcement(build=build, weights=weights))

    if not args.yes:
        # Non-interactive without --yes must not silently proceed, and must not hang
        # waiting on a stdin nobody is attached to.
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if build:
        clone_pinned()
        portable.patch(VENDOR)
        install_packages()
    if weights:
        install_weights()
    print("\nDone. Open the viewer's Rig tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
