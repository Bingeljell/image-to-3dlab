#!/usr/bin/env python3
"""Download Hunyuan3D-MLX weights from Hugging Face.

Fetches shape-stage weights (2.1, 2.0, 2.0-turbo -- pick one with --model, or omit for
all three) into shape/weights/, and paint-stage weights into paint/weights/. Same source
repos Xiong's own dl_any.py/dl_modelscope.py pull from ModelScope, just via Hugging Face
Hub instead -- ModelScope was a region workaround for the original author, not something
needed here; verified 2026-08-19 that both tencent/Hunyuan3D-2 and tencent/Hunyuan3D-2.1
are directly reachable.

2.1 ships only a .ckpt on HF; this converts it to the .safetensors format the shape
pipeline actually loads, via shape/scripts/convert_v21_ckpt.py.

RealESRGAN super-res weights (paint/weights/realesrgan/rrdbnet.npz) aren't part of the
official Tencent HF repos and aren't fetched by this script -- run (needs a torch venv,
dev-time only): `paint/scripts/convert_realesrgan.py`. It downloads the official
xinntao/Real-ESRGAN release and converts it; see that script's docstring.

Usage:
    shape/.venv/bin/python download_weights.py [--model 2.0] [--skip-paint]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHAPE_WEIGHTS = REPO / "shape" / "weights"
PAINT_WEIGHTS = REPO / "paint" / "weights"

# (hf_repo, hf_subdir, local_group_dir) -- local_group_dir matches the layout
# hunyuan_mlx_xiong_generate.py's SHAPE_MODELS / generate_api.py's
# HUNYUAN_XIONG_SHAPE_MODELS already expect.
# Roughly what each piece costs on disk, measured 2026-09-21 (paint re-measured
# 2026-09-29 after the VAE/DINO/UNet-safetensors additions). Approximate on purpose and
# only ever printed, never checked: the point is that nobody starts a 13 GB fetch without
# being told it is 13 GB.
APPROX_GB = {"2.1": 6.9, "2.0": 4.6, "2.0-turbo": 4.6, "paint": 13.1}

# Pinned revisions for every paint fetch, so a fresh install reproduces the
# layout run_paint_pbr.py was verified against. Bump deliberately.
PAINT_REVISION_21 = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
SHAPE_VAE_REVISION = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
DINO_REVISION = "611a9d42f2335e0f921f1e313ad3c1b7178d206d"

DEFAULT_MODEL = "2.0"

SHAPE_HF_SOURCES = {
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "Hunyuan3D-2.1"),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", "Hunyuan3D-2"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo", "Hunyuan3D-2"),
}


def download_shape(model: str) -> None:
    from huggingface_hub import hf_hub_download

    hf_repo, hf_dir, local_group = SHAPE_HF_SOURCES[model]
    local_root = SHAPE_WEIGHTS / local_group
    dest = local_root / hf_dir
    dest.mkdir(parents=True, exist_ok=True)

    hf_hub_download(hf_repo, f"{hf_dir}/config.yaml", local_dir=local_root)

    if model == "2.1":
        # 2.1 only ships a .ckpt on HF -- convert once, same as the existing local copy.
        ckpt = hf_hub_download(hf_repo, f"{hf_dir}/model.fp16.ckpt", local_dir=local_root)
        target = dest / "model.fp16.safetensors"
        if not target.is_file():
            print(f"converting {ckpt} -> {target} ...", flush=True)
            subprocess.run(
                [sys.executable, str(REPO / "shape" / "scripts" / "convert_v21_ckpt.py"),
                 ckpt, str(target)],
                check=True,
            )
        # The checkpoint is the download; the safetensors is what the loader opens. Keeping
        # both stored the same 6.9 GB model twice, which nobody noticed until a disk audit
        # on 2026-09-21. Removed only once the conversion is on disk and non-empty, so an
        # interrupted convert still leaves something to retry from.
        discard_converted_checkpoint(Path(ckpt), target)
    else:
        hf_hub_download(hf_repo, f"{hf_dir}/model.fp16.safetensors", local_dir=local_root)

    print(f"{model}: ready at {dest}")


def discard_converted_checkpoint(ckpt: Path, converted: Path) -> bool:
    """Delete `ckpt` once `converted` exists and is plausibly complete.

    Returns whether anything was removed, so a caller (and a test) can tell the difference
    between "cleaned up" and "left alone because the conversion looks unfinished".
    """
    if not converted.is_file() or converted.stat().st_size <= 0:
        return False
    if not ckpt.is_file() or ckpt.resolve() == converted.resolve():
        return False
    freed = ckpt.stat().st_size
    ckpt.unlink()
    print(f"removed {ckpt.name} ({freed / 1024 ** 3:.1f} GB): superseded by "
          f"{converted.name}", flush=True)
    return True


def announce(models: list[str], *, paint: bool) -> float:
    """Say what is about to be fetched, and how much of it, before fetching anything.

    Not a prompt. Someone who typed this command has already consented; what they have
    not necessarily been told is the size, and a silent multi-gigabyte download is how a
    disk fills up unexplained. Returns the total so a caller can assert on it.
    """
    lines = [f"  {m:<10} {APPROX_GB.get(m, 0):>5.1f} GB  ({SHAPE_HF_SOURCES[m][0]})"
             for m in models]
    total = sum(APPROX_GB.get(m, 0) for m in models)
    if paint:
        lines.append(f"  {'paint':<10} {APPROX_GB['paint']:>5.1f} GB  (tencent/Hunyuan3D-2.1)")
        total += APPROX_GB["paint"]
    print("Downloading Hunyuan3D weights from Hugging Face:", flush=True)
    print("\n".join(lines), flush=True)
    print(f"  {'total':<10} {total:>5.1f} GB (approximate)\n", flush=True)
    print("These are Tencent Hunyuan Community License weights, not licensed for use in "
          "the EU, the UK or South Korea.\n", flush=True)
    return total


def _converter_deps_present() -> bool:
    """True iff this interpreter can run the UNet .bin -> .safetensors converter."""
    try:
        import huggingface_hub  # noqa: F401
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        return True
    except ImportError:
        return False


def download_paint() -> None:
    # Fail BEFORE any multi-GB download if the conversion step cannot run in
    # this interpreter: upstream ships the paint UNet only as a torch .bin,
    # so paint setup always needs torch + safetensors at least once.
    if not _converter_deps_present():
        sys.exit(
            "the paint download path needs huggingface_hub + torch + safetensors "
            "in THIS python (the paint UNet ships as a torch .bin that must be "
            "converted). Run this script with e.g. the env used for "
            "convert_realesrgan.py (plus huggingface_hub), "
            "or pre-convert "
            f"{PAINT_WEIGHTS / 'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors'}".strip() +
            " from its .bin with paint/scripts/convert_unet_bin.py first."
        )
    from huggingface_hub import hf_hub_download, snapshot_download

    # Everything run_paint_pbr.py opens, fetched at pinned revisions:
    #   1. paintpbr-v2-1/* from tencent/Hunyuan3D-2.1 (VAE? no -- see 2; UNet ships
    #      as a torch .bin, converted to safetensors below)
    #   2. the paint VAE is ONLY in tencent/Hunyuan3D-2 under hunyuan3d-paint-v2-0/vae
    #      (same SD2.1 VAE the 2.1 pipeline expects); the 2.1 repo never shipped it
    #   3. dinov2/model.safetensors is NOT in any Tencent repo (the old comment
    #      claiming it ships inside paintpbr-v2-1/dinov2/ was wrong -- verified
    #      against upstream file lists 2026-09-28). The paint code is an HF
    #      Dinov2Model port at giant config, so fetch facebook/dinov2-giant and
    #      drop it in 1:1; run_paint_pbr.py transposes the conv at load.
    snapshot_download(
        "tencent/Hunyuan3D-2.1",
        allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        local_dir=PAINT_WEIGHTS,
        revision=PAINT_REVISION_21,
    )
    pbr = PAINT_WEIGHTS / "hunyuan3d-paintpbr-v2-1"
    for vae_file in (
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        hf_hub_download(
            "tencent/Hunyuan3D-2",
            f"hunyuan3d-paint-v2-0/{vae_file}",
            local_dir=PAINT_WEIGHTS,
            revision=SHAPE_VAE_REVISION,
        )
    hf_hub_download(
        "facebook/dinov2-giant",
        "model.safetensors",
        local_dir=pbr / "dinov2",
        revision=DINO_REVISION,
    )
    unet_st = pbr / "unet" / "diffusion_pytorch_model.safetensors"
    unet_bin = pbr / "unet" / "diffusion_pytorch_model.bin"
    if unet_bin.is_file():
        # The converter is the validation gate: it accepts an existing
        # destination only if it matches the .bin on keys/shapes/dtypes/values,
        # and regenerates anything else. The source .bin is always retained.
        print("validating/converting UNet .bin -> .safetensors "
              "(torch + safetensors required)", flush=True)
        subprocess.run(
            [sys.executable, str(REPO / "paint" / "scripts" / "convert_unet_bin.py"),
             str(unet_bin), str(unet_st)],
            check=True,
        )
    print(f"paint weights: ready at {PAINT_WEIGHTS}")
    print(
        "NOTE: RealESRGAN weights (weights/realesrgan/rrdbnet.npz) are NOT covered by "
        "this script -- run paint/scripts/convert_realesrgan.py separately (needs torch)."
    )


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, separately from running it.

    Extracted so a test can assert the defaults without downloading 13 GB to find out
    what they are; the default model is the whole point of this script's recent fix.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", choices=sorted(SHAPE_HF_SOURCES), default=DEFAULT_MODEL,
        help=f"which shape model to download (default: {DEFAULT_MODEL}, the route the "
             f"pipeline uses); pass --all for every one",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="download every shape model, not just the default. Adds about "
             f"{APPROX_GB['2.1'] + APPROX_GB['2.0-turbo']:.0f} GB you probably do not need",
    )
    parser.add_argument("--skip-paint", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Defaulting to every model was the bug: the README documents this exact command and
    # calls it "~13 GB", while all three shape checkpoints plus paint come to ~24 GB.
    # Nobody chose that, and the extra two models go unused by the default route.
    models = sorted(SHAPE_HF_SOURCES) if args.all else [args.model]
    announce(models, paint=not args.skip_paint)

    for model in models:
        download_shape(model)
    if not args.skip_paint:
        download_paint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
