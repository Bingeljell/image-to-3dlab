#!/usr/bin/env python3
"""Run a frozen-shape TRELLIS.2 Stage-3 material probe on CUDA.

This intentionally stops before UV baking/export.  It answers whether the generated PBR
field is already dark on CUDA while holding the source image, shape latent, sampler, and
texture seed fixed against a local MPS probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(Path(__file__).resolve().parent / "autotune_cache.json"),
    )

    import torch
    from PIL import Image
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.latents, map_location="cpu", weights_only=False)
    resolution = int(bundle["res"])
    if resolution != 512:
        raise ValueError(f"this parity probe expects a frozen 512 shape, got {resolution}")

    started = time.time()
    # The source already has useful transparency, so official preprocessing never invokes
    # background removal.  Avoid constructing gated RMBG-2.0 while leaving the official
    # TRELLIS pipeline, weights, preprocessing, and sampling implementation untouched.
    with patch.object(rembg, "BiRefNet", new=lambda **_kwargs: None):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    pipeline_seconds = time.time() - started

    raw_image = Image.open(args.image)
    image = pipeline.preprocess_image(raw_image)
    processed_path = output_dir / "preprocessed-black-matte.png"
    image.save(processed_path)

    cond_started = time.time()
    cond = pipeline.get_cond([image], 512)
    cond_seconds = time.time() - cond_started

    shape_slat = SparseTensor(
        feats=bundle["shape_slat_feats"].cuda(),
        coords=bundle["coords"].cuda(),
    )
    sampler = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.6, 0.9),
        "rescale_t": 3.0,
    }
    torch.manual_seed(args.texture_seed)
    torch.cuda.manual_seed_all(args.texture_seed)

    sample_started = time.time()
    tex_slat = pipeline.sample_tex_slat(
        cond,
        pipeline.models["tex_slat_flow_model_512"],
        shape_slat,
        sampler,
    )
    sample_seconds = time.time() - sample_started

    latent_path = output_dir / "cuda-texture-latent.pt"
    torch.save(
        {
            "tex_slat_feats": tex_slat.feats.detach().cpu(),
            "coords": tex_slat.coords.detach().cpu(),
            "res": resolution,
            "texture_seed": args.texture_seed,
            "sampler": sampler,
        },
        latent_path,
    )

    decode_started = time.time()
    _meshes, subs = pipeline.decode_shape_slat(shape_slat, resolution)
    voxel = pipeline.decode_tex_slat(tex_slat, subs)[0]
    decode_seconds = time.time() - decode_started
    attrs = voxel.feats.detach().cpu().float()
    coords = voxel.coords[:, 1:].detach().cpu()
    layout = pipeline.pbr_attr_layout

    material_path = output_dir / "cuda-material-field.pt"
    torch.save(
        {
            "attrs": attrs,
            "coords": coords,
            "attr_layout": layout,
            "res": resolution,
        },
        material_path,
    )
    report = {
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "source_sha256": sha256(args.image),
        "latents_sha256": sha256(args.latents),
        "processed_sha256": sha256(processed_path),
        "resolution": resolution,
        "texture_seed": args.texture_seed,
        "sampler": sampler,
        "voxel_attributes": {
            "count": int(attrs.shape[0]),
            "base_color_mean": attrs[:, layout["base_color"]].mean(0).tolist(),
            "base_color_median": attrs[:, layout["base_color"]].median(0).values.tolist(),
            "metallic_median": float(attrs[:, layout["metallic"]].median()),
            "roughness_median": float(attrs[:, layout["roughness"]].median()),
            "alpha_median": float(attrs[:, layout["alpha"]].median()),
        },
        "texture_latent": {
            "mean": float(tex_slat.feats.float().mean()),
            "std": float(tex_slat.feats.float().std()),
            "sha256": sha256(latent_path),
        },
        "timings_seconds": {
            "pipeline_load": pipeline_seconds,
            "conditioning": cond_seconds,
            "stage3_sample": sample_seconds,
            "material_decode": decode_seconds,
            "total": time.time() - started,
        },
    }
    report_path = output_dir / "cuda-probe.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
