#!/usr/bin/env python3
"""Resample only TRELLIS.2's Stage-3 material field on a cached shape latent.

This keeps sparse structure, shape sampling, and final geometry fixed. It reloads the source
image conditioning, samples a new texture SLat with an explicit independent texture seed,
decodes the voxel PBR attributes, and writes a split cache consumed by trellis_rebake.py.

Example:

    env PYTHONUNBUFFERED=1 vendor/trellis-mac/.venv/bin/python \
      scripts/trellis_stage3.py \
      output/snag_same_seed_hf/snag_seed614089393_latents.pt \
      output/snag_same_seed_hf/stage3/default_tseed614089393_material.pt \
      --geometry-decode output/snag_same_seed_hf/snag_seed614089393_decode.pt \
      --texture-seed 614089393

The output does not duplicate vertices/faces. It records ``geometry_ref`` and contains only
the candidate attrs/coords plus layout and voxel size. Re-bake it exactly like a full decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR = REPO / "vendor" / "trellis-mac"

REQUIRED_LATENT_KEYS = {
    "shape_slat_feats",
    "coords",
    "res",
    "pipeline_type",
    "images",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_params(args: argparse.Namespace) -> dict:
    return {
        "steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "guidance_rescale": args.guidance_rescale,
        "guidance_interval": (args.guidance_interval[0], args.guidance_interval[1]),
        "rescale_t": args.rescale_t,
    }


def validate_bundle(payload: dict) -> None:
    missing = REQUIRED_LATENT_KEYS - payload.keys()
    if missing:
        raise ValueError(f"latent bundle missing: {', '.join(sorted(missing))}")
    feats = payload["shape_slat_feats"]
    coords = payload["coords"]
    if getattr(feats, "ndim", None) != 2 or getattr(coords, "ndim", None) != 2:
        raise ValueError("shape_slat_feats and coords must both be rank-2 tensors")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"latent row mismatch: {feats.shape[0]} features vs {coords.shape[0]} coords"
        )
    if coords.shape[1] != 4:
        raise ValueError(f"expected batched BXYZ coordinates with 4 columns, got {coords.shape}")
    if int(payload["res"]) not in (512, 1024, 1536):
        raise ValueError(f"unsupported latent resolution: {payload['res']}")


def default_geometry_decode(latent_path: Path) -> Path:
    name = latent_path.name
    if name.endswith("_latents.pt"):
        return latent_path.with_name(name.removesuffix("_latents.pt") + "_decode.pt")
    return latent_path.with_name(latent_path.stem + "_decode.pt")


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("latents", type=Path)
    parser.add_argument("output", type=Path, help="split material cache (.pt)")
    parser.add_argument("--geometry-decode", type=Path)
    parser.add_argument("--image", type=Path,
                        help="conditioning image; defaults to the first cached image")
    parser.add_argument("--texture-seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-strength", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--guidance-interval", type=float, nargs=2, default=(0.6, 0.9))
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--tex-latent-output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR,
                        help="TRELLIS wrapper containing TRELLIS.2 and .venv")
    parser.add_argument("--sparse-attn-backend", default="sdpa",
                        choices=("sdpa", "metal_flash"))
    parser.add_argument("--sample-only", action="store_true",
                        help="save the texture latent and stop before material decoding")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and print the experiment without loading models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.latents.is_file():
        raise SystemExit(f"missing latent bundle: {args.latents}")

    # Backend configuration must precede every torch/TRELLIS import.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", args.sparse_attn_backend)
    vendor = args.vendor_root.resolve()
    trellis = vendor / "TRELLIS.2"
    stubs = vendor / "stubs"
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(vendor / "cache/flex_gemm_autotune.json"),
    )
    import torch

    bundle = torch.load(args.latents, map_location="cpu", weights_only=False)
    validate_bundle(bundle)
    geometry_path = args.geometry_decode or default_geometry_decode(args.latents)
    image_path = args.image or Path(bundle["images"][0])
    if not geometry_path.is_file():
        raise SystemExit(f"missing geometry decode: {geometry_path}")
    if not image_path.is_file():
        raise SystemExit(f"missing conditioning image: {image_path}")

    params = sampler_params(args)
    experiment = {
        "shape_seed": bundle.get("seed"),
        "texture_seed": args.texture_seed,
        "pipeline_type": bundle["pipeline_type"],
        "resolution": int(bundle["res"]),
        "sampler": params,
        "latents": repo_relative(args.latents),
        "geometry_decode": repo_relative(geometry_path),
        "image": repo_relative(image_path),
        "output": repo_relative(args.output),
        "vendor_root": str(vendor),
        "sparse_attention_backend": args.sparse_attn_backend,
    }
    print(json.dumps(experiment, indent=2), flush=True)
    if args.dry_run:
        return 0

    try:
        import flex_gemm  # noqa: F401
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

    sys.path.insert(0, str(trellis))
    if stubs.is_dir():
        sys.path.append(str(stubs))
    from PIL import Image
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    started = time.time()
    raw_image = Image.open(image_path)
    has_transparent_alpha = (
        raw_image.mode == "RGBA" and raw_image.getextrema()[3][0] < 255
    )
    print("Loading TRELLIS.2 pipeline...", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        load_rembg=not has_transparent_alpha,
    )
    pipeline.to(torch.device("mps"))
    load_seconds = time.time() - started
    print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

    image = pipeline.preprocess_image(raw_image)
    cond_started = time.time()
    # pipeline.run always wraps even a single image before get_cond; the extractor
    # deliberately accepts a list of PIL images, not one PIL object.
    cond = pipeline.get_cond([image], 1024)
    cond_seconds = time.time() - cond_started
    print(f"Image conditioning encoded in {cond_seconds:.1f}s", flush=True)

    device = torch.device("mps")
    shape_slat = SparseTensor(
        feats=bundle["shape_slat_feats"].to(device),
        coords=bundle["coords"].to(device),
    )
    torch.manual_seed(args.texture_seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(args.texture_seed)

    sample_started = time.time()
    tex_slat = pipeline.sample_tex_slat(
        cond,
        pipeline.models["tex_slat_flow_model_1024"],
        shape_slat,
        params,
    )
    sample_seconds = time.time() - sample_started
    print(f"Stage-3 material sampled in {sample_seconds:.1f}s", flush=True)

    tex_latent_path = args.tex_latent_output or args.output.with_name(
        args.output.stem.removesuffix("_material") + "_tex_latent.pt"
    )
    tex_latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tex_slat_feats": tex_slat.feats.cpu(),
            "coords": tex_slat.coords.cpu(),
            "res": int(bundle["res"]),
            "texture_seed": args.texture_seed,
            "sampler": params,
            "shape_latents_sha256": sha256(args.latents),
        },
        tex_latent_path,
    )
    print(f"Texture latent cached: {tex_latent_path}", flush=True)

    if args.sample_only:
        metadata = {
            "schema_version": 1,
            "experiment": experiment,
            "timings_seconds": {
                "pipeline_load": load_seconds,
                "conditioning": cond_seconds,
                "stage3_sample": sample_seconds,
                "total": time.time() - started,
            },
            "artifacts": {
                "texture_latent": {
                    "path": repo_relative(tex_latent_path),
                    "sha256": sha256(tex_latent_path),
                    "bytes": tex_latent_path.stat().st_size,
                }
            },
        }
        metadata_path = args.metadata or args.output.with_suffix(".json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print("Sample-only run complete; material decode intentionally skipped.", flush=True)
        print(f"Metadata: {metadata_path}", flush=True)
        return 0

    decode_started = time.time()
    _meshes, subs = pipeline.decode_shape_slat(shape_slat, int(bundle["res"]))
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    voxel = tex_voxels[0]
    decode_seconds = time.time() - decode_started
    print(f"Material field decoded in {decode_seconds:.1f}s", flush=True)

    geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
    material_payload = {
        "geometry_ref": repo_relative(geometry_path),
        "attrs": voxel.feats.cpu(),
        "coords": voxel.coords[:, 1:].cpu(),
        "layout": geometry["layout"],
        "voxel_size": geometry["voxel_size"],
        "texture_seed": args.texture_seed,
        "sampler": params,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(material_payload, args.output)

    attrs = material_payload["attrs"]
    stats = {
        "count": int(attrs.shape[0]),
        "base_color_mean": [float(value) for value in attrs[:, :3].mean(0)],
        "base_color_median": [float(value) for value in attrs[:, :3].median(0).values],
        "metallic_median": float(attrs[:, 3].median()),
        "roughness_median": float(attrs[:, 4].median()),
        "alpha_median": float(attrs[:, 5].median()),
    }
    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "timings_seconds": {
            "pipeline_load": load_seconds,
            "conditioning": cond_seconds,
            "stage3_sample": sample_seconds,
            "material_decode": decode_seconds,
            "total": time.time() - started,
        },
        "voxel_attributes": stats,
        "artifacts": {
            "material_cache": {
                "path": repo_relative(args.output),
                "sha256": sha256(args.output),
                "bytes": args.output.stat().st_size,
            },
            "texture_latent": {
                "path": repo_relative(tex_latent_path),
                "sha256": sha256(tex_latent_path),
                "bytes": tex_latent_path.stat().st_size,
            },
        },
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Material cache: {args.output}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
