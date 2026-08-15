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
VENDOR = REPO / "vendor" / "trellis-mac"
TRELLIS = VENDOR / "TRELLIS.2"
STUBS = VENDOR / "stubs"

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


def _encode_coords(coords, grid_size: int):
    coords = coords.to(dtype=__import__("torch").int64)
    return (coords[:, 0] * grid_size + coords[:, 1]) * grid_size + coords[:, 2]


def _decode_coords(codes, grid_size: int):
    x = codes // (grid_size * grid_size)
    remainder = codes % (grid_size * grid_size)
    y = remainder // grid_size
    z = remainder % grid_size
    return __import__("torch").stack((x, y, z), dim=1)


def derive_subdivision_guide_tensors(
    final_coords, shape_coords, resolution: int, levels: int = 4
):
    """Recover decoder subdivision guides from an already-decoded sparse field.

    Both shape and material decoders follow the same four binary sparse upsampling
    decisions. The frozen decode's final voxel coordinates therefore encode those
    decisions exactly; recovering them avoids re-running the 40+ GiB shape decoder.
    Returns CPU ``(features, coordinates)`` pairs in decoder order.
    """
    import torch

    base_grid = resolution // (2 ** levels)
    if base_grid <= 0 or base_grid * (2 ** levels) != resolution:
        raise ValueError(f"resolution {resolution} is incompatible with {levels} levels")
    if final_coords.ndim != 2 or final_coords.shape[1] != 3:
        raise ValueError("final_coords must be an N x 3 tensor")
    if shape_coords.ndim != 2 or shape_coords.shape[1] != 4:
        raise ValueError("shape_coords must be an N x 4 BXYZ tensor")
    if torch.any(shape_coords[:, 0] != 0):
        raise ValueError("only single-batch Stage-3 caches are supported")

    # Build the active coordinate set at each decoder resolution. Packed 1-D
    # codes make unique/search operations far cheaper than N x 3 torch.unique.
    active_codes = {levels: torch.unique(_encode_coords(final_coords, resolution))}
    for level in range(levels - 1, 0, -1):
        child_grid = base_grid * (2 ** (level + 1))
        child_xyz = _decode_coords(active_codes[level + 1], child_grid)
        parent_grid = child_grid // 2
        active_codes[level] = torch.unique(
            _encode_coords(child_xyz // 2, parent_grid)
        )

    parent_coords = shape_coords.to(dtype=torch.int32, device="cpu")
    guides = []
    for level in range(levels):
        parent_grid = base_grid * (2 ** level)
        child_grid = parent_grid * 2
        child_xyz = _decode_coords(active_codes[level + 1], child_grid)
        parent_xyz = child_xyz // 2
        target_parent_codes = _encode_coords(parent_xyz, parent_grid)
        parent_codes = _encode_coords(parent_coords[:, 1:], parent_grid)
        sorted_parent_codes, parent_order = torch.sort(parent_codes)
        positions = torch.searchsorted(sorted_parent_codes, target_parent_codes)
        if torch.any(positions >= sorted_parent_codes.numel()):
            raise ValueError(f"decoded coordinates escape the shape lattice at level {level}")
        rows = parent_order[positions]
        if not torch.equal(parent_codes[rows], target_parent_codes):
            raise ValueError(f"decoded coordinates do not match shape parents at level {level}")

        child_bits = child_xyz % 2
        subindices = (
            child_bits[:, 0] + 2 * child_bits[:, 1] + 4 * child_bits[:, 2]
        ).to(torch.int64)
        features = torch.zeros((parent_coords.shape[0], 8), dtype=torch.bool)
        features[rows, subindices] = True
        guides.append((features, parent_coords))

        # SparseChannel2Spatial emits children by parent row and then subindex.
        # Preserve that order for the coordinates expected by the next guide.
        emitted_order = torch.argsort(rows * 8 + subindices)
        ordered_children = child_xyz[emitted_order].to(torch.int32)
        parent_coords = torch.cat(
            (
                torch.zeros((ordered_children.shape[0], 1), dtype=torch.int32),
                ordered_children,
            ),
            dim=1,
        )

    if parent_coords.shape[0] != final_coords.shape[0]:
        raise ValueError("recovered guide hierarchy does not reproduce final voxel count")
    return guides


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
    parser.add_argument(
        "--resume-texture-latent",
        type=Path,
        help="reuse a cached texture latent and skip conditioning/sampling",
    )
    parser.add_argument("--metadata", type=Path)
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
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
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
    }
    if args.resume_texture_latent:
        if not args.resume_texture_latent.is_file():
            raise SystemExit(f"missing cached texture latent: {args.resume_texture_latent}")
        experiment["resumed_texture_latent"] = repo_relative(args.resume_texture_latent)
    print(json.dumps(experiment, indent=2), flush=True)
    if args.dry_run:
        return 0

    try:
        import flex_gemm  # noqa: F401
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

    sys.path.insert(0, str(TRELLIS))
    sys.path.append(str(STUBS))
    from trellis2 import models as trellis_models
    from trellis2.modules.sparse import SparseTensor

    started = time.time()
    device = torch.device("mps")
    tex_latent_path = args.tex_latent_output or args.output.with_name(
        args.output.stem.removesuffix("_material") + "_tex_latent.pt"
    )
    if args.resume_texture_latent:
        print("Loading TRELLIS.2 material decoder...", flush=True)
        tex_decoder = trellis_models.from_pretrained(
            "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
        )
        tex_decoder.eval()
        pipeline = None
        load_seconds = time.time() - started
        print(f"Material decoder loaded in {load_seconds:.1f}s", flush=True)
        cached_tex = torch.load(
            args.resume_texture_latent, map_location="cpu", weights_only=False
        )
        tex_slat = SparseTensor(
            feats=cached_tex["tex_slat_feats"].to(device),
            coords=cached_tex["coords"].to(device),
        )
        tex_latent_path = args.resume_texture_latent
        cond_seconds = 0.0
        sample_seconds = 0.0
        print(f"Texture latent resumed: {tex_latent_path}", flush=True)
    else:
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        from PIL import Image

        print("Loading TRELLIS.2 pipeline...", flush=True)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.to(device)
        tex_decoder = None
        load_seconds = time.time() - started
        print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

        shape_slat = SparseTensor(
            feats=bundle["shape_slat_feats"].to(device),
            coords=bundle["coords"].to(device),
        )
        image = pipeline.preprocess_image(Image.open(image_path))
        cond_started = time.time()
        # pipeline.run always wraps even a single image before get_cond; the extractor
        # deliberately accepts a list of PIL images, not one PIL object.
        cond = pipeline.get_cond([image], 1024)
        cond_seconds = time.time() - cond_started
        print(f"Image conditioning encoded in {cond_seconds:.1f}s", flush=True)

        torch.manual_seed(args.texture_seed)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(args.texture_seed)

        sample_started = time.time()
        # pipeline.run is no-grad, but its individual helpers are not. A standalone
        # Stage-3 runner must preserve that contract or MPS retains every denoising
        # activation and later decoders exhaust unified memory.
        with torch.no_grad():
            tex_slat = pipeline.sample_tex_slat(
                cond,
                pipeline.models["tex_slat_flow_model_1024"],
                shape_slat,
                params,
            )
        sample_seconds = time.time() - sample_started
        print(f"Stage-3 material sampled in {sample_seconds:.1f}s", flush=True)

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

        # Sampling briefly holds large attention buffers on MPS. The next phase is
        # decoder-heavy, so release the conditioning and shape inputs first.
        del cond, image, shape_slat
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    decode_started = time.time()
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
    print("Recovering sparse subdivision guides from frozen geometry...", flush=True)
    guide_tensors = derive_subdivision_guide_tensors(
        geometry["coords"], bundle["coords"], int(bundle["res"])
    )
    subs = [
        SparseTensor(features.to(device), coords.to(device))
        for features, coords in guide_tensors
    ]
    del guide_tensors
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    with torch.no_grad():
        if pipeline is not None:
            tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
        else:
            tex_decoder.to(device)
            tex_voxels = tex_decoder(tex_slat, guide_subs=subs) * 0.5 + 0.5
            tex_decoder.cpu()
    voxel = tex_voxels[0]
    decode_seconds = time.time() - decode_started
    print(f"Material field decoded in {decode_seconds:.1f}s", flush=True)

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
