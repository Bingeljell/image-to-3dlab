#!/usr/bin/env python3
"""Full image -> GLB generation through the CLEAN ``trellis-space-mac`` port on Apple Silicon.

This is the Mac end-to-end CLI wrapper the clean port lacked. The upstream Gradio ``app.py``
already contains the full path -- ``pipeline.run()`` (Stage 1 sparse -> Stage 2 shape ->
Stage 3 material) -> ``decode_latent`` -> ``o_voxel.postprocess.to_glb`` -- but it is CUDA/HF-Space
bound (``ATTN_BACKEND=flash_attn``, ``.cuda()``, ``@spaces.GPU``, Gradio). This script runs the
same path on MPS/SDPA, reusing the proven scaffold from ``trellis_stage3.py``.

**It mirrors the demo's DEFAULT parameters exactly** (see ``DEMO_PARAMS``). Do not experiment with
these until a clean baseline is established -- the demo produces good output, so any deviation is
unnecessary variance. The demo's ``decimation_target=300000`` + ``remesh=True`` is *why* its meshes
are clean (~300k faces, few holes) rather than the 3M-face, holey output of the old ``trellis-mac``
port at ``remesh=off``.

Run it with the clean port's interpreter (not the old port's, not the dev venv)::

    env PYTHONUNBUFFERED=1 \
      vendor/upstream-audit-worktree/vendor/trellis-space-mac/.venv/bin/python \
      vendor/upstream-audit-worktree/scripts/trellis_space_generate.py \
      assets_to_test/cute-creature-lucian.png output/space_baseline/lucian.glb

Verify the environment first -- seconds, no model load, no sampling -- before the ~25-min run::

    ... trellis_space_generate.py --check

Saved alongside the GLB: ``<out>_latents.pt`` (schema-compatible with ``trellis_stage3.py`` /
``trellis_rebake.py`` so Stage-3 resamples and rebakes need no re-sampling) and a ``<out>.json``
manifest with the exact params and per-stage timings.

License note: the input must carry a transparent alpha foreground. With alpha, the background
remover (rembg / BRIA RMBG) is never loaded (``load_rembg=False``), honoring the repo's BRIA
guardrail. A non-alpha input would load it, so this script refuses one unless ``--allow-rembg``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR = REPO / "vendor" / "trellis-space-mac"

# --- The upstream Gradio demo defaults, verbatim (app.py gr.Slider ``value=``). ---
# resolution "1024" -> pipeline_type "1024_cascade"; seed 0; extract at 300k faces / 2048 texture.
DEMO_PARAMS: dict[str, Any] = {
    "resolution": "1024",
    "seed": 0,
    "decimation_target": 300_000,
    "texture_size": 2048,
    "sparse_structure": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
    "shape_slat": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
    "tex_slat": {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
    "remesh": {"remesh": True, "remesh_band": 1, "remesh_project": 0},
}

PIPELINE_TYPE_BY_RESOLUTION = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}

# nvdiffrast index limit; app.py calls mesh.simplify(this) before to_glb.
NVDIFFRAST_FACE_LIMIT = 16_777_216
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


# ----------------------------------------------------------------------------------------------
# Pure, importable helpers (no torch/PIL at import time, so the unit test needs neither).
# ----------------------------------------------------------------------------------------------
def pipeline_type_for_resolution(resolution: str) -> str:
    """Map the demo's resolution radio ("512"/"1024"/"1536") to the pipeline_type run() wants."""
    try:
        return PIPELINE_TYPE_BY_RESOLUTION[str(resolution)]
    except KeyError:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of "
            f"{sorted(PIPELINE_TYPE_BY_RESOLUTION)}"
        ) from None


def sampler_params(stage: str) -> dict[str, float]:
    """The four sampler params (steps/guidance_strength/guidance_rescale/rescale_t) for a stage.

    ``stage`` is one of ``sparse_structure`` / ``shape_slat`` / ``tex_slat``. Returns a fresh dict
    of the demo defaults -- the single source of truth is ``DEMO_PARAMS`` so tests catch drift.
    """
    if stage not in ("sparse_structure", "shape_slat", "tex_slat"):
        raise ValueError(f"unknown sampler stage {stage!r}")
    return dict(DEMO_PARAMS[stage])


def alpha_is_transparent(mode: str, alpha_min: int | None) -> bool:
    """True when an image actually carries a transparent foreground.

    ``alpha_min`` is the minimum value of the alpha channel (None if there is no alpha channel).
    Mirrors app.py/trellis_stage3: RGBA with any pixel below fully opaque means real alpha.
    """
    return mode == "RGBA" and alpha_min is not None and alpha_min < 255


def build_manifest(
    *,
    image: str,
    output: str,
    params: dict[str, Any],
    pipeline_type: str,
    seed: int,
    timings: dict[str, float],
    artifacts: dict[str, Any],
    load_rembg: bool,
    sparse_attn_backend: str,
) -> dict[str, Any]:
    """Assemble the run manifest. Pure: all inputs in, one dict out (testable without torch)."""
    return {
        "schema_version": 1,
        "generator": "trellis_space_generate.py",
        "port": "trellis-space-mac (clean upstream HF Space, MPS)",
        "input": image,
        "output": output,
        "device": "mps",
        "attn_backend": "sdpa",
        "sparse_attn_backend": sparse_attn_backend,
        "load_rembg": load_rembg,
        "seed": seed,
        "pipeline_type": pipeline_type,
        "params": params,
        "timings_seconds": timings,
        "artifacts": artifacts,
    }


def configure_environment(vendor_root: Path, sparse_attn_backend: str) -> None:
    """Set the Mac backend env BEFORE any torch/TRELLIS import. Idempotent (setdefault)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", sparse_attn_backend)
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(vendor_root / "cache" / "flex_gemm_autotune.json"),
    )


def verify_paths(vendor_root: Path) -> list[str]:
    """Cheap filesystem checks: the clean port is present and looks built. Returns problems."""
    problems: list[str] = []
    checks = {
        ".venv python": vendor_root / ".venv" / "bin" / "python",
        "TRELLIS.2": vendor_root / "TRELLIS.2",
        "pipeline module": vendor_root / "TRELLIS.2" / "trellis2" / "pipelines"
        / "trellis2_image_to_3d.py",
        "patched mesh/base.py": vendor_root / "TRELLIS.2" / "trellis2" / "representations"
        / "mesh" / "base.py",
        "o_voxel": vendor_root / "deps" / "trellis2-apple" / "o-voxel" / "o_voxel"
        / "postprocess.py",
        "remesh.metal": vendor_root / "deps" / "mtlmesh" / "src" / "metal" / "remesh.metal",
    }
    for name, path in checks.items():
        if not path.exists():
            problems.append(f"missing {name}: {path}")
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


# ----------------------------------------------------------------------------------------------
# The heavy path. Everything torch/TRELLIS-shaped is imported lazily inside these functions so
# importing this module (for the unit test) stays free.
# ----------------------------------------------------------------------------------------------
def check_environment(vendor_root: Path, sparse_attn_backend: str) -> int:
    """Fast env gate: imports resolve, MPS is up, the run/to_glb surface exists. No model load."""
    problems = verify_paths(vendor_root)
    if problems:
        for p in problems:
            print(f"  FAIL: {p}", flush=True)
        return 1

    configure_environment(vendor_root, sparse_attn_backend)
    sys.path.insert(0, str(vendor_root / "TRELLIS.2"))
    stubs = vendor_root / "stubs"
    if stubs.is_dir():
        sys.path.append(str(stubs))

    import torch

    if not torch.backends.mps.is_available():
        print("  FAIL: torch MPS backend not available", flush=True)
        return 1

    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    import o_voxel  # noqa: F401

    missing = [m for m in ("run", "decode_latent", "get_cond", "preprocess_image")
               if not hasattr(Trellis2ImageTo3DPipeline, m)]
    if missing:
        print(f"  FAIL: pipeline missing methods: {missing}", flush=True)
        return 1
    if not hasattr(o_voxel.postprocess, "to_glb"):
        print("  FAIL: o_voxel.postprocess.to_glb not found", flush=True)
        return 1
    if os.environ.get("ATTN_BACKEND") != "sdpa":
        print(f"  FAIL: ATTN_BACKEND={os.environ.get('ATTN_BACKEND')!r}, expected 'sdpa'", flush=True)
        return 1

    print("  PASS: clean port present, MPS available, run/decode_latent/to_glb resolved, "
          "ATTN_BACKEND=sdpa", flush=True)
    return 0


def generate(
    image_path: Path,
    output_path: Path,
    vendor_root: Path,
    *,
    seed: int,
    sparse_attn_backend: str,
    allow_rembg: bool,
    save_latents: bool,
    resolution: str = DEMO_PARAMS["resolution"],
    decimation_target: int = DEMO_PARAMS["decimation_target"],
    texture_size: int = DEMO_PARAMS["texture_size"],
) -> dict[str, Any]:
    """Run the full clean-port pipeline on MPS and export a GLB. Returns the manifest dict."""
    configure_environment(vendor_root, sparse_attn_backend)
    sys.path.insert(0, str(vendor_root / "TRELLIS.2"))
    stubs = vendor_root / "stubs"
    if stubs.is_dir():
        sys.path.append(str(stubs))

    try:
        import flex_gemm  # noqa: F401
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

    import torch
    from PIL import Image
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    import o_voxel

    raw_image = Image.open(image_path)
    extrema = raw_image.getextrema()
    alpha_min = extrema[3][0] if raw_image.mode == "RGBA" else None
    has_alpha = alpha_is_transparent(raw_image.mode, alpha_min)
    if not has_alpha and not allow_rembg:
        raise SystemExit(
            f"{image_path} has no transparent alpha foreground. Loading the background remover "
            "(rembg/BRIA) would be required; pass --allow-rembg to permit it, or pre-mask the image."
        )
    load_rembg = not has_alpha

    pipeline_type = pipeline_type_for_resolution(resolution)
    ss = sampler_params("sparse_structure")
    shape = sampler_params("shape_slat")
    tex = sampler_params("tex_slat")

    started = time.time()
    print(f"Loading TRELLIS.2 pipeline (load_rembg={load_rembg})...", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", load_rembg=load_rembg
    )
    pipeline.to(torch.device("mps"))
    load_seconds = time.time() - started
    print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

    image = pipeline.preprocess_image(raw_image)

    # The demo seeds once before Stages 1->2->3; run() does torch.manual_seed(seed) internally.
    # Seed the MPS generator too so the single-seed behaviour matches on Apple Silicon.
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    run_started = time.time()
    outputs, latents = pipeline.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params=ss,
        shape_slat_sampler_params=shape,
        tex_slat_sampler_params=tex,
        pipeline_type=pipeline_type,
        return_latent=True,
    )
    run_seconds = time.time() - run_started
    print(f"pipeline.run() (stages 1-3) done in {run_seconds:.1f}s", flush=True)

    shape_slat, tex_slat, res = latents

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    if save_latents:
        latents_path = output_path.with_name(output_path.stem + "_latents.pt")
        torch.save(
            {
                "shape_slat_feats": shape_slat.feats.cpu(),
                "coords": shape_slat.coords.cpu(),
                "tex_slat_feats": tex_slat.feats.cpu(),
                "res": int(res),
                "pipeline_type": pipeline_type,
                "seed": seed,
                "images": [str(image_path)],
            },
            latents_path,
        )
        artifacts["latents"] = _artifact(latents_path)
        print(f"Latents cached: {latents_path}", flush=True)

    decode_started = time.time()
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    mesh.simplify(NVDIFFRAST_FACE_LIMIT)
    decode_seconds = time.time() - decode_started
    print(f"decode_latent done in {decode_seconds:.1f}s", flush=True)

    bake_started = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=AABB,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=DEMO_PARAMS["remesh"]["remesh"],
        remesh_band=DEMO_PARAMS["remesh"]["remesh_band"],
        remesh_project=DEMO_PARAMS["remesh"]["remesh_project"],
        use_tqdm=True,
    )
    glb.export(str(output_path), extension_webp=True)
    bake_seconds = time.time() - bake_started
    print(f"to_glb + export done in {bake_seconds:.1f}s -> {output_path}", flush=True)
    artifacts["glb"] = _artifact(output_path)

    timings = {
        "pipeline_load": load_seconds,
        "run_stages_1_3": run_seconds,
        "decode_latent": decode_seconds,
        "to_glb": bake_seconds,
        "total": time.time() - started,
    }
    effective_params = {**DEMO_PARAMS, "resolution": resolution,
                        "decimation_target": decimation_target,
                        "texture_size": texture_size}
    manifest = build_manifest(
        image=str(image_path),
        output=str(output_path),
        params=effective_params,
        pipeline_type=pipeline_type,
        seed=seed,
        timings=timings,
        artifacts=artifacts,
        load_rembg=load_rembg,
        sparse_attn_backend=sparse_attn_backend,
    )
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total: {timings['total']:.1f}s", flush=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, nargs="?", help="input image (transparent alpha foreground)")
    parser.add_argument("output", type=Path, nargs="?", help="GLB to write")
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR,
                        help="the clean trellis-space-mac build (with .venv + deps)")
    parser.add_argument("--resolution", choices=("512", "1024", "1536"),
                        default=DEMO_PARAMS["resolution"],
                        help="demo resolution / pipeline cascade")
    parser.add_argument("--seed", type=int, default=DEMO_PARAMS["seed"])
    parser.add_argument("--decimation-target", type=int, default=DEMO_PARAMS["decimation_target"],
                        help="final face/vertex budget the mesh is simplified DOWN to "
                             "(our app.py demo default 300000; the live HF demo may use 3000000)")
    parser.add_argument("--texture-size", type=int, default=DEMO_PARAMS["texture_size"])
    parser.add_argument("--sparse-attn-backend", default="sdpa", choices=("sdpa", "metal_flash"))
    parser.add_argument("--allow-rembg", action="store_true",
                        help="permit loading the background remover for a non-alpha input")
    parser.add_argument("--no-save-latents", dest="save_latents", action="store_false",
                        help="do not write <out>_latents.pt")
    parser.add_argument("--check", action="store_true",
                        help="verify the environment and exit (no model load, no sampling)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vendor_root = args.vendor_root.resolve()

    if args.check:
        return check_environment(vendor_root, args.sparse_attn_backend)

    if args.image is None or args.output is None:
        raise SystemExit("image and output are required (or pass --check)")
    if not args.image.is_file():
        raise SystemExit(f"missing input image: {args.image}")

    generate(
        args.image,
        args.output,
        vendor_root,
        seed=args.seed,
        sparse_attn_backend=args.sparse_attn_backend,
        allow_rembg=args.allow_rembg,
        save_latents=args.save_latents,
        resolution=args.resolution,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
