#!/usr/bin/env python3
"""Full image -> textured GLB through Tencent's own Hunyuan3D-2.1 on an NVIDIA card.

Shape then paint, as upstream's ``demo.py`` does, on the checkout that
``bootstrap_hunyuan_cuda.py`` builds in ``vendor/hunyuan-cuda/``:

1. ``Hunyuan3DDiTFlowMatchingPipeline`` turns the picture into an untextured mesh.
2. The shape model is unloaded (shape ~10 GB and paint ~21 GB do not fit a 24 GB card
   together; upstream's demo keeps both loaded and needs ~29 GB).
3. ``Hunyuan3DPaintPipeline`` paints PBR textures onto it and writes the GLB.

Run it with that checkout's interpreter::

    vendor/hunyuan-cuda/.venv/bin/python scripts/hunyuan_cuda_generate.py in.png out.glb
    vendor/hunyuan-cuda/.venv/bin/python scripts/hunyuan_cuda_generate.py --check

Writes ``<out>.glb``, ``<out>.json`` (the run manifest) and ``<out>.provenance.json``
(the licence record). ``--shape-only`` stops after step 1.

Images without a real cut-out are matted by our own remover (``image_to_3dlab/matte.py``)
first. Upstream's demo converts every image to RGBA before asking whether it is RGB, so
its own remover never runs and a photo's background becomes geometry.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bootstrap_hunyuan_cuda as bootstrap  # noqa: E402, RUF100

from image_to_3dlab.provenance import (  # noqa: E402, RUF100
    LICENSES,
    MATTE_LICENSES,
    sha256_file,
)

DEFAULT_VENDOR = bootstrap.VENDOR
BUILT_MARKER_NAME = bootstrap.BUILT_MARKER.name

# Upstream's demo.py and pipeline defaults. Change them here, and only on purpose.
DEFAULTS: dict[str, Any] = {
    "seed": 1234,
    "steps": 50,
    "octree_resolution": 384,
    "guidance_scale": 5.0,
    "max_num_view": 6,
    "paint_resolution": 512,
}
VALID_OCTREE = (256, 384, 512)
VALID_VIEWS = range(6, 10)
VALID_PAINT_RESOLUTION = (512, 768)


# ----------------------------------------------------------------------------------------
# Pure, importable helpers: no torch at import time, so the unit tests need none.
# ----------------------------------------------------------------------------------------
def checkout(vendor_root: Path) -> Path:
    return vendor_root / bootstrap.CHECKOUT.name


def configure_environment(vendor_root: Path,
                          environ: dict[str, str] | None = None) -> dict[str, str]:
    """Point upstream's shape loader at the weights the bootstrap fetched. Before import."""
    environ = os.environ if environ is None else environ
    # the installed route follows the chosen models folder; any other checkout keeps its own
    models = bootstrap.MODELS if vendor_root == bootstrap.VENDOR else vendor_root / "models"
    environ["HY3DGEN_MODELS"] = str(models)
    environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environ


def import_paths(vendor_root: Path) -> list[Path]:
    """What upstream's demo.py puts on sys.path: the checkout, hy3dshape, hy3dpaint."""
    root = checkout(vendor_root)
    return [root, root / "hy3dshape", root / "hy3dpaint"]


def verify_paths(vendor_root: Path) -> list[str]:
    """Cheap filesystem checks that the checkout is present and finished building."""
    root = checkout(vendor_root)
    checks = {
        ".venv python": vendor_root / ".venv" / "bin" / "python",
        "shape pipeline": root / "hy3dshape" / "hy3dshape" / "pipelines.py",
        "paint pipeline": root / "hy3dpaint" / "textureGenPipeline.py",
        "paint config": root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml",
        "completed build (scripts/bootstrap_hunyuan_cuda.py)": vendor_root / BUILT_MARKER_NAME,
    }
    return [f"missing {name}: {path}" for name, path in checks.items() if not path.exists()]


def realesrgan_path(vendor_root: Path) -> Path:
    return vendor_root / bootstrap.REALESRGAN.relative_to(bootstrap.VENDOR)


def paint_config_overrides(vendor_root: Path) -> dict[str, str]:
    """Absolute paths for the fields upstream's config leaves relative to its own cwd."""
    root = checkout(vendor_root)
    return {
        "realesrgan_ckpt_path": str(realesrgan_path(vendor_root)),
        "multiview_cfg_path": str(root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"),
        "custom_pipeline": str(root / "hy3dpaint" / "hunyuanpaintpbr"),
    }


def validate(settings: dict[str, Any]) -> dict[str, Any]:
    """Defaults filled in, and every value checked, before a model loads."""
    merged = {**DEFAULTS, **settings}
    if merged["octree_resolution"] not in VALID_OCTREE:
        raise ValueError(f"octree_resolution must be one of {VALID_OCTREE}")
    if merged["max_num_view"] not in VALID_VIEWS:
        raise ValueError("max_num_view must be 6 to 9")
    if merged["paint_resolution"] not in VALID_PAINT_RESOLUTION:
        raise ValueError(f"paint_resolution must be one of {VALID_PAINT_RESOLUTION}")
    if merged["steps"] < 1:
        raise ValueError("steps must be at least 1")
    return merged


def work_dir(output: Path) -> Path:
    """Where the intermediate meshes go. Upstream's paint writes beside its input mesh."""
    return output.with_name(f"{output.stem}__hunyuan")


def matted_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}__matted.png")


def build_manifest(*, image: str, output: str, params: dict[str, Any], shape_only: bool,
                   timings: dict[str, float], artifacts: dict[str, Any],
                   matte_model: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "hunyuan_cuda_generate.py",
        "backend": "hunyuan3d-2.1",
        "port": "Tencent-Hunyuan/Hunyuan3D-2.1 (official, CUDA)",
        "device": "cuda",
        "image": image,
        "output": output,
        "params": params,
        "shape_only": shape_only,
        "matte_model": matte_model,
        "timings": timings,
        "artifacts": artifacts,
    }


def build_provenance(*, image: Path, output: Path, parameters: dict[str, Any],
                     matte_model: str | None, shape_only: bool,
                     backend_revision: str | None) -> dict[str, Any]:
    """The `.provenance.json` licence record, in `provenance.finalize_output`'s schema."""
    profile = LICENSES["hunyuan-comfyui"]  # the Hunyuan3D-2.1 licence profile
    components: list[dict[str, Any]] = []
    if matte_model:
        components.append({
            "component": f"rembg/{matte_model}",
            "purpose": "background removal",
            "license": MATTE_LICENSES.get(matte_model, "see rembg"),
            "commercial_status": "commercial-clear",
        })
    if not shape_only:
        components += [
            {"component": "facebook/dinov2-giant", "purpose": "paint image conditioning",
             "license": "Apache-2.0", "commercial_status": "commercial-clear"},
            {"component": "RealESRGAN_x4plus", "purpose": "texture upscaling",
             "license": "BSD-3-Clause", "commercial_status": "commercial-clear"},
        ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(image), "sha256": sha256_file(image), "source": "user-provided"},
        "output": {"path": str(output), "sha256": sha256_file(output),
                   "classification": profile.classification},
        "model": {"backend": "hunyuan3d-2.1", "route": "nvidia-cuda", "parameters": parameters},
        "license": {"name": profile.license_name, "url": profile.license_url,
                    "conditions": list(profile.conditions)},
        "components": components,
        "software": {"backend_revision": backend_revision},
    }


def provenance_path(output: Path) -> Path:
    return output.with_suffix(".provenance.json")


def backend_revision(vendor_root: Path) -> str | None:
    try:
        return json.loads((vendor_root / BUILT_MARKER_NAME).read_text()).get("commit")
    except (OSError, ValueError, AttributeError):
        return None


# ----------------------------------------------------------------------------------------
# The heavy path. torch and upstream are imported lazily.
# ----------------------------------------------------------------------------------------
def prepare_image(image_path: Path, output: Path, *, force_matte: bool) -> tuple[Path, str | None]:
    """The image to generate from, and the remover that cut it (None if we did not)."""
    from PIL import Image

    from image_to_3dlab.matte import cut_out, fallback_note, is_matted

    with Image.open(image_path) as opened:
        if not force_matte and is_matted(opened):
            return image_path, None
        cut, model = cut_out(opened.convert("RGB"))
    destination = matted_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cut.save(destination)
    print(f"[hunyuan-cuda] matted with {model} -> {destination}", flush=True)
    if fallback_note(model):
        print(f"[hunyuan-cuda] note: {fallback_note(model)}", flush=True)
    return destination, model


def _enter_upstream(vendor_root: Path) -> None:
    configure_environment(vendor_root)
    for path in reversed(import_paths(vendor_root)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:  # upstream's own shim for basicsr on newer torchvision; optional there too
        from torchvision_fix import apply_fix
        apply_fix()
    except ImportError:
        pass


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def check_environment(vendor_root: Path) -> int:
    """Seconds, no model load: paths, CUDA, and the imports a run needs."""
    problems = verify_paths(vendor_root)
    if not realesrgan_path(vendor_root).is_file():
        problems.append(f"missing RealESRGAN weights: {realesrgan_path(vendor_root)}")
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}", flush=True)
        return 1
    _enter_upstream(vendor_root)
    import torch

    if not torch.cuda.is_available():
        print("  FAIL: torch cannot see a CUDA device", flush=True)
        return 1
    import custom_rasterizer  # noqa: F401  compiled by the bootstrap
    from DifferentiableRenderer import mesh_inpaint_processor  # noqa: F401
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    from textureGenPipeline import (  # noqa: F401
        Hunyuan3DPaintConfig,
        Hunyuan3DPaintPipeline,
    )

    print(f"  PASS: {torch.cuda.get_device_name(0)}, shape + paint pipelines and both "
          "compiled extensions import", flush=True)
    return 0


# Said before each slow load: on a real pod the shape model took ~2.5 min to come off disk
# with nothing on screen, and the run looked stuck.
LOADING_SHAPE = "loading the shape model (about 7 GB from disk, 1-3 minutes)..."
LOADING_PAINT = "loading the paint models (about 2 minutes)..."


def generate_shape(image: Path, mesh_path: Path, settings: dict[str, Any]) -> float:
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from PIL import Image

    started = time.time()
    print(LOADING_SHAPE, flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(bootstrap.HUNYUAN_REPO)
    print(f"loaded shape pipeline in {time.time() - started:.1f}s", flush=True)
    with Image.open(image) as opened:
        picture = opened.convert("RGBA")
    mesh = pipeline(
        image=picture,
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance_scale"],
        octree_resolution=settings["octree_resolution"],
        generator=torch.manual_seed(settings["seed"]),
    )[0]
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(mesh_path))
    seconds = time.time() - started
    print(f"shape generated in {seconds:.1f}s -> {mesh_path}", flush=True)
    del pipeline, mesh
    _free_gpu()
    return seconds


def generate_paint(image: Path, mesh_path: Path, output: Path, settings: dict[str, Any],
                   vendor_root: Path) -> float:
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    started = time.time()
    config = Hunyuan3DPaintConfig(settings["max_num_view"], settings["paint_resolution"])
    for name, value in paint_config_overrides(vendor_root).items():
        setattr(config, name, value)
    print(LOADING_PAINT, flush=True)
    pipeline = Hunyuan3DPaintPipeline(config)
    print(f"mesh loaded; paint models ready in {time.time() - started:.1f}s", flush=True)
    obj = work_dir(output) / "textured_mesh.obj"
    pipeline(mesh_path=str(mesh_path), image_path=str(image), output_mesh_path=str(obj))
    painted = obj.with_suffix(".glb")
    if not painted.is_file():
        raise SystemExit(f"paint finished but wrote no GLB at {painted}")
    output.parent.mkdir(parents=True, exist_ok=True)
    painted.replace(output)
    seconds = time.time() - started
    print(f"paint stage done in {seconds:.1f}s -> {output}", flush=True)
    del pipeline
    _free_gpu()
    return seconds


def generate(image_path: Path, output: Path, vendor_root: Path, settings: dict[str, Any], *,
             shape_only: bool = False, force_matte: bool = False) -> None:
    settings = validate(settings)
    problems = verify_paths(vendor_root)
    if problems:
        raise SystemExit("Hunyuan3D-2.1 is not installed:\n  " + "\n  ".join(problems) +
                         "\nRun: python scripts/bootstrap_hunyuan_cuda.py")
    prepared, matte_model = prepare_image(image_path, output, force_matte=force_matte)
    _enter_upstream(vendor_root)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see a CUDA device; check the driver and the venv")

    started = time.time()
    mesh_path = output if shape_only else work_dir(output) / "shape.glb"
    timings = {"shape": generate_shape(prepared, mesh_path, settings)}
    if not shape_only:
        timings["paint"] = generate_paint(prepared, mesh_path, output, settings, vendor_root)
    timings["total"] = time.time() - started

    artifacts = {"glb": {"path": str(output), "sha256": sha256_file(output)}}
    manifest = build_manifest(image=str(image_path), output=str(output), params=settings,
                              shape_only=shape_only, timings=timings, artifacts=artifacts,
                              matte_model=matte_model)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record = build_provenance(image=image_path, output=output, parameters=settings,
                              matte_model=matte_model, shape_only=shape_only,
                              backend_revision=backend_revision(vendor_root))
    provenance_path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {output.with_suffix('.json')}", flush=True)
    print(f"Provenance: {provenance_path(output)}", flush=True)
    print(f"DONE in {timings['total']:.1f}s -> {output}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, nargs="?", help="input image")
    parser.add_argument("output", type=Path, nargs="?", help="GLB to write")
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR,
                        help="the checkout built by bootstrap_hunyuan_cuda.py")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--octree-resolution", type=int, choices=VALID_OCTREE,
                        default=DEFAULTS["octree_resolution"])
    parser.add_argument("--guidance-scale", type=float, default=DEFAULTS["guidance_scale"])
    parser.add_argument("--max-num-view", type=int, choices=list(VALID_VIEWS),
                        default=DEFAULTS["max_num_view"])
    parser.add_argument("--paint-resolution", type=int, choices=VALID_PAINT_RESOLUTION,
                        default=DEFAULTS["paint_resolution"])
    parser.add_argument("--shape-only", action="store_true",
                        help="stop after the untextured mesh")
    parser.add_argument("--matte", action="store_true",
                        help="cut the subject out with our remover even if it looks matted")
    parser.add_argument("--check", action="store_true",
                        help="verify the environment and exit (no model load)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vendor_root = args.vendor_root.resolve()
    if args.check:
        return check_environment(vendor_root)
    if args.image is None or args.output is None:
        raise SystemExit("image and output are required (or pass --check)")
    if not args.image.is_file():
        raise SystemExit(f"missing input image: {args.image}")
    if args.output.suffix.lower() != ".glb":
        raise SystemExit("the output must be a .glb path")
    settings = {"seed": args.seed, "steps": args.steps,
                "octree_resolution": args.octree_resolution,
                "guidance_scale": args.guidance_scale, "max_num_view": args.max_num_view,
                "paint_resolution": args.paint_resolution}
    generate(args.image, args.output, vendor_root, settings, shape_only=args.shape_only,
             force_matte=args.matte)
    return 0


def finish(code: int) -> None:
    """Exit with `code` once output is flushed, skipping interpreter teardown.

    A finished run (GLB, manifest and provenance written) segfaulted during Python's
    shutdown on a real NVIDIA pod, as bpy and open3d tore down, and the viewer then read
    the run as failed. Nothing is left to clean up by then, so skip the teardown.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    finish(main())
