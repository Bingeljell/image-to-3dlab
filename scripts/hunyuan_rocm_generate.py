#!/usr/bin/env python3
"""Full image -> textured GLB through Tencent's own Hunyuan3D-2.1 on an AMD card.

Shape then paint, as upstream's ``demo.py`` does, on the checkout that
``bootstrap_hunyuan_rocm.py`` builds in ``vendor/hunyuan-rocm/``:

1. ``Hunyuan3DDiTFlowMatchingPipeline`` turns the picture into an untextured mesh.
2. The shape model is unloaded (shape ~10 GB and paint ~21 GB do not fit a 24 GB card
   together; upstream's demo keeps both loaded and needs ~29 GB).
3. ``Hunyuan3DPaintPipeline`` paints PBR textures onto it and writes the GLB.

Run it with that checkout's interpreter::

    vendor/hunyuan-rocm/.venv/bin/python scripts/hunyuan_rocm_generate.py in.png out.glb
    vendor/hunyuan-rocm/.venv/bin/python scripts/hunyuan_rocm_generate.py --check

Writes ``<out>.glb``, ``<out>.json`` (the run manifest) and ``<out>.provenance.json``
(the licence record). ``--shape-only`` stops after step 1.

**What differs from the NVIDIA run, and why.** Nothing in upstream needed porting: the
shape stage is plain PyTorch and the paint stage's one CUDA extension builds with hipcc.
Three run-time choices are ours.

- **No ``expandable_segments``.** The NVIDIA route sets it, and on ROCm it has produced
  silent NaNs on more than one stack. A run that finishes and writes nothing is much worse
  than one that is a little slower, so it is simply not set.
- **Paint settings are fitted to the card.** Upstream's defaults want about 21 GB, which a
  20 GB card does not have. ``fit_paint_settings`` walks down a ladder of views, texture
  resolution and attention chunk until it fits what is free. The numbers are starting
  points, not guarantees: lower them further if it still runs out.
- **Shape asks for ``enable_flashvdm``** where the card can afford it. It is pure PyTorch
  despite the name, and the default decoder queries every point of a 385^3 grid.

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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bootstrap_hunyuan_rocm as bootstrap  # noqa: E402, RUF100

from image_to_3dlab import gpu_memory  # noqa: E402, RUF100
from image_to_3dlab.provenance import (  # noqa: E402, RUF100
    LICENSES,
    MATTE_LICENSES,
    sha256_file,
)

DEFAULT_VENDOR = bootstrap.VENDOR
BUILT_MARKER_NAME = bootstrap.BUILT_MARKER.name
GB = 1024 ** 3

# Upstream's demo.py and pipeline defaults. Change them here, and only on purpose.
DEFAULTS: dict[str, Any] = {
    "seed": 1234,
    "steps": 50,
    "octree_resolution": 384,
    "guidance_scale": 5.0,
    "max_num_view": 6,
    # Upstream's default is 512. This is not a memory compromise: 512 bakes to an
    # all-black texture on this card. See VALID_PAINT_RESOLUTION.
    "paint_resolution": 256,
}
VALID_OCTREE = (256, 384, 512)
# Wider than the NVIDIA route's 6..9: a small card needs to be allowed to ask for fewer.
VALID_VIEWS = range(1, 10)
# 128 and 256 only, and not because upstream says so.
#
# Upstream offers 512 and 768. On this card those produce a texture that is uniformly
# black: the paint UNet's multiview output comes back with a maximum of 0.02 across the
# whole image at 512, where at 256 it is a normal picture with no non-finite values at all.
# The views are black before the bake, so the bake is innocent, and it is not the attention
# patch -- chunked and unchunked agree at both sizes.
#
# It is NOT an out-of-memory: the unchunked 512px run finishes paint in 43 s with 20 GB
# free. It is also not reliably reproducible -- one instrumented run gave a correct texture
# at 512 with the same seed, differing only in that it held a reference to the UNet output,
# which shifts PyTorch's attention-backend choice. The paint path runs float16 with a
# bfloat16 UNet and pins no attention backend, so this looks like a marginal numerical
# sensitivity at the larger latent rather than a hard limit. Unproven.
#
# A route that is right 9 times in 10 is worse than one that is right every time, so 256 is
# what this asks for until someone pins the actual cause.
VALID_PAINT_RESOLUTION = (128, 256)

# The attention chunk, in query tokens. Read by the patch, not by this script.
CHUNK_ENV = "HY3D_SDPA_QUERY_CHUNK"

# The largest texture this route asks for, in one place so the ladder, the CLI and the
# warning below cannot drift apart.
MAX_PAINT_RESOLUTION = max(VALID_PAINT_RESOLUTION)

# A baked sheet whose every pixel is under this is not a dark object, it is the failure
# above: measured 0.000 mean and 0.000 standard deviation. A real texture of a dark object
# still varies.
BLANK_TEXTURE_LEVEL = 0.02

# (minimum free GB, views, texture resolution, attention chunk or 0 for off), best first.
#
# Paint at upstream's defaults wants about 21 GB. Measured community figures for ROCm put
# 6 views at 512 near 10.5 GB with chunking and unchunked well past 20 GB, so the ladder
# trades views first (they multiply the batch), then resolution, and only then starts
# chunking, which costs a little speed for a lot of headroom.
#
# No rung asks for more than 256px: see VALID_PAINT_RESOLUTION for why that is the
# ceiling on this card rather than a memory question.
FIT_LADDER: tuple[tuple[int, int, int, int], ...] = (
    (24, 6, 256, 0),
    (20, 6, 256, 4096),
    (16, 4, 256, 2048),
    (12, 4, 256, 1024),
    (8, 2, 256, 512),
    (0, 1, 128, 512),
)


def texture_is_blank(image_path: Path, level: float = BLANK_TEXTURE_LEVEL) -> bool:
    """Whether a baked texture is the all-black failure rather than a dark picture.

    Measured mean and standard deviation both come out as 0.000 for the failure, while a
    real texture of a dark object still varies: the threshold is on the spread, not the
    brightness, so a genuinely black model is not mistaken for this.
    """
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
    return float(pixels.std()) <= level and float(pixels.max()) <= level


# ----------------------------------------------------------------------------------------
# Pure, importable helpers: no torch at import time, so the unit tests need none.
# ----------------------------------------------------------------------------------------
def checkout(vendor_root: Path) -> Path:
    return vendor_root / bootstrap.CHECKOUT.name


def fit_paint_settings(free_bytes: int | None,
                       wanted_views: int | None = None,
                       wanted_resolution: int | None = None
                       ) -> dict[str, Any]:
    """Paint settings that fit the card, and the attention chunk to run them with.

    `None` free memory means *unknown*, not *none*: a box whose tools are missing still
    generates, so the ladder starts one step below the best rather than refusing. An
    explicit --max-num-view or --paint-resolution wins over the ladder, because a caller
    who asked for one meant it.
    """
    if free_bytes is None:
        rung = FIT_LADDER[1]
    else:
        gigabytes = free_bytes // GB
        rung = next(row for row in FIT_LADDER if gigabytes >= row[0])
    _, views, resolution, chunk = rung
    return {
        "max_num_view": wanted_views or views,
        "paint_resolution": wanted_resolution or resolution,
        "sdpa_query_chunk": chunk,
    }


def plan_paint(settings: dict[str, Any], free_bytes: int | None
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run's settings, checked, with paint fitted to the card; and the fit on its own.

    What the caller asked for is read *before* defaults are filled in. Read after, the
    defaults look like a request, win over the ladder, and only the chunk is ever fitted.
    """
    fitted = fit_paint_settings(free_bytes, settings.get("max_num_view"),
                                settings.get("paint_resolution"))
    return {**validate(settings), **fitted}, fitted


def _torch_free_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None
    return None


def measured_free_bytes(torch_free: Callable[[], int | None] = _torch_free_bytes,
                        smi_free: Callable[[], int | None] = gpu_memory.free_vram_bytes
                        ) -> int | None:
    """Free GPU memory as HIP sees it, falling back to the vendor tool.

    Torch first, because it answers for the memory a run can actually allocate. On an APU
    (Strix Halo) "VRAM" is a 1 GB slice and the real pool is shared system memory, and a
    ROCm with only `amd-smi` has no `rocm-smi` to ask at all.
    """
    found = torch_free()
    return found if found is not None else smi_free()


def describe_fit(settings: dict[str, Any], free_bytes: int | None) -> str:
    """One line for the log, saying what was chosen and why, so a slow run is explicable."""
    chunk = settings["sdpa_query_chunk"]
    free = "unknown" if free_bytes is None else gpu_memory.human_bytes(free_bytes)
    return (f"paint: {settings['max_num_view']} views at {settings['paint_resolution']}px, "
            f"attention chunk {chunk or 'off'}, free VRAM {free}")


def configure_environment(vendor_root: Path, chunk: int = 0,
                          environ: dict[str, str] | None = None) -> dict[str, str]:
    """Point upstream's shape loader at the weights the bootstrap fetched. Before import.

    Deliberately does not set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments`` the way the
    NVIDIA route does: on ROCm it has produced silent NaNs, and a run that writes nothing
    is a far worse outcome than a few percent of speed.
    """
    environ = os.environ if environ is None else environ
    environ["HY3DGEN_MODELS"] = str(vendor_root / bootstrap.MODELS.name)
    if chunk:
        environ[CHUNK_ENV] = str(chunk)
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
        "completed build (scripts/bootstrap_hunyuan_rocm.py)": vendor_root / BUILT_MARKER_NAME,
    }
    return [f"missing {name}: {path}" for name, path in checks.items() if not path.exists()]


def realesrgan_path(vendor_root: Path) -> Path:
    return vendor_root / bootstrap.REALESRGAN.relative_to(bootstrap.VENDOR)


def hf_hub_dir(environ: dict[str, str] | None = None) -> Path:
    """The Hugging Face cache upstream's paint stage reads, honouring the usual variables."""
    environ = os.environ if environ is None else environ
    if environ.get("HF_HUB_CACHE"):
        return Path(environ["HF_HUB_CACHE"])
    home = environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _cached(hub: Path, repo: str, inside: str) -> bool:
    """Whether `inside` (a file or folder) is in any snapshot of `repo` in the cache."""
    snapshots = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
    return any(snapshots.glob(f"*/{inside}"))


def missing_weights(vendor_root: Path, hub: Path | None = None,
                    shape_only: bool = False) -> list[str]:
    """Weights the run would otherwise fetch on its own, 8-15 GB of them, unannounced.

    Upstream downloads whatever it cannot find. That is the one thing this repo never lets
    happen without a choice (CLAUDE.md), so a gap is a refusal that names the bootstrap.
    """
    hub = hf_hub_dir() if hub is None else hub
    shape = vendor_root / bootstrap.MODELS.name / bootstrap.HUNYUAN_REPO
    checks = {
        "Hunyuan3D-2.1 shape model": (shape / "hunyuan3d-dit-v2-1").is_dir(),
        "Hunyuan3D-2.1 shape VAE": (shape / "hunyuan3d-vae-v2-1").is_dir(),
    }
    if not shape_only:
        checks.update({
            "Hunyuan3D-2.1 PBR paint": _cached(hub, bootstrap.HUNYUAN_REPO,
                                                "hunyuan3d-paintpbr-v2-1"),
            "DINOv2-giant image encoder": _cached(hub, bootstrap.DINO_REPO,
                                                  "model.safetensors"),
            "RealESRGAN upscaler": realesrgan_path(vendor_root).is_file(),
        })
    return [f"missing weights: {name}" for name, present in checks.items() if not present]


WEIGHTS_HINT = ("Fetch them on purpose (about 19.5 GB, Tencent Hunyuan Community License, "
                "not licensed in the EU, UK or South Korea):\n"
                "  python scripts/bootstrap_hunyuan_rocm.py --weights-only")


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
        raise ValueError("max_num_view must be between 1 and 9")
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
                   matte_model: str | None, fitted: dict[str, Any],
                   free_bytes: int | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "hunyuan_rocm_generate.py",
        "backend": "hunyuan3d-2.1",
        "port": "Tencent-Hunyuan/Hunyuan3D-2.1 (official, ROCm)",
        "device": "hip",
        "image": image,
        "output": output,
        "params": params,
        "fitted": fitted,
        "free_vram_bytes": free_bytes,
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
        "created_at": datetime.now(UTC).isoformat(),
        "input": {"path": str(image), "sha256": sha256_file(image), "source": "user-provided"},
        "output": {"path": str(output), "sha256": sha256_file(output),
                   "classification": profile.classification},
        "model": {"backend": "hunyuan3d-2.1", "route": "amd-rocm", "parameters": parameters},
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
    print(f"[hunyuan-rocm] matted with {model} -> {destination}", flush=True)
    if fallback_note(model):
        print(f"[hunyuan-rocm] note: {fallback_note(model)}", flush=True)
    return destination, model


def _enter_upstream(vendor_root: Path, chunk: int = 0) -> None:
    configure_environment(vendor_root, chunk)
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
    # Emptying the cache is spelled the same under ROCm: torch keeps the CUDA namespace.
    torch.cuda.empty_cache()


def check_environment(vendor_root: Path) -> int:
    """Seconds, no model load: paths, the GPU, and the imports a run needs."""
    problems = verify_paths(vendor_root) + missing_weights(vendor_root)
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}", flush=True)
        if any(problem.startswith("missing weights") for problem in problems):
            print(WEIGHTS_HINT, flush=True)
        return 1
    _enter_upstream(vendor_root)
    import torch

    if not torch.cuda.is_available():
        print("  FAIL: torch cannot see a GPU (is this a ROCm build of PyTorch?)", flush=True)
        return 1
    if not getattr(torch.version, "hip", None):
        # The exact failure this route exists to prevent: a CUDA torch on an AMD card
        # reports no device, and the install bug that produced one was real.
        print("  FAIL: this torch is a CUDA build, not a ROCm one. Re-run "
              "scripts/bootstrap_hunyuan_rocm.py --code-only", flush=True)
        return 1
    import custom_rasterizer  # noqa: F401  compiled by the bootstrap with hipcc
    from DifferentiableRenderer import mesh_inpaint_processor  # noqa: F401
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    from textureGenPipeline import (  # noqa: F401
        Hunyuan3DPaintConfig,
        Hunyuan3DPaintPipeline,
    )

    free = measured_free_bytes()
    print(f"  PASS: {torch.cuda.get_device_name(0)} (ROCm "
          f"{torch.version.hip}), {gpu_memory.human_bytes(free) if free else 'VRAM unknown'}, "
          "shape + paint pipelines and both compiled extensions import", flush=True)
    print(f"  fit:  {describe_fit(fit_paint_settings(free), free)}", flush=True)
    return 0


# Said before each slow load: on a real run the shape model took minutes to come off disk
# with nothing on screen, and the run looked stuck.
LOADING_SHAPE = "loading the shape model (about 7 GB from disk, 1-3 minutes)..."
LOADING_PAINT = "loading the paint models (about 2 minutes)..."


def generate_shape(image: Path, mesh_path: Path, settings: dict[str, Any],
                   flash_vdm: bool = False) -> float:
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from PIL import Image

    started = time.time()
    print(LOADING_SHAPE, flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(bootstrap.HUNYUAN_REPO)
    if flash_vdm:
        # Pure PyTorch despite the name: it asks the decoder for only the promising
        # points instead of every point of the grid, which is the difference between a
        # few minutes and most of half an hour.
        try:
            pipeline.enable_flashvdm()
            print("flashvdm enabled", flush=True)
        except Exception as error:
            print(f"flashvdm unavailable ({error}); continuing without it", flush=True)
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
    _refuse_a_blank_texture(work_dir(output), settings)
    painted.replace(output)
    seconds = time.time() - started
    print(f"paint stage done in {seconds:.1f}s -> {output}", flush=True)
    del pipeline
    _free_gpu()
    return seconds


def _refuse_a_blank_texture(where: Path, settings: dict[str, Any]) -> None:
    """Stop rather than ship a GLB whose texture is uniformly black.

    Found by running this on a real card: at 512px the paint stage returns without an
    error and writes a perfectly valid GLB whose base colour is entirely black. That reads
    in the viewer as a finished model with no texture, and nothing says why.
    """
    baked = where / "textured_mesh.jpg"
    if not baked.is_file() or not texture_is_blank(baked):
        return
    asked = settings["paint_resolution"]
    lower = MAX_PAINT_RESOLUTION if asked > MAX_PAINT_RESOLUTION else min(VALID_PAINT_RESOLUTION)
    raise SystemExit(
        f"The paint stage produced an all-black texture at {asked}px, which is the known "
        f"failure at larger sizes on this card -- not a problem with the picture. Re-run "
        f"with --paint-resolution {lower}. The GLB has not been written.")


def generate(image_path: Path, output: Path, vendor_root: Path, settings: dict[str, Any], *,
             shape_only: bool = False, force_matte: bool = False,
             free_bytes: int | None = None, flash_vdm: bool = False) -> None:
    settings, fitted = plan_paint(settings, free_bytes)
    problems = verify_paths(vendor_root)
    if problems:
        raise SystemExit("Hunyuan3D-2.1 is not installed:\n  " + "\n  ".join(problems) +
                         "\nRun: python scripts/bootstrap_hunyuan_rocm.py")
    absent = missing_weights(vendor_root, shape_only=shape_only)
    if absent:
        raise SystemExit("Refusing to let upstream download weights unannounced:\n  " +
                         "\n  ".join(absent) + "\n" + WEIGHTS_HINT)
    print(describe_fit(fitted, free_bytes), flush=True)
    prepared, matte_model = prepare_image(image_path, output, force_matte=force_matte)
    _enter_upstream(vendor_root, fitted["sdpa_query_chunk"])
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see a GPU; check that this venv has the ROCm build")

    started = time.time()
    mesh_path = output if shape_only else work_dir(output) / "shape.glb"
    timings = {"shape": generate_shape(prepared, mesh_path, settings, flash_vdm=flash_vdm)}
    if not shape_only:
        timings["paint"] = generate_paint(prepared, mesh_path, output, settings, vendor_root)
    timings["total"] = time.time() - started

    artifacts = {"glb": {"path": str(output), "sha256": sha256_file(output)}}
    manifest = build_manifest(image=str(image_path), output=str(output), params=settings,
                              shape_only=shape_only, timings=timings, artifacts=artifacts,
                              matte_model=matte_model, fitted=fitted, free_bytes=free_bytes)
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
                        help="the checkout built by bootstrap_hunyuan_rocm.py")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--octree-resolution", type=int, choices=VALID_OCTREE,
                        default=DEFAULTS["octree_resolution"])
    parser.add_argument("--guidance-scale", type=float, default=DEFAULTS["guidance_scale"])
    parser.add_argument("--max-num-view", type=int, choices=list(VALID_VIEWS), default=None,
                        help="views to render; by default fitted to the card's free VRAM")
    parser.add_argument("--paint-resolution", type=int, choices=list(VALID_PAINT_RESOLUTION),
                        default=None, help="texture resolution; by default fitted to VRAM")
    parser.add_argument("--shape-only", action="store_true",
                        help="stop after the untextured mesh")
    parser.add_argument("--matte", action="store_true",
                        help="cut the subject out with our remover even if it looks matted")
    parser.add_argument("--flash-vdm", action="store_true",
                        help="ask the shape decoder only for promising points (much faster)")
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
    settings: dict[str, Any] = {"seed": args.seed, "steps": args.steps,
                                "octree_resolution": args.octree_resolution,
                                "guidance_scale": args.guidance_scale}
    if args.max_num_view is not None:
        settings["max_num_view"] = args.max_num_view
    if args.paint_resolution is not None:
        settings["paint_resolution"] = args.paint_resolution
    generate(args.image, args.output, vendor_root, settings, shape_only=args.shape_only,
             force_matte=args.matte, free_bytes=measured_free_bytes(),
             flash_vdm=args.flash_vdm)
    return 0


def finish(code: int) -> None:
    """Exit with `code` once output is flushed, skipping interpreter teardown.

    A finished run (GLB, manifest and provenance written) segfaulted during Python's
    shutdown as bpy and open3d tore down, and the viewer then read the run as failed.
    Nothing is left to clean up by then, so skip the teardown.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    finish(main())