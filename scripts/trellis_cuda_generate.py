#!/usr/bin/env python3
"""Full image -> GLB generation through Microsoft's own TRELLIS.2 on an NVIDIA card.

The CUDA twin of ``trellis_space_generate.py``: the same pipeline calls and the same demo
settings (it imports ``DEMO_PARAMS`` and the sampling helper from that script, so the two
cannot drift), run on the official ``microsoft/TRELLIS.2`` checkout that
``bootstrap_trellis_cuda.py`` builds in ``vendor/trellis-cuda/``. The path is upstream's
``app.py``: ``pipeline.run()`` (sparse structure -> shape -> material) -> ``decode_latent``
-> ``mesh.simplify`` to nvdiffrast's limit -> ``o_voxel.postprocess.to_glb``.

Run it with that checkout's interpreter::

    vendor/trellis-cuda/.venv/bin/python scripts/trellis_cuda_generate.py in.png out.glb
    vendor/trellis-cuda/.venv/bin/python scripts/trellis_cuda_generate.py --check

Writes ``<out>.glb``, ``<out>.json`` (the run manifest, same shape as the Mac route's) and
``<out>.provenance.json`` (the licence record). ``--from-latents`` resumes a run whose
decode or bake failed.

**Licence guardrail.** Upstream loads BRIA RMBG-2.0 as its background remover. This script
refuses to run unless ``patch_trellis_cuda_no_bria.py`` has removed that load, and it also
makes the remover class raise if anything tries to build it anyway. Images without a real
cut-out are matted by our own remover (``image_to_3dlab/matte.py``) first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import patch_trellis_cuda_no_bria as bria_patch  # noqa: E402, RUF100
import patch_trellis_rocm as rocm_patch  # noqa: E402, RUF100
import trellis_route  # noqa: E402, RUF100
import trellis_space_generate as space  # noqa: E402, RUF100

from image_to_3dlab.provenance import (  # noqa: E402, RUF100
    LICENSES,
    MATTE_LICENSES,
    sha256_file,
)

#: This script's defaults are the CUDA route; the AMD twin passes ROCM through the same
#: functions, so the pipeline calls and the record shapes cannot drift between them.
ROUTE = trellis_route.CUDA
DEMO_PARAMS = space.DEMO_PARAMS
BUILT_MARKER_NAME = ROUTE.marker_name


# ----------------------------------------------------------------------------------------
# Pure, importable helpers: no torch at import time, so the unit tests need none.
# ----------------------------------------------------------------------------------------
def checkout(vendor_root: Path) -> Path:
    return vendor_root / "TRELLIS.2"


def configure_environment(environ: dict[str, str] | None = None,
                          route: trellis_route.Route = ROUTE) -> dict[str, str]:
    """Pin the attention backends BEFORE any TRELLIS import. Returns the environment it set."""
    return route.environ(environ)


def bria_refusal(vendor_root: Path) -> str | None:
    """Why this checkout may not run, or None. The licence guardrail, checked first."""
    if bria_patch.is_patched(checkout(vendor_root)):
        return None
    return (f"{checkout(vendor_root)} still loads BRIA RMBG-2.0, which this repo's "
            "generation pipeline must never load. Apply the guardrail first:\n"
            f"    python scripts/patch_trellis_cuda_no_bria.py --root {checkout(vendor_root)}")


def uses_aule(route: trellis_route.Route) -> bool:
    """Whether this route selects AuleAttention, which only the ROCm patches make exist."""
    return "aule" in route.backend_env.values()


def aule_patch_refusal(vendor_root: Path, route: trellis_route.Route) -> str | None:
    """Why an AMD checkout cannot attend, or None. A file check: seconds, no torch.

    Without the patches upstream ignores the unknown backend name and the run dies inside
    the first attention, minutes in, on a flash-attn import nobody asked for.
    """
    if not uses_aule(route) or rocm_patch.is_patched(vendor_root):
        return None
    return (f"{vendor_root} lacks the AMD attention patches (the aule backend). Apply them:\n"
            f"    python scripts/patch_trellis_rocm.py --root {vendor_root}")


AULE_MISSING = ("aule (AuleAttention) does not import in this venv, and it is the only "
                "attention backend on AMD. Re-run: python scripts/bootstrap_trellis_rocm.py "
                "--code-only")


class BriaRefused:
    """Stands in for upstream's remover class, so building it fails instead of fetching."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("image-to-3dlab: BRIA RMBG-2.0 must never be loaded "
                           f"(asked with {args or kwargs})")


def forbid_bria(rembg_module: Any) -> None:
    """Replace every remover class upstream exposes with `BriaRefused`.

    Belt and braces behind the source patch: if upstream ever adds a second place that
    builds the remover, this turns a silent download into an error.
    """
    package = rembg_module.__name__
    for name, value in list(vars(rembg_module).items()):
        if isinstance(value, type) and str(getattr(value, "__module__", "")).startswith(package):
            setattr(rembg_module, name, BriaRefused)


def verify_paths(vendor_root: Path, route: trellis_route.Route = ROUTE) -> list[str]:
    """Cheap filesystem checks that the checkout is present and finished building."""
    root = checkout(vendor_root)
    checks = {
        ".venv python": vendor_root / ".venv" / "bin" / "python",
        "TRELLIS.2 pipeline": root / "trellis2" / "pipelines" / "trellis2_image_to_3d.py",
        "o-voxel source": root / "o-voxel",
        f"completed build ({route.bootstrap})": vendor_root / route.marker_name,
    }
    return [f"missing {name}: {path}" for name, path in checks.items() if not path.exists()]


def matted_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}__matted.png")


def build_manifest(*, image: str, output: str, params: dict[str, Any], pipeline_type: str,
                   seed: int, timings: dict[str, float], artifacts: dict[str, Any],
                   matte_model: str | None,
                   route: trellis_route.Route = ROUTE) -> dict[str, Any]:
    """The run manifest: the Mac route's shape, with this route's facts in place of the Mac's."""
    manifest = space.build_manifest(
        image=image, output=output, params=params, pipeline_type=pipeline_type, seed=seed,
        timings=timings, artifacts=artifacts, load_rembg=False,
        sparse_attn_backend=route.backend_env["SPARSE_ATTN_BACKEND"],
    )
    manifest.update({
        "generator": route.generator,
        "port": route.port,
        "device": route.device,
        "attn_backend": route.backend_env["ATTN_BACKEND"],
        "sparse_conv_backend": route.backend_env["SPARSE_CONV_BACKEND"],
        "matte_model": matte_model,
    })
    return manifest


def build_provenance(*, image: Path, output: Path, parameters: dict[str, Any],
                     matte_model: str | None, backend_revision: str | None,
                     route: trellis_route.Route = ROUTE) -> dict[str, Any]:
    """The `.provenance.json` licence record, in `provenance.finalize_output`'s schema."""
    profile = LICENSES["trellis2"]
    components: list[dict[str, Any]] = []
    if matte_model:
        components.append({
            "component": f"rembg/{matte_model}",
            "purpose": "background removal",
            "license": MATTE_LICENSES.get(matte_model, "see rembg"),
            "commercial_status": "commercial-clear",
        })
    components += [
        {
            "component": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "purpose": "image conditioning",
            "license": "DINOv3 License",
            "commercial_status": "commercial-conditional",
            "license_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
        },
        {
            "component": "BRIA RMBG-2.0",
            "purpose": "background removal",
            "loaded": False,
            "commercial_status": "blocked",
            "note": "Explicitly disabled by scripts/patch_trellis_cuda_no_bria.py.",
        },
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(image), "sha256": sha256_file(image), "source": "user-provided"},
        "output": {"path": str(output), "sha256": sha256_file(output),
                   "classification": profile.classification},
        "model": {"backend": "trellis2", "route": route.provenance_route,
                  "parameters": parameters},
        "license": {"name": profile.license_name, "url": profile.license_url,
                    "conditions": list(profile.conditions)},
        "components": components,
        "software": {"backend_revision": backend_revision},
    }


def provenance_path(output: Path) -> Path:
    return output.with_suffix(".provenance.json")


def backend_revision(vendor_root: Path, route: trellis_route.Route = ROUTE) -> str | None:
    marker = vendor_root / route.marker_name
    try:
        return json.loads(marker.read_text()).get("commit")
    except (OSError, ValueError, AttributeError):
        return None


# ----------------------------------------------------------------------------------------
# The heavy path. torch/TRELLIS are imported lazily.
# ----------------------------------------------------------------------------------------
def prepare_image(image_path: Path, output: Path, *, force_matte: bool,
                  allow_uncut: bool,
                  route: trellis_route.Route = ROUTE) -> tuple[Path, str | None]:
    """Return the image to generate from and the remover that cut it (None if we did not).

    A real cut-out goes straight in. Anything else is cut out here, by our own remover,
    because with BRIA patched out upstream has nothing to cut it with.
    """
    import numpy as np
    from PIL import Image

    from image_to_3dlab.matte import cut_out, fallback_note, is_matted

    with Image.open(image_path) as opened:
        if force_matte or not is_matted(opened):
            cut, model = cut_out(opened.convert("RGB"))
            destination = matted_path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cut.save(destination)
            print(f"[{route.name}] matted with {model} -> {destination}", flush=True)
            if fallback_note(model):
                print(f"[{route.name}] note: {fallback_note(model)}", flush=True)
            return destination, model
        border = space.border_opaque_fraction(np.array(opened.convert("RGBA"))[..., 3])
    if border > space.BORDER_OPAQUE_LIMIT and not allow_uncut:
        raise SystemExit(space.uncut_foreground_message(image_path, border))
    return image_path, None


def load_pipeline(vendor_root: Path, route: trellis_route.Route = ROUTE):
    refusal = bria_refusal(vendor_root) or aule_patch_refusal(vendor_root, route)
    if refusal:
        raise SystemExit(refusal)
    configure_environment(route=route)
    sys.path.insert(0, str(checkout(vendor_root)))

    import torch
    from trellis2.pipelines import rembg
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see a CUDA device; check the driver and the venv")
    forbid_bria(rembg)
    model = route.model_path()
    print(f"Loading TRELLIS.2 pipeline ({route.provenance_route}, weights from {model}, "
          "BRIA disabled)...", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
    if pipeline.rembg_model is not None:
        raise SystemExit("refusing: the pipeline built a background remover despite the patch")
    pipeline.cuda()
    return pipeline


def check_environment(vendor_root: Path, route: trellis_route.Route = ROUTE) -> int:
    """Seconds, no model load: paths, the BRIA guardrail, the device, and the imports a run
    needs. Run this before a generation, not after one has failed."""
    problems = verify_paths(vendor_root, route)
    for refusal in (bria_refusal(vendor_root), aule_patch_refusal(vendor_root, route)):
        if refusal:
            problems.append(refusal)
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}", flush=True)
        return 1
    configure_environment(route=route)
    sys.path.insert(0, str(checkout(vendor_root)))
    import torch

    if not torch.cuda.is_available():
        print("  FAIL: torch cannot see a CUDA device", flush=True)
        return 1
    if uses_aule(route):
        try:
            import aule  # noqa: F401  the attention backend the patches select
        except ImportError:
            print(f"  FAIL: {AULE_MISSING}", flush=True)
            return 1
    import o_voxel
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    missing = [m for m in ("run", "decode_latent", "preprocess_image")
               if not hasattr(Trellis2ImageTo3DPipeline, m)]
    if missing or not hasattr(o_voxel.postprocess, "to_glb"):
        print(f"  FAIL: TRELLIS.2 surface missing: {missing or ['to_glb']}", flush=True)
        return 1
    print(f"  PASS: {torch.cuda.get_device_name(0)}, BRIA patched out, "
          "run/decode_latent/to_glb resolved", flush=True)
    return 0


def _decode_and_bake(pipeline, shape_slat, tex_slat, res, output: Path, *,
                     decimation_target: int, texture_size: int) -> dict[str, float]:
    import o_voxel
    import torch

    started = time.time()
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    mesh.simplify(space.NVDIFFRAST_FACE_LIMIT)  # nvdiffrast limit, as upstream's app.py
    decode_seconds = time.time() - started
    print(f"decode_latent done in {decode_seconds:.1f}s", flush=True)

    started = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=space.AABB,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=DEMO_PARAMS["remesh"]["remesh"],
        remesh_band=DEMO_PARAMS["remesh"]["remesh_band"],
        remesh_project=DEMO_PARAMS["remesh"]["remesh_project"],
        use_tqdm=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output), extension_webp=True)
    torch.cuda.empty_cache()
    bake_seconds = time.time() - started
    print(f"bake (to_glb + export) done in {bake_seconds:.1f}s -> {output}", flush=True)
    return {"decode_latent": decode_seconds, "bake": bake_seconds}


def _write_records(*, image: Path, output: Path, params: dict[str, Any], pipeline_type: str,
                   seed: int, timings: dict[str, float], artifacts: dict[str, Any],
                   matte_model: str | None, vendor_root: Path,
                   route: trellis_route.Route = ROUTE) -> None:
    manifest = build_manifest(image=str(image), output=str(output), params=params,
                              pipeline_type=pipeline_type, seed=seed, timings=timings,
                              artifacts=artifacts, matte_model=matte_model, route=route)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record = build_provenance(image=image, output=output, parameters=params,
                              matte_model=matte_model,
                              backend_revision=backend_revision(vendor_root, route),
                              route=route)
    provenance_path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {output.with_suffix('.json')}", flush=True)
    print(f"Provenance: {provenance_path(output)}", flush=True)


def generate(image_path: Path, output: Path, vendor_root: Path, *, seed: int,
             resolution: str, decimation_target: int, texture_size: int,
             save_latents: bool, force_matte: bool = False,
             allow_uncut: bool = False,
             route: trellis_route.Route = ROUTE) -> None:
    refusal = bria_refusal(vendor_root)
    if refusal:
        raise SystemExit(refusal)
    prepared, matte_model = prepare_image(image_path, output, force_matte=force_matte,
                                          allow_uncut=allow_uncut, route=route)
    pipeline_type = space.pipeline_type_for_resolution(resolution)

    started = time.time()
    pipeline = load_pipeline(vendor_root, route)
    load_seconds = time.time() - started
    print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

    import torch
    from PIL import Image

    with Image.open(prepared) as opened:
        image = pipeline.preprocess_image(opened.convert("RGBA"))

    run_started = time.time()
    shape_slat, tex_slat, res = space.sample_latents_without_decode(
        pipeline, image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params=space.sampler_params("sparse_structure"),
        shape_slat_sampler_params=space.sampler_params("shape_slat"),
        tex_slat_sampler_params=space.sampler_params("tex_slat"),
        pipeline_type=pipeline_type,
    )
    run_seconds = time.time() - run_started
    print(f"pipeline.run() sampling (stages 1-3) done in {run_seconds:.1f}s", flush=True)

    # Checkpoint before decode, same schema as the Mac route, so a failed decode resumes.
    output.parent.mkdir(parents=True, exist_ok=True)
    latents_path = output.with_name(output.stem + "_latents.pt")
    torch.save({"shape_slat_feats": shape_slat.feats.cpu(), "coords": shape_slat.coords.cpu(),
                "tex_slat_feats": tex_slat.feats.cpu(), "res": int(res),
                "pipeline_type": pipeline_type, "seed": seed, "images": [str(image_path)]},
               latents_path)

    stage_timings = _decode_and_bake(pipeline, shape_slat, tex_slat, res, output,
                                     decimation_target=decimation_target,
                                     texture_size=texture_size)
    artifacts: dict[str, Any] = {"glb": {"path": str(output), "sha256": sha256_file(output)}}
    if save_latents:
        artifacts["latents"] = {"path": str(latents_path), "sha256": sha256_file(latents_path)}
    elif output.is_file():
        latents_path.unlink(missing_ok=True)

    params = {**DEMO_PARAMS, "resolution": resolution, "decimation_target": decimation_target,
              "texture_size": texture_size}
    timings = {"pipeline_load": load_seconds, "run_stages_1_3": run_seconds, **stage_timings,
               "total": time.time() - started}
    _write_records(image=image_path, output=output, params=params,
                   pipeline_type=pipeline_type, seed=seed, timings=timings,
                   artifacts=artifacts, matte_model=matte_model, vendor_root=vendor_root,
                   route=route)
    print(f"Total: {timings['total']:.1f}s", flush=True)


def generate_from_latents(latents_path: Path, output: Path, vendor_root: Path, *,
                          decimation_target: int, texture_size: int,
                          route: trellis_route.Route = ROUTE) -> None:
    """Resume from `<out>_latents.pt`: skip sampling, decode and bake only."""
    started = time.time()
    pipeline = load_pipeline(vendor_root, route)
    import torch
    from trellis2.modules.sparse import SparseTensor

    bundle = torch.load(latents_path, map_location="cpu", weights_only=False)
    shape_slat = SparseTensor(feats=bundle["shape_slat_feats"].cuda(),
                              coords=bundle["coords"].cuda())
    tex_slat = shape_slat.replace(bundle["tex_slat_feats"].cuda())
    stage_timings = _decode_and_bake(pipeline, shape_slat, tex_slat, int(bundle["res"]), output,
                                     decimation_target=decimation_target,
                                     texture_size=texture_size)
    image = Path(str(bundle.get("images", [latents_path])[0]))
    params = {**DEMO_PARAMS, "decimation_target": decimation_target,
              "texture_size": texture_size, "resumed_from_latents": str(latents_path)}
    _write_records(image=image if image.is_file() else latents_path, output=output,
                   params=params,
                   pipeline_type=bundle.get("pipeline_type", "1024_cascade"),
                   seed=int(bundle.get("seed", -1)),
                   timings={**stage_timings, "total": time.time() - started},
                   artifacts={"glb": {"path": str(output), "sha256": sha256_file(output)}},
                   matte_model=None, vendor_root=vendor_root, route=route)


def parse_args(argv: list[str] | None = None,
               route: trellis_route.Route = ROUTE) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, nargs="?", help="input image")
    parser.add_argument("output", type=Path, nargs="?", help="GLB to write")
    parser.add_argument("--vendor-root", type=Path, default=route.vendor,
                        help=f"the checkout built by {route.bootstrap}")
    parser.add_argument("--resolution", choices=("512", "1024", "1536"),
                        default=DEMO_PARAMS["resolution"])
    parser.add_argument("--seed", type=int, default=DEMO_PARAMS["seed"])
    parser.add_argument("--decimation-target", type=int,
                        default=DEMO_PARAMS["decimation_target"])
    parser.add_argument("--texture-size", type=int, default=DEMO_PARAMS["texture_size"])
    parser.add_argument("--matte", action="store_true",
                        help="cut the subject out with our remover even if it looks matted")
    parser.add_argument("--allow-uncut", action="store_true",
                        help="permit an alpha image whose background was never cut out")
    parser.add_argument("--no-save-latents", dest="save_latents", action="store_false",
                        help="delete <out>_latents.pt once the GLB exists")
    parser.add_argument("--from-latents", type=Path, default=None,
                        help="resume from a cached *_latents.pt; give the GLB path positionally")
    parser.add_argument("--check", action="store_true",
                        help="verify the environment and exit (no model load)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, route: trellis_route.Route = ROUTE) -> int:
    args = parse_args(argv, route)
    vendor_root = args.vendor_root.resolve()
    if args.check:
        return check_environment(vendor_root, route)
    if args.from_latents is not None:
        out = args.output or args.image
        if out is None or not args.from_latents.is_file():
            raise SystemExit("give an existing --from-latents bundle and the output GLB path")
        generate_from_latents(args.from_latents, out, vendor_root,
                              decimation_target=args.decimation_target,
                              texture_size=args.texture_size, route=route)
        return 0
    if args.image is None or args.output is None:
        raise SystemExit("image and output are required (or pass --check)")
    if not args.image.is_file():
        raise SystemExit(f"missing input image: {args.image}")
    generate(args.image, args.output, vendor_root, seed=args.seed,
             resolution=args.resolution, decimation_target=args.decimation_target,
             texture_size=args.texture_size, save_latents=args.save_latents,
             force_matte=args.matte, allow_uncut=args.allow_uncut, route=route)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
