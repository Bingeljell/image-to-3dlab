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
``trellis_rebake.py`` so Stage-3 resamples and rebakes need no re-sampling), ``<out>_decode.pt``
(the decoded mesh, so ``--from-decode`` re-bakes without re-decoding or reloading the model) and
a ``<out>.json`` manifest with the exact params and per-stage timings.

MPS bake path: app.py pre-simplifies the decoded mesh to nvdiffrast's 2**24 index cap before
``to_glb``, and the demo never meets the ~20M-face raw decode. cumesh's Metal simplify is not
robust at that scale on MPS (the decode->GLB blocker). This script instead CPU pre-caps with
``fast_simplification`` to ``--pre-cap`` (4,000,000 -- the old port's proven threshold for the
Metal to_glb/mtlbvh path) and hands ``to_glb`` CPU tensors, which is what the metal backend
expects internally.

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

# app.py pre-simplifies the decoded mesh to nvdiffrast's 2**24 index cap before to_glb. On MPS
# that call is both pointless (no nvdiffrast) and crashes: cumesh's Metal simplify_step is not
# robust on ~20M-face meshes (off-by-one AcceleratorError). We skip it and CPU pre-cap instead.
NVDIFFRAST_FACE_LIMIT = 16_777_216  # informational only; deliberately not called on MPS
PRE_CAP_FACES_DEFAULT = 4_000_000  # old port's proven-safe budget for the Metal to_glb path
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


def valid_face_mask(faces, num_vertices: int):
    """Per-face boolean mask: True where every vertex index is in [0, num_vertices).

    Some MPS decodes emit a degenerate face carrying a -1 (or otherwise out-of-range) index;
    cumesh's Metal ``simplify_step`` rejects it ("vertex index out of range") and aborts the whole
    bake. Dropping those faces before simplify is the fix. Accepts anything numpy views as a 2D int
    array -- a list of triples, a numpy array, or a CPU tensor's ``.numpy()``.
    """
    import numpy as np

    arr = np.asarray(faces)
    return ((arr >= 0) & (arr < num_vertices)).all(axis=1)


def precap_ratio(num_faces: int, pre_cap: int) -> float:
    """Fraction of faces ``fast_simplification`` must remove to land at ``pre_cap``.

    ``fast_simplification.simplify(verts, faces, ratio)`` removes ``ratio`` (0..1) of the faces,
    so the ratio for reaching ``pre_cap`` is ``1 - pre_cap/num_faces``; 0.0 when already at or
    under the cap (no-op). Pure and importable, so the unit test needs neither numpy nor torch.
    """
    if pre_cap <= 0:
        raise ValueError(f"pre_cap must be positive, got {pre_cap}")
    if num_faces <= pre_cap:
        return 0.0
    return 1.0 - (pre_cap / num_faces)


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
    """Pin the Mac backend env BEFORE any torch/TRELLIS import.

    ATTN_BACKEND / SPARSE_ATTN_BACKEND are hard-assigned, not setdefault: a stale value in
    the inherited environment would silently select a different backend. (The web UI once
    inherited SPARSE_CONV_BACKEND=none and the pipeline died importing the nonexistent
    conv_none module.) sdpa is the validated backend; --sparse-attn-backend is the only way
    to change it.
    """
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = sparse_attn_backend
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(vendor_root / "cache" / "flex_gemm_autotune.json"),
    )


def require_flex_gemm() -> None:
    """Import flex_gemm and pin SPARSE_CONV_BACKEND, or fail with an actionable error.

    flex_gemm is the only Metal sparse-conv backend in this port (spconv/torchsparse are
    CUDA). 'none' is NOT a valid backend: trellis2 imports ``conv_<CONV>`` and there is no
    conv_none module, so a silent fallback to 'none' dies mid-load and the model loader
    then tries to re-download checkpoints with a mangled repo id. Fail loudly instead, so
    the real problem (e.g. a moved venv whose compiled kernels embed stale rpaths) is
    visible at load time.
    """
    try:
        import flex_gemm  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "flex_gemm (the only Metal sparse-conv backend in this port) failed to import: "
            f"{exc}. 'none' is not a valid SPARSE_CONV_BACKEND (no conv_none module exists). "
            "If the built venv was moved or relocated, its compiled kernels embed absolute "
            "rpaths; rebuild it in place: uv pip install --python .venv/bin/python "
            "--no-build-isolation --force-reinstall deps/mtlgemm"
        ) from exc
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"


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

    try:
        import fast_simplification  # noqa: F401
    except ImportError:
        print("  FAIL: fast_simplification not importable (needed for the CPU pre-cap before to_glb)",
              flush=True)
        return 1

    import o_voxel
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

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


def filter_degenerate_faces(mesh) -> int:
    """Drop decode faces with out-of-range indices (the -1 marker some MPS decodes emit).

    Returns the count removed and mutates ``mesh.faces`` in place (settable, as base.py does).
    """
    import torch

    num_vertices = int(mesh.vertices.shape[0])
    mask = valid_face_mask(mesh.faces.cpu().numpy(), num_vertices)
    removed = int((~mask).sum())
    if removed:
        keep = torch.as_tensor(mask, device=mesh.faces.device)
        mesh.faces = mesh.faces[keep]
    return removed


def _decode_mesh(
    pipeline, shape_slat, tex_slat, res, *,
    pipeline_type: str, seed: int, image_path: Path,
) -> tuple[dict[str, Any], float]:
    """decode_latent -> drop degenerate faces -> CPU bundle. Returns (bundle, decode_seconds).

    The bundle is exactly the ``--from-decode`` cache schema (verts/faces/attrs/coords on CPU
    plus attr_layout/res/pipeline_type/seed/images), so saving it IS writing the cache.
    """
    decode_started = time.time()
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    removed = filter_degenerate_faces(mesh)
    if removed:
        print(f"  filtered {removed} degenerate face(s) (out-of-range/-1 index) before bake",
              flush=True)
    decode_seconds = time.time() - decode_started
    print(f"decode_latent (+face filter) done in {decode_seconds:.1f}s", flush=True)
    return (
        {
            "vertices": mesh.vertices.detach().cpu(),
            "faces": mesh.faces.detach().cpu(),
            "attrs": mesh.attrs.detach().cpu(),
            "coords": mesh.coords.detach().cpu(),
            "attr_layout": pipeline.pbr_attr_layout,
            "res": int(res),
            "pipeline_type": pipeline_type,
            "seed": seed,
            "images": [str(image_path)],
        },
        decode_seconds,
    )


def _release_mps_memory() -> None:
    """gc + release MPS cached memory after the caller drops the pipeline reference.

    Keeping the 4B model resident while fast_simplification allocates its working set on a
    ~27M-face mesh once segfaulted the C extension (KERN_INVALID_ADDRESS in _simplify.so); the
    same pre-cap succeeds with the model unloaded, so this is memory pressure, not a size
    ceiling. The bake (to_glb) does not need the pipeline, so callers ``del pipeline`` then
    call this before ``_bake_export``.
    """
    import gc

    gc.collect()
    import torch

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def filter_out_of_range_faces(faces, num_vertices: int):
    """Drop faces referencing vertex indices outside [0, num_vertices).

    ``fast_simplification`` occasionally emits a small number of corrupt indices on >~20M-face
    inputs (usually a few dozen, rarely catastrophic); mtlbvh's BVH build segfaults on them.
    Returns (kept_faces, removed_count). Pure enough to unit test without torch.
    """
    import torch

    mask = valid_face_mask(faces.cpu().numpy(), num_vertices)
    removed = int((~mask).sum())
    if removed:
        faces = faces[torch.as_tensor(mask, device=faces.device)]
    return faces, removed


def _precap_subprocess(vertices, faces, pre_cap: int, max_attempts: int = 5):
    """CPU ``fast_simplification`` in a FRESH interpreter, with verify-and-retry.

    Two independent reasons the decimation is isolated and verified:

    1. fast_simplification segfaults/bus-errors on >~20M-face meshes in any process that has
       imported o_voxel's deps (cumesh/cv2/Metal reserve address space and shift the heap); the
       same pre-cap runs in a clean interpreter.
    2. Above ~20M input faces its output is nondeterministically corrupt -- usually a few dozen
       out-of-range face indices (filtered by the caller), rarely millions (retried here). Each
       attempt runs in its own subprocess so a crash is just a failed attempt, not a lost bake.

    Returns (verts, faces) CPU tensors. Raises RuntimeError after ``max_attempts`` consecutive
    corrupt/crashed attempts.
    """
    import subprocess
    import tempfile

    import torch

    code = (
        "import sys, torch, fast_simplification\n"
        "b = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "v, f = b['vertices'].numpy(), b['faces'].numpy()\n"
        "sv, sf = fast_simplification.simplify(v, f, 1.0 - int(sys.argv[3]) / f.shape[0])\n"
        "torch.save((torch.from_numpy(sv).float(), torch.from_numpy(sf.astype('int32'))), "
        "sys.argv[2])\n"
    )
    last_error = "no attempts made"
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "precap_in.pt"
            out_path = Path(tmp) / "precap_out.pt"
            torch.save({"vertices": vertices, "faces": faces}, in_path)
            try:
                subprocess.run(
                    [sys.executable, "-c", code, str(in_path), str(out_path), str(pre_cap)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                verts_out, faces_out = torch.load(out_path, map_location="cpu", weights_only=False)
            except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
                last_error = f"crash/failure: {exc}"
                print(f"  pre-cap attempt {attempt}/{max_attempts} {last_error}; retrying",
                      flush=True)
                continue
        num_faces = int(faces_out.shape[0])
        bad = int((~valid_face_mask(faces_out.numpy(), verts_out.shape[0])).sum())
        if bad / num_faces > 0.001:  # catastrophic corruption (usual case is a few dozen)
            last_error = f"{bad:,} out-of-range faces ({bad / num_faces:.1%})"
            print(f"  pre-cap attempt {attempt}/{max_attempts}: {last_error}; retrying", flush=True)
            continue
        if attempt > 1:
            print(f"  pre-cap succeeded on attempt {attempt}", flush=True)
        return verts_out, faces_out
    raise RuntimeError(f"pre-cap failed after {max_attempts} attempts; last: {last_error}")


def _bake_export(
    vertices, faces, attrs, coords, attr_layout, res, output_path: Path, *,
    decimation_target: int, texture_size: int, pre_cap: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """CPU pre-cap the decoded mesh, then Metal to_glb -> GLB export. Shared by every entry point.

    The raw decode is ~10M verts / ~20M faces. cumesh's Metal simplify and mtlbvh are not robust
    at that scale (the MPS decode->GLB blocker), so we (a) pre-cap with CPU ``fast_simplification``
    to ``pre_cap`` -- the threshold the old port proved safe for this exact to_glb -- and
    (b) hand to_glb CPU tensors: the metal backend sets ``device='cpu'`` internally and builds
    aabb/voxel_size on ``coords.device``, so MPS inputs would mismatch. attrs/coords are the
    voxel field and pass through the pre-cap untouched (to_glb samples the volume, it does not
    read per-vertex attributes).
    """
    import o_voxel

    started = time.time()
    verts_cpu = vertices.detach().cpu()
    faces_cpu = faces.detach().cpu()
    num_faces = int(faces_cpu.shape[0])
    if num_faces > pre_cap:
        ratio = precap_ratio(num_faces, pre_cap)
        print(f"  CPU pre-cap: {num_faces:,} -> ~{pre_cap:,} faces "
              f"(fast_simplification remove {ratio:.2%}, subprocess)", flush=True)
        simp_verts, simp_faces = _precap_subprocess(verts_cpu, faces_cpu, pre_cap)
        verts_cpu = simp_verts
        faces_cpu, removed_bad = filter_out_of_range_faces(simp_faces, verts_cpu.shape[0])
        if removed_bad:
            print(f"  filtered {removed_bad} corrupt face(s) from pre-cap output", flush=True)
        print(f"  pre-cap done: {verts_cpu.shape[0]:,} verts, {faces_cpu.shape[0]:,} faces",
              flush=True)
    else:
        print(f"  mesh at/under pre-cap ({num_faces:,} <= {pre_cap:,}); no pre-cap needed", flush=True)

    glb = o_voxel.postprocess.to_glb(
        vertices=verts_cpu,
        faces=faces_cpu,
        attr_volume=attrs.detach().cpu(),
        coords=coords.detach().cpu(),
        attr_layout=attr_layout,
        grid_size=res,
        aabb=AABB,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=DEMO_PARAMS["remesh"]["remesh"],
        remesh_band=DEMO_PARAMS["remesh"]["remesh_band"],
        remesh_project=DEMO_PARAMS["remesh"]["remesh_project"],
        use_tqdm=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_path), extension_webp=True)
    bake_seconds = time.time() - started
    print(f"bake (pre-cap + to_glb + export) done in {bake_seconds:.1f}s -> {output_path}",
          flush=True)
    return ({"bake": bake_seconds}, {"glb": _artifact(output_path)})


def _decode_and_cache(
    pipeline, shape_slat, tex_slat, res, output_path: Path, *,
    save_decode: bool, decode_path: Path | None,
    pipeline_type: str, seed: int, image_path: Path,
) -> tuple[dict[str, Any], float]:
    """decode -> filter -> (save the CPU bundle) -> return (bundle, decode_seconds).

    The returned bundle is the ``--from-decode`` cache schema. The CALLER must free the 4B
    model before the bake: keeping it resident while fast_simplification allocates its working
    set on a 20M+ face mesh once segfaulted the C extension (see ``_release_mps_memory``), and
    the bake does not need the pipeline.
    """
    bundle, decode_seconds = _decode_mesh(
        pipeline, shape_slat, tex_slat, res,
        pipeline_type=pipeline_type, seed=seed, image_path=image_path,
    )
    if save_decode:
        if decode_path is None:
            decode_path = output_path.with_name(output_path.stem + "_decode.pt")
        import torch

        torch.save(bundle, decode_path)
        print(f"Decode cached: {decode_path}", flush=True)
    return bundle, decode_seconds


def load_pipeline(vendor_root: Path, sparse_attn_backend: str, load_rembg: bool):
    """Configure the Mac env, import TRELLIS, load the 4B pipeline onto MPS. Returns the pipeline."""
    configure_environment(vendor_root, sparse_attn_backend)
    sys.path.insert(0, str(vendor_root / "TRELLIS.2"))
    stubs = vendor_root / "stubs"
    if stubs.is_dir():
        sys.path.append(str(stubs))
    require_flex_gemm()

    import torch
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    print(f"Loading TRELLIS.2 pipeline (load_rembg={load_rembg})...", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", load_rembg=load_rembg
    )
    pipeline.to(torch.device("mps"))
    return pipeline


def generate_from_latents(
    latents_path: Path,
    output_path: Path,
    vendor_root: Path,
    *,
    sparse_attn_backend: str,
    save_decode: bool,
    pre_cap: int,
    decimation_target: int = DEMO_PARAMS["decimation_target"],
    texture_size: int = DEMO_PARAMS["texture_size"],
) -> dict[str, Any]:
    """Resume from a cached ``*_latents.pt``: skip the ~78-min sampling, just decode + bake.

    Reconstructs the shape/texture SLat from the saved bundle exactly as app.py's ``unpack_state``
    does (but on MPS), then runs the shared decode+bake path.
    """
    started = time.time()
    pipeline = load_pipeline(vendor_root, sparse_attn_backend, load_rembg=False)
    load_seconds = time.time() - started
    print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

    # trellis2 is only importable after load_pipeline has put TRELLIS.2 on sys.path.
    import torch
    from trellis2.modules.sparse import SparseTensor

    bundle = torch.load(latents_path, map_location="cpu", weights_only=False)
    device = torch.device("mps")
    shape_slat = SparseTensor(
        feats=bundle["shape_slat_feats"].to(device),
        coords=bundle["coords"].to(device),
    )
    tex_slat = shape_slat.replace(bundle["tex_slat_feats"].to(device))
    res = int(bundle["res"])
    pipeline_type = bundle.get("pipeline_type", pipeline_type_for_resolution(DEMO_PARAMS["resolution"]))

    decode_path = None
    if save_decode:
        decode_path = output_path.with_name(output_path.stem + "_decode.pt")
    image_path = Path(str(bundle["images"][0])) if bundle.get("images") else Path("<from latents>")
    mesh_bundle, decode_seconds = _decode_and_cache(
        pipeline, shape_slat, tex_slat, res, output_path,
        save_decode=save_decode, decode_path=decode_path,
        pipeline_type=pipeline_type, seed=int(bundle.get("seed", -1)),
        image_path=image_path,
    )
    del pipeline  # free the 4B model before the memory-heavy bake (see _release_mps_memory)
    _release_mps_memory()
    bake_timings, bake_artifacts = _bake_export(
        mesh_bundle["vertices"], mesh_bundle["faces"], mesh_bundle["attrs"],
        mesh_bundle["coords"], mesh_bundle["attr_layout"], mesh_bundle["res"], output_path,
        decimation_target=decimation_target, texture_size=texture_size, pre_cap=pre_cap,
    )
    decode_timings = {"decode_latent": decode_seconds, **bake_timings}
    artifacts = bake_artifacts
    artifacts["latents"] = _artifact(latents_path)
    if decode_path is not None and decode_path.is_file():
        artifacts["decode"] = _artifact(decode_path)

    timings = {"pipeline_load": load_seconds, **decode_timings, "total": time.time() - started}
    effective_params = {**DEMO_PARAMS, "decimation_target": decimation_target,
                        "texture_size": texture_size}
    manifest = build_manifest(
        image=str(bundle.get("images", ["<from latents>"])[0]),
        output=str(output_path),
        params=effective_params,
        pipeline_type=pipeline_type,
        seed=int(bundle.get("seed", -1)),
        timings=timings,
        artifacts=artifacts,
        load_rembg=False,
        sparse_attn_backend=sparse_attn_backend,
    )
    manifest["resumed_from_latents"] = str(latents_path)
    manifest_path = output_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total (decode+bake, no resample): {timings['total']:.1f}s", flush=True)
    return manifest


def generate(
    image_path: Path,
    output_path: Path,
    vendor_root: Path,
    *,
    seed: int,
    sparse_attn_backend: str,
    allow_rembg: bool,
    save_latents: bool,
    save_decode: bool,
    pre_cap: int,
    resolution: str = DEMO_PARAMS["resolution"],
    decimation_target: int = DEMO_PARAMS["decimation_target"],
    texture_size: int = DEMO_PARAMS["texture_size"],
) -> dict[str, Any]:
    """Run the full clean-port pipeline on MPS and export a GLB. Returns the manifest dict."""
    from PIL import Image

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
    pipeline = load_pipeline(vendor_root, sparse_attn_backend, load_rembg)
    load_seconds = time.time() - started
    print(f"Pipeline loaded in {load_seconds:.1f}s", flush=True)

    import torch

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

    decode_path = None
    if save_decode:
        decode_path = output_path.with_name(output_path.stem + "_decode.pt")
    mesh_bundle, decode_seconds = _decode_and_cache(
        pipeline, shape_slat, tex_slat, res, output_path,
        save_decode=save_decode, decode_path=decode_path,
        pipeline_type=pipeline_type, seed=seed, image_path=image_path,
    )
    del pipeline  # free the 4B model before the memory-heavy bake (see _release_mps_memory)
    _release_mps_memory()
    bake_timings, bake_artifacts = _bake_export(
        mesh_bundle["vertices"], mesh_bundle["faces"], mesh_bundle["attrs"],
        mesh_bundle["coords"], mesh_bundle["attr_layout"], mesh_bundle["res"], output_path,
        decimation_target=decimation_target, texture_size=texture_size, pre_cap=pre_cap,
    )
    decode_timings = {"decode_latent": decode_seconds, **bake_timings}
    artifacts.update(bake_artifacts)
    if decode_path is not None and decode_path.is_file():
        artifacts["decode"] = _artifact(decode_path)

    timings = {
        "pipeline_load": load_seconds,
        "run_stages_1_3": run_seconds,
        **decode_timings,
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


def generate_from_decode(
    decode_path: Path,
    output_path: Path,
    vendor_root: Path,
    *,
    decimation_target: int,
    texture_size: int,
    pre_cap: int,
) -> dict[str, Any]:
    """Resume from a cached ``*_decode.pt``: no model load, no decode -- just pre-cap + bake.

    The decode bundle carries everything to_glb needs (verts/faces/attrs/coords + attr_layout),
    so this is the ~1-min-per-attempt GLB packaging loop for iterating on bake options.
    """
    import torch

    configure_environment(vendor_root, "sdpa")
    started = time.time()
    bundle = torch.load(decode_path, map_location="cpu", weights_only=False)
    res = int(bundle["res"])
    bake_timings, bake_artifacts = _bake_export(
        bundle["vertices"], bundle["faces"], bundle["attrs"], bundle["coords"],
        bundle["attr_layout"], res, output_path,
        decimation_target=decimation_target, texture_size=texture_size, pre_cap=pre_cap,
    )
    bake_artifacts["decode"] = _artifact(decode_path)
    timings = {"load_decode_bundle": time.time() - started, **bake_timings,
               "total": time.time() - started}
    manifest = build_manifest(
        image=str(bundle.get("images", ["<from decode>"])[0]),
        output=str(output_path),
        params={**DEMO_PARAMS, "decimation_target": decimation_target,
                "texture_size": texture_size, "pre_cap_faces": pre_cap},
        pipeline_type=bundle.get(
            "pipeline_type", pipeline_type_for_resolution(DEMO_PARAMS["resolution"])),
        seed=int(bundle.get("seed", -1)),
        timings=timings,
        artifacts=bake_artifacts,
        load_rembg=False,
        sparse_attn_backend="sdpa",
    )
    manifest["resumed_from_decode"] = str(decode_path)
    manifest_path = output_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total (pre-cap + bake, no decode): {timings['total']:.1f}s", flush=True)
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
    parser.add_argument("--no-save-decode", dest="save_decode", action="store_false",
                        default=True,
                        help="do not write <out>_decode.pt (the resumeable decoded-mesh cache)")
    parser.add_argument("--pre-cap", type=int, default=PRE_CAP_FACES_DEFAULT,
                        help="CPU fast_simplification face budget before the Metal to_glb "
                             "(default %(default)s; the old port's proven-safe threshold)")
    parser.add_argument("--from-decode", type=Path, default=None,
                        help="resume from a cached *_decode.pt: no model load, no decode, only "
                             "CPU pre-cap + to_glb bake. Give the output GLB path positionally.")
    parser.add_argument("--from-latents", type=Path, default=None,
                        help="resume from a cached *_latents.pt: skip the ~78-min sampling, only "
                             "decode + bake. Give the output GLB path as the positional argument.")
    parser.add_argument("--check", action="store_true",
                        help="verify the environment and exit (no model load, no sampling)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vendor_root = args.vendor_root.resolve()

    if args.check:
        return check_environment(vendor_root, args.sparse_attn_backend)

    if args.from_decode is not None:
        out = args.output or args.image  # accept the GLB path in either positional slot
        if out is None:
            raise SystemExit("provide the output GLB path (positional) with --from-decode")
        if not args.from_decode.is_file():
            raise SystemExit(f"missing decode bundle: {args.from_decode}")
        generate_from_decode(
            args.from_decode, out, vendor_root,
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            pre_cap=args.pre_cap,
        )
        return 0

    if args.from_latents is not None:
        out = args.output or args.image  # accept the GLB path in either positional slot
        if out is None:
            raise SystemExit("provide the output GLB path (positional) with --from-latents")
        if not args.from_latents.is_file():
            raise SystemExit(f"missing latents bundle: {args.from_latents}")
        generate_from_latents(
            args.from_latents, out, vendor_root,
            sparse_attn_backend=args.sparse_attn_backend,
            save_decode=args.save_decode,
            pre_cap=args.pre_cap,
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
        )
        return 0

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
        save_decode=args.save_decode,
        pre_cap=args.pre_cap,
        resolution=args.resolution,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
