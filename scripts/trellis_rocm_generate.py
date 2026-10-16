#!/usr/bin/env python3
"""Full image -> GLB generation through Microsoft's own TRELLIS.2 on an AMD card.

The ROCm twin of ``trellis_cuda_generate.py``. Every pipeline call, demo setting, decode,
bake, manifest and provenance rule is imported from that script rather than copied, so the
two cards cannot drift: this one supplies ``trellis_route.ROCM`` and nothing else. The path is
upstream's ``app.py``: ``pipeline.run()`` (sparse structure -> shape -> material) ->
``decode_latent`` -> ``mesh.simplify`` to nvdiffrast's limit ->
``o_voxel.postprocess.to_glb``.

Run it with that checkout's interpreter::

    vendor/trellis-rocm/.venv/bin/python scripts/trellis_rocm_generate.py in.png out.glb
    vendor/trellis-rocm/.venv/bin/python scripts/trellis_rocm_generate.py --check

Writes ``<out>.glb``, ``<out>.json`` (the run manifest) and ``<out>.provenance.json`` (the
licence record). ``--from-latents`` resumes a run whose decode or bake failed.

**Two differences from the NVIDIA route, both deliberate.** Attention runs on AuleAttention,
a Triton flash attention with no CUDA extension to compile, because flash-attn cannot be
built here (see ``scripts/patch_trellis_rocm.py``). And no allocator configuration is set:
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` produced silent NaNs on ROCm rather than
an error, which is worse than not setting it.

**Licence guardrail.** As on the NVIDIA route, upstream loads BRIA RMBG-2.0 and this script
refuses to run unless ``patch_trellis_cuda_no_bria.py`` has removed that load.
"""

from __future__ import annotations

import functools
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import trellis_cuda_generate as shared  # noqa: E402, RUF100
import trellis_route  # noqa: E402, RUF100

#: Everything shared with the NVIDIA route, re-exported so this script reads as one file.
ROUTE = trellis_route.ROCM
DEMO_PARAMS = shared.DEMO_PARAMS
DEFAULT_VENDOR = ROUTE.vendor


def _for_this_route(function):
    """Pin this route into anything that takes one.

    Without this the re-exported functions keep their NVIDIA defaults, and the first sign is
    silent: `configure_environment()` sets ATTN_BACKEND=flash_attn, which does not exist on
    an AMD card, and the run dies in the first attention rather than at startup.
    """
    if "route" not in inspect.signature(function).parameters:
        return function
    return functools.partial(function, route=ROUTE)


for _name in (
    "backend_revision",
    "bria_refusal",
    "build_manifest",
    "build_provenance",
    "check_environment",
    "checkout",
    "configure_environment",
    "forbid_bria",
    "generate",
    "generate_from_latents",
    "load_pipeline",
    "matted_path",
    "parse_args",
    "prepare_image",
    "provenance_path",
    "verify_paths",
):
    globals()[_name] = _for_this_route(getattr(shared, _name))


def main(argv: list[str] | None = None) -> int:
    return shared.main(argv, route=ROUTE)


if __name__ == "__main__":
    raise SystemExit(main())