#!/usr/bin/env python3
"""Let Pixal3D load only the checkpoints a run actually needs.

    python scripts/patch_pixal3d_model_subset.py
    PIXAL3D_SKIP_MODELS=tex_slat_flow_model_1024,tex_slat_decoder ... generate_mps.py ...

**Why.** `Pixal3DImageTo3DPipeline` loads every model named in the weights' `pipeline.json`
before it does anything, and at 1024 that is roughly 22 GB of bf16:

    sparse_structure_flow_model      5.36 GB
    shape_slat_flow_model_512        5.55 GB
    shape_slat_flow_model_1024       5.55 GB
    tex_slat_flow_model_1024         5.55 GB
    shape_slat_decoder / tex_slat_decoder / sparse_structure_decoder   ~2 GB

On a 32 GB unified-memory Mac that leaves nothing for the sampler, and the run dies during
load — twice here before this patch existed. The port's author reports using an M5 Max, so
this is a small-machine problem rather than a defect.

`pipelines/base.py` already honours a class attribute, `model_names_to_load`, and skips any
checkpoint not in it. This patch simply makes that list subtractable from the environment,
so a geometry-only run can leave the texture stack on disk — 6.5 GB of the 22 — without
touching the default behaviour of anything else.

Skipping a model the run then needs fails at that stage, not silently: the pipeline looks
the model up by name. Pair `PIXAL3D_SKIP_MODELS` with the matching CLI flag, e.g. the
texture models with `--no-texture`.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone. Safe to
run repeatedly; it detects its own marker and does nothing on a second run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .pixal3d_compat import (
        ENV_VAR,
        helper_matches,
        install_helper,
        replace_payload,
    )
else:
    from pixal3d_compat import (
        ENV_VAR,
        helper_matches,
        install_helper,
        replace_payload,
    )

MARKER = "i2l_model_subset"
HELPER_SOURCE = Path(__file__).with_name("pixal3d_compat.py")

DEFAULT_TARGET = Path("vendor/pixal3d-mac/pixal3d/pipelines/pixal3d_image_to_3d.py")

ANCHOR = """        'tex_slat_decoder',
    ]
"""

INJECTION = '''    # i2l_model_subset: apply the checkpoint selection before loading weights.
    from ._assetfurnace_pixal3d_compat import model_names_for_run as _i2l_select
    model_names_to_load = _i2l_select(model_names_to_load)
    del _i2l_select
'''
LEGACY_END = '    del _i2l_os, _i2l_skip\n'
LEGACY_DIGEST = "044a8822bc71e3012a0bab7506b9a4e8635d15fcda43ed6aef553f6f6f8bf357"


def is_patched(source: str) -> bool:
    return source.count(ANCHOR) == 1 and ANCHOR + INJECTION in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    return replace_payload(source, ANCHOR, INJECTION, MARKER, LEGACY_END, LEGACY_DIGEST,
                           "anchor not found or changed: the audited Pixal3D checkpoint selection block.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text(encoding="utf-8")

    if args.check:
        ready = is_patched(source) and helper_matches(HELPER_SOURCE, args.target)
        print(("patched: " if ready else "NOT PATCHED: ") + str(args.target))
        return 0 if ready else 1

    patched = apply(source)
    helper_changed = install_helper(HELPER_SOURCE, args.target)
    if patched == source and not helper_changed:
        print(f"already patched: {args.target}")
        return 0
    if patched != source:
        args.target.write_text(patched, encoding="utf-8")
    print(f"patched {args.target}: {ENV_VAR} now subtracts from model_names_to_load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
