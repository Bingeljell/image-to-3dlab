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

MARKER = "i2l_model_subset"
ENV_VAR = "PIXAL3D_SKIP_MODELS"

DEFAULT_TARGET = Path("vendor/pixal3d-mac/pixal3d/pipelines/pixal3d_image_to_3d.py")

ANCHOR = """        'tex_slat_decoder',
    ]
"""

INJECTION = f'''
    # {MARKER}: drop checkpoints this run will not use, before they are read. The list
    # above is ~22 GB at 1024, which does not fit beside a sampler on a 32 GB Mac.
    # `pipelines/base.py` skips any model absent from this attribute, so subtracting
    # here is enough — nothing downstream needs to know.
    import os as _i2l_os

    _i2l_skip = {{
        _name.strip()
        for _name in _i2l_os.environ.get("{ENV_VAR}", "").split(",")
        if _name.strip()
    }}
    if _i2l_skip:
        print(f"[{MARKER}] not loading: {{sorted(_i2l_skip)}}", flush=True)
        # A plain loop, not a comprehension: comprehensions get their own scope and
        # cannot see class-body names, so the obvious one-liner raises NameError at
        # import time. Caught by a test rather than by a dead run.
        _i2l_kept = []
        for _i2l_name in model_names_to_load:
            if _i2l_name not in _i2l_skip:
                _i2l_kept.append(_i2l_name)
        model_names_to_load = _i2l_kept
        del _i2l_kept, _i2l_name
    del _i2l_os, _i2l_skip
'''


def remaining_models(names: list[str], skip: str) -> list[str]:
    """The models still loaded after applying a `PIXAL3D_SKIP_MODELS` value.

    Pure, and the same decision the injected code makes, so it can be tested without
    importing a pipeline or spending a run.
    """
    dropped = {name.strip() for name in skip.split(",") if name.strip()}
    return [name for name in names if name not in dropped]


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    if ANCHOR not in source:
        raise SystemExit(
            "anchor not found: the pipeline's `model_names_to_load` list no longer ends "
            "with 'tex_slat_decoder'. Check what it loads now before assuming this patch "
            "still selects the right subset."
        )
    return source.replace(ANCHOR, ANCHOR + INJECTION, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text()

    if args.check:
        print(("patched: " if is_patched(source) else "NOT PATCHED: ") + str(args.target))
        return 0 if is_patched(source) else 1

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched)
    print(f"patched {args.target}: {ENV_VAR} now subtracts from model_names_to_load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
