#!/usr/bin/env python3
"""Stop Pixal3D loading BRIA RMBG-2.0, before it ever downloads it.

    python scripts/patch_pixal3d_rembg.py            # applies to vendor/pixal3d-mac
    python scripts/patch_pixal3d_rembg.py --check    # exit 1 if not patched

**Why.** Pixal3D's matting is a `BiRefNet` wrapper whose own default is the permissive
upstream model, but the `pipeline.json` shipped with the Tencent weights overrides it:

    "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}}

So an out-of-the-box run downloads and loads BRIA RMBG-2.0 into this repo's generation
path, which is exactly what the licence guardrail in CLAUDE.md forbids. The weights config
is fetched fresh from Hugging Face and is not ours to edit durably, so the substitution
belongs in the loader, where it holds no matter what config asks.

**What it substitutes.** `ZhengPeng7/BiRefNet`, MIT, which is the model BRIA fine-tuned to
produce RMBG-2.0 — the closest permissive equivalent rather than a different approach to
matting. It is also what this file already defaults to, so the patch restores the author's
default rather than inventing one.

**What this costs the comparison.** Matting changes the input image, so a Pixal3D run
patched this way is not a pixel-exact like-for-like against a TRELLIS run that used
different background removal. For judging one model against another that is acceptable;
for a controlled comparison, feed both an identical pre-cut RGBA and skip matting entirely.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone. Safe to
run repeatedly: it detects its own marker and does nothing on a second run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "i2l_rembg_licence"
SUBSTITUTE = "ZhengPeng7/BiRefNet"
FORBIDDEN_PREFIX = "briaai/"

DEFAULT_TARGET = Path("vendor/pixal3d-mac/pixal3d/pipelines/rembg/BiRefNet.py")

ANCHOR = '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'

INJECTION = f'''        # {MARKER}: the weights' pipeline.json asks for "briaai/RMBG-2.0", whose
        # licence this repo does not accept in its generation path. Substitute the
        # permissive upstream model BRIA fine-tuned to make it. Never remove silently:
        # the point is that no run can load BRIA by way of a config we do not control.
        if model_name.startswith("{FORBIDDEN_PREFIX}"):
            print(
                f"[{MARKER}] refusing {{model_name}}; using {SUBSTITUTE} instead",
                flush=True,
            )
            model_name = "{SUBSTITUTE}"
'''


def permissive_model_name(model_name: str) -> str:
    """The model that may actually be loaded, given what a config asked for.

    Pure, and the same decision the injected code makes — which is why it can be tested
    here rather than only by running a generation.
    """
    if model_name.startswith(FORBIDDEN_PREFIX):
        return SUBSTITUTE
    return model_name


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    if ANCHOR not in source:
        raise SystemExit(
            "anchor not found: BiRefNet.__init__ no longer has the expected signature. "
            "Check how the matting model is selected before assuming this patch works — "
            "an unpatched run downloads BRIA RMBG-2.0."
        )
    return source.replace(ANCHOR, ANCHOR + INJECTION, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--check", action="store_true",
        help="Report whether the patch is present and exit non-zero if it is not, so a "
             "runner can refuse to generate rather than quietly loading BRIA",
    )
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text()

    if args.check:
        if is_patched(source):
            print(f"patched: {args.target}")
            return 0
        print(f"NOT PATCHED: {args.target} — a run would load BRIA RMBG-2.0")
        return 1

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched)
    print(f"patched {args.target}: {FORBIDDEN_PREFIX}* -> {SUBSTITUTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
