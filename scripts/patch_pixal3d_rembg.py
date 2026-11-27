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

if __package__:
    from .pixal3d_compat import (
        FORBIDDEN_PREFIX,
        SUBSTITUTE,
        helper_matches,
        install_helper,
        replace_payload,
    )
else:
    from pixal3d_compat import (
        FORBIDDEN_PREFIX,
        SUBSTITUTE,
        helper_matches,
        install_helper,
        replace_payload,
    )

MARKER = "i2l_rembg_licence"
HELPER_SOURCE = Path(__file__).with_name("pixal3d_compat.py")

DEFAULT_TARGET = Path("vendor/pixal3d-mac/pixal3d/pipelines/rembg/BiRefNet.py")

ANCHOR = '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'

INJECTION = '''        # i2l_rembg_licence: guard against the forbidden briaai/ model family.
        from ._assetfurnace_pixal3d_compat import guard_rembg_model
        model_name = guard_rembg_model(model_name)
'''
LEGACY_END = '            model_name = "ZhengPeng7/BiRefNet"\n'
LEGACY_DIGEST = "441cb01263380bd9edf3305e6b8a0004669da5d86854e7ebe2aade727d08259c"


def is_patched(source: str) -> bool:
    return source.count(ANCHOR) == 1 and ANCHOR + INJECTION in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    return replace_payload(source, ANCHOR, INJECTION, MARKER, LEGACY_END, LEGACY_DIGEST,
                           "anchor not found or changed: verify the audited BiRefNet constructor before generation.")


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
    source = args.target.read_text(encoding="utf-8")

    if args.check:
        if is_patched(source) and helper_matches(HELPER_SOURCE, args.target):
            print(f"patched: {args.target}")
            return 0
        print(f"NOT PATCHED: {args.target} — refresh the BRIA guard before generation")
        return 1

    patched = apply(source)
    helper_changed = install_helper(HELPER_SOURCE, args.target)
    if patched == source and not helper_changed:
        print(f"already patched: {args.target}")
        return 0
    if patched != source:
        args.target.write_text(patched, encoding="utf-8")
    print(f"patched {args.target}: {FORBIDDEN_PREFIX}* -> {SUBSTITUTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
