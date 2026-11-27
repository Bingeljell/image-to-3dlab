#!/usr/bin/env python3
"""Make Pixal3D's low-VRAM mode reachable, via `PIXAL3D_LOW_VRAM=1`.

    python scripts/patch_pixal3d_low_vram.py
    PIXAL3D_LOW_VRAM=1 ... generate_mps.py ...

**Why.** `generate_mps.py` already ships `_install_low_vram_hooks`, which releases the MPS
allocator cache after each heavy stage so peak reserved memory drops. It is gated behind
`getattr(args, "low_vram", False)` — and no CLI flag or environment variable ever sets
that, so the function is unreachable. On a 32 GB Mac the 1024 pipeline loads ~22 GB of
weights and is then killed partway through shape sampling, which is exactly the case those
hooks were written for.

This patch leaves the existing `args.low_vram` path intact and adds the environment
variable beside it, so a run can opt in without editing code. Costs some re-allocation
latency between stages, which is why it stays opt-in rather than becoming the default.

Worth upstreaming: the hooks are the author's own work, and this only exposes them.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone. Safe to
run repeatedly; it detects its own marker and does nothing on a second run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .pixal3d_compat import LOW_VRAM_ENV, helper_matches, install_helper
else:
    from pixal3d_compat import LOW_VRAM_ENV, helper_matches, install_helper

MARKER = "i2l_low_vram"
ENV_VAR = LOW_VRAM_ENV
HELPER_SOURCE = Path(__file__).with_name("pixal3d_compat.py")

DEFAULT_TARGET = Path("vendor/pixal3d-mac/generate_mps.py")

ANCHOR = '    if getattr(args, "low_vram", False):\n'

LEGACY = (
    f'    # {MARKER}: the hooks below were unreachable — nothing ever set args.low_vram.\n'
    f'    if getattr(args, "low_vram", False) or os.environ.get("{ENV_VAR}", "").strip() '
    f'not in ("", "0", "false", "False"):\n'
)
REPLACEMENT = f'''    # {MARKER}: apply the existing option or {ENV_VAR}.
    from _assetfurnace_pixal3d_compat import low_vram_for_run
    if low_vram_for_run(args):
'''


def is_patched(source: str) -> bool:
    return source.count(REPLACEMENT) == 1


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    anchor = LEGACY if LEGACY in source else ANCHOR
    if source.count(anchor) != 1:
        raise SystemExit(
            "anchor not found exactly once: verify the audited low-VRAM hook before patching."
        )
    return source.replace(anchor, REPLACEMENT, 1)


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
    print(f"patched {args.target}: {ENV_VAR}=1 now installs the low-VRAM hooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
