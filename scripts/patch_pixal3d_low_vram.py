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

MARKER = "i2l_low_vram"
ENV_VAR = "PIXAL3D_LOW_VRAM"

DEFAULT_TARGET = Path("vendor/pixal3d-mac/generate_mps.py")

ANCHOR = '    if getattr(args, "low_vram", False):\n'

REPLACEMENT = (
    f'    # {MARKER}: the hooks below were unreachable — nothing ever set args.low_vram.\n'
    f'    if getattr(args, "low_vram", False) or os.environ.get("{ENV_VAR}", "").strip() '
    f'not in ("", "0", "false", "False"):\n'
)


def low_vram_requested(args_value: bool, env_value: str | None) -> bool:
    """Whether low-VRAM mode should be installed.

    Pure, and the same decision the injected condition makes. The falsy spellings matter:
    `PIXAL3D_LOW_VRAM=0` in a shell profile must not silently turn it on.
    """
    if args_value:
        return True
    if env_value is None:
        return False
    return env_value.strip() not in ("", "0", "false", "False")


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    if ANCHOR not in source:
        raise SystemExit(
            "anchor not found: generate_mps.py no longer gates the low-VRAM hooks on "
            "`getattr(args, \"low_vram\", False)`. Check whether it grew a real flag "
            "before applying this."
        )
    return source.replace(ANCHOR, REPLACEMENT, 1)


def uses_os_module(source: str) -> bool:
    """The injected condition calls `os.environ`, so the host file must import os.

    Patched code must never assume the host's imports — that assumption has cost a whole
    generation run in this repo before.
    """
    return "\nimport os\n" in source or source.startswith("import os\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text(encoding="utf-8")

    if args.check:
        print(("patched: " if is_patched(source) else "NOT PATCHED: ") + str(args.target))
        return 0 if is_patched(source) else 1

    if not uses_os_module(source):
        raise SystemExit(
            "generate_mps.py does not import os at module level; the injected condition "
            "would raise NameError at runtime."
        )

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched {args.target}: {ENV_VAR}=1 now installs the low-VRAM hooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
