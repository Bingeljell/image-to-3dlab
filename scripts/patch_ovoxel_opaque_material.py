#!/usr/bin/env python3
"""Match the official TRELLIS GLB's opaque, single-sided material flags."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    REPO / "vendor/trellis-mac/deps/trellis2-apple/o-voxel/o_voxel/postprocess.py",
    REPO / "vendor/trellis-mac/.venv/lib/python3.11/site-packages/o_voxel/postprocess.py",
)
OLD_ALPHA = """    # Auto-detect transparency from baked alpha values
    alpha_valid = alpha[mask]
    if alpha_valid.size > 0 and alpha_valid.min() < 250:
        alpha_mode = 'BLEND'
        if verbose:
            print(f"Detected transparency (alpha min={alpha_valid.min()}), using BLEND mode")
    else:
        alpha_mode = 'OPAQUE'
"""
NEW_ALPHA = """    # TRELLIS demo exports these reconstructed objects as opaque. Texture padding and
    # unsampled texels legitimately contain alpha=0; treating their minimum as evidence
    # of a transparent material makes the solid rear shell reveal interior surfaces.
    alpha_mode = 'OPAQUE'
"""


def patch(source: str) -> str:
    if NEW_ALPHA not in source:
        if OLD_ALPHA not in source:
            raise RuntimeError("alpha mode anchor missing")
        source = source.replace(OLD_ALPHA, NEW_ALPHA, 1)
    source = source.replace("        doubleSided=True,\n", "        doubleSided=False,\n", 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path, action="append")
    args = parser.parse_args()

    states = []
    for target in tuple(args.target or DEFAULT_TARGETS):
        if not target.is_file():
            states.append(f"MISSING {target}")
            continue
        source = target.read_text()
        applied = NEW_ALPHA in source and "        doubleSided=False,\n" in source
        if applied:
            states.append(f"APPLIED {target}")
        elif args.check:
            states.append(f"ABSENT {target}")
        else:
            target.write_text(patch(source))
            states.append(f"PATCHED {target}")
    print("\n".join(states))
    return 0 if all(state.startswith(("APPLIED", "PATCHED")) for state in states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
