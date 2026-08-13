#!/usr/bin/env python3
"""Re-enable the decode-time mesh cleanup that `mps_compat.py` turns into no-ops.

    python scripts/patch_trellis_enable_cleanup.py            # enable
    python scripts/patch_trellis_enable_cleanup.py --check    # report, change nothing
    python scripts/patch_trellis_enable_cleanup.py --revert    # put the stubs back

**What is disabled.** `patches/mps_compat.py` inserts a bare `return` at the top of
`MeshBase.fill_holes`, `.remove_faces` and `.simplify`, with the note *"Metal cumesh
segfaults on large decode meshes."* The bodies are intact underneath; they are simply
unreachable. `o_voxel.postprocess.to_glb` calls these **eight times before UV unwrapping**
(four `fill_holes`, three `simplify`, one `remove_faces`), so every asset this repo has
produced was unwrapped and baked from an uncleaned mesh.

**Why it matters.** The Hugging Face reference assets have **1** boundary edge. Ours run
from 891 to 48,261 across 23 fox generations, and none has ever been closed. The reference
is the same documented flow with these methods alive.

**Why this is a real risk, not a formality.** The stub's stated reason may still hold: the
Metal `cumesh` port segfaulted on the ~400K-vertex decode mesh, and removing the 200k face
cap made those meshes *larger*, not smaller. A crash is a legitimate outcome and tells us
the pure-Python route is required instead. Run it and find out — but expect the crash to be
possible and do not treat a segfault as a surprise.

`vendor/` is git-ignored, so this must be re-applied after every bootstrap, exactly like
`patch_trellis_face_cap.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "vendor/trellis-mac/TRELLIS.2/trellis2/representations/mesh/base.py"

# (method signature, the stub line that follows it). The signatures are the anchors; if one
# is missing the port has changed shape and we must not guess.
STUBS = (
    (
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n",
        "        return  # Skip — Metal cumesh segfaults on large decode meshes\n",
    ),
    (
        "    def remove_faces(self, face_mask: torch.Tensor):\n",
        "        return\n",
    ),
    (
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n",
        "        return\n",
    ),
)

MARKER = "        # image-to-3dlab: cleanup re-enabled by patch_trellis_enable_cleanup.py\n"


def find_stubs(source: str) -> list[str]:
    """Which methods are currently stubbed. Empty means cleanup is live."""
    stubbed = []
    for signature, stub in STUBS:
        if signature not in source:
            raise RuntimeError(
                f"anchor missing from the port: {signature.strip()!r}. "
                "The vendored file has changed shape; inspect it before patching."
            )
        if source.split(signature, 1)[1].startswith(stub):
            stubbed.append(signature.strip())
    return stubbed


def enable(source: str) -> tuple[str, list[str]]:
    """Remove the early `return` from each stubbed method. Idempotent."""
    changed = []
    for signature, stub in STUBS:
        head, tail = source.split(signature, 1)
        if tail.startswith(stub):
            source = head + signature + MARKER + tail[len(stub):]
            changed.append(signature.strip())
    return source, changed


def revert(source: str) -> tuple[str, list[str]]:
    """Put the stubs back, so a segfault can be undone without a re-bootstrap."""
    changed = []
    for signature, stub in STUBS:
        head, tail = source.split(signature, 1)
        if tail.startswith(MARKER):
            source = head + signature + stub + tail[len(MARKER):]
            changed.append(signature.strip())
    return source, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report state, change nothing")
    parser.add_argument("--revert", action="store_true", help="restore the stubs")
    parser.add_argument("--path", type=Path, default=TARGET)
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"not found: {args.path}\nRun the TRELLIS bootstrap first.", file=sys.stderr)
        return 2

    source = args.path.read_text()
    stubbed = find_stubs(source)

    if args.check:
        if stubbed:
            print(f"STUBBED (cleanup disabled): {', '.join(stubbed)}")
        else:
            print("ENABLED (cleanup runs)")
        return 0

    updated, changed = (revert if args.revert else enable)(source)
    if not changed:
        print("already " + ("stubbed" if args.revert else "enabled") + " — no change")
        return 0
    args.path.write_text(updated)
    verb = "re-stubbed" if args.revert else "enabled"
    print(f"{verb}: {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
