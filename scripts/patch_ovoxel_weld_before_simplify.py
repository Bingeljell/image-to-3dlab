#!/usr/bin/env python3
"""Weld coincident vertices before every `simplify()` in o_voxel's `to_glb`.

**The defect.** `o_voxel/postprocess.py` calls `repair_non_manifold_edges()` immediately
before each `simplify()` (lines ~225/232 and ~269/276). That repair works by splitting
vertices -- cumesh's own docstring: *"Repair Non-manifold edges by splitting vertices.
This creates duplicate vertices with the same coordinates."* QEM edge collapse cannot
collapse across a duplicate pair, so the simplifier tears the surface open around every
seam the repair just introduced.

Measured on the thorn-knot Snag: entering the final simplify the mesh sits at 7.8% of
faces touching a boundary; one call takes it to **44.7%**. The identical call on a welded
mesh gives 8.4%. This is what made every downstream repair futile -- Solidify, welding,
voxel remeshing, visibility culling -- because all of them were working on a mesh that had
been shredded by a step we control.

**Why it hid for two days:** our own tear metric welds before measuring, so it stitched
the cuts back together and reported the mesh healthy at every step until the very end.

The weld is exact, not tolerance-based. The duplicates are bit-identical because they were
made by splitting one point, so exact matching finds them precisely -- and a tolerance
weld would additionally fuse genuinely distinct nearby surfaces, which is the adjacent-coil
fusing that made voxel remeshing unusable here.

Safe to run repeatedly; it detects its own marker and does nothing on a second run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = "_i2l_weld"

DEFAULT_TARGET = Path(
    "vendor/trellis-mac/.venv/lib/python3.11/site-packages/o_voxel/postprocess.py"
)

_HELPER = '''

# --- image-to-3dlab: weld before simplify ------------------------------------------
# repair_non_manifold_edges() splits vertices, creating exact duplicates. QEM collapse
# cannot cross them, so simplify() tears the surface apart at every seam. Weld first.
def {marker}(mesh):
    import sys

    import torch

    repo = {repo!r}
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.mesh_weld import duplicate_count, weld_vertices

    vertices, faces = mesh.read()
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    duplicates = duplicate_count(v)
    if duplicates == 0:
        return
    welded_v, welded_f = weld_vertices(v, f)
    mesh.init(
        torch.as_tensor(welded_v, dtype=torch.float32).contiguous(),
        torch.as_tensor(welded_f, dtype=torch.int32).contiguous(),
    )
    print(
        f"  [image-to-3dlab] welded {{duplicates}} duplicate vertices "
        f"({{len(f)}} -> {{len(welded_f)}} faces) before simplify"
    )
# --- end image-to-3dlab -------------------------------------------------------------
'''

_SIMPLIFY = re.compile(r"^(\s*)mesh\.simplify\(", re.MULTILINE)


def already_patched(source: str) -> bool:
    return MARKER in source


def apply_patch(source: str, repo: str) -> str:
    """Insert the helper and a call to it before every `mesh.simplify(` line."""
    if already_patched(source):
        return source

    calls = _SIMPLIFY.findall(source)
    if not calls:
        raise RuntimeError(
            "anchor not found: no `mesh.simplify(` call in the target. The vendored "
            "o_voxel has changed shape; re-read postprocess.py before patching."
        )

    patched = _SIMPLIFY.sub(
        lambda m: f"{m.group(1)}{MARKER}(mesh)\n{m.group(1)}mesh.simplify(", source
    )

    # Put the helper after the import block so it is defined before any call site.
    lines = patched.splitlines(keepends=True)
    last_import = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            last_import = i
        if line.lstrip().startswith("def "):
            break
    helper = _HELPER.format(marker=MARKER, repo=repo)
    lines.insert(last_import + 1, helper)
    return "".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--revert", action="store_true", help="restore the .orig backup")
    args = p.parse_args()

    target = args.target.expanduser().resolve()
    backup = target.with_suffix(target.suffix + ".i2l-orig")

    if args.revert:
        if not backup.is_file():
            raise SystemExit(f"no backup at {backup}")
        target.write_text(backup.read_text())
        print(f"reverted {target} from {backup}")
        return 0

    if not target.is_file():
        raise SystemExit(f"target not found: {target}")
    source = target.read_text()
    if already_patched(source):
        print(f"already patched: {target}")
        return 0

    if not backup.is_file():
        backup.write_text(source)
    target.write_text(apply_patch(source, str(args.repo.resolve())))
    print(f"patched {target}\nbackup at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
