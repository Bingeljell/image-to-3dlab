#!/usr/bin/env python3
"""Use the measured-safe cleanup order for Metal remesh output."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    REPO / "vendor/trellis-mac/deps/trellis2-apple/o-voxel/o_voxel/postprocess.py",
    REPO / "vendor/trellis-mac/.venv/lib/python3.11/site-packages/o_voxel/postprocess.py",
)
ANCHOR = """        # Clean up topology before simplification
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f\"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces\")
"""
PATCHED = """        # Clean up topology before simplification
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        # The Metal repair step splits non-manifold seams into coincident boundary
        # vertices. Geometrically the shell is already closed; filling those
        # topological seams adds overlapping caps and creates visible openings.
        # Exact-position analysis: 0 boundary edges here, 77,358 after fill_holes.
        if verbose:
            print(f\"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces\")
"""
POST_MARKER = '_i2l_save_checkpoint("after_simplify_cleanup", mesh=mesh)'
POST_CLEANUP = """

        # Simplification can leave small torn fragments around former non-manifold
        # seams. Remove those fragments, but do not call Metal fill_holes: that call
        # caps coincident seam boundaries and makes the geometric topology worse.
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        _i2l_save_checkpoint("after_simplify_cleanup", mesh=mesh)
"""
CHECKPOINT_ANCHOR = '        _i2l_save_checkpoint("after_simplify", mesh=mesh)\n'
DIAG_ANCHOR = """        if _os.environ.get("I2L_REMESH_DIAG"):
            _i2l_vol("AFTER SIMPLIFY")
"""


def patch(source: str) -> str:
    if PATCHED not in source:
        if ANCHOR not in source:
            raise RuntimeError("remesh cleanup anchor missing")
        source = source.replace(ANCHOR, PATCHED, 1)
    if POST_MARKER not in source:
        anchor = DIAG_ANCHOR if DIAG_ANCHOR in source else CHECKPOINT_ANCHOR
        if anchor not in source:
            raise RuntimeError("post-simplify anchor missing")
        source = source.replace(anchor, anchor + POST_CLEANUP, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path, action="append")
    args = parser.parse_args()

    states = []
    for target in tuple(args.target or DEFAULT_TARGETS):
        if not target.is_file():
            states.append(f"MISSING {target}")
            continue
        source = target.read_text()
        if PATCHED in source and POST_MARKER in source:
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
