#!/usr/bin/env python3
"""Add opt-in exact geometry checkpoints to o_voxel's remesh branch.

Set `I2L_CHECKPOINT_PREFIX=output/checkpoints/fox` on a rebake to produce:

- `fox_after_remesh.pt`
- `fox_after_cleanup.pt`
- `fox_after_simplify.pt`
- `fox_after_uv.pt`

Normal runs are unchanged. Each file contains CPU `vertices` and `faces` tensors. The
checkpoints localize where visible culled-shell damage first appears without rerunning
diffusion. `vendor/` is ignored, so reapply this after bootstrap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    REPO / "vendor/trellis-mac/deps/trellis2-apple/o-voxel/o_voxel/postprocess.py",
    REPO / "vendor/trellis-mac/.venv/lib/python3.11/site-packages/o_voxel/postprocess.py",
)
MARKER = "from ._i2l_trellis_export import save_checkpoint as _i2l_save_checkpoint"
HELPER_SOURCE = Path(__file__).with_name("trellis_export_compat.py")
HELPER_NAME = "_i2l_trellis_export.py"
HELPER_ANCHOR = "# Mesh processing — cumesh auto-selects Metal/CUDA\n"

LEGACY_HELPER = '''

def _i2l_save_checkpoint(tag, mesh=None, vertices=None, faces=None):
    """Persist exact pre-export geometry when I2L_CHECKPOINT_PREFIX is set."""
    import os
    from pathlib import Path

    prefix = os.environ.get("I2L_CHECKPOINT_PREFIX")
    if not prefix:
        return
    if mesh is not None:
        vertices, faces = mesh.read()
    if vertices is None or faces is None:
        raise RuntimeError(f"checkpoint {tag} has no geometry")
    path = Path(f"{prefix}_{tag}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vertices": vertices.detach().cpu().contiguous(),
            "faces": faces.detach().cpu().contiguous(),
        },
        path,
    )
    print(f"[i2l] checkpoint {tag}: {path}", flush=True)
'''

HELPER = f"\n{MARKER}\n"

CALLS = (
    (
        '            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")\n',
        '        _i2l_save_checkpoint("after_remesh", mesh=mesh)\n',
    ),
    (
        '            print(f"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")\n',
        '        _i2l_save_checkpoint("after_cleanup", mesh=mesh)\n',
    ),
    (
        '            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")\n',
        '        _i2l_save_checkpoint("after_simplify", mesh=mesh)\n',
    ),
    (
        "    out_vmaps = out_vmaps.to(device)\n",
        '    if remesh:\n        _i2l_save_checkpoint("after_uv", vertices=out_vertices, faces=out_faces)\n',
    ),
)


def patch(source: str) -> str:
    if LEGACY_HELPER in source:
        return source.replace(LEGACY_HELPER, HELPER, 1)
    if MARKER in source:
        return source
    if HELPER_ANCHOR not in source:
        raise RuntimeError("mesh backend anchor missing")
    source = source.replace(HELPER_ANCHOR, HELPER + "\n" + HELPER_ANCHOR, 1)
    for anchor, addition in CALLS:
        if anchor not in source:
            raise RuntimeError(f"postprocess anchor missing: {anchor.strip()}")
        source = source.replace(anchor, anchor + addition, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path, action="append")
    args = parser.parse_args()
    targets = tuple(args.target or DEFAULT_TARGETS)

    states = []
    payload = HELPER_SOURCE.read_bytes()
    for target in targets:
        if not target.is_file():
            states.append(f"MISSING {target}")
            continue
        source = target.read_text()
        helper = target.with_name(HELPER_NAME)
        helper_current = helper.is_file()
        if MARKER in source and helper_current:
            states.append(f"APPLIED {target}")
        elif args.check:
            status = "APPLIED" if LEGACY_HELPER in source else "ABSENT"
            states.append(f"{status} {target}")
        else:
            patched = patch(source)
            if not helper_current:
                helper.write_bytes(payload)
            if patched != source:
                target.write_text(patched)
            states.append(f"PATCHED {target}")
    print("\n".join(states))
    return 0 if all(not state.startswith(("MISSING", "ABSENT")) for state in states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
