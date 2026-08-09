#!/usr/bin/env python3
"""Let `o_voxel.postprocess.to_glb` forward xatlas packing options.

`cumesh`'s `MtlMesh.uv_unwrap` already accepts `xatlas_pack_charts_kwargs`, and
`xatlas.PackOptions` already has a `brute_force` flag. `to_glb` sits between them and
passes neither, so the flag is unreachable from the generator.

It is worth reaching. Measured on the hero fox through this exact path — cumesh chart
clustering, then `xatlas.pack_charts` — `brute_force=True` moves atlas coverage from
52.90% to 58.76%: an 11% relative gain from one flag, with no geometry change and no
regeneration. Every other packing and chart-clustering knob tested was noise. xatlas
documents the option as "slower"; on 101k faces it cost one second.

This patch is additive and defaults to the previous behaviour: with no
`xatlas_pack_charts_kwargs` argument, `to_glb` calls `uv_unwrap` exactly as before.

Note this covers the Metal path only. `postprocess_cpu.to_glb` calls
`xatlas.parametrize` directly and takes no packing options; the CPU fallback keeps
default packing.

The target lives in the TRELLIS venv's site-packages, so reinstalling `o_voxel` reverts
it — re-run this script after any reinstall. Re-running is idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "vendor" / "trellis-mac" / ".venv"

MARKER = "image-to-3dlab: forward xatlas packing options"

SIGNATURE_NEEDLE = """    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
"""

SIGNATURE_REPLACEMENT = """    mesh_cluster_smooth_strength=1,
    # image-to-3dlab: forward xatlas packing options. None keeps xatlas' defaults.
    xatlas_pack_charts_kwargs: dict = None,
    verbose: bool = False,
"""

CALL_NEEDLE = """        return_vmaps=True,
        verbose=verbose,
    )"""

CALL_REPLACEMENT = """        xatlas_pack_charts_kwargs=dict(xatlas_pack_charts_kwargs or {}),
        return_vmaps=True,
        verbose=verbose,
    )"""


def find_postprocess(venv: Path) -> Path:
    """Locate the installed `o_voxel/postprocess.py` under a venv's site-packages."""
    matches = sorted(venv.glob("lib/python*/site-packages/o_voxel/postprocess.py"))
    if not matches:
        raise RuntimeError(
            f"o_voxel is not installed under {venv}. "
            "Run scripts/bootstrap_trellis_macos.sh first."
        )
    return matches[0]


def patch(path: Path) -> None:
    source = path.read_text()

    if MARKER in source:
        return

    if SIGNATURE_NEEDLE not in source:
        raise RuntimeError(f"expected to_glb signature not found in {path}")
    if CALL_NEEDLE not in source:
        raise RuntimeError(f"expected uv_unwrap call not found in {path}")

    source = source.replace(SIGNATURE_NEEDLE, SIGNATURE_REPLACEMENT, 1)
    source = source.replace(CALL_NEEDLE, CALL_REPLACEMENT, 1)
    path.write_text(source)


def main() -> int:
    target = find_postprocess(VENV)
    patch(target)
    print(f"Patched {target} to forward xatlas packing options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
