#!/usr/bin/env python3
"""Re-bake a GLB from a cached decode, without re-sampling or loading the model.

Run with the TRELLIS venv, since it needs `o_voxel`:

    vendor/trellis-mac/.venv/bin/python scripts/trellis_rebake.py decode.pt out.glb \\
        --remesh --decimation-target 500000 --texture-size 3072

**Why.** A full generation is ~20 minutes and only the last ~4 are `to_glb`. Every open
question is a `to_glb` question — remesh on or off, `decimation_target` in the units it
actually wants (vertices, not faces), texture size, Branch 1 versus Branch 2. This turns
each of those from a 20-minute round trip into a few minutes, which is the difference
between testing one hypothesis an hour and testing ten.

Mirrors the official demo's `extract_glb` (`TRELLIS.2/app.py:472`), which is why its
sliders re-extract instantly. Defaults here follow that function exactly — `remesh=True`,
`remesh_band=1`, `remesh_project=0`, `extension_webp=True` — so `--remesh` reproduces the
reference path rather than our own.

Produce the input with `generate.py --dump-decode decode.pt`
(see `scripts/patch_trellis_dump_decode.py`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _geometry_ref_path(cache_path: Path, reference: str) -> Path:
    """Resolve a split material cache's geometry reference portably."""
    path = Path(reference)
    if path.is_absolute():
        return path
    repo_relative = REPO / path
    if repo_relative.is_file():
        return repo_relative
    return cache_path.parent / path


def load_payload(path: Path, torch_module) -> dict:
    """Load either a traditional full decode or a split Stage-3 material cache.

    Stage-3 candidates share hundreds of megabytes of vertices and faces. A split cache
    stores only the new attrs/coords and points at the original geometry decode.
    """
    payload = torch_module.load(path, weights_only=False)
    reference = payload.get("geometry_ref")
    if reference is None:
        return payload

    geometry_path = _geometry_ref_path(path, reference)
    if not geometry_path.is_file():
        raise FileNotFoundError(
            f"geometry_ref {reference!r} from {path} resolved to missing {geometry_path}"
        )
    geometry = torch_module.load(geometry_path, weights_only=False)
    merged = dict(geometry)
    merged.update({key: value for key, value in payload.items() if key != "geometry_ref"})
    return merged


def build_to_glb_kwargs(payload: dict, args: argparse.Namespace) -> dict:
    """Assemble the `to_glb` call. Pure, so the argument mapping is testable.

    `decimation_target` is passed through untouched and documented as **vertices**: the
    upstream docstring says "target number of vertices for mesh simplification", while the
    port's own flag is named `--bake-target-faces` and feeds a face count straight into it.
    That unit confusion propagated into our manifests; this script does not repeat it.
    """
    return {
        "vertices": payload["vertices"],
        "faces": payload["faces"],
        "attr_volume": payload["attrs"],
        "coords": payload["coords"],
        "attr_layout": payload["layout"],
        "voxel_size": payload["voxel_size"],
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "remesh": args.remesh,
        "remesh_band": args.remesh_band,
        "remesh_project": args.remesh_project,
        "verbose": True,
    }


def summarise(payload: dict) -> str:
    return (
        f"{payload['vertices'].shape[0]:,} vertices, "
        f"{payload['faces'].shape[0]:,} faces"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decode", type=Path, help="cached decode from --dump-decode")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument(
        "--remesh", action="store_true",
        help="use to_glb's Branch 2 (dual-contouring rebuild). The reference demo has this "
             "ON and hardcoded; Branch 1 (clean+simplify) is what we have always run",
    )
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.0)
    parser.add_argument(
        "--decimation-target", type=int, default=500_000,
        help="target VERTICES (not faces). Demo slider default is 500000",
    )
    parser.add_argument("--texture-size", type=int, default=3072)
    parser.add_argument(
        "--no-webp", action="store_true",
        help="skip WebP texture compression. The reference exports with it, and its "
             "EXT_texture_webp is how we identified which pipeline made the controls",
    )
    args = parser.parse_args()

    if not args.decode.is_file():
        parser.error(f"{args.decode} does not exist — run generate.py --dump-decode first")

    import o_voxel
    import torch

    payload = load_payload(args.decode, torch)
    print(f"Loaded decode: {summarise(payload)}")
    print(f"Branch: {'2 (remesh/DC rebuild)' if args.remesh else '1 (clean + simplify)'}")

    started = time.time()
    glb = o_voxel.postprocess.to_glb(**build_to_glb_kwargs(payload, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(args.output), extension_webp=not args.no_webp)
    print(f"\nWrote {args.output} in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
