#!/usr/bin/env python3
"""Isolate why Metal narrow-band DC remeshing emits a wireframe instead of a surface.

Run with the TRELLIS venv (needs `cumesh`):

    vendor/trellis-mac/.venv/bin/python scripts/diagnose_remesh_kernel.py

**The symptom.** `remesh=True` produces a cage: correct silhouette, correct vertex
placement, no faces between the struts. Signed volume collapses to ~0 (-0.00028 on the
moss fox against a healthy +0.0052).

**RESOLVED 2026-08-13 — it is the GPU watchdog, not hashmap logic.** `generate.py`'s own
`_watchdog_help_message` says it: *"the macOS GPU watchdog killing a long-running Metal
kernel... The Metal error prints to stderr but does not raise a Python exception, so
execution continues with empty tensors."* A killed lookup dispatch leaves the output at the
`0xffffffff` sentinel `_init_hashmap` pre-fills, so every quad fails the validity test and
only the struts survive. Running these tests took the machine down via a WindowServer
watchdog cascade (IOGPU/AGX in the crash report). **Run headless, or with
`MTL_CAPTURE_ENABLED=1`, which extends the watchdog.**

Kept because the round-trip harness is still the right way to confirm a fix, and because
the two hangs it produced (0% CPU, no output, no error) are the watchdog's signature.

**Why the hashmap was the suspect.** `metal_remeshing.py:169` reads

    connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)

so a quad is emitted only when all four neighbouring voxels resolve in the hash map. A
lookup that wrongly reports "not found" drops the quad but keeps its edges — which is
precisely a wireframe of the right shape.

Two tests, cheapest first:

1. **Hashmap round-trip.** Insert known coordinates, look those exact coordinates back up,
   count misses. A correct hashmap returns every key it was given. Any miss here is the bug.
2. **Minimal repro.** Remesh an analytic sphere. If a sphere also cages, the defect is
   independent of our meshes and the repro fits in a bug report.

Both are bounded and printed as numbers rather than judged by eye.
"""

from __future__ import annotations

import argparse
import sys


def summarise_lookup(found, expected_count: int) -> dict:
    """Miss statistics for a hashmap round-trip.

    `0xffffffff` is the sentinel `metal_remeshing.py` tests against, so it is the value
    that matters rather than any negative convention.
    """
    import torch

    missing = int((found == 0xFFFFFFFF).sum().item())
    negative = int((found.to(torch.int64) < 0).sum().item())
    return {
        "queried": expected_count,
        "missing": missing,
        "negative": negative,
        "miss_rate": missing / expected_count if expected_count else 0.0,
    }


def quad_survival(miss_rate: float) -> float:
    """Share of quads surviving, if four independent lookups must all succeed.

    Included because the intuition is badly wrong: a 5% per-lookup miss rate leaves 81% of
    quads, but 30% leaves 24% — a mesh that reads as a cage. It converts a lookup number
    into the thing actually observed.
    """
    return (1.0 - miss_rate) ** 4


def test_hashmap(resolution: int, count: int) -> dict:
    import torch
    from cumesh import _C
    from cumesh.metal_remeshing import _init_hashmap

    device = torch.device("mps")
    generator = torch.Generator(device="cpu").manual_seed(0)
    # Deduplicate on CPU. torch.unique(dim=0) on MPS stalls indefinitely at this size --
    # 0% CPU, no output, no error -- which is a hang in the harness, not in the kernel.
    coords = torch.unique(
        torch.randint(0, resolution, (count, 3), dtype=torch.int32, generator=generator),
        dim=0,
    ).to(device)
    n = coords.shape[0]
    print(f"  [prepared {n:,} unique keys]", flush=True)

    batched = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    # Same allocation the real path uses: capacity 2*N, keys pre-filled with the sentinel.
    hashmap = _init_hashmap(resolution, 2 * n, device)
    _C.hashmap_insert_3d_idx_as_val(*hashmap, batched, resolution, resolution, resolution)
    print('  [inserted]', flush=True)

    found = _C.hashmap_lookup_3d(*hashmap, batched, resolution, resolution, resolution)
    print('  [looked up]', flush=True)
    stats = summarise_lookup(found, n)
    stats["resolution"] = resolution
    stats["quad_survival"] = quad_survival(stats["miss_rate"])

    # Round-tripping the keys must also return the right index, not merely "present".
    if stats["missing"] == 0:
        expected = torch.arange(n, device=found.device, dtype=found.dtype)
        stats["wrong_index"] = int((found != expected).sum().item())
    return stats


def test_sphere(resolution: int) -> dict:
    """Remesh an analytic sphere. A sphere is closed, convex and trivially manifold."""
    import torch
    import trimesh
    from cumesh.metal_remeshing import remesh_narrow_band_dc

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    vertices = torch.tensor(sphere.vertices, dtype=torch.float32, device="mps")
    faces = torch.tensor(sphere.faces, dtype=torch.int32, device="mps")

    out_v, out_f = remesh_narrow_band_dc(
        vertices, faces,
        center=torch.zeros(3, device="mps"),
        scale=1.2, resolution=resolution, band=1, project_back=0.0, verbose=False,
    )
    result = trimesh.Trimesh(
        vertices=out_v.cpu().numpy(), faces=out_f.cpu().numpy(), process=False
    )
    return {
        "in_faces": len(sphere.faces),
        "out_vertices": len(result.vertices),
        "out_faces": len(result.faces),
        "volume": float(result.volume),
        "expected_volume": float(sphere.volume),
        "watertight": bool(result.is_watertight),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--keys", type=int, default=200_000)
    parser.add_argument("--skip-sphere", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST 1 — hashmap round-trip (insert known keys, look the same keys up)")
    print("=" * 70)
    try:
        stats = test_hashmap(args.resolution, args.keys)
        for key, value in stats.items():
            print(f"  {key:16s} {value}")
        if stats["missing"]:
            print(f"\n  ** {stats['miss_rate']:.1%} of keys not found. With four lookups per")
            print(f"  ** quad that leaves {stats['quad_survival']:.1%} of faces — a wireframe.")
        else:
            print("\n  Hashmap is clean. The defect is downstream: look at "
                  "simple_dual_contour's out_intersected.")
    except Exception as exc:  # noqa: BLE001 - a crash here is itself the finding
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    if args.skip_sphere:
        return 0

    print()
    print("=" * 70)
    print("TEST 2 — minimal repro: remesh an analytic sphere")
    print("=" * 70)
    try:
        stats = test_sphere(args.resolution)
        for key, value in stats.items():
            print(f"  {key:16s} {value}")
        ratio = stats["volume"] / stats["expected_volume"] if stats["expected_volume"] else 0
        print(f"\n  volume ratio: {ratio:.4f}  (1.0 = correct solid, ~0 = cage)")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
