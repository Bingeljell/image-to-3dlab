# Resume here — 2026-08-13 evening

This supersedes the earlier 13 August handovers for the remesh investigation. Read
`HANDOVER-remesh-cage.md` for eliminated hypotheses, but use this file for current state.

For the shorter review-ready conclusion, read
[`ROOT-CAUSE-trellis-mac-hollow-fox.md`](ROOT-CAUSE-trellis-mac-hollow-fox.md). This file remains
the detailed operational handover.

## Goal and acceptance test

The Hugging Face GLBs are capability controls, not pixel-perfect targets. A different seed
may change pose, proportions, colour and small details. Success means the result authentically
represents the source and remains visually solid from every angle with backface culling on.
No visible skull/body holes, exposed interior surfaces, cage, or large inverted regions.
Formal watertightness is diagnostic rather than the acceptance gate.

## What is proven

The raw Fox decode contains strong shape and material information, but that does not prove
that its topology matches CUDA or is engine-ready. The official demo also sends its decoded
mesh through `to_glb(remesh=True, remesh_band=1, remesh_project=0)`.

The Mac postprocessor definitely had a production-scale BVH traversal defect. The Metal BVH
used a 24-entry fixed stack. A four-way BVH over the 12,829,054-face Fox can require roughly
34 pending nodes, so the stack silently dropped live branches. Exact triangle centroids —
points which must have distance zero — measured p50 0.0030036, with 87.86% exceeding the
remesh epsilon 0.00097942. The earlier 82K-face sphere test was too shallow to expose it.

The unsigned-distance Metal kernel now uses the existing stackless escape-index traversal.
The other traversal paths use a 48-entry stack. On the full Fox, the same centroid invariant
now gives p50 4.06e-9, p99 2.65e-8, max 4.74e-8, and 0% over epsilon.

## Current local Metal changes

`vendor/` is git-ignored, so these are on disk but must be made into a reproducible patch
before any bootstrap:

- `mtlbvh/src/metal/bvh.metal`: stackless unsigned-distance traversal; `FixedStack48` for
  ray/signed/normal traversal.
- `mtlbvh/src/metal_bvh.mm`: local `@autoreleasepool` around CPU/unified-memory Metal command
  buffers used by chunked queries.
- `mtlbvh/bvh.py`: production queries chunked (default 65,536; test used 8,192).
- `mtlmesh/src/metal/remesh.metal`: guard `udf[0xFFFFFFFF]` on hashmap misses. This is a real
  OOB defect but was measured not to be the cage cause.
- `cumesh/metal_remeshing.py`: optional `I2L_REMESH_DIAG` tracing.
- `scripts/measure_bvh_on_surface.py`: production-scale regression test.

After rebuilding the native extension, explicitly ad-hoc sign both copies with
`codesign --force --sign -`. A linker-only signature passed `codesign --verify` but macOS
still killed Python at import with `CODESIGNING: Invalid Page`. The explicit signature fixed
the import. Do not overwrite a loaded `.so`; stop all Python workers first.

Current hashes after the successful run:

- `mtlbvh.metallib`: `31569460696a93754241dd96a0a3b9b6776655499adb9c75403ba2d060a64f5c`
- explicitly signed `mtlbvh/_C...so`: `97e02dea25b886731f12e18fb70379be712e4f40738c6e20e0b62dbc7e4e2bad`

## Successful isolated rebake

Only one process was run. Four earlier apparent failures were confounded by orphaned rebakes
whose session handles had been discarded. The isolated run completed in 412.3 seconds:

`output/bvh_stack_fix/fox_remesh_fixed.glb`

Important intermediate facts:

- all 23,367,184 final sparse-grid distance queries completed;
- BVH hashmap lookups missed 0 of 51,185,944;
- 12,796,486 of 12,796,486 candidate quads survived;
- after remesh: 12,679,082 vertices / 25,592,972 faces;
- final GLB: 588,061 stored vertices / 447,800 faces.

## Checkpoint result: two topology defects localized

Exact tensor checkpoints were captured from one cached Fox rebake. Position welding and
geometric face deduplication proved that dual contouring itself produced a closed shell:

| stage | boundary/open edges | important observation |
|---|---:|---|
| immediately after remesh | 0 | extraction is geometrically closed |
| after `repair_non_manifold_edges` | 0 | vertex splitting preserves geometric closure |
| after `remove_small_connected_components` | 0 | still closed |
| after the old pre-simplify `fill_holes` | 77,358 | this call creates the damage |
| simplified without that fill | 3,407 | QEM leaves smaller torn fragments |
| post-simplify cleanup, without fill | 408 | better than the HF control's measured 690 |
| after UV/export duplication | 408 | UV/export adds no openings |

The safe Metal order is therefore:

1. remove duplicate faces, repair non-manifold edges and remove small components;
2. do **not** run `fill_holes` on the repair-split pre-simplify mesh;
3. weld, simplify, then repeat duplicate/non-manifold/small-component cleanup;
4. again do **not** run Metal `fill_holes`.

`scripts/patch_ovoxel_skip_spurious_fill.py` reproduces this order. The final geometry
checkpoint has 484,826 faces, 408 boundary edges, 3,960 non-manifold edges, no geometric
duplicate faces and consistent winding.

## Decisive visual defect: wrong glTF transparency

The topology improvement was real, but the close rear render revealed the decisive reason
the textured Fox appeared hollow while flat grey and normals looked good. The Mac exporter
inspected the minimum baked alpha value and emitted:

- local: `alphaMode=BLEND`, `doubleSided=true`;
- HF control: `alphaMode=OPAQUE`, `doubleSided=false`.

Both base-colour textures contain alpha-zero texels. They are UV padding/unsampled texels,
not proof that the reconstructed object is meant to be transparent. `BLEND` made the opaque
rear shell reveal the eyes and mouth behind it, producing exactly the false front-facing Fox
reported by the user.

This was proven with the *same final geometry and same rear camera* side-by-side: the opaque
copy shows the back of the head; the blend copy shows the face through it. No inference is
involved in that comparison.

The exporter now always matches the official demo's opaque, single-sided material policy.
`scripts/patch_ovoxel_opaque_material.py` reproduces the source/runtime patch, and
`scripts/fix_glb_opaque_material.py` repairs already-generated GLBs without a rebake.

## Current Fox result

Use:

`output/final_cleanup_fix/fox_fixed_opaque.glb`

Do not use `fox_fixed.glb`; it has the corrected geometry but intentionally retains the old
transparent flags as the A/B control. The fixed result was inspected closely from the true
rear with backface culling enabled. It shows the solid rear head rather than the front face.

Checkpoint tooling remains available through `scripts/patch_ovoxel_remesh_checkpoints.py`
and `scripts/mesh_checkpoint_forensics.py`. No further Fox GPU run is needed. The next useful
work is to run the same opaque/material and culled-rear acceptance check on Snag and the other
previously broken inputs using their cached decodes where available.
