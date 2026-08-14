# TRELLIS.2 upstream-first Mac port audit

**Started:** 2026-08-14  
**Branch:** `audit/upstream-rebase`  
**Worktree:** `vendor/upstream-audit-worktree`

This worktree exists so the working `feat/tear-provenance` branch, its ignored model
artifacts, and its modified Metal installation remain untouched. Do not develop the audit
inside the original checkout.

## Objective

Reconstruct the Mac port from a pinned Microsoft TRELLIS.2 checkout, then replay Pedro,
Shiv, and image-to-3dlab changes as distinct, reviewable layers. Preserve upstream inference
semantics unless a Metal compatibility requirement is demonstrated by a test.

This is not a blind update to the newest upstream revision. The first target is the exact
Microsoft commit used by the successful local installation. Moving that pin happens only
after the reconstructed build matches the current Fox, Forest Flicker, and Snag baselines.

## Source graph

```text
Microsoft TRELLIS.2 @ 75fbf018
        |
        +-- Shiv MPS source patcher and generation harness @ 0b8efd4
        |
        +-- Pedro Metal backends
              trellis2-apple @ 6055b868
              mtlbvh        @ 23f441c
              mtldiffrast   @ 4668cd9
              mtlgemm       @ 867aec8
              mtlmesh       @ 212079e
        |
        +-- image-to-3dlab fixes, cached experiment lanes, and tests
```

Exact URLs and full hashes are recorded in
`audit/trellis-port/upstreams.lock.json`.

## Initial evidence

The installed `TRELLIS.2` directory is itself a Microsoft clone at `75fbf018`; Shiv's setup
does not replace the complete core with Pedro's fork. It applies `patches/mps_compat.py` to
Microsoft and installs Pedro's repositories as backend dependencies.

Current working deltas from their respective clean commits:

| layer | changed source files | additions | deletions |
|---|---:|---:|---:|
| Microsoft core after all local patches | 9 | 176 | 28 |
| Shiv harness `generate.py` after local patches | 1 | 167 | 11 |
| Pedro `o_voxel/postprocess.py` after local patches | 1 | 47 | 11 |
| Pedro `mtlbvh` after local patches | 3 | 79 | 21 |
| Pedro `mtlmesh` after local patches | 2 | 36 | 0 |
| Pedro `mtldiffrast` | 0 | 0 | 0 |
| Pedro `mtlgemm` | 0 | 0 | 0 |

A clean replay of Shiv's `mps_compat.py` against the pinned Microsoft checkout isolates the
additional TRELLIS-core delta to:

- `trellis2/pipelines/trellis2_image_to_3d.py`
- `trellis2/pipelines/trellis2_texturing.py`
- `trellis2/representations/mesh/base.py`

The first two contain local background/multiview behavior. The last re-enables decoder mesh
cleanup on the Metal backend. These must be reviewed independently from Shiv's MPS attention,
DINO device-routing, sparse-convolution, and mesh-extraction compatibility changes.

## Patch-layer policy

1. **Upstream inference contract** — image preprocessing, DINO features, sampler equations,
   normalization, latent layouts, and decoder equations. Changes here require numerical
   parity tests.
2. **MPS compatibility** — device routing, supported dtypes, sparse attention/convolution,
   memory release, and unsupported CUDA calls. Changes must preserve the upstream equation.
3. **Metal geometry/export** — BVH, remesh, simplification, UV unwrap, attribute sampling,
   opacity, and GLB packaging. These cannot be allowed to alter Stage-1/2/3 latents.
4. **Harness/product layer** — CLI flags, checkpoints, manifests, viewers, diagnostics, and
   cached Stage-3/rebake lanes. This layer should not patch model internals.

## Validation gates

The audit branch began clean at `2866d02`. Baseline test result:

```text
434 passed, 2 skipped
```

The reconstructed port must then pass, in order:

1. patch application and import tests from a fresh checkout;
2. DINO preprocessing/conditioning determinism on the three golden source PNGs;
3. sparse-attention equivalence tests on single-batch and padded multi-batch inputs;
4. Stage-3 sampler/default/latent schema contract tests;
5. cached decode → GLB Fox topology and backface-culling checks;
6. cached Snag material → GLB channel and spatial-colour checks;
7. one fresh end-to-end golden run only after all cheaper gates pass.

## Current verdict

An upstream-first reconstruction is feasible and preferable to continuing with untracked
in-place patches. It is not yet safe to claim that a fresh upstream checkout fixes material
fidelity: Shiv's SDPA attention replacement participates directly in Stage 3 and needs a
numerical equivalence test, while the observed brown/green variation may still be normal
sampling variance.
