# TRELLIS.2 upstream-first Mac port audit

**Started:** 2026-08-14  
**Branch:** `audit/upstream-rebase`  
**Worktree:** `vendor/upstream-audit-worktree`

This worktree exists so the working `feat/tear-provenance` branch, its ignored model
artifacts, and its modified Metal installation remain untouched. Do not develop the audit
inside the original checkout.

## Objective

Reconstruct the Mac port directly from the source snapshot used by Microsoft's official
Hugging Face Space, then add only audited device/backend substitutions. Pedro and Shiv are
valuable prior work, but neither is the semantic foundation: Pedro's repositories provide
Metal implementations and Shiv's patcher is a compatibility inventory to review item by
item. Preserve official inference semantics unless a Metal requirement is demonstrated by
a test.

This is not a blind update to the newest GitHub revision. The first parity target is the
pinned Space snapshot, because that is the executable implementation behind the user's
control path. Microsoft's GitHub tree is tracked separately as an official library and
training-code comparison. A pin moves only after the reconstructed build matches the Fox,
Forest Flicker, and Snag contracts.

## Control specification

| priority | source | pin | role |
|---:|---|---|---|
| 1 | Microsoft Hugging Face Space | `ebf60b20` | executable demo behavior |
| 2 | Microsoft GitHub repository | `75fbf018` | official library and training-code comparison |
| 3 | Pedro Metal repositories | see lock file | required Metal backends/reimplementations |
| 4 | Shiv `trellis-mac` | `0b8efd4` | patch inventory and harness reference only |
| 5 | image-to-3dlab | audit branch | tests, diagnostics, cached experiments, product layer |

The Space and GitHub trees are **not identical** at these pins. They differ in the FDG VAE,
pipeline loader/base, image-to-3D pipeline, texturing-pipeline organization, and PBR
renderer; GitHub additionally contains datasets and trainers. The image-to-3D diff inspected
so far is mostly loader/refactor work, but parity cannot assume the two snapshots are
interchangeable.

## Source graph

```text
Microsoft HF Space @ ebf60b20  (executable parity spec)
        |
        +-- minimal audited MPS/device substitutions
        |
        +-- Pedro Metal backends where CUDA operators need replacements
        |     trellis2-apple @ 6055b868
        |     mtlbvh        @ 23f441c
        |     mtldiffrast   @ 4668cd9
        |     mtlgemm       @ 867aec8
        |     mtlmesh       @ 212079e
        |
        +-- image-to-3dlab parity tests and optional product features

Microsoft GitHub @ 75fbf018  (official comparison)
Shiv trellis-mac @ 0b8efd4   (compatibility checklist, not base)
```

Exact URLs and full hashes are recorded in
`audit/trellis-port/upstreams.lock.json`.

## Initial evidence

The installed `TRELLIS.2` directory is itself a Microsoft GitHub clone at `75fbf018`;
Shiv's setup does not contain or replace the complete core. It clones Microsoft, applies
`patches/mps_compat.py`, and installs Pedro's repositories as backend dependencies. This
explains the current port, but it does not make Shiv's patcher the source of truth.

The official Space was cloned independently and pinned at `ebf60b20`. Its `app.py` loads
`microsoft/TRELLIS.2-4B`, selects FlashAttention on CUDA, disables low-VRAM mode, runs the
three official samplers, and extracts through `o_voxel.postprocess.to_glb`. It simplifies
the decoded mesh to the renderer limit before preview/export; that pre-export step is not
the learned source of material semantics.

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

1. **Space inference contract** — image preprocessing, DINO features, sampler equations,
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

After adding the fresh-checkout multi-view patch regression test, the audit branch reports:

```text
435 passed, 2 skipped
```

The reconstructed port must then pass, in order:

1. patch application and import tests from a fresh checkout;
2. DINO preprocessing/conditioning determinism on the three golden source PNGs;
3. sparse-attention equivalence tests on single-batch and padded multi-batch inputs;
4. Stage-3 sampler/default/latent schema contract tests;
5. cached decode → GLB Fox topology and backface-culling checks;
6. cached Snag material → GLB channel and spatial-colour checks;
7. one fresh end-to-end golden run only after all cheaper gates pass.

The material-specific numerical gates are expanded in
`docs/TRELLIS2-PAPER-MATERIAL-FIDELITY.md`.

## Current verdict

An official-Space-first reconstruction is feasible and preferable to continuing with
untracked in-place patches. It is not yet safe to claim that a fresh source replay fixes
material fidelity: the MPS SDPA replacement participates directly in Stage 3 and needs a
numerical equivalence test, while the observed brown/green variation may still include
normal sampling variance. The paper nevertheless narrows the likely fault domain to Stage-3
conditioning/attention/precision rather than UV baking.
