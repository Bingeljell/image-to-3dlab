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
untracked in-place patches. The first full-size attention experiment found a concrete limit
in Pedro's fused kernel rather than a material-fidelity improvement:

- the Snag shape has `22,894` sparse tokens in one sequence;
- TRELLIS.2-4B uses `1,536 / 12 = 128` values per attention head;
- Pedro's tiled Metal kernel accepts only head dimensions through `64`;
- at `128` it silently selects a per-query kernel that loops over every key serially;
- Pedro's parity tests cover head dimensions `32` and `64`, and its largest benchmark uses
  `2,048` tokens per sequence.

The real Stage-3 run completed zero of 12 steps in `29m44s` before interruption. The stack
stopped at the first cross-attention prefix transfer, which is an MPS synchronization point;
the preceding full self-attention dispatch was still outstanding. This is an unsupported
production shape, not evidence that the sampler or Python process crashed.

The clean port now uses sequence-wise PyTorch SDPA for TRELLIS.2-4B. It has the same
block-diagonal equation as upstream FlashAttention without padding or cross-sequence mixing.
A real-head-width MPS integration gate passed with maximum errors of `5.96046e-08` for sparse
self-attention and `4.47035e-08` for DINO cross-attention. Selecting `metal_flash` with a
head dimension above `64` now fails immediately instead of entering the serial fallback.

This clears attention semantics as the likely cause for a normal single-image run. Material
fidelity is still not proved: DINO drift is only about `1e-6` RMSE, while the hosted and local
texture samples may differ because Stage 3 is generative. The next expensive gate is one
official-Space-first Stage-3 sample through the validated SDPA path, followed by a latent
comparison before decoding or GLB baking.
