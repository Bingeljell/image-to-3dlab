# Handover — official-Space-first TRELLIS.2 Mac port

**Date:** 2026-08-14  
**Status:** active audit; clean port boots and its practical MPS attention path has numerical parity  
**Start here:** this document, then `docs/UPSTREAM-REBASE-AUDIT.md`

## Product goal

Build a reliable Apple Silicon port directly from the Microsoft TRELLIS.2 source used by the
official Hugging Face demo. The required result is an authentic, solid image-to-3D asset: it
does not need to match the source pixel-for-pixel, but holes, see-through backs, destroyed
geometry, and globally wrong material intent are failures.

Shiv and Pedro are audit inputs, not the semantic base. Credit their work, reuse necessary
Metal libraries, but do not copy Shiv's patched TRELLIS tree into the new core.

## Worktree safety

Do all upstream reconstruction work here:

```text
/Users/nikhilshahane/projects/image-to-3dlab/vendor/upstream-audit-worktree
branch: audit/upstream-rebase
```

Do **not** edit or commit the original checkout while this audit is active:

```text
/Users/nikhilshahane/projects/image-to-3dlab
branch: feat/tear-provenance
current status fingerprint: 685b79231640fbf71b149455f9b44e89041687769c84624a1786d5757e617694
```

Another session may be using the original checkout, so a changed fingerprint is not evidence
that this audit modified it. Inspect and preserve all existing changes.

Existing audit commits:

```text
396d9ee docs: pin Space parity spec and paper findings
34042cc fix: make multiview patch replayable
56521c0 docs: start upstream-first TRELLIS port audit
```

The current audit changes listed below are not committed yet.

## Pinned source graph

| source | commit | role |
|---|---|---|
| Microsoft HF Space | `ebf60b20fc5a4607f90a1c11c0aab0ceeda5429d` | executable parity specification |
| Microsoft GitHub | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` | official library comparison |
| Shiv `trellis-mac` | `0b8efd434f195e7225199b669165bc764d67a404` | compatibility checklist only |
| Pedro `trellis2-apple` | `6055b868734af6e12769d229d90580e775fae9f0` | Metal integration reference |
| Pedro `mtlbvh` | `23f441c470ce1f537e1fd836f3ffb5b8245f7975` | Metal BVH |
| Pedro `mtldiffrast` | `4668cd91cb6d27f5e264731f94a06841fbf7aab8` | Metal rasterizer |
| Pedro `mtlgemm` | `867aec8234299a7fe1ede7f802c8debe5a939a82` | sparse GEMM/attention kernels |
| Pedro `mtlmesh` | `212079e55772cff3d648a21372392c37e0643f3b` | Metal mesh/voxel operators |
| `utils3d` | `9a4eb15e4021b67b12c460c7057d642626897ec8` | pinned dependency |

The full URLs and hashes live in `audit/trellis-port/upstreams.lock.json`.

## Clean build on disk

The ignored reconstruction is here:

```text
vendor/trellis-space-mac/
  TRELLIS.2/       # official HF Space checkout, patched by our replayable scripts
  deps/            # clean pinned Pedro/utils3d clones
  .venv/           # fresh Python 3.11 environment
  cache/
```

It was built from scratch with the Space dependency versions, including PyTorch `2.11.0`,
torchvision `0.26.0`, transformers `4.57.3`, trimesh `4.10.1`, and kornia `0.8.2`. All five
Metal components compiled locally: `mtlbvh`, `mtldiffrast`, `mtlmesh/cumesh`,
`mtlgemm/flex_gemm`, and O-Voxel.

`scripts/bootstrap_trellis_space_macos.py` recreates this layout. It does not clone or apply
files from Shiv's repository.

## Changes implemented in the replayable port

### Microsoft core compatibility

`scripts/patch_trellis_space_core.py` applies only audited compatibility changes:

- MPS pipeline/device routing;
- DINO tensor/device routing and DINOv3 transformer-layer compatibility;
- sparse tensor and mesh accelerator aliases;
- an optional background-removal load;
- a correct sequence-wise PyTorch SDPA sparse-attention backend;
- Pedro's fused Metal attention as an explicit limited backend.

The optional background-removal change matters because `briaai/RMBG-2.0` is gated. An RGBA
input with real transparency never uses the remover, so `trellis_stage3.py` now skips loading
that unused model. Preprocessing pixels and Trellis equations remain unchanged. Inputs without
transparency still require the remover.

### Metal geometry/export compatibility

`scripts/patch_trellis_metal_backends.py` replays the production fixes already proved in the
working port:

- chunked MTLBVH queries;
- a 48-entry traversal stack and stackless unsigned-distance path;
- Objective-C autorelease pooling;
- the real `mtlmesh` hashmap-miss bounds guard;
- a sparse O-Voxel coordinate map instead of an approximately 8 GiB dense `1024^3` map;
- solid opaque export, `doubleSided: false`, and seam-safe cleanup.

Diagnostic-only environment hooks are intentionally excluded from the release patch.

## Main finding from today's full-size attention test

Pedro's fused attention is numerically correct on its tests but cannot run the real
TRELLIS.2-4B Stage-3 workload efficiently.

Measured production shape:

```text
Snag tokens:            22,894, one sequence
model width:             1,536
attention heads:            12
values per head:            128
```

Pedro's optimized tiled kernel supports head dimensions only through `64`. At `128`, its C++
dispatcher silently chooses a naïve Metal kernel where every query thread loops over every
key serially. Pedro's parity tests cover widths `32/64`; its largest benchmark has `2,048`
tokens per sequence. A source comment incorrectly says TRELLIS uses width `64` everywhere.

The first official-Space-first Snag Stage-3 attempt used `metal_flash`. It loaded correctly,
encoded DINO, reached `Sampling texture SLat`, then completed **zero of 12 steps in 29m44s**.
The Apple GPU had been observed at 100%, the Python process remained alive, and Ctrl-C later
produced a stack at the first cross-attention prefix transfer. That transfer synchronizes MPS,
so the preceding full self-attention dispatch was still outstanding. This was an effective
kernel hang/scale failure, not a Python crash.

The run exited with code `130`. It wrote no useful texture latent.

The clean fix is now in place:

- `sdpa` evaluates every packed sequence independently, preserving upstream block-diagonal
  semantics without padding or a dense mask;
- for the normal one-image case, it makes one direct SDPA call and introduces no padding;
- `metal_flash` now raises immediately when the head dimension exceeds `64`, instead of
  silently entering the serial fallback;
- `trellis_stage3.py` defaults to `sdpa`.

MPS integration result at the real 128-wide head size:

```text
self-attention max error:   5.96046e-08
cross-attention max error:  4.47035e-08
verdict: PASS
```

Command:

```bash
env PYTHONUNBUFFERED=1 vendor/trellis-space-mac/.venv/bin/python \
  scripts/check_trellis_space_attention.py \
  --root vendor/trellis-space-mac --backend sdpa
```

## Other parity findings

### DINO conditioning

The exact Snag source preprocessing is byte-identical between the current and reconstructed
ports:

```text
preprocessed shape: [948, 948, 3]
SHA-256: e82e137f53cd30cc3efd5ff6b6b0a46c7206308354cecf218c262a44ef1a3ef5
DINO output shape: [1, 4101, 1024]
feature max difference: 3.6716461e-05
feature mean absolute difference: 8.525938e-07
feature RMSE: 1.16389e-06
```

Dependency drift needed pinning, but this tiny DINO difference is unlikely to explain the
missing brown-vine/green-moss separation by itself.

### Stage-3 seeds are not stage-local replays

The official demo calls `torch.manual_seed(seed)` once before Stages 1→2→3. Resetting the
same integer immediately before a Stage-3-only sample creates different texture noise.

For the cached Snag shape, the full-run texture latent and the independent reset-before-Stage3
latent have identical coordinates but:

```text
max difference:       12.4588
mean absolute:         1.54939
RMSE:                  1.95977
cosine similarity:     0.741134
```

Therefore `texture_seed=614089393` in the Stage-3-only runner is a new material candidate, not
a replay of the hosted demo's Stage-3 noise. Seed variance remains a real explanation for some
local/control material difference.

### Where material semantics live

The paper/source audit confirms that Stage 3 predicts the native 3D PBR field from the DINO
image features plus the generated geometry latent. GLB extraction only interpolates and
packages that field; it cannot invent a brown/moss semantic split that Stage 3 omitted.

## Existing Snag artifacts to preserve

```text
/Users/nikhilshahane/projects/image-to-3dlab/output/snag_same_seed_hf/
  snag_seed614089393_latents.pt             # 5.9 MB; shape + original texture latent
  snag_seed614089393_decode.pt              # 668 MB; decoded geometry/material
  snag_seed614089393.glb                    # accepted solid local baseline
  stage3/default_tseed614089393_tex_latent.pt
  stage3/default_tseed614089393_material.pt
  stage3/default_tseed614089393_preview.glb
```

Hosted control:

```text
/Users/nikhilshahane/projects/image-to-3dlab/assets_to_test/trellis-snag-huggingface.glb
```

Source:

```text
/Users/nikhilshahane/projects/image-to-3dlab/assets_to_test/3-4th-snag-roots-alpha.png
```

The existing independent Stage-3 SDPA sample took about `8m08s`; decode was later resumed in
a fresh process and took about `49s`. Keep sample and decode in separate processes to avoid
the previous memory failure.

## Current tests

Completed after the SDPA/fail-fast correction:

```text
13 focused Python tests passed
128-wide MPS self/cross-attention integration passed
```

Earlier clean-build gates also passed Pedro's `24` fused-attention parity cases and the DINO
comparison. Those prove the fused kernel's small-shape math, not its production scalability.

The full repository suite has **not** been rerun after the newest uncommitted changes. The
last full result before them was `435 passed, 2 skipped`.

## Uncommitted audit files

Tracked modifications:

```text
audit/trellis-port/upstreams.lock.json
docs/TRELLIS2-PAPER-MATERIAL-FIDELITY.md
docs/UPSTREAM-REBASE-AUDIT.md
scripts/trellis_stage3.py
tests/test_trellis_stage3.py
```

New files:

```text
audit/trellis-port/requirements-macos.in
scripts/bootstrap_trellis_space_macos.py
scripts/check_trellis_space_attention.py
scripts/check_trellis_space_dino.py
scripts/patch_trellis_metal_backends.py
scripts/patch_trellis_space_core.py
tests/test_bootstrap_trellis_space_macos.py
tests/test_check_trellis_space_attention.py
tests/test_check_trellis_space_dino.py
tests/test_patch_trellis_metal_backends.py
tests/test_patch_trellis_space_core.py
```

This handover file is also new.

## Exact next steps

1. Run the complete CPU/static suite from the audit worktree:

   ```bash
   /Users/nikhilshahane/projects/image-to-3dlab/.venv/bin/python -m pytest -q
   git diff --check
   ```

2. Commit the bootstrap/core/backend/test documentation in logical commits. Do not include
   ignored model weights, virtual environments, or generated outputs.

3. With the user's approval for another GPU pass, run one clean official-Space-first Stage-3
   sample through validated SDPA:

   ```bash
   env PYTHONUNBUFFERED=1 vendor/trellis-space-mac/.venv/bin/python \
     scripts/trellis_stage3.py \
     /Users/nikhilshahane/projects/image-to-3dlab/output/snag_same_seed_hf/snag_seed614089393_latents.pt \
     output/upstream_audit/snag_sdpa_material.pt \
     --geometry-decode /Users/nikhilshahane/projects/image-to-3dlab/output/snag_same_seed_hf/snag_seed614089393_decode.pt \
     --image assets_to_test/3-4th-snag-roots-alpha.png \
     --texture-seed 614089393 \
     --vendor-root vendor/trellis-space-mac \
     --sparse-attn-backend sdpa \
     --sample-only
   ```

   Expected sampler time is roughly 8–10 minutes, not 30+ minutes per step.

4. Compare the resulting texture latent with both cached local latents before decoding. If the
   clean SDPA latent is materially different, decode it in a fresh process, then make a cheap
   preview GLB. If it is numerically the same as the existing independent SDPA candidate, do
   not waste a decode/bake; source reconstruction is not the material difference.

5. If clean SDPA reproduces the old candidate, move to controlled Stage-3 seed/guidance
   selection on the frozen shape. Do not return to remesh, BVH, lighting, or GLB packaging as
   explanations for missing semantic colour separation.

## Short verdict

The Microsoft-first Mac rebuild is viable. We found and removed one bad assumption in Pedro's
attention backend: its fast kernel does not support TRELLIS.2-4B's real 128-wide heads. The
validated SDPA replacement is fast enough and numerically correct. We have **not yet proved**
that rebuilding from the official Space source improves Snag's material; the next 8–10 minute
SDPA latent comparison is the deciding experiment.
