# State of the Repo — 2026-08-15

**Purpose:** a single "read this first" snapshot so we stop re-deriving the same facts. Written
after a session that burned hours on a confusion this doc exists to prevent: **there are two
TRELLIS ports, and the one `pipeline.py` runs by default is the OLD, pre-fix one.**

---

## TL;DR (the four things that cost time)

1. **Two TRELLIS ports.** `pipeline.py --run-manifest` runs the **old** `vendor/trellis-mac`
   (repo root). The **clean** rebuilt port is `vendor/upstream-audit-worktree/vendor/`**`trellis-space-mac`**.
2. **The clean port has no Mac full-generation CLI yet.** Its `run()` exists inside the upstream
   pipeline, but the only entry (`app.py`) is CUDA/HF-Space-bound. So **every fresh image→GLB to
   date, including the 2026-08-15 Lucian, went through the OLD port.** The next step is to build
   `scripts/trellis_space_generate.py`.
3. **Judge culled, by eye. HOLES are the defect, not bodies.** A good asset can have 50
   disconnected bodies (forest does). Measure holes with a **position-only** vertex merge.
4. **Don't confuse the subdirs.** The audit worktree *also* contains a bare `vendor/trellis-mac`
   (no venv, no deps) — that is NOT the build. The build is `trellis-space-mac`.

---

## Worktrees

| Checkout | Path | Branch |
|---|---|---|
| Main | `/Users/nikhilshahane/projects/image-to-3dlab` | `feat/tear-provenance` |
| Audit | `…/image-to-3dlab/vendor/upstream-audit-worktree` | `audit/upstream-rebase` |

They have **diverged 3 commits each way** (siblings, neither strictly ahead). `vendor/` is
git-ignored in both, so each worktree has its own vendored ports and its own patch state.

---

## The two TRELLIS ports

### OLD — `vendor/trellis-mac` (repo root)
- Clone of `shivampkumar/trellis-mac`. Built (`.venv`, ~147 MB `deps/`, mtlbvh + cumesh
  `.metallib`), and face-cap patched (`generate.py:339`, `I2L_PRE_CAP` default 4,000,000).
- **This is the pre-fix port.** `image_to_3dlab/cli.py` defaults `--trellis-repo vendor/trellis-mac`,
  so `pipeline.py --run-manifest` runs it. **All fresh image→GLB assets so far came from here**,
  followed by the rebake/restore lane below.
- Winding repair (`trellis_backend.py:_repair_winding`) only fixes **global** orientation
  (flips if signed volume < 0 via `trimesh.fix_normals`, `process=False`); it does not weld or
  close holes.

### CLEAN — `vendor/upstream-audit-worktree/vendor/trellis-space-mac`
- Reconstructed **Microsoft HF Space** source, built and MPS-adapted. `.venv` present; kernels
  compiled: `deps/{mtlbvh, mtldiffrast, mtlgemm, mtlmesh, trellis2-apple, utils3d}`.
- Patched for mesh quality + Metal: `TRELLIS.2/trellis2/representations/mesh/base.py`
  (cumesh `fill_holes` enabled), `deps/trellis2-apple/o-voxel/o_voxel/postprocess.py` &
  `convert/flexible_dual_grid.py`, `deps/mtlmesh/src/metal/remesh.metal`, mtlbvh
  `bvh.metal`/`metal_bvh.mm`, sparse `attention/full_attn.py`/`basic.py`/`config.py`, pipelines.
- **Verified:** MPS/device routing, sparse-attention parity (self/cross max err ~5e-8, PASS at the
  real 128-wide head size), DINO conditioning parity.
- **mtlgemm caveat:** its fast-attn kernel is 64-wide-head only; TRELLIS uses 128. The clean port
  uses **SDPA + fail-fast**, not Pedro's slow fallback (`ATTN_BACKEND=sdpa`).
- **Status: uncommitted** in the audit worktree.

### What the clean port has vs. lacks
- **Has:** the full upstream `Trellis2ImageTo3DPipeline.run()` (Stage 1 sparse → Stage 2 shape →
  Stage 3 material → decode) plus `app.py`, which calls `run()` (l.406) then
  `o_voxel.postprocess.to_glb()` (l.530).
- **Lacks:** a Mac CLI. `app.py` is CUDA/Space-bound (`ATTN_BACKEND=flash_attn` l.8, `pipeline.cuda()`
  l.671, `device='cuda'` tensors, `@spaces.GPU`, Gradio session handling).
- **Exercised so far:** clean checkout, compiled Metal deps, MPS routing, BVH/remesh/export fixes,
  attention + DINO parity, and `scripts/trellis_stage3.py` (Stage-3 resample from cached shape latents).

---

## Next step (greenlit 2026-08-15) — `scripts/trellis_space_generate.py`

Build the Mac end-to-end wrapper in the **audit worktree** (`audit/upstream-rebase`), next to the
other `trellis_space` scripts. It reuses `trellis_stage3.py`'s proven MPS scaffold and calls the
**full** `run()` instead of Stage-3-only:

1. `ATTN_BACKEND=sdpa` (+ `SPARSE_ATTN_BACKEND`); load clean patched pipeline via
   `Trellis2ImageTo3DPipeline.from_pretrained(..., load_rembg=not has_transparent_alpha)`; `.to("mps")`.
2. Accept an image; `preprocess_image` + `get_cond`.
3. `pipeline.run(...)` on MPS with **app.py's demo params** (mirror l.406–420 exactly — no
   parameter experimentation until a baseline exists).
4. Save shape/material latents + decoded geometry.
5. `o_voxel.postprocess.to_glb(...)` with app.py's demo params (l.530–545).
6. Write manifest + timings.

Follow repo conventions: **importable core** functions + a **unit test** (imports resolve,
`ATTN_BACKEND=sdpa`, `run`/`to_glb` signatures, alpha guardrail) as the cheap env-check **before**
the ~25-min run. Validate on Lucian; judge **culled** against forest/Flicker.

---

## How the *good* reference assets were actually made (the lane)

Not raw generation — a post-process lane on top of the old port:
`generate.py` → **`scripts/trellis_rebake.py`** (cumesh cleanup + decimate, from a cached
`--dump-decode`) → **`scripts/restore_pbr_material.py`** (re-attaches the metallicRoughness map).

| Asset | Path | Faces | Holes | Bodies |
|---|---|---:|---:|---:|
| Flicker r0.7 (finished) | `output/pbr_restore/flicker_r0.7.glb` | 293,488 | **1,153** | 1 |
| Forest (finished) | `output/forest_validation/forest_fixed_opaque.glb` | 487,087 | **7** | 50 |
| Flicker 767 (raw demo-match) | `output/conditional/…767aff9faf21.glb` | 2,984,561 | 3,997 | 1 |

`r0.45/r0.7/r1.0` are **roughness variants** of one clean mesh (restore step), not remesh levels.

---

## Metrics that matter (and the trap)

- **Culled render is the honest test** — glTF is double-sided, so a hollow/holey mesh looks fine
  in preview and fails only under backface culling (which every game engine does).
- **Holes (boundary edges) are the defect; bodies are not** — forest has 50 bodies and looks great.
- **Measure with a position-only merge:** `mesh.merge_vertices(merge_tex=True, merge_norm=True)`.
  Default `merge_vertices()` keeps UV-seam/normal splits, so every seam reads as a fake hole
  (this trap made a clean control look as holey as ours — the "224,057 vs the real 1" incident).

---

## This session's outputs — flagged as OLD-PORT, not representative

Made through the **old** `trellis-mac` port; keep for comparison, do **not** treat as the clean
port's quality:

| Asset | Faces | Holes | Bodies | Note |
|---|---:|---:|---:|---|
| `…cute-creature-lucian__…e76a41ea3355.glb` (demo-match) | 2,978,176 | 19,452 | 41 | old port; furry subject decodes messy |
| Lucian / controller fast-look (512, 200k) | — | — | — | crushed by 200k budget; ignore |

`output/cleanup_test/lucian_cleaned_geom.glb` = a failed post-hoc trimesh cleanup (19,452→18,061
holes). Post-processing the final GLB can't do what the in-decode cumesh rebake does.

---

## Run reference

```bash
# clean-port interpreter
vendor/upstream-audit-worktree/vendor/trellis-space-mac/.venv/bin/python

# attention parity check (PASS = SDPA works at 128-wide heads)
env PYTHONUNBUFFERED=1 vendor/upstream-audit-worktree/vendor/trellis-space-mac/.venv/bin/python \
  vendor/upstream-audit-worktree/scripts/check_trellis_space_attention.py \
  --root vendor/trellis-space-mac --backend sdpa
```

- **Clean-port scripts** (audit worktree): `bootstrap_trellis_space_macos.py`,
  `patch_trellis_space_core.py`, `patch_trellis_metal_backends.py`,
  `check_trellis_space_attention.py`, `check_trellis_space_dino.py`, `trellis_stage3.py`.
- **Clean-port docs** (audit worktree): `HANDOVER-upstream-space-mac-port.md`,
  `UPSTREAM-REBASE-AUDIT.md`, `TRELLIS2-PAPER-MATERIAL-FIDELITY.md`.

---

## Housekeeping / open threads

- **Repo cleanup wanted.** A large pile of untracked files; the worktree confusion is a symptom.
- `output/snag_same_seed_hf/` holds Snag caches/GLBs. A fused-attention experiment produced no
  useful new latent.
- `assets_to_test/cute-creature-lucian.png` and the colour-guidance handover came from **another
  agent**, not the kernel work.
- The clean port's changes are **uncommitted**; the replayable scripts/tests are what get committed.

---

## Clean-port wrapper: built, works through decode, BLOCKED on MPS GLB packaging (2026-08-15 eve)

We built the Mac end-to-end wrapper (the "next step" above): **`vendor/upstream-audit-worktree/
scripts/trellis_space_generate.py`** + `tests/test_trellis_space_generate.py` (19 passing).
`--check` passes (MPS up, `run`/`decode_latent`/`to_glb` resolve, `ATTN_BACKEND=sdpa`). It has a
full-run mode and a `--from-latents` resume mode. It runs the demo path on MPS/SDPA with demo params.

### What works
- **Sampling + decode run.** First Lucian run (seed 0, res 1024, decimation 300k, tex 2048,
  remesh on) completed all of `pipeline.run()` — stages 1–3 in **4658.8s (~78 min)**: sparse 1:39,
  shape-coarse 2:34, **shape-fine 41:55**, tex-SLat 26:27. (~3.5× slower than the old port; the
  fine Shape-SLat + tex-SLat are attention-bound on SDPA — the unbuilt flash-attn kernel is the
  known "2× lever," a build task not a flag.)
- **The 78 min is banked.** Latents cached at `output/space_baseline/lucian_latents.pt` (5.6 MB).
  `--from-latents` skips sampling and only re-runs decode + bake (~6 min: 79s load + ~4 min decode).

### The blocker: post-decode mesh ops crash on MPS
The `decode → GLB` path dies in **cumesh's Metal `simplify`** on the raw multi-million-face decode
(`decode_latent` yields ~10M verts / ~20M faces before remesh; the HF demo GLBs are ~282k *after*
remesh, so a huge raw decode is normal):

1. **Full run** — `mesh.simplify(16777216)` (app.py's nvdiffrast cap) → cumesh Metal `simplify_step`:
   `RuntimeError: face 121458 has vertex index out of range (…, -1) for V=10045263`. A degenerate
   face carrying a `-1` index.
2. **Fix tried** — `filter_degenerate_faces` (drop faces with any index `<0` or `>=num_vertices`).
   Removed **1,679** faces; got past the `-1`.
3. **Resume run** — `mesh.simplify` then failed differently:
   `torch.AcceleratorError: index 20549376 is out of bounds: range 0 to 20549376` — an internal
   off-by-one in the Metal `simplify_step` on the ~20M-element mesh.
4. (Also fixed en route: an import-ordering bug in the resume path — `trellis2` imported before
   `load_pipeline` put `TRELLIS.2` on `sys.path`.)

### Hypothesis
**cumesh's Metal `simplify` is not robust on multi-million-face meshes.** app.py's
`mesh.simplify(16777216)` works on CUDA (cumesh CUDA backend) but hits two distinct crashes on MPS.
This is precisely the "Mac end-to-end GLB packaging is still TODO" — the decode→GLB path needs
MPS-specific handling app.py never needed. Corroborating: the **old port's own comment** —
*"Pre-simplify mesh to avoid mtlbvh crash on large meshes"* — and it used **CPU
`fast_simplification`**, not the Metal simplifier, to dodge exactly this class of large-mesh Metal
crash. Worth reporting upstream to the clean-port/Pedro team as an MPS `simplify_step` robustness bug.

### Suggested fixes (the plan)
1. **Cache the decoded mesh** (post-`decode_latent`, post-filter) to a `.pt` + add a `--from-decode`
   mode, so each GLB-packaging attempt is ~1 min (no re-decode) instead of ~6.
2. **Replace the crashing Metal `mesh.simplify(NVDIFFRAST_FACE_LIMIT)` with CPU `fast_simplification`**
   (v0.2.0, confirmed installed in the clean-port venv) pre-capping to an mtlbvh-safe size (~4M, the
   old port's threshold), then `to_glb`. Safe because `to_glb` takes the voxel `attrs`/`coords`
   **separately** from mesh `verts`/`faces` — decimating verts/faces (dropping per-vertex attrs)
   doesn't touch the PBR field, and `remesh=True` rebuilds topology regardless. `to_glb` has no
   large-mesh guard (only `>8 triangles`).
3. **Watch device placement for `to_glb`** — the old port moved mesh tensors to **CPU** before
   `to_glb` to avoid device mismatch; the clean-port MPS path may need the same.

### Facts on hand
- `fast_simplification` **is** in the clean-port venv (v0.2.0).
- `to_glb`/`mtlbvh` assert only `triangles > 8`; no max-face guard.
- Latents cached; decode is deterministic to re-run.
- Files touched this session (uncommitted, audit worktree): `scripts/trellis_space_generate.py`,
  `tests/test_trellis_space_generate.py`. Baseline outputs under `output/space_baseline/`.
