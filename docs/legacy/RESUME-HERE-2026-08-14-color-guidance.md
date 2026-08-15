# RESUME HERE — Snag colour solved (conceptually) + the RunPod pivot

**Date:** 2026-08-14
**Worktree:** `vendor/upstream-audit-worktree` (branch `audit/upstream-rebase`)
**Read first:** this file, then `docs/HANDOVER-upstream-space-mac-port.md`,
`docs/UPSTREAM-REBASE-AUDIT.md`.

## TL;DR

The long-running Snag "too green / too brown, no separation" problem is **understood and the
lever is found**:

- The **green was never a Stage-3 flaw** — our texture latent decodes to a *brown* material.
  The green we chased for days lived in one bad **bake** (`snag_seed614089393.glb`); a newer
  decode of the same latent bakes brown.
- The brown/green **moss separation is controlled by Stage-3 `guidance_strength`**. We
  bracketed it: **guidance 1.0 = too brown (0% green), 3.0 = too green + far too dark (8%),
  target (hosted) ≈ 3% green at normal brightness → sweet spot ~1.5–2.0.**
- The **HF demo has no finish/grade lane** — `app.py` = preprocess → 3 stages → `to_glb`. So
  the moss must come from the Stage-3 sample, not post-processing. A global grade cannot add
  moss; only a moss-mask grade (hero-asset fallback) or a better sample can.

**Keep the local Mac pipeline — it is the product.** Snag/Flicker/Fox are good on geometry AND
texture; the moss is last-mile, and it is a **parameter** (`--guidance-strength`) the local
runner already exposes, not a code fix. The Mac hit **three memory walls** searching for it
(25/42/55 GB), which is slow — so use a **RunPod CUDA box (RTX A6000, 48 GB, ~$0.33/hr)** as a
**wind tunnel** to find the guidance value in ~30s/try, then bring the number home. RunPod is a
search harness + one-time CUDA control, **not** a replacement.

## What we proved (with numbers)

Same frozen Snag shape, same seed (614089393), Stage-3 only:

| run | green-moss voxel frac | base-color mean RGB | metallic | verdict |
|---|---|---|---|---|
| g1.0 (demo default) | **0%** | [0.166, 0.139, 0.081] | 0.70 | too brown, moss gone |
| **hosted (target)** | ~3% | [0.129, 0.127, 0.044] | — | balanced bark + moss |
| g3.0 (full interval) | **2.55%** (voxel) / 7.95% (baked albedo) | **[0.009, 0.008, 0.001]** | 0.91 | moss back but too green + ~15× too dark |

- Latent cosines: g1.0 ≈ prior candidate B (0.9999); **g3.0 vs g1.0 = 0.82** → guidance
  materially changed the sample (it is an active lever, not inert).
- Upstream audit (this session's earlier gates) already cleared: preprocessing preserves the
  brown (SHA-identical to source interior), DINO parity ~1e-6, attention parity ~6e-8, and the
  clean official-Space reconstruction reproduces our latent (0.9999). So nothing *upstream of
  the Stage-3 sample* is the cause.

## The three Mac memory walls (and the workarounds)

1. **Stage-3 sample — 25 GB.** MPS has no FlashAttention; PyTorch SDPA materialises the full
   `22894² × 12 heads × 4B ≈ 25 GB` score matrix → SIGABRT. **Fixed** by query-chunking the
   `sdpa` branch (`SDPA_Q_CHUNK`, default 1024) — numerically exact (parity unchanged at
   5.96e-8 / 4.47e-8). Applied in `scripts/patch_trellis_space_core.py` **and** the vendored
   `.../sparse/attention/full_attn.py`. **Uncommitted.**
2. **Decode — 42 GB.** The full shape decoder OOMs at the ~42 GB MPS watermark. **Use** the OLD
   `scripts/trellis_stage3.py --resume-texture-latent`: loads only the fp16 tex decoder and
   recovers subdivision guides from the frozen geometry decode (`derive_subdivision_guide_tensors`),
   skipping the shape decoder. ~47s.
3. **`to_glb` bake — 55 GB.** The raw 20M-face mesh blows up `to_glb`. **Use**
   `scripts/trellis_rebake.py --geometry-glb <accepted.glb>` to bake the material onto an
   already-simplified mesh (baseline's 2.8M faces). `--remesh` stays OFF (broken on this port).

Timings on Snag (cached shape): g1.0 sample ~28 min (140s/step ×12); g3.0 ~54 min (CFG doubles
per-step to ~274s). Decode ~47s. Bake ~27s. Face-count is a downstream `--decimation-target`
knob (our 2.8M vs hosted 284k = under-decimation), not a sample cost.

## RunPod as wind tunnel (not a replacement)

The local Mac pipeline stays the product. RunPod exists only to make the guidance *search* cheap
and to give a one-time CUDA ground-truth — the winning `--guidance-strength` value drops straight
back into the local runner (same model, same latents, identical result; Mac just slower).

CUDA has FlashAttention, so **upstream runs unmodified** — *none* of the Mac patches, chunked
SDPA, or Metal backends are needed (they exist only to fake CUDA on Metal). Setup = clone official
TRELLIS.2 → `pip install` (use a template with **flash-attn prebuilt**) → download
`microsoft/TRELLIS.2-4B` weights to a **persistent volume** → run. Card: **RTX A6000, 48 GB,
~$0.33/hr** (48 GB matters — the decode needs ~40 GB; a 24 GB 4090 would OOM it, matching the
demo's A100-40 GB profile).

Bonus: pure upstream on CUDA is **demo-identical**, giving the CUDA control we've never had —
it settles whether the lost moss is a Mac-port artifact or just the seed/guidance.

## Exact next steps

1. Spin up RunPod A6000, weights on a persistent volume, upstream installed.
2. **Guidance sweep 1.5 / 1.75 / 2.0** on Snag (same source image), full guidance interval
   (0.0,1.0), 12 steps. Pick the run matching hosted's ~3% green at normal brightness.
3. If it lands slightly dark, a brightness grade is fine (brightness is gradeable; spatial moss
   is not). Confirm brown-bark + green-moss separation in the viewer, culled.
4. **Bring the winning guidance value home to the local Mac runner** (`--guidance-strength <v>`)
   and produce the final Snag locally; re-check Flicker/Fox at the same setting. Set
   `--decimation-target` ~300–500k for game-ready GLBs. RunPod stays optional (large batches or
   more parameter searches), never a dependency.

## Artifacts produced this session

- `output/upstream_audit/snag_sdpa_g1.0_tex_latent.pt` — clean SDPA Stage-3, guidance 1.0.
- `output/upstream_audit/snag_sdpa_g3.0_tex_latent.pt` — guidance 3.0.
- `output/upstream_audit_view/snag_g3.0_material.pt`, `snag_g3.0_decode.pt`, `snag_g3.0.glb`
  (30 MB, near-black — the over-guided result).
- `output/upstream_audit/snag_preproc.png` — the preprocessed image DINO sees (brown intact).

## Memory written (fresh session will load these)

`guidance-is-the-moss-separation-lever`, `green-was-a-bake-artifact-not-stage3`,
`demo-has-no-finish-lane-separation-is-stage3`, `mps-no-flash-kernel-speed-and-decode-memory`,
`runpod-4090-may-replace-the-mac-port`.
