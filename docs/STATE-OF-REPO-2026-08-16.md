# State of the Repo — 2026-08-16

**Purpose:** the "read this first" baseline after the 2026-08-15 session that unblocked the
clean port's decode→GLB bake. Previous snapshot and all older docs: [`legacy/`](legacy/).

## The headline

**The clean TRELLIS.2 port produces GLBs end-to-end on Apple Silicon.** Three assets baked in
one session: Lucian, controller, Flicker. The decode→GLB blocker (cumesh Metal simplify
crashing on ~20M-face meshes) is fixed — see [MPS-BAKE-FIXES-2026-08-15.md](MPS-BAKE-FIXES-2026-08-15.md)
for the five bugs found and the assumptions that were wrong.

## The two ports

| Port | Path | Status |
|---|---|---|
| **Clean** (upstream Microsoft HF Space, MPS-adapted) | `vendor/trellis-space-mac` (built by the bootstrap) | **The path forward.** Full image→GLB works |
| **Old** (Shiv `trellis-mac`, pre-fix) | `vendor/trellis-mac` | Still `pipeline.py`'s default CLI backend; legacy lane |

The clean port's code, bootstrap and patches are in this repository (on `main`); the built
environment (`vendor/trellis-space-mac/`) is git-ignored and rebuilt by
`scripts/bootstrap_trellis_space_macos.py` + the `patch_trellis_*` scripts.

## How to run image → GLB (clean port)

### Fresh machine / fresh clone (reproducible)

```bash
git clone <repo> && cd image-to-3dlab
# requires: uv, Python 3.11, Xcode command-line tools (builds the Metal kernels; ~1h)
python scripts/bootstrap_trellis_space_macos.py
# first run downloads the ~14 GB TRELLIS.2-4B weights automatically
env PYTHONUNBUFFERED=1 vendor/trellis-space-mac/.venv/bin/python \
  scripts/trellis_space_generate.py input.png output/out.glb
```

The bootstrap clones the pinned upstream sources into `vendor/trellis-space-mac/` (exactly
where the generator's default `--vendor-root` points) and builds the Metal kernels. The
vendored build is git-ignored by design — the bootstrap + patch scripts are the
reproducible path.

**This machine** already has a build at `vendor/trellis-space-mac/` (moved here from the
legacy audit worktree), so the fresh-machine recipe works as-is with no flags.

- `--check` verifies the environment first (seconds, no model load).
- The input must have a transparent alpha foreground (BRIA guardrail); RGBA with real alpha
  never loads the background remover.
- Sampling runs 12 demo-default steps per stage; total runtime scales with the subject's
  sparse-structure token count (see timings).
- Outputs: `<out>.glb`, `<out>_latents.pt`, `<out>_decode.pt`, `<out>.json` manifest.

### Resume modes (skip expensive work)

| Flag | Skips | Cost |
|---|---|---|
| `--from-latents <latents.pt>` | Sampling (stages 1–3) | decode + bake only |
| `--from-decode <decode.pt>` | Sampling **and** decode and model load | pre-cap + bake only (~1 min setup) |

### The bake path (why it's robust now)

decode → filter degenerate faces → cache CPU bundle → free the 4B model → **CPU
`fast_simplification` pre-cap in a subprocess** (verify-and-retry, then filter residual
corrupt indices) → Metal `to_glb` (CPU tensors) → export. Full reasoning in
[MPS-BAKE-FIXES-2026-08-15.md](MPS-BAKE-FIXES-2026-08-15.md).

## Measured timings (M5, 32 GB)

| Asset | Tokens | Sampling | Decode | Bake | Total | Holes |
|---|---|---|---|---|---|---|
| Lucian | 21,765 | 78 min (banked) | 189 s | 443 s | ~12 min from cache | 6,733 (86% pinholes) |
| Controller | ~12k | 23.5 min | 38 s | 141 s | 27.9 min | 591 |
| Flicker | ~8k | 10.6 min | 24 s | 85 s | 13.8 min | 561 |

Flicker's clean-port output is geometrically near-identical to the HF demo control
(`assets_to_test/trellis-flicker-huggingface.glb`): same 278k-face budget, volume within 5%.

**Speed model:** sampling time is attention-bound and grows with token count² — a ~22k-token
subject (Lucian) costs ~78 min on MPS SDPA while an ~8k-token subject (Flicker) costs ~11 min.
The HF demo does the same work in minutes because CUDA flash attention is far faster at 22k
tokens.

## Known gaps / open threads

1. **Holes** (561–6,733 vs the HF demo's 1–14): mostly pinholes; the pre-cap + DC-remesh path
   loses some fidelity vs running `to_glb` on the full mesh (CUDA can). Worth a targeted look.
2. **Texture drift** vs the HF demo: Stage 3 runs at guidance 1.0 (pure sampling), so texture
   is seed-sensitive; MPS RNG ≠ CUDA RNG at the same seed. Frozen-shape seed search
   (`scripts/trellis_stage3.py`) is the lever.
3. **Speed**: the real lever is extending Pedro's fused `mtlgemm` kernel head-dim 64 → 128
   (TRELLIS.2-4B uses 128); multiplier grows with n².
4. **Old-port CLI**: `pipeline.py` still defaults to `vendor/trellis-mac`; the old lane
   (rebake/restore) still produced the pre-baseline references.
5. **Housekeeping**: repo cleanup is ongoing (this baseline); `vendor/` stays git-ignored, so
   the clean-port patch state lives in the replayable `scripts/patch_trellis_*` scripts.

## Session assets

All in `output/space_baseline/` (git-ignored): `lucian.glb`, `controller.glb`, `flicker.glb`
+ manifests + latents + decode caches. The HF demo controls are in `assets_to_test/`
(`trellis-flicker-huggingface.glb`, `trellis-mossfox-huggingface.glb`,
`trellis-snag-huggingface.glb`).
