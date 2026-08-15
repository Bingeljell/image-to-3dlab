# State of the Repo — 2026-08-16

**Purpose:** the "read this first" baseline after the 2026-08-15 session that unblocked the
clean port's decode→GLB bake. Previous snapshot and all older docs: [`legacy/`](legacy/).

## The headline

**The clean TRELLIS.2 port produces GLBs end-to-end on Apple Silicon — from the web UI.**
Three assets baked in one session (Lucian, controller, Flicker) via the CLI, and the
controller was then generated entirely through the browser (Generate tab + Setup card).
The decode→GLB blocker (cumesh Metal simplify crashing on ~20M-face meshes) is fixed —
see [MPS-BAKE-FIXES-2026-08-15.md](MPS-BAKE-FIXES-2026-08-15.md) for the full fix chain
(bake path, venv relocation, env hygiene) and the assumptions that were wrong.

## Resume here (2026-08-16 evening)

- **Web UI works end-to-end** (`python viewer/serve.py` → Generate → Setup card → run).
  The controller generated through the browser completed cleanly: 289,522 faces, 600
  holes, correct volume.
- **The open quality issue is texture, not geometry.** The controller's buttons (ABXY) came
  out glossy black instead of coloured, and the model reads darker than the source.
  Stage-3 material generation is the suspect (see "Known gaps" — the seed-search lever is
  ready).
- **fast_simplification is flakier than the docs said**: it also SIGBUS'd at 9.97M input
  faces (below the "20M" note) in a clean subprocess — the verify-and-retry caught it
  (attempt 3 succeeded). It is a safety net, not a cure.
- **The viewer now shows Decode/Bake stage progress** (tqdm `it/s` parsing + banner fix;
  committed `1607bfd`, viewer restarted). Next run will display all stages.

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
| Controller (CLI) | ~12k | 23.5 min | 38 s | 141 s | 27.9 min | 591 |
| Controller (web UI) | ~12k | ~23 min | ~40 s | ~10 min (heavier remesh) | ~35 min | 600 |
| Flicker | ~8k | 10.6 min | 24 s | 85 s | 13.8 min | 561 |

Flicker's clean-port output is geometrically near-identical to the HF demo control
(`assets_to_test/trellis-flicker-huggingface.glb`): same 278k-face budget, volume within 5%.

**Speed model:** sampling time is attention-bound and grows with token count² — a ~22k-token
subject (Lucian) costs ~78 min on MPS SDPA while an ~8k-token subject (Flicker) costs ~11 min.
The HF demo does the same work in minutes because CUDA flash attention is far faster at 22k
tokens.

## Known gaps / open threads

1. **Texture fidelity vs the source (the current priority).** Two concrete symptoms from the
   web-UI controller run: the model reads **darker** than the source, and the coloured ABXY
   buttons came out **glossy black**. Stage 3 (tex SLat) runs at guidance 1.0 — pure
   sampling, so the result is seed/RNG-sensitive (MPS RNG ≠ CUDA RNG at the same seed).
   The frozen-shape seed search (`scripts/trellis_stage3.py`, ~8 min per Stage-3-only run on
   cached latents) is the ready lever; a seed sweep on the cached controller latents is the
   obvious first experiment.
2. **Holes** (561–6,733 vs the HF demo's 1–14): mostly pinholes; the pre-cap + DC-remesh path
   loses some fidelity vs running `to_glb` on the full mesh (CUDA can).
3. **Speed**: the real lever is extending Pedro's fused `mtlgemm` kernel head-dim 64 → 128
   (TRELLIS.2-4B uses 128); multiplier grows with n².
4. **fast_simplification reliability**: flaky in content-dependent ways below the documented
   "20M" note (SIGBUS at 9.97M on the controller decode). The verify-and-retry subprocess
   handles it; a deterministic decimator would be the long-term fix.
5. **Old-port CLI**: `pipeline.py` still defaults to `vendor/trellis-mac`; the old lane
   (rebake/restore) still produced the pre-baseline references.

## Session assets

All in `output/space_baseline/` (git-ignored): `lucian.glb`, `controller.glb`, `flicker.glb`
+ manifests + latents + decode caches. The HF demo controls are in `assets_to_test/`
(`trellis-flicker-huggingface.glb`, `trellis-mossfox-huggingface.glb`,
`trellis-snag-huggingface.glb`).
