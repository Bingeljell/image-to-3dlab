# Spec: Web "Generate" view (image → 3D) — for Luna to build

**Status:** design only, 2026-08-15. **Do not build from this doc without the owner's go-ahead.**
**Audience:** Luna (implementer).

## Goal

A browser view to run **image → 3D** interactively — upload an image, pick settings, watch live
progress with an ETA, and see the resulting model in place — instead of hand-writing a manifest and
reading a terminal log. The CLI stays; this is a friendlier front door onto the **same** CLI.

It runs the clean-port wrapper we just built:
`vendor/upstream-audit-worktree/scripts/trellis_space_generate.py` (MPS/SDPA, demo params).

## Hard constraints (read first)

- **This is a NEW MODE of the existing viewer, not a replacement.** The compare viewer
  (`viewer/index.html`, served by `viewer/serve.py`) must keep working exactly as-is. Add a
  **mode switch** (`Compare | Generate`); the owner will tweak Compare separately. **Do not
  overwrite or refactor the compare viewer's behavior.**
- **One generation at a time.** MPS cannot run two TRELLIS jobs at once (OOM / contention). The
  backend must serialize: reject or queue a second request while one is running.
- **Honor the BRIA guardrail.** The wrapper already refuses a non-alpha image unless
  `--allow-rembg`. The web layer must surface that refusal, not silently pass `--allow-rembg`.
- **Demo params are the default and the recommendation.** Expose knobs, but label the demo
  defaults and warn against experimentation until a baseline exists (see the CLI's `DEMO_PARAMS`).
- **Long jobs.** A run is ~25–85 min on this Mac (subject-dependent; the fine Shape-SLat pass
  dominates). The UI must be built for a long, resumable-looking wait, not a spinner that feels hung.

## Layout

Two columns. Left = input + settings. Right = progress, which becomes the model when done.

```
┌──────────────────────────────┐  ┌───────────────────────────────────────────┐
│  ▣  Drop / choose an image   │  │  RIGHT PANE (one box, two lifetimes)        │
│     (PNG w/ alpha preferred) │  │                                             │
│     [ thumbnail preview ]    │  │  WHILE RUNNING → progress:                  │
│                              │  │    Overall  ▓▓▓▓▓▓░░░░░  62%   ~18 min left  │
│  ── Settings ──              │  │    ● Load ............ done  1m23s           │
│  Resolution  [1024 ▼]        │  │    ● Sparse struct ... done  1m39s           │
│  Seed        [ 0 ]           │  │    ● Shape SLat coarse done  2m34s           │
│  Decimation  [ 300000 ]      │  │    ● Shape SLat fine ▓▓▓░ 5/12 · ~6 min      │
│  Texture     [ 2048 ▼]       │  │    ○ Texture SLat .... queued                │
│  [ ] allow rembg (no alpha)  │  │    ○ Decode / remesh . queued                │
│                              │  │                                             │
│      [  Generate  ]          │  │  WHEN DONE → the same box shows the GLB      │
│                              │  │  in an orbit viewer (reuse Compare's loader) │
└──────────────────────────────┘  └───────────────────────────────────────────┘
```

- **Left, top:** drop-zone / file-picker. Show a thumbnail and a small **alpha badge**
  ("transparent foreground ✓" / "no alpha — will refuse unless you allow rembg").
- **Left, below:** settings form (see *Settings*). A **Generate** button, disabled until an image
  is chosen and while a job runs.
- **Right:** a single box. During the run it is the **progress panel**; on completion it swaps to
  an **orbit viewer** of the finished GLB. Keep a small "Download GLB / manifest" affordance.

## Settings (fields, defaults, guardrails)

Map 1:1 to the CLI flags. Defaults are the demo defaults.

| Field | Control | Default | Notes |
|---|---|---|---|
| Resolution | select 512 / 1024 / 1536 | **1024** | → `pipeline_type` 512 / 1024_cascade / 1536_cascade |
| Seed | int | **0** | demo default |
| Decimation target | int | **300000** | final face/vertex cap; **non-binding with remesh** — see note |
| Texture size | select 1024 / 2048 / 3072 / 4096 | **2048** | demo default |
| Allow rembg | checkbox | **off** | only enable for a non-alpha image; loads the background remover |

> **Decimation note for the UI copy:** with `remesh=True` (always on, demo-faithful) the final face
> count is set by the remesh resolution (~282k at res 1024), and `decimation_target` only caps it.
> So 300k vs 3M produce the same ~282k output. Present it as an advanced/rarely-touched field.

`remesh=True`, `remesh_band=1`, `remesh_project=0` are fixed demo params — **not** user-exposed.

## Backend

The compare viewer is served by `viewer/serve.py` (a stdlib `http.server` static server). Extend it
(or add a sibling module it imports) with a small job API. **Keep the static-serving behavior intact.**

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/generate` | multipart: `image` + settings JSON → starts a job, returns `{job_id}`. 409 if one is already running. |
| `GET` | `/api/generate/{job_id}/events` | **SSE** stream of progress events (below). Closes on completion/error. |
| `GET` | `/api/generate/{job_id}/result.glb` | the finished GLB (served once `done`). |
| `GET` | `/api/generate/{job_id}/manifest.json` | the run manifest the CLI wrote. |
| `POST` | `/api/generate/{job_id}/cancel` | terminate the subprocess (SIGTERM the process group). |

### Job lifecycle

1. Save the uploaded image to a temp path. Detect alpha; if none and `allow_rembg` is false, return
   a 422 with the guardrail message (do not start the subprocess).
2. Spawn `trellis_space_generate.py` with the clean-port interpreter
   (`vendor/upstream-audit-worktree/vendor/trellis-space-mac/.venv/bin/python`), passing the image,
   an output path under `output/space_web/<job_id>/model.glb`, and the settings as flags.
   Run it in its **own process group** so cancel can kill the whole tree. `env PYTHONUNBUFFERED=1`.
3. Read the subprocess's combined stdout+stderr line-by-line (tqdm uses `\r`; split on both `\n`
   and `\r`). Parse into progress events; push to any connected SSE client and also append to a log
   file so a reconnecting client can catch up.
4. On exit 0: emit `done` with the GLB + manifest URLs. On non-zero/timeout: emit `error` with the
   last ~40 log lines.

### Progress event schema (SSE `data:` JSON)

```jsonc
{
  "phase": "load|sparse_structure|shape_slat_coarse|shape_slat_fine|texture_slat|decode|bake|done|error",
  "step": 5, "total": 12,            // omit for non-stepped phases (load/decode/bake)
  "s_per_it": 119.8,                 // live rate from tqdm, when available
  "stage_eta_seconds": 838,          // this stage's remaining (tqdm gives it directly)
  "total_eta_seconds": 1080,         // see ETA logic
  "stage_pct": 41, "overall_pct": 62,
  "elapsed_seconds": 3120,
  "message": "Sampling shape SLat"   // human label
}
```

## Progress + ETA logic (the part to get right)

**Parse the wrapper's output.** Two kinds of lines:

- **Stage banners** from the wrapper's own `print`s: `Loading TRELLIS.2 pipeline...`,
  `pipeline.run() (stages 1-3) done in ...`, `decode_latent done in ...`,
  `to_glb + export done in ... -> <path>`, `Total: <n>s`.
- **tqdm bars** from TRELLIS internals, e.g.
  `Sampling shape SLat:  33%|███▎ | 4/12 [07:59<15:58, 119.80s/it]`
  Parse: **label** (`Sampling shape SLat`), **step/total** (`4/12`), **elapsed** (`07:59`),
  **remaining** (`15:58`), **rate** (`119.80s/it`). tqdm hands you the per-stage ETA directly.

**Disambiguate the two Shape-SLat passes.** `Sampling shape SLat` appears **twice** in a
1024_cascade run — the first completed bar is the **coarse** pass, the second is the **fine** pass.
Track a stage cursor; the second time you see that label starting from 0/12, it's `shape_slat_fine`.

**Stage sequence (1024_cascade):**
`load → sparse_structure → shape_slat_coarse → shape_slat_fine → texture_slat → decode → bake`.
`load`, `decode`, `bake` are non-stepped (show an indeterminate bar; `bake` has its own sub-tqdm
for BVH / remesh / simplify / xatlas you can surface if you want).

**ETA = current stage remaining (from tqdm) + baseline estimate for the not-yet-started stages.**

- Keep a **baseline profile** (JSON) of per-stage durations. Seed it from a real run — the Lucian
  clean-port baseline is a good first profile (see that run's manifest `timings_seconds`). Ship it
  as `viewer/generate_baseline.json`.
- `total_eta = stage_eta_from_tqdm(current) + Σ baseline[future_stages]`.
- **Refine live:** when a stage finishes, overwrite that stage's baseline with its actual duration,
  so the estimate for a *re-run* and for later stages tightens.
- **Communicate uncertainty:** the fine Shape-SLat pass is subject-dependent and dominates the total
  (it was 2 min for Flicker, 42 min for Lucian). Until that pass reports a stable `s/it` (say, 2–3
  iterations in), show the total ETA as a **range** or a "still estimating…" state rather than a
  false-precise number.

**UI copy the owner asked for:** per active stage show `Shape SLat 5/12 · ~6 min left`, and up top
`Total ~N min` (or `~N–M min` while estimating). Completed stages show their actual duration.

## Reuse, don't reinvent

- The **GLB orbit viewer** already exists in the compare viewer (three.js loader, backface-cull
  toggle, camera). Extract/share that component so the "done" state renders the model with the same
  loader — including the **backface-cull toggle**, since culled is our acceptance test.
- The **manifest** the CLI writes (`<out>.json`) already has params + timings; show a compact
  summary (faces, holes if you compute them client-side, total time) beside the model.

## Milestones for Luna

1. **Mode switch** in the viewer shell (`Compare | Generate`); Generate mode renders an empty
   two-column layout. Compare untouched.
2. **Backend job API** in/alongside `serve.py`: `POST /api/generate` spawns the wrapper, `GET …/events`
   streams raw log lines (no parsing yet). One-job lock + cancel.
3. **Progress parser + ETA** (the schema above) feeding real progress bars.
4. **Done state** swaps the right pane to the shared GLB viewer + manifest summary + downloads.
5. **Polish:** alpha badge, guardrail messaging, reconnect-catches-up, error surface.

## Non-goals (for now)

- Multi-job queue UI (one-at-a-time lock is enough).
- Editing the compare viewer's compare/overlay features.
- License-class sorting / provenance sidecars (the research CLI path handles that; the web baseline
  writes the wrapper's manifest only).
- The old `trellis-mac` port — this view targets the clean `trellis-space-mac` wrapper only.
```
