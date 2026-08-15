# Architecture

## Overview

One CLI turns a single image into a 3D asset (`.glb`) via one of three backends. Every run
produces the asset **and** a `.provenance.json` sidecar, sorted into a license-class folder.

```
image ──► pipeline.py ──► image_to_3dlab.cli.main
                              │
              ┌───────────────┼───────────────────────────┐
              │ 1. resolve args / run manifest             │
              │ 2. validate_run_policy (license gate)      │  ◄── refuses before any model runs
              │ 3. dispatch to a backend                   │
              │ 4. finalize_output (move + provenance)     │
              └───────────────┬───────────────────────────┘
                              ▼
     ┌────────────┬───────────────────────┬──────────────────────┐
     │  --fast    │      --quality        │      --trellis        │
     │  SF3D      │  Hunyuan3D / ComfyUI  │  TRELLIS.2 (Mac port) │
     │ in-process │  HTTP to :8188        │  subprocess venv      │
     └────────────┴───────────────────────┴──────────────────────┘
                              ▼
             output/<license-folder>/<name>__<backend>__<class>__<id>.glb
             + .provenance.json sidecar
```

## Request flow (`image_to_3dlab/cli.py`)

1. **Argument resolution.** Either a positional image + exactly one mode flag, or
   `--run-manifest` (schema v1), which sets the backend, image path, and parameters from JSON.
2. **Intent + policy.** Intent (`use_case`, `distribution`, `commercial_intent`) and the
   license policy come from the manifest (or safe defaults). `validate_run_policy` runs
   **before** any model work and can abort the run (exit 2).
3. **Dispatch.** The selected backend generates into `output/.working/`.
4. **Finalize.** `finalize_output` moves the asset into its license folder with a
   provenance-bearing filename and writes the sidecar.

## Backends (`image_to_3dlab/*_backend.py`)

- **SF3D (`--fast`)** — imports the vendored `sf3d` package and runs on MPS in-process.
  Auto-retries on CPU if MPS reports OOM (`_is_mps_oom`). rembg/U²-Net removes the background.
- **Hunyuan3D (`--quality`)** — `ComfyUIClient` talks to a local ComfyUI at `:8188`: uploads
  the image, patches the graph's `LoadImage`, queues `/prompt`, polls `/history`, downloads
  the first 3D result. No raw Python Hunyuan backend — the API workflow is the contract.
- **TRELLIS.2 (`--trellis`)** — shells into a **separate** `vendor/trellis-mac` venv so its
  pinned deps stay isolated. Pre-mattes the input to RGBA (U²-Net). Refuses to run unless the
  BRIA-disable patch is present. Reports whether the Metal or CPU/KDTree texture path was used.

## Provenance & licensing (`image_to_3dlab/provenance.py`)

- `LICENSES` maps each backend to a `LicenseProfile` (classification, output folder, license
  name/url, conditions).
- `validate_run_policy` enforces the declared policy up front (e.g. blocks Hunyuan for
  worldwide game distribution; honors `allow_conditional`).
- `finalize_output` writes a schema-v1 sidecar: input/output SHA-256, intent, model
  parameters, per-component licenses (incl. an explicit `BRIA loaded: false` record for
  TRELLIS), package versions, backend git revision, and the source manifest hash.

## Key invariants

- The license gate runs before model execution — keep it that way.
- The TRELLIS BRIA guardrail is a hard stop, not a warning.
- Manifest schema and provenance schema are versioned; bumps are breaking (SemVer major).
- `vendor/` and `output/` are git-ignored; backends are bootstrapped locally.
