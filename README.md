# Image to 3D Lab

**Turn a single image into a textured 3D model — locally on Apple Silicon, with a
license-provenance record for every result.**

Drop in a picture of a character or object; get back a `.glb` (with PBR texture) plus a
`.provenance.json` sidecar recording exactly how it was made and under which licenses.
Everything runs on your Mac — nothing is uploaded to a cloud service.

Powered by **TRELLIS.2** (Microsoft's image-to-3D model), ported to Apple Silicon with
Metal kernels. Output is geometrically close to the official Hugging Face demo: the Flicker
control matches the demo's face budget (278k vs 282k faces) and volume (within 5%).

---

## Quick start — web UI (recommended)

```bash
git clone <repo> && cd image-to-3dlab
python viewer/serve.py
# opens http://127.0.0.1:8777/viewer/index.html
```

Go to **Generate**. A **Setup** card tells you exactly where your machine stands:

- **Clean-port build** — if missing, click **Run setup** (bootstraps the Metal port, ~1h,
  needs `uv`, Python 3.11 and Xcode command-line tools), or run it manually:
  `python scripts/bootstrap_trellis_space_macos.py`
- **Model weights** — shows what's cached in `~/.cache/huggingface`; the first run
  downloads the ~14 GB TRELLIS.2-4B weights automatically

Then drop a **pre-masked PNG** (transparent background), pick resolution/seed, hit
**Generate**. Progress streams live; the GLB lands in `output/space_web/`. You can also
**Compare** two models side by side in the same viewer.

## CLI

Same engine without the browser:

```bash
# after the bootstrap:
vendor/trellis-space-mac/.venv/bin/python scripts/trellis_space_generate.py input.png output/out.glb
```

- `--check` verifies the environment first (seconds, no model load).
- Resume modes skip the expensive parts:
  - `--from-latents out_latents.pt` — skip sampling (stages 1–3), re-decode + bake
  - `--from-decode out_decode.pt` — skip sampling, decode **and** model load (bake only)
- Every run writes `<out>.glb`, `<out>_latents.pt`, `<out>_decode.pt`, and a `.json` manifest
  with exact params and per-stage timings.

## Requirements

| Thing | Why |
|---|---|
| Apple Silicon Mac (M-series) | Metal kernels; 32 GB unified memory recommended |
| macOS + Xcode command-line tools | compiles the Metal shaders during setup |
| `uv` | builds the reproducible Python environment |
| Python 3.11 | pinned by the bootstrap |
| ~14 GB disk | TRELLIS.2-4B weights (auto-downloaded once) |

## How the runs behave

- Sampling is attention-bound and scales with the subject's sparse structure: a simple
  subject (~8k tokens) takes ~14 min end-to-end; a complex one (~22k tokens, e.g. a fluffy
  creature) ~78 min. The HF demo is faster only because CUDA flash attention beats MPS SDPA
  at large token counts.
- Decode + bake adds a few minutes; the decode is cached, so re-bakes are ~1 min of setup.
- Known gaps vs the HF demo: slight texture drift (Stage-3 seed sensitivity — the frozen
  shape seed-search in `scripts/trellis_stage3.py` is the lever) and mostly-pinhole holes
  (561 vs 1 on Flicker).

## Licensing & provenance (non-negotiable)

- **TRELLIS.2** code and weights: MIT. **DINOv3** image encoder: separate DINOv3 License —
  so TRELLIS output is classified `commercial-conditional`.
- **BRIA RMBG-2.0 is disabled** by patch and must stay unloaded. Inputs must carry a real
  transparent alpha foreground; the pipeline refuses anything else unless you explicitly
  pass `--allow-rembg`.
- Every run emits a `.provenance.json` sidecar (hashes, settings, license classification,
  component licenses).

## Development

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q        # 490+ tests; backends that load real models stay manual
ruff check .
```

Conventions: Conventional Commits, Keep a Changelog (`CHANGELOG.md`), test-first (a unit
test is minutes; a generation run is 15–80). Judge assets **backface-culled, by eye** —
glTF is double-sided by default, so a hollow mesh looks fine in preview and fails only in a
game engine. Measure holes with a **position-only** vertex merge
(`merge_vertices(merge_tex=True, merge_norm=True)`).

## Legacy backends

The older `pipeline.py` CLI wraps three interchangeable backends — Stable Fast 3D (fast),
Hunyuan3D via local ComfyUI (quality), and the community trellis-mac port — with the same
provenance sidecar. It still works, but the clean TRELLIS.2 port above is the current path.
