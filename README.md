# Image to 3D Lab



**Turn a single image into a textured 3D model — locally on Apple Silicon, with a
license-provenance record for every result.**

Apple Silicon deserves more love in the 3D and Imagen community. So this is an attempt at that. 


Drop in a picture of a character or object; get back a `.glb` (with PBR texture) plus a
`.provenance.json` sidecar recording exactly how it was made and under which licenses. This should make your game-dev or whatever else you're up to easier to manage.
Everything runs on your Mac — nothing is uploaded to a cloud service.

Four backends, one Generate page. Sadly life is full of trade-offs, so pick the tradeoff you want (lol):

| Backend | Best for | Setup | License |
|---|---|---|---|
| **Hunyuan3D-MLX (Xiong, full pipeline)** ⭐ | Fast, clean results — recommended default | Clone-and-go: code is tracked in this repo, weights download separately | MIT (code); Tencent Community License (weights) |
| **Hunyuan3D-MLX (dgrauet shape + Xiong paint)** | The single cleanest shape we've tested, at the cost of manual setup | Vendor-cloned, manual | Tencent Community License (code + weights) |
| **TRELLIS.2** | Highest fidelity, closest to the official demo | One-button bootstrap from the web UI (~1h) | MIT + DINOv3 License |
| **Stable Fast 3D** | Fastest, lower fidelity | Vendor-cloned, manual | Stability AI Community License |

⭐ Start with Hunyuan3D-MLX (Xiong, full pipeline) — it's the quickest to get running from a
fresh clone and gives strong results (~9 min shape+paint end to end at its default model).
Reach for TRELLIS.2 when fidelity matters more than speed. But be warned, Trellis texture has minor drift (a Metal port artifact). 
Working to see how we can be more colour accurate. 

---

## Quick start — web UI (recommended)

```bash
git clone <repo> && cd image-to-3dlab
python viewer/serve.py
# opens http://127.0.0.1:8777/viewer/index.html
```

Go to **Generate**, pick a backend from the dropdown. Each one has its own **Setup**
status telling you exactly what's missing:

- **Hunyuan3D-MLX (Xiong, full pipeline)** — the code is already there (tracked in this
  repo at `hunyuan_mlx/`). Run once per machine:
  ```bash
  uv sync --project hunyuan_mlx/shape
  uv sync --project hunyuan_mlx/paint
  hunyuan_mlx/shape/.venv/bin/python hunyuan_mlx/download_weights.py
  ```
  Downloads three shape models (2.1, 2.0, 2.0-turbo — 2.0 is the default and the
  recommended one) plus paint weights from Hugging Face, ~13 GB total for the default
  model. Full detail, including the one extra manual step for RealESRGAN super-res
  weights: [`docs/hunyuan-mlx-recipes.md`](docs/hunyuan-mlx-recipes.md).
- **TRELLIS.2** — click **Run setup** (bootstraps the Metal port, ~1h, needs `uv`,
  Python 3.11 and Xcode command-line tools), or run it manually:
  `python scripts/bootstrap_trellis_space_macos.py`. First run downloads the ~14 GB
  TRELLIS.2-4B weights automatically.
- **Hunyuan3D-MLX (dgrauet shape + Xiong paint)** and **Stable Fast 3D** — no automated
  setup or documented setup guide yet; background and licensing in
  [`docs/info_and_credits.md`](docs/info_and_credits.md), but expect to read the source
  (`scripts/hunyuan_mlx_generate.py`, `viewer/generate_api.py`) to set these up by hand.

Then drop a **pre-masked PNG** (transparent background), pick your settings, hit
**Generate**. Progress streams live; the GLB lands in `output/`. You can also **Compare**
two models side by side in the same viewer.

## CLI

Same engines without the browser.

**Hunyuan3D-MLX (Xiong, full pipeline):**
```bash
hunyuan_mlx/shape/.venv/bin/python scripts/hunyuan_mlx_xiong_generate.py \
    input.png output.glb --model 2.0
```

**TRELLIS.2** (after the bootstrap):
```bash
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
| Apple Silicon Mac (M-series) | Metal kernels / MLX; 32 GB unified memory recommended |
| macOS + Xcode command-line tools | compiles the Metal shaders during TRELLIS setup |
| `uv` | builds the reproducible Python environments |
| Python 3.11 (TRELLIS) / 3.12 (Hunyuan3D-MLX) | pinned by each backend's own setup |
| ~13 GB disk | Hunyuan3D-MLX 2.0 shape + paint weights (auto-downloaded once) |
| ~14 GB disk | TRELLIS.2-4B weights (auto-downloaded once, if using TRELLIS) |

## How the runs behave

- **Hunyuan3D-MLX (Xiong, 2.0, default settings):** ~9 min end to end (shape + paint) on
  a real benchmark run. 2.0-turbo trades some fine-detail cleanliness for ~2-3 min shape.
  See [`docs/hunyuan-mlx-recipes.md`](docs/hunyuan-mlx-recipes.md) for the full model
  comparison.
- **TRELLIS.2:** sampling is attention-bound and scales with the subject's sparse
  structure — a simple subject (~8k tokens) takes ~14 min end-to-end; a complex one
  (~22k tokens, e.g. a fluffy creature) ~78 min on my m5 w/ 32 gigs of unified memory. This is infinitely faster on CUDA / Nvidia. 
  Decode + bake adds a few minutes; the decode is cached, so re-bakes are ~1 min of setup. Known gaps vs the HF demo: slight
  texture drift and mostly-pinhole holes.

## Licensing & provenance (non-negotiable)

- **TRELLIS.2** code and weights: MIT. **DINOv3** image encoder: separate DINOv3 License —
  so TRELLIS output is classified `commercial-conditional`.
- **Hunyuan3D-2 / 2.1 model weights** (used by both Hunyuan3D-MLX backends): Tencent
  Hunyuan Community License — **not licensed for use in the EU, UK, or South Korea**;
  verify exact terms per model before any redistribution-sensitive use.
- **Hunyuan3D-MLX (Xiong, full pipeline) code**: MIT — tracked in this repo at
  `hunyuan_mlx/`, safe to clone and modify freely (weights are the license-restricted
  part, downloaded separately).
- **Hunyuan3D-MLX (dgrauet shape) code**: Tencent Hunyuan Community License, not MIT —
  the code itself, not just the weights, carries the same restriction. Stays vendor-cloned
  rather than tracked in this repo for that reason.
- **BRIA RMBG-2.0 is disabled** by patch and must stay unloaded in the TRELLIS pipeline.
  Inputs must carry a real transparent alpha foreground; the pipeline refuses anything else
  unless you explicitly pass `--allow-rembg`.
- Every run emits a `.provenance.json` sidecar (hashes, settings, license classification,
  component licenses).

Full credits and per-backend detail: [`docs/info_and_credits.md`](docs/info_and_credits.md).

## Development

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q        # 551 tests; backends that load real models stay manual
ruff check .
```

Conventions: Conventional Commits, Keep a Changelog (`CHANGELOG.md`), test-first. Judge assets
**backface-culled, by eye** — glTF is double-sided by default, so a hollow mesh looks fine
in preview and fails only in a game engine. Measure holes with a **position-only** vertex
merge (`merge_vertices(merge_tex=True, merge_norm=True)`).
