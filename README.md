# Image to 3D Lab

![Three source images above the textured 3D models generated from them: a photoreal warrior bust, a stylised garden gnome, and a multi-object shoe-house diorama](docs/images/one-image-in-textured-model-out.jpg)

**Turn a single image into a textured 3D model, locally on Apple Silicon, with a
license-provenance record for every result.**

Apple Silicon deserves more love in the 3D and Imagen community. So this is an attempt at that. 


Drop in a picture of a character or object; get back a `.glb` (with PBR texture) plus a
`.provenance.json` sidecar recording exactly how it was made and under which licenses. This should make your game-dev or whatever else you're up to easier to manage.
Everything runs on your Mac; nothing is uploaded to a cloud service.

> **Apple Silicon only, for now.** Every backend here is built on MLX or Metal, so Windows
> and Linux/NVIDIA machines cannot run it yet. This repo wraps other people's ports rather
> than writing its own, so NVIDIA support means picking the right CUDA backend and wiring
> it in. Aiming for **30 September 2026**.

Five backends, one Generate page. Sadly life is full of trade-offs, so pick the tradeoff you want (lol):

| Backend | Best for | Setup | License |
|---|---|---|---|
| **Pixal3D (C++/GGML, Metal)** ⭐ | Best results we have; one pass, ~6 min, no repaint needed | One script: `scripts/bootstrap_pixal3d_cpp.sh` (needs Xcode's Metal compiler, 8.1 GB weights) | MIT (code + flow weights); DINOv3 License (bundled encoder) |
| **Hunyuan3D-MLX (Xiong, full pipeline)** | Fast, clean results | Clone-and-go: code is tracked in this repo, weights download separately | MIT (code); Tencent Community License (weights) |
| **Hunyuan3D-MLX (dgrauet shape + Xiong paint)** | The single cleanest shape we've tested, at the cost of manual setup | Vendor-cloned, manual | Tencent Community License (code + weights) |
| **TRELLIS.2** | Highest fidelity, closest to the official demo | One-button bootstrap from the web UI (~1h) | MIT + DINOv3 License |
| **Stable Fast 3D** | Fastest, lower fidelity | Vendor-cloned, manual | Stability AI Community License |

⭐ Start with **Pixal3D**. It is TRELLIS.2's backbone with pixel-aligned, camera-aware
conditioning, and on the assets tested here it produced correct saturated colour in a single
pass where TRELLIS.2 bleached flat illustrations badly enough to need a separate repaint
stage, in 5m50s against 14 min. Raise `--gss` to 10; at the 7.5 default a thin sword blade
went missing entirely. Numbers, caveats and the Mac-port comparison:
[`docs/pixal3d-evaluation-2026-09-20.md`](docs/pixal3d-evaluation-2026-09-20.md).

<p align="center">
  <img src="docs/images/turntable-pixal3d-warrior.webp" width="360"
       alt="A full 360-degree turn of the generated warrior bust, showing textured geometry from every side">
  <br>
  <sub>The warrior above, turned through 360°. Pixal3D, one pass, no repaint stage.<br>
  Every model on this page came from a single image on an M-series Mac.</sub>
</p>

Hunyuan3D-MLX (Xiong, full pipeline) remains the quickest to get running from a fresh clone
(~9 min shape+paint end to end at its default model).
Reach for TRELLIS.2 when fidelity matters more than speed. Its material model can produce
severe colour drift on flat/vector-style illustrations; prefer photographs or softly lit
3D-style references. This is an input-dependent upstream model behaviour, not a Metal-port
artifact. See the [investigation and input guidance](docs/trellis2-flat-illustration-colour-drift.md).

---

## Quick start: web UI (recommended)

```bash
git clone https://github.com/Bingeljell/image-to-3dlab.git && cd image-to-3dlab
python3 -m venv .venv && .venv/bin/pip install Pillow
.venv/bin/python viewer/serve.py
# opens http://127.0.0.1:8777/viewer/index.html
```

That's it. No other deps needed until you pick a backend below.

Go to **Generate**, pick a backend from the dropdown. Each one has its own **Setup**
status telling you exactly what's missing:

- **Hunyuan3D-MLX (Xiong, full pipeline)**: the code is already there (tracked in this
  repo at `hunyuan_mlx/`). Run once per machine:
  ```bash
  uv sync --project hunyuan_mlx/shape
  uv sync --project hunyuan_mlx/paint
  hunyuan_mlx/shape/.venv/bin/python hunyuan_mlx/download_weights.py
  ```
  Downloads the 2.0 shape model plus the paint weights from Hugging Face, about 13 GB,
  and prints the sizes before it starts. `--model 2.1` or `--model 2.0-turbo` fetches a
  different one; `--all` fetches every shape model, which is about 24 GB and more than
  the default route uses. Full detail, including the one extra manual step for RealESRGAN super-res
  weights: [`docs/hunyuan-mlx-recipes.md`](docs/hunyuan-mlx-recipes.md).
- **Pixal3D**: one script, no venv of its own:
  ```bash
  scripts/bootstrap_pixal3d_cpp.sh
  ```
  Builds [`raven38/pixal3d.cpp`](https://github.com/raven38/pixal3d.cpp) with Metal (the
  backend is automatic on Apple builds) and fetches the 8.1 GB Q8_0 single-view weight set.
  Needs Xcode's Metal compiler, not just the command-line tools; the script prints the two
  commands that fix that if it is missing. No Hugging Face token: the matting and image
  encoders are ungated mirrors, and BRIA RMBG-2.0 is never used.
- **TRELLIS.2**: click **Run setup** (bootstraps the Metal port, ~1h, needs `uv`,
  Python 3.11 and Xcode command-line tools), or run it manually:
  `python scripts/bootstrap_trellis_space_macos.py`. First run downloads the ~14 GB
  TRELLIS.2-4B weights automatically. Selecting an image also runs an optional local
  TinyCLIP style advisory; its small checkpoint downloads on first use and never blocks
  generation.
- **Hunyuan3D-MLX (dgrauet shape + Xiong paint)** and **Stable Fast 3D**: no automated
  setup or documented setup guide yet; background and licensing in
  [`docs/info_and_credits.md`](docs/info_and_credits.md), but expect to read the source
  (`scripts/hunyuan_mlx_generate.py`, `viewer/generate_api.py`) to set these up by hand.

Then drop a **pre-masked PNG** (transparent background), pick your settings, hit
**Generate**. Progress streams live; the GLB lands in `output/`. You can also **Compare**
two models side by side in the same viewer.

## CLI

Same engines without the browser.

**Pixal3D:**
```bash
python scripts/pixal3d_generate.py input.png output.glb --res 1024 --seed 42
```
A pre-matted RGBA image skips background removal entirely and keeps the cutout identical to
whatever else you ran on it.

**Hunyuan3D-MLX (Xiong, full pipeline):**
```bash
hunyuan_mlx/shape/.venv/bin/python scripts/hunyuan_mlx_xiong_generate.py \
    input.png output.glb --model 2.0
```

### Reproducible runs

For a run you can audit or repeat later, use a manifest; it records the input, the
backend, every parameter and the licensing intent the run was gated on:

```bash
cp manifests/example-trellis2.json manifests/my-run.json
# point "input.path" at your own image, then:
python pipeline.py --run-manifest manifests/my-run.json
```

Paths inside a manifest resolve relative to the manifest file, not your working
directory. Manifests you write land in `manifests/` and stay local; only the template is
tracked. See [`manifests/README.md`](manifests/README.md).

**TRELLIS.2** (after the bootstrap):
```bash
vendor/trellis-space-mac/.venv/bin/python scripts/trellis_space_generate.py input.png output/out.glb
```
- `--check` verifies the environment first (seconds, no model load).
- Resume modes skip the expensive parts:
  - `--from-latents out_latents.pt`: skip sampling (stages 1–3), re-decode + bake
  - `--from-decode out_decode.pt`: skip sampling, decode **and** model load (bake only)
- Every run writes `<out>.glb`, `<out>_latents.pt`, `<out>_decode.pt`, and a `.json` manifest
  with exact params and per-stage timings.
- In the web UI, a failed TRELLIS decode or bake retains `<out>_latents.pt` even when
  **Debug** is off, so the expensive sampling stage can be resumed. Successful non-debug
  runs clean up the checkpoint after the GLB is safely written.

## Finishing an asset

Generated assets arrive dense and heavy, often ~900k faces and 30+ MB, nearly all of it
uncompressed texture. The **Finish** page in the viewer, and the same chain on the CLI,
brings that down without a visible quality cost:

```bash
python scripts/retopo_repaint.py generated.glb source.png finished.glb \
    --faces 40000 --skip-paint
```

Three stages, each skippable:

1. **Retopologise**: voxel-remesh, then decimate. The ordering matters: decimating the raw
   mesh shatters thin geometry, measured.
2. **Repaint** (optional): hands the clean mesh to Hunyuan 2.1 PBR and paints from the
   source art. Use it when the generator's own texture is wrong; Pixal3D output usually does
   not need it, so `--skip-paint` finishes in seconds instead of ~6 minutes.
3. **Compress**: re-encodes the textures. The paint stage emits two uncompressed 4096²
   PNGs; core-glTF JPEG at 2048 measures below the renderer's own sampling noise and takes a
   typical asset from 32 MB to under 5.

Every run writes a JSON record of the settings used, so a batch of finished assets is
comparable rather than each one being tuned by hand.

## Where this is going

[`docs/browser-workshop.md`](docs/browser-workshop.md) is the product and architecture
boundary for the browser workshop: upload a creature image, generate a 3D asset, make it
deformable with a known rig, paint it, author an animation, export a GLB. Read it before
adding to `viewer/`.

## Blender animation recipes

The reusable Blender tooling lives in `scripts/blender_*.py`: import, inspect,
stage, bake, render, rig and rebind helpers that work on any mesh this pipeline
produces. **[`scripts/README.md`](scripts/README.md) indexes every tool in the
repository**, grouped by what you are trying to do, and is kept honest by a test
that reads each script's own docstring.

Per-creature rigs and animations are **not** shipped. They lived here once and
were model-specific references rather than drop-in tools, so they now sit in a
git-ignored `characters/<name>/` folder alongside their tests. The techniques are
documented in `docs/`; the creature-specific scripts are ours, not yours.

The [reusable quadruped gait plan](docs/reusable-gait-quadruped-trot.md) records
the accepted trot, the implementation handoff and the validation needed before
claiming support across Rigify basic-quadruped characters.

## Requirements

**Not supported on Windows or Linux/NVIDIA yet.** The backends are MLX and Metal builds, so
there is no route that finishes on those machines; the setup page will tell you so rather
than starting a download it cannot use. NVIDIA is next, by wiring in an existing CUDA
backend: the work here is the tuning and the pipeline on top, not the port underneath.
Aiming for **30 September 2026**.

| Thing | Why |
|---|---|
| Apple Silicon Mac (M-series) | Metal kernels / MLX; 32 GB unified memory recommended |
| macOS + Xcode command-line tools | compiles the Metal shaders during TRELLIS setup |
| `uv` | builds the reproducible Python environments |
| Python 3.11 (TRELLIS) / 3.12 (Hunyuan3D-MLX) | pinned by each backend's own setup |
| ~13 GB disk | Hunyuan3D-MLX 2.0 shape + paint weights (auto-downloaded once) |
| ~14 GB disk | TRELLIS.2-4B weights (auto-downloaded once, if using TRELLIS) |
| ~94 MB download | TinyCLIP flat-input advisor (local and non-blocking) |

## How the runs behave

- **Pixal3D:** ~6 min end to end at res 1024, including model load, on a 32 GB M-series Mac.
  Stage split on a real run: sparse structure 77s, shape SLAT 512 then the 1024 cascade 136s,
  decode 13s, texture 63s, postprocess 25s. Peak memory is modest; the Q8_0 weight set is
  8.1 GB against 24 for the PyTorch port, which does not fit 32 GB at all.
- **Hunyuan3D-MLX (Xiong, 2.0, default settings):** ~9 min end to end (shape + paint) on
  a real benchmark run. 2.0-turbo trades some fine-detail cleanliness for ~2-3 min shape.
  See [`docs/hunyuan-mlx-recipes.md`](docs/hunyuan-mlx-recipes.md) for the full model
  comparison.
- **TRELLIS.2:** sampling is attention-bound and scales with the subject's sparse
  structure; a simple subject (~8k tokens) takes ~14 min end-to-end; a complex one
  (~22k tokens, e.g. a fluffy creature) ~78 min on my m5 w/ 32 gigs of unified memory. This is infinitely faster on CUDA / Nvidia.
  Setting **Attention backend** to `mlx` routes attention through MLX's fused Metal kernel
  and cut a 1024 run from 34.3 to 14.3 min; output at a fixed seed is unchanged. Only worth
  it at 1024 and above. See [`docs/mlx-attention-2026-09-20.md`](docs/mlx-attention-2026-09-20.md). 
  Decode + bake adds a few minutes; with Debug enabled the decode is cached, so re-bakes
  are ~1 min of setup. Known gaps vs the HF demo: slight
  texture drift, severe colour failures on some flat/vector inputs, and mostly-pinhole
  holes. See the [TRELLIS.2 input guidance](docs/trellis2-flat-illustration-colour-drift.md).

## Licensing & provenance (non-negotiable)

- **Pixal3D** code and flow weights: MIT. The Q8_0 bundle also carries the **DINOv3** image
  encoder under its own licence, so treat its output the same as TRELLIS's. Background
  removal uses ungated `ZhengPeng7/BiRefNet`, never BRIA RMBG-2.0.
- **TRELLIS.2** code and weights: MIT. **DINOv3** image encoder: separate DINOv3 License,
  so TRELLIS output is classified `commercial-conditional`.
- **TinyCLIP ViT-8M/16** input advisor: MIT. It only warns about risky input style and is
  not part of the generated artifact.
- **Hunyuan3D-2 / 2.1 model weights** (used by both Hunyuan3D-MLX backends): Tencent
  Hunyuan Community License; **not licensed for use in the EU, UK, or South Korea**;
  verify exact terms per model before any redistribution-sensitive use.
- **Hunyuan3D-MLX (Xiong, full pipeline) code**: MIT, tracked in this repo at
  `hunyuan_mlx/`, safe to clone and modify freely (weights are the license-restricted
  part, downloaded separately).
- **Hunyuan3D-MLX (dgrauet shape) code**: Tencent Hunyuan Community License, not MIT;
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
**backface-culled, by eye**. glTF is double-sided by default, so a hollow mesh looks fine
in preview and fails only in a game engine. Measure holes with a **position-only** vertex
merge (`merge_vertices(merge_tex=True, merge_norm=True)`).

## Credits

This repo trains nothing and invents nothing; it builds upon other people's models and
work. What it *does* add is filling the gaps that exist to make some of these models work
on Apple Silicon, and improving the overall experience. Grateful to everyone who built
before me; they are named and credited in
[`docs/info_and_credits.md`](docs/info_and_credits.md). Also a special thanks to Claude
and Codex for being my partners through this! Not just helping me build, but teaching me
so much along the way. Yes, I just credited AI.
