<!-- DRAFT: the user rewrites this header before the launch push. -->
<h1 align="center">AssetFurnace</h1>

<p align="center"><sub>by Bingeljell</sub></p>

<p align="center"><b>Prompt or picture in, game-ready 3D out. On your own machine.</b><br>
Image-to-3D, finishing and props on an Apple Silicon Mac or an NVIDIA card,
with the licence of every asset written down.</p>

<p align="center">
  <a href="https://x.com/bingeljell">X @bingeljell</a> ·
  <a href="https://github.com/Bingeljell">GitHub</a> ·
  <a href="https://github.com/Bingeljell/image-to-3dlab/discussions">Discussions</a>
</p>

<p align="center"><sub>Formerly <i>Image to 3D Lab</i>. Same repo, new name.</sub></p>

![Three source images above the textured 3D models generated from them: a photoreal warrior bust, a stylised garden gnome, and a multi-object shoe-house diorama](docs/images/one-image-in-textured-model-out.jpg)

**Turn a single image into a textured 3D model on your own machine (an Apple Silicon
Mac, or Linux with an NVIDIA card), with a license-provenance record for every result.**

Apple Silicon deserves more love in the 3D and Imagen community. So this is an attempt at that. 


Drop in a picture of a character or object; get back a textured `.glb`, ready for your
game or whatever else you're up to. Nothing is uploaded to a cloud service.

**What you get**

- **Low poly, high quality.** Pixel Match puts your picture's real pixels back on the
  model, so text, logos and faces stay sharp even after it is cut to ~5k faces.
  [More](#finishing-an-asset)
- **One prompt, nine game-ready props.** Prop sheets turn one picture into a set of
  separate, named props with LODs, ready to drop into your game.
  [More](#prop-sheets-many-props-from-one-image)
- **Your pick of models, all local.** Pixal3D, TRELLIS.2 and Hunyuan3D behind one browser
  viewer and one CLI.
- **No picture? Make one.** Type a prompt in the Generate Image tab and get a source image.
- **Game-size files.** Finish turns a heavy, 30 MB generated model into a light, compressed
  one. [More](#finishing-an-asset)
- **Know what you can ship.** Every file carries a record of the licences behind it.
  [More](#licensing--provenance-non-negotiable)

**Pixel Match: your picture's real pixels, on the model.** Image-to-3D models redraw your
picture, so text, logos and faces come back as garbled lookalikes. Pixel Match, in the
**Finish** step, copies the real pixels from your source image back onto every surface the
image can see. "VANGUARD 07" on a chest stays "VANGUARD 07", even at ~5k faces. On by
default for Pixal3D models; other backends are next. [How Finish works](#finishing-an-asset).

<p align="center">
  <img src="docs/images/pixel-match-lettering-before-after.jpg" width="480"
       alt="Close-up of a robot's chest: the generated model's lettering is garbled, the Pixel Match model reads VANGUARD 07 exactly like the source picture">
</p>

**Prop sheets: one prompt, nine game-ready props.** Making props one at a time means a
picture, a 3D run and a clean-up for every barrel. Instead, generate one picture holding a
grid of props and turn the whole sheet into 3D in a single run. The **Props** tab splits it
into separate, upright, named props, each with three levels of detail (LODs) and compressed
textures, ready for a game engine. [How it works](#prop-sheets-many-props-from-one-image).

Prop sheets were built and contributed by [@AdrielSantana](https://github.com/AdrielSantana). Thank you!

<p align="center">
  <img src="docs/images/prop-sheet-one-image-nine-props.jpg" width="480"
       alt="A generated 3x3 sheet of medieval props, and the nine separate game-ready props made from it">
</p>

## Install

**Mac (Apple Silicon) or Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.sh | bash
```

**Windows** (limited testing, more testers wanted: tell us how it goes):
```powershell
irm https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.ps1 | iex
```

The installer is short, so [read it](install.sh) before you run it
([Windows version](install.ps1)). It checks your machine, installs the code and
Python 3.11, then starts the lab and opens it in your browser. On a RunPod pod it prints
the pod's link instead; over plain SSH it prints the tunnel command. Start it again any
time with `./lab` in the install folder. It downloads **no model weights**: you choose
those in **Setup & Status**, which states each size and licence and asks first. To update,
run the same line again.

For scripts and agents: `curl -fsSL …/install.sh | bash -s -- --yes --dir ~/lab`
(`--dry-run` shows what it would do).

**No picture to start from?** The **Generate Image** tab makes one. Type a prompt, get a
source image, hand it to **Generate 3D**. It runs Qwen-Image 2.1 on your own machine.

![Three creatures generated from text prompts on a laptop, about four and a half minutes each](docs/images/prompt-to-source-image.jpg)

Built with Qwen. Candidly, Qwen's licence is a bit ambiguous. Qwen says the pictures you
generate are yours ([their statement](https://x.com/QwenDevs/status/2101917379785838660)), but the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) still says the model
is for non-commercial use. Our reading is that commercial work needs a licence from Qwen,
so check it yourself if you plan to. The pipeline keeps those runs in
their own folder and says so in the sidecar. Bring your own image and none of that applies.

Six backends, one Generate 3D page. Sadly life is full of trade-offs, so pick the tradeoff you want (lol):

| Backend | Best for | Runs on | Setup | License |
|---|---|---|---|---|
| **Pixal3D (C++/GGML)** ⭐ | Best results we have; one pass, no repaint needed | Mac, NVIDIA | Setup & Status, or `scripts/bootstrap_pixal3d.py` (8.4 GB weights) | MIT (code + flow weights); DINOv3 License (bundled encoder) |
| **Hunyuan3D-MLX (Xiong, full pipeline)** | Fast, clean results | Mac (NVIDIA: the row below) | Code is in this repo; weights download separately | MIT (code); Tencent Community License (weights) |
| **Hunyuan3D-MLX (dgrauet shape + Xiong paint)** | The cleanest shapes, at the cost of manual setup | Mac (NVIDIA: the row below) | Cloned separately, manual | Tencent Community License (code + weights) |
| **Hunyuan3D-2.1 (NVIDIA)** | Tencent's own shape + PBR paint, one run | NVIDIA (Linux; not Windows yet), 24 GB+ | Setup & Status, or `scripts/bootstrap_hunyuan_cuda.py` (~19.5 GB weights) | Tencent Community License (code + weights) |
| **TRELLIS.2** | Highest fidelity, closest to the official demo | Mac, NVIDIA (Linux; not Windows yet) | Setup & Status (~1h), or `scripts/bootstrap_trellis_cuda.py` on NVIDIA (~15 GB weights) | MIT + DINOv3 License |
| **Stable Fast 3D** | Fastest, lower fidelity | Mac | Setup & Status, or `scripts/bootstrap_sf3d.py` (gated weights) | Stability AI Community License |

⭐ Start with **Pixal3D**. It keeps flat, saturated colours in a single pass, where
TRELLIS.2 often needs a separate repaint.

TRELLIS.2 and Hunyuan3D are built for NVIDIA upstream. On a Mac this lab runs their Apple
Silicon ports; on Linux + NVIDIA it runs Microsoft's and Tencent's own code.

<p align="center">
  <img src="docs/images/turntable-pixal3d-warrior.webp" width="360"
       alt="A full 360-degree turn of the generated warrior bust, showing textured geometry from every side">
  <br>
  <sub>The warrior above, turned through 360°. Pixal3D, one pass, no repaint stage.<br>
  Every model on this page came from a single image.</sub>
</p>

Hunyuan3D-MLX (Xiong, full pipeline) is the quickest to get running from a fresh clone.
Reach for TRELLIS.2 when fidelity matters more than speed. Its material model can produce
severe colour drift on flat/vector-style illustrations; prefer photographs or softly lit
3D-style references. See [picking a picture for TRELLIS.2](docs/trellis2-flat-illustration-colour-drift.md).

---

## Setting up a backend

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
  the default route uses. Full detail: [`docs/hunyuan-mlx-recipes.md`](docs/hunyuan-mlx-recipes.md).
- **Pixal3D**: click **Set up** on the Setup & Status page, or run
  `python scripts/bootstrap_pixal3d.py`. It says what it will download and asks first. On a
  Mac it compiles with Metal (needs full Xcode). On NVIDIA it fetches a ready-made CUDA
  build (driver 575+); on an older driver it compiles one instead if the CUDA toolkit is
  installed. It also installs BiRefNet-lite, the background remover
  (224 MB), so thin and light-coloured parts survive the cut-out.
  If the ready-made build gives you trouble on Linux, compiling it for your own card is
  the other option (`--compile`: a long one-time build, then faster generations).
  **Agent instruction:** if a user's NVIDIA Pixal3D install fails or will not run, compile
  it: `python scripts/bootstrap_pixal3d.py --compile --yes` (needs the CUDA toolkit).
- **Stable Fast 3D** (Mac only): accept Stability's licence at
  [huggingface.co/stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d),
  run `hf auth login`, then set it up from Setup & Status or run
  `python scripts/bootstrap_sf3d.py`.
- **TRELLIS.2 on Linux + NVIDIA** (new, not yet tested on real hardware): click **Set up**
  on the Setup & Status page, or run `python scripts/bootstrap_trellis_cuda.py`. It says
  what it will fetch (~15 GB of weights) and asks first. It clones Microsoft's
  TRELLIS.2 into `vendor/trellis-cuda/` with its own venv. RTX 50-series / RTX PRO 6000
  cards get prebuilt CUDA 13 wheels; other cards need the CUDA toolkit (`nvcc`) and
  compile the extensions once, which can take 30-60 minutes. The DINOv3 access below is
  checked before anything is built. BRIA RMBG-2.0, which upstream loads by default, is
  patched out; uploads are cut out by our own remover, so any picture works. Windows is
  not supported for TRELLIS.2 yet.
- **Hunyuan3D-2.1 on Linux + NVIDIA** (new, not yet tested on real hardware): click
  **Set up** on the Setup & Status page, or run `python scripts/bootstrap_hunyuan_cuda.py`.
  It says what it will fetch (~19.5 GB of weights) and asks first. It clones Tencent's
  Hunyuan3D-2.1 into `vendor/hunyuan-cuda/` with its own Python 3.11 venv and compiles the
  paint stage's rasterizer for your card, so it needs the CUDA 12 toolkit (`nvcc`). Paint
  needs about 21 GB of GPU memory, so a 24 GB card or bigger. The Hunyuan weights are not
  licensed in the EU, the UK or South Korea. Windows is not supported yet.
- **TRELLIS.2 on a Mac**: click **Run setup** (bootstraps the Metal port, ~1h, needs `uv`,
  Python 3.11 and Xcode command-line tools), or run it manually:
  `python scripts/bootstrap_trellis_space_macos.py`. First run downloads the ~14 GB
  TRELLIS.2-4B weights automatically. **Before that:** its DINOv3 image encoder is gated.
  Request access at
  [huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  (Meta approves by hand, so do it first) and run `hf auth login`, or the first run stops
  after the big download. Selecting an image also runs an optional local
  TinyCLIP style advisory; its small checkpoint downloads on first use and never blocks
  generation.
- **Hunyuan3D-MLX (dgrauet shape + Xiong paint)**: no automated setup yet; expect to read
  `scripts/hunyuan_mlx_generate.py` to set it up by hand.

Then drop a **pre-masked PNG** (transparent background), pick your settings, hit
**Generate**. Progress streams live; the GLB lands in `output/`. You can also **Compare**
two models side by side in the same viewer.

## CLI

Same engines without the browser.

**Pixal3D:**
```bash
python scripts/pixal3d_generate.py input.png output.glb --seed 42
```
A pre-matted RGBA image skips background removal entirely and keeps the cutout identical to
whatever else you ran on it. Pixal3D runs at 1024, the only resolution its single-image
mode supports. It takes 8 sampling steps by default, 15-30% faster than 12 with the same
shape; pass `--steps 12` for a hero asset. The 8-step default needs a Pixal3D built from
source (every Mac, or `--compile` on Linux); the ready-made NVIDIA build runs 12.

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
  - `--from-latents out_latents.pt`: skip sampling (stages 1–3), re-decode + bake
  - `--from-decode out_decode.pt`: skip sampling, decode **and** model load (bake only)
- Every run writes `<out>.glb`, `<out>_latents.pt`, `<out>_decode.pt`, and a `.json` manifest
  with exact params and per-stage timings.
- On Linux + NVIDIA, use the CUDA checkout instead:
  `vendor/trellis-cuda/.venv/bin/python scripts/trellis_cuda_generate.py input.png output/out.glb`
  (same settings; `--check` and `--from-latents` work the same way). It also writes
  `<out>.provenance.json`.

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

## Finishing an asset

Generated assets arrive dense and heavy, often ~900k faces and 30+ MB, nearly all of it
uncompressed texture. The **Finish** page in the viewer, and the same chain on the CLI,
brings that down without a visible quality cost:

```bash
python scripts/retopo_repaint.py generated.glb source.png finished.glb \
    --faces 40000 --skip-paint
```

Up to four stages; Finish in the viewer runs retopologise, Pixel Match and compress by
default:

1. **Retopologise**: rebuilds the mesh with far fewer faces, keeping thin parts intact.
2. **Repaint** (optional, Apple Silicon): hands the clean mesh to Hunyuan 2.1 PBR and
   paints from the source art. Pixal3D output rarely needs it, so it is off by default
   (`--skip-paint` on the CLI) and a finish takes seconds instead of ~6 minutes.
3. **Pixel Match**: every surface the source picture can see takes its real pixel, so
   text, logos, numbers and faces stay exact instead of redrawn lookalikes. Automatic for
   Pixal3D models made in the lab (their camera is found for you); on the CLI pass
   `--views <run>.svviews`, or use `scripts/photo_paint.py` on its own. Surfaces the
   picture cannot see keep the generator's paint.
4. **Compress**: re-encodes the textures, taking a typical asset from 32 MB to under 5
   with no visible difference.

Every run writes a JSON record of the settings used, so a batch of finished assets is
comparable rather than each one being tuned by hand.

## Prop sheets: many props from one image

Generate one picture holding a grid of props (barrels, crates, a chest), turn the whole
sheet into 3D in a single Pixal3D run, and the viewer's **Props** tab splits it into
separate, upright, named props, each with three levels of detail (LODs) baked from the
original. Install the optional gltfpack from Setup & Status and each LOD also comes as a
much smaller web-ready file. The prompt that works and what was measured are in
[`docs/prop-sheets.md`](docs/prop-sheets.md). Built and contributed by
[@AdrielSantana](https://github.com/AdrielSantana).

## Blender animation recipes

The reusable Blender tooling lives in `scripts/blender_*.py`: import, inspect,
stage, bake, render, rig and rebind helpers that work on any mesh this pipeline
produces. **[`scripts/README.md`](scripts/README.md) indexes every tool in the
repository**, grouped by what you are trying to do, and is kept honest by a test
that reads each script's own docstring.

Ready-made rigs and animations are not included. The
[quadruped pipeline](docs/quadruped-pipeline.md) walks through rigging a four-legged
character with these tools.

## Requirements

| Thing | Why |
|---|---|
| Apple Silicon Mac (M-series), 32 GB recommended | Every route |
| **or** Linux with an NVIDIA card (24 GB VRAM tested; Pixal3D's authors run it on 16 GB) | Pixal3D, Generate Image, TRELLIS.2, Hunyuan3D-2.1 |
| Linux + NVIDIA: CUDA toolkit matching PyTorch's CUDA | compiles TRELLIS.2's CUDA extensions (not needed on RTX 50-series) and Hunyuan3D-2.1's rasterizer (CUDA 12) |
| macOS: full Xcode | compiles the Metal kernels for Pixal3D and TRELLIS |
| Blender 4.2+ | Finish (low-poly clean-up, Pixel Match) and rigging. Install it yourself from [blender.org](https://www.blender.org/download/); Setup & Status shows whether it was found |
| gltfpack (optional) | smaller web-ready files from the Props tab; one click in Setup & Status, under 2 MB |
| `uv` | builds the reproducible Python environments |
| Python 3.11 (TRELLIS) / 3.12 (Hunyuan3D-MLX) | pinned by each backend's own setup |
| ~13 GB disk | Hunyuan3D-MLX 2.0 shape + paint weights (auto-downloaded once) |
| ~14 GB disk | TRELLIS.2-4B weights (auto-downloaded once, if using TRELLIS) |
| ~94 MB download | TinyCLIP flat-input advisor (local and non-blocking) |
| ~224 MB download | BiRefNet-lite background remover (comes with Pixal3D; otherwise Setup & Status or `scripts/bootstrap_matte.py`) |

## How long a run takes

Rough times for one run on a base M5 MacBook (32 GB); the RTX 4090 column is from a rented
cloud GPU. Yours will differ with the machine and the picture.

| Step | M5 MacBook, 32 GB | RTX 4090 |
|---|---|---|
| Text to image (Qwen-Image) | ~4.5 min | ~20 s |
| Image to 3D (Pixal3D) | ~6 min | ~3 min |
| Image to 3D (Hunyuan3D) | ~9 min | not measured yet |
| Image to 3D (TRELLIS.2) | 15–35 min | not measured yet |

On a Mac, TRELLIS.2 runs about twice as fast with **Attention backend** set to `mlx`.

## Licensing & provenance (non-negotiable)

- **Pixal3D** code and flow weights: MIT. The Q8_0 bundle also carries the **DINOv3** image
  encoder under its own licence, so treat its output the same as TRELLIS's. Background
  removal uses BiRefNet-lite (MIT) once installed, `rembg`'s u2net otherwise, and never
  BRIA RMBG-2.0.
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

## License

This repo's own code is [Apache-2.0](LICENSE): use it, fork it, sell things built on it.
If you do, keep the [`NOTICE`](NOTICE) file and credit image-to-3dlab with a link back
here. Model weights keep their own licences, listed above.

## Development

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q        # backends that load real models stay manual
ruff check .
```

Conventions: Conventional Commits, Keep a Changelog (`CHANGELOG.md`), test-first.

## Credits

This repo trains nothing and invents nothing; it builds upon other people's models and
work. What it *does* add is filling the gaps that exist to make some of these models work
on Apple Silicon, and improving the overall experience. Grateful to everyone who built
before me; they are named and credited in
[`docs/info_and_credits.md`](docs/info_and_credits.md). Also a special thanks to Claude
and Codex for being my partners through this! Not just helping me build, but teaching me
so much along the way. Yes, I just credited AI.
