# Image to 3D Lab

**Turn a single image into a textured 3D model — locally, on Apple Silicon — with a
license-provenance record for every result.**

Point it at a picture of a character or object and it produces a `.glb` 3D asset plus a
`.provenance.json` sidecar that records exactly how it was made and under which licenses.
Everything runs on your Mac; nothing is uploaded to a cloud service.

## What you can do

- **Generate a 3D model from one image** through three interchangeable backends — pick speed
  or quality.
- **Keep every result traceable** — runs are driven by versioned manifests and emit a
  provenance sidecar (input/output hashes, settings, license classification, component
  licenses).
- **Preview in Blender** with a one-command render script (matte or metallic lighting).
- **Rig and animate a model** — *in progress*; see [`docs/rigging-plan.md`](docs/rigging-plan.md).

## The three backends

- **Fast — Stable Fast 3D:** runs directly through Python on MPS, with automatic CPU retry
  when MPS reports an out-of-memory error.
- **Quality — Hunyuan3D:** an API-format workflow runs in a local ComfyUI instance. This
  keeps CUDA/NVCC-only rasterizer details out of this repository.
- **Experimental quality — TRELLIS.2:** runs through the community Apple Silicon port with
  BRIA background removal forcibly disabled.

## Quick start

```bash
# 1. install the fast backend (full details under "Install SF3D" below)
chmod +x scripts/bootstrap_macos.sh && ./scripts/bootstrap_macos.sh
source .venv/bin/activate
huggingface-cli login                    # for the gated SF3D model

# 2. turn an image into a 3D model
python pipeline.py your-image.png --fast
```

That writes a `.glb` plus a `.provenance.json` sidecar into `output/conditional/`. For
reproducible, traceable runs, prefer a manifest:

```bash
python pipeline.py --run-manifest manifests/moss-fox-showcase.json
```

The sections below cover the full prerequisites, the TRELLIS and Hunyuan backends, Blender
previews, and TRELLIS material modes.

## How to use this repo

Generating the mesh is **one step out of seven**. A raw generated asset has a correct
silhouette, roughly correct colour, and no surface at all — every part of it is uniformly
matte, so nothing catches light and the whole thing reads as one lump. The remaining six
steps are what make it look like a thing rather than a shape.

```
  source image
       |
   [1] generate ........ pipeline.py --run-manifest      <- the AI model
       |
   [2] close holes ..... blender_solidify.py             <- everything below
   [3] grade colour .... colour_match_albedo.py             is this repo
   [4] bake AO ......... blender_bake_ao.py
   [5] mask features ... feature_mask.py
   [6] build surface ... surface_detail.py
   [7] render & judge .. headless Blender
       |
  finished .glb  (+ .provenance.json from step 1)
```

**Only step 1 depends on the backend.** Steps 2–7 need nothing but a GLB with UVs and a
base colour texture, so they apply equally to SF3D or Hunyuan output. Full detail,
worked commands, and the traps are in **[`docs/finishing.md`](docs/finishing.md)**.

Two things to know before you start:

- **The recipe is per-subject and you pick it by eye.** Bark wants matte with strong
  derived relief; glazed ceramic wants gloss and *no* relief, because a normal map
  derived from albedo turns painted markings into dents. Settings do not transfer
  between subjects — neither do colour-grade strengths.
- **Judge visually, not numerically.** Render a lineup of variants side by side, zoom in,
  and orbit. Several confident conclusions on this project have been overturned by
  looking at the render.

### Run manifests

A manifest is the reproducible record of a run — prefer it over ad-hoc CLI flags.
Manifests live in `manifests/` and currently describe **generation only** (schema v1):

```json
{
  "schema_version": 1,
  "use_case": "showcase",
  "distribution": "private",
  "commercial_intent": false,
  "input": { "path": "../assets_to_test/3-4th-snag-roots-alpha.png", "source": "user-provided" },
  "model": {
    "backend": "trellis2",
    "id": "microsoft/TRELLIS.2-4B",
    "parameters": {
      "seed": 42, "pipeline_type": "1024_cascade", "texture_size": 2048,
      "bake_target_faces": 100000, "steps": 12, "material_mode": "matte"
    }
  },
  "license_policy": { "allow_conditional": true, "allow_research_only": false },
  "output": { "directory": "output" }
}
```

**Planned — finishing is not yet covered by the manifest.** Steps 2–6 are currently loose
scripts run by hand, which means a finished asset is only reproducible by someone
retracing the commands. The proposed schema v2 addition:

```json
"finishing": {
  "solidify": { "enabled": true },
  "colour": { "strength": 0.6 },
  "ao": { "distance_frac": 0.035, "strength": 0.35 },
  "material": { "roughness": [0.55, 0.95], "normal_strength": 6.0 },
  "features": [ { "name": "eye", "roughness": 0.3 } ]
}
```

Tracked with the rest of the packaging work — see
[`docs/finishing.md`](docs/finishing.md) ("Not done yet").

## Prerequisites

- macOS on Apple Silicon, 32 GB unified memory recommended
- Python 3.10 or 3.11
- Homebrew and Xcode Command Line Tools
- Full Xcode plus the Metal Toolchain for accelerated TRELLIS texture export
- Hugging Face access to the gated `stabilityai/stable-fast-3d` model

The bootstrap intentionally uses `python3.11` or `python3.10`; SF3D's pinned native
dependencies are not currently a good fit for the system's Python 3.14.

## Install SF3D

```bash
chmod +x scripts/bootstrap_macos.sh
./scripts/bootstrap_macos.sh
source .venv/bin/activate
huggingface-cli login
```

The bootstrap clones the official SF3D source into ignored `vendor/`, builds its Metal
texture baker with `USE_CUDA=0 USE_METAL=1`, and installs the macOS (non-GPU) `rembg` path.
It disables pip build isolation for SF3D's local extensions so they can compile against the
PyTorch already installed in the environment.
If stable PyTorch has an MPS regression, install current nightly wheels in the activated
environment before re-running the last install step:

```bash
python -m pip install --pre --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Run

```bash
python pipeline.py input.png --fast
python pipeline.py input.png --fast --cpu
```

For traceable runs, prefer a versioned JSON manifest. The output filename and generated
`.provenance.json` sidecar retain the intended use, license classification, hashes, settings,
and backend revision:

```bash
python pipeline.py --run-manifest manifests/moss-fox-showcase.json
```

SF3D writes `output/<name>_sf3d.glb` plus the background-removed input. Tune memory with
`--texture-resolution 512`; use `--remesh triangle --target-vertices 20000` for a lighter
game mesh. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before importing PyTorch.

For Hunyuan3D, start ComfyUI on port 8188, install and validate your chosen Hunyuan custom
nodes in its UI, then export the working graph in API format. See `workflows/README.md`.

```bash
python pipeline.py input.png --quality \
  --workflow workflows/my_hunyuan_api.json
```

The client verifies ComfyUI, uploads the source image, patches the graph's `LoadImage`,
queues `/prompt`, polls `/history/<prompt_id>`, and downloads the first 3D result via `/view`.
Use `--image-node` and `--output-node` when the graph is not unambiguous.

## Development

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python pipeline.py --help
```

No raw Hunyuan Python backend is provided. Its texture stack and supported custom nodes
change independently, so a tested ComfyUI API workflow is the explicit integration contract.

## TRELLIS.2 licensing-safe setup

The Apple Silicon port normally configures BRIA RMBG-2.0, whose self-hosted weights are
non-commercial. Our bootstrap disables that model and requires a pre-matted RGBA input,
produced with the commercially compatible U²-Net preprocessing lane:

```bash
chmod +x scripts/bootstrap_trellis_macos.sh
./scripts/bootstrap_trellis_macos.sh
```

Without full Xcode and its Metal Toolchain this uses the port's supported KDTree/PyTorch
fallback. Install the required compiler component with:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain
```

Xcode's optional Predictive Code Completion Model is not used by this project and may be
skipped. The bootstrap detects `/Applications/Xcode.app`, targets it explicitly, and installs
the Metal backends when its compiler is available. Run a traceable accelerated test with:

```bash
python pipeline.py --run-manifest manifests/moss-fox-trellis-metal-seed42-showcase.json
```

Or run the lower-cost CPU-bake manifest with:

```bash
python pipeline.py --run-manifest manifests/moss-fox-trellis-showcase-fast-bake.json
```

The CPU fallback defaults to a 50,000-triangle UV-bake budget because 200,000 triangles can
make `xatlas` spend tens of minutes charting foliage-like meshes. Override it with
`--trellis-bake-target-faces`; the selected budget and actual `metal-o-voxel` or
`kdtree-cpu` texture backend are recorded in provenance. TRELLIS output is classified
`commercial-conditional`: TRELLIS.2 is MIT, but its DINOv3 image encoder has a separate
license. BRIA remains blocked and unloaded.

### Material handling

TRELLIS exports each material as `alphaMode=BLEND` with a metallic-roughness map, which
renders as transparent, mirror-like shards and hides the baked albedo. By default the
pipeline rewrites each exported GLB's material so it renders correctly, patching only the
glTF JSON chunk — geometry and texture buffers are left byte-for-byte intact:

- `--trellis-material-mode matte` (default) forces `alphaMode=OPAQUE` and drops metalness.
  Best for organic subjects whose shading is already baked into the albedo (foliage, fur).
- `--trellis-material-mode pbr` forces `alphaMode=OPAQUE` but keeps the baked
  metallic-roughness, so genuinely metallic subjects (brass, chrome) keep their sheen.
- `--trellis-raw-material` skips normalization entirely and keeps the raw export.

In a manifest, set `"material_mode"` (or `"normalize_material": false`) under
`model.parameters`. The outcome is recorded in the provenance sidecar as
`material_normalized` and `material_mode`.

## Rendering previews

`scripts/blender_render_asset.py` renders cardinal previews of a generated GLB through a
local Blender instance running the [BlenderMCP](https://github.com/ahujasid/blender-mcp)
addon (a socket server on port 9876). Launch Blender, enable the addon, and start its
server (N-panel → BlenderMCP → Connect), then:

```bash
python scripts/blender_render_asset.py \
  output/conditional/<asset>.glb output/diagnostics --label myasset --env dark
```

The script starts from a clean scene, imports and grounds the asset (trusting the glTF
importer's Y-up→Z-up conversion), builds a camera and three-point lighting, and writes
`output/diagnostics/myasset_<view>.png` for five views. Use `--env studio` for metallic
(`pbr`) assets so they have a lit environment to reflect; `--env dark` (default) suits
matte assets.

## Reproducing the example assets

The `manifests/` directory holds versioned example runs for the bundled `moss-fox` and
`clockwork-pangolin` inputs, including SF3D and TRELLIS variants and matte/`pbr`/texture
-resolution options. For example:

```bash
python pipeline.py --run-manifest manifests/clockwork-pangolin-trellis-seed42-pbr.json
python scripts/blender_render_asset.py \
  output/conditional/clockwork-pangolin__trellis2__*.glb output/diagnostics \
  --label pangolin --env studio
```
