# Image to 3D Lab

A local Apple Silicon pipeline with three backends:

- **Fast:** Stable Fast 3D runs directly through Python on MPS, with automatic CPU retry
  when MPS reports an out-of-memory error.
- **Quality:** an API-format Hunyuan3D workflow runs in a local ComfyUI instance. This keeps
  CUDA/NVCC-only rasterizer details out of this repository.
- **Experimental quality:** TRELLIS.2 runs through the community Apple Silicon port with
  BRIA background removal forcibly disabled.

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
