# Viewer workbench

Status as of **2026-08-14**. The diagnostic mesh viewer (`viewer/index.html`, served by
`viewer/serve.py` on `127.0.0.1:8777`) has grown from a URL-only compare page into an
interactive loader and source-comparison tool. This is **step 1 of the product plan's first
build order** — "extract/grow the viewer" — done as an in-place enhancement rather than a
module extraction. See [PRODUCT-PLAN-image-to-3d-workbench.md](PRODUCT-PLAN-image-to-3d-workbench.md)
for where this is heading (the four-stage Source → Generate → Inspect → Finish workbench).

Committed as `f9bb54e` on `feat/tear-provenance` (two files: `viewer/index.html` and
`viewer/IndexedOBJLoader.js`, the latter previously untracked despite being imported).

## What it does now

Up to **three panes**, side by side. Each pane holds either a **3D model** or a **source
image**. Panes stay independent but their cameras sync (toggle-able) so a comparison is honest.

### Three ways to load, and they no longer overlap in meaning

| You want to… | Do this |
|---|---|
| **Compare side by side** | Click the in-layout **＋ Add** pane (or `＋ add pane`, top-left), then pick a model *or* an image |
| **Overlay a source onto a model** to align it | Click **⧉ overlay** on that model pane, or drag an image directly onto it |
| **Remove a pane** | The **✕**, top-right of the pane |

The old top-right button used to read **"src"**, which read like "load a source" but actually
overlaid onto the current pane — the single biggest point of confusion. It is now **⧉ overlay**,
and the discoverable way to load side by side is a visible **"＋ Add a model or source image"**
placeholder pane that occupies each free slot until all three are full.

### Loading mechanics

- **Formats.** Models: `GLB`, `GLTF`, `OBJ`. Source images: `PNG`, `JPG`, `WebP`, `GIF`, `BMP`,
  `AVIF`. Adding a model format is one line (`MODEL_EXTS`) plus a vendored loader import.
  STL/PLY/FBX are not yet wired.
- **Local files** load via blob URLs — no server round-trip. **Multi-file bundles** work: drag a
  whole folder (or select every file) and a `.gltf`'s `.bin` + textures resolve by basename
  through a three.js `LoadingManager`.
- **Drop routing is DWIM.** A drop containing a model file loads the model (any images in that
  same drop are treated as its textures). A drop with *no* model file treats each image as its
  own source pane. Dropping an image *onto a loaded model pane* makes it an overlay, not a
  replacement.
- **Query string still works** unchanged: `?a=path&b=path&c=path` (+ `la`/`lb`/`lc` labels),
  relative to the repo root. An image extension makes an image pane; anything else a model. This
  keeps `serve.py --open` and browser automation driving the viewer exactly as before. Example:
  `?a=assets_to_test/moss-fox-mv-front.png&b=assets_to_test/trellis-mossfox-huggingface.glb`.

### Inspection controls (unchanged, carried over)

Backface culling (default **ON** — a double-sided preview cannot reveal a hollow mesh),
wireframe, flat-grey, normals, the `front / side / back / az210 / az130` angle presets, spin,
sync-cams, and per-pane face/vert stats. Models are normalised into a unit box so a size
difference cannot masquerade as a quality difference. Source-image panes support pan (drag),
zoom (wheel), and reset (double-click), and report pixel dimensions + zoom level.

### The overlay (onion-skin) alignment tool

Pinning a source image over a model gives a fade-able overlay with an opacity slider. The two
are framed independently, so they do **not** auto-register — you orbit/zoom the model until it
lines up under the fixed reference. This is the by-eye closeness-to-source check the repo treats
as the acceptance bar, made interactive.

## Resource behaviour (deliberate)

Rendering is **fully event-driven**. Nothing repaints unless a camera moves, a display state
changes, or spin is on. When idle, **no `requestAnimationFrame` is scheduled at all** — verified
in-browser: 0 frames/second idle, exactly 1 frame per repaint request, and it settles straight
back to 0 after OrbitControls damping eases out. Image panes are static DOM and never enter the
render loop.

This is not just battery politeness. A free-running 60fps loop over three ~300k-face textured
meshes sustains GPU load that tightens the macOS GPU watchdog — the same watchdog that kills the
remesh kernel mid-dispatch and, on 2026-08-13, took the machine down. A viewer used to diagnose
that failure must not contribute to it. (The earlier version was "render-on-demand" but still ran
a perpetual rAF that woke every frame to check a dirty flag; this version schedules nothing when
idle.)

## Load-bearing implementation notes

- **Slots vs. views.** `slots[0..2]` may each be a model view, an image view, or `null`. `views`
  is the compacted list of *model* views — the only ones the 3D render/frame/sync loops touch.
- **Blob lifecycle.** Object URLs are revoked when the last view sharing a drop's blob set is
  cleared, so you can swap models repeatedly without leaking GPU contexts or memory. Clearing a
  model pane disposes its `WebGLRenderer` and forces context loss.
- **OBJ coordinate transform.** `IndexedOBJLoader` streams large indexed OBJ decodes (v/f only)
  without exploding every face into new vertices; the raw OBJ is rotated `-π/2` about X to match
  o_voxel's `(x, y, z) → (x, z, -y)` GLB export, so a raw decode and its processed GLB share an
  up axis.

## Not done yet

- STL / PLY / FBX loaders (each one vendored loader away).
- No connection to generation — this is still a **viewer**, not the workbench. The next plan
  steps are the project/cache model and a job runner around the CLI and `trellis_rebake.py`
  (`POST /api/projects/{id}/generate` etc.), so a generated output lands in a pane automatically.
- Overlay alignment is manual; there is no saved camera or registration between source and model.
