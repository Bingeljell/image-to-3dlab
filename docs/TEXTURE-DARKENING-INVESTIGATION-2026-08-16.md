# Texture darkening/desaturation — session facts (2026-08-16)

Raw observations from this session, in the order they were gathered. No conclusions drawn
here — see the author's own read of this before treating anything below as a verdict.

## Starting point

Controller, Storm Ram, and Bloomglass (all generated on our Mac/MPS port, seed=0, demo-default
params) all came out darker and/or more metallic than their source images.

## Step 1 — bake/postprocess path

Measured `base_color` on the raw voxel attrs (pre-bake) vs. the final baked GLB texture for
Bloomglass:

| | base_color mean (0-255) |
|---|---|
| Raw voxel attrs | 139.0 |
| Final baked texture | 139.9 |

Re-baked Bloomglass's cached decode with `--pre-cap` raised above its raw face count (the CPU
pre-simplify step skipped entirely):

| | base_color mean (0-255) |
|---|---|
| With pre-cap (original) | 140.0 |
| Without pre-cap | 139.9 |

## Step 2 — image preprocessing

Reproduced `pipeline.preprocess_image()`'s logic (bbox crop + alpha premultiply) standalone on
the real Bloomglass input. Output image inspected directly: correctly bright, pale-lavender,
matches the concept art.

## Step 3 — Controller reseed + guidance experiments (MPS)

Resampled Stage 3 only (frozen shape latents) three times:

| texture_seed | guidance_strength | base_color_mean (0-255) |
|---|---|---|
| 0 (original full run) | 1.0 | 7.9 |
| 424242 | 1.0 | 3.3 |
| 777 | 1.0 | 1.2 |
| 999 | 3.0 | 1.2 |

## Step 4 — CUDA control run (Bloomglass, seed=0)

Ran the real upstream `microsoft/TRELLIS.2` repo on an RTX PRO 6000 (Blackwell), same
conditioning image, same seed (0), same params as our MPS run. Infra details in
`RUNPOD-CUDA-DIAGNOSTIC-2026-08-16.md`.

| | base_color_mean (0-1) | metallic_median | GLB |
|---|---|---|---|
| Our MPS run | 0.545 | 0.78 | [`bloomglass.glb`](../output/bloomglass/bloomglass.glb) |
| Real CUDA, seed=0 | 0.633 | 0.97 | [`cuda_bloomglass.glb`](../output/bloomglass/cuda_reference/cuda_bloomglass.glb) |
| Reference GLB | ~0.79 | ~0.52 | [`trellis-bloom-glass.glb`](../assets_to_test/trellis-bloom-glass.glb) |

Rendered side by side in the viewer: `output/bloomglass/cuda_reference/three_way_comparison.jpg`.

## Step 5 — lighting check (Blender `studio` env, headless)

Re-rendered Bloomglass (ours, real-CUDA-seed0, reference) through Blender's `studio`
environment (lifted world + ray-traced reflections) instead of the bare-lit web viewer.

- Our MPS run: reads as pale lavender/grey under `studio` lighting (differs visibly from how
  it reads in `viewer/index.html`, which has no environment map).
- Real CUDA seed=0: reads as mirror-polished chrome under `studio` lighting too (same as in
  the web viewer).
- Reference: pale lavender under `studio` lighting.

Files: `output/bloomglass/blender_studio/*.png`.

Ran the same `studio`-lighting check on Controller (our MPS run only, no CUDA comparison at
this step): reads as pure black under `studio` lighting, same as in the web viewer.
`output/space_baseline/blender_studio/controller_studio_*.png`.

## Step 6 — nine-run seed sweep (real CUDA)

Three assets (Storm Ram, Controller, Bloomglass) x three seeds each, drawn with
`random.SystemRandom()`:

| tier | seed |
|---|---|
| large | 78,426,575 |
| very_large | 965,379,546 |
| uber_large | 1,976,445,060 |

All nine ran the real upstream repo, same `DEMO_PARAMS` (12 steps/stage, `guidance_strength`
1.0 for tex_slat), on an RTX 5090 (Blackwell). Full stats JSON + GLBs in `output/seed_sweep/`.

**Storm Ram:**

| seed tier | base_color_mean (0-255) | metallic_median | GLB |
|---|---|---|---|
| large | 106.5 | 0.0015 | [`storm_ram_seed78426575_large.glb`](../output/seed_sweep/storm_ram_seed78426575_large.glb) |
| very_large | 111.9 | 0.0498 | [`storm_ram_seed965379546_very_large.glb`](../output/seed_sweep/storm_ram_seed965379546_very_large.glb) |
| uber_large | 86.9 | 0.0002 | [`storm_ram_seed1976445060_uber_large.glb`](../output/seed_sweep/storm_ram_seed1976445060_uber_large.glb) |

Our MPS run (seed=0) for comparison: [`storm_ram.glb`](../output/storm_ram/storm_ram.glb).
Viewer screenshot, all three seeds: `output/seed_sweep/storm_ram_all_3_cuda_seeds.jpg`.

**Controller:**

| seed tier | base_color_mean (0-255) | metallic_median | visible button color (viewer, zoomed) | GLB |
|---|---|---|---|---|
| large | 3.9 | 0.47 | yellow, red, green, blue-ish all visible | [`controller_seed78426575_large.glb`](../output/seed_sweep/controller_seed78426575_large.glb) |
| very_large | 6.6 | 0.45 | yellow, purple, green, red all visible | [`controller_seed965379546_very_large.glb`](../output/seed_sweep/controller_seed965379546_very_large.glb) |
| uber_large | 2.8 | 0.23 | mostly black, no distinguishable button colors | [`controller_seed1976445060_uber_large.glb`](../output/seed_sweep/controller_seed1976445060_uber_large.glb) |

Note: `base_color_mean` for `large` (3.9) is lower than the original seed=0 run's (7.9)
despite the visible-color assessment ranking `large` above the original. Our MPS run
(seed=0) for comparison: [`controller.glb`](../output/space_baseline/controller.glb). Viewer
screenshot, all three seeds: `output/seed_sweep/controller_all_3_cuda_seeds.jpg`. MPS-vs-CUDA
at seed=0: `output/seed_sweep/controller_mps_vs_cuda_large.jpg`.

**Bloomglass:**

| seed tier | base_color_mean (0-255) | metallic_median | GLB |
|---|---|---|---|
| large | 136.3 | 0.63 | [`bloomglass_seed78426575_large.glb`](../output/seed_sweep/bloomglass_seed78426575_large.glb) |
| very_large | 95.4 | 0.59 | [`bloomglass_seed965379546_very_large.glb`](../output/seed_sweep/bloomglass_seed965379546_very_large.glb) |
| uber_large | 111.2 | 0.86 | [`bloomglass_seed1976445060_uber_large.glb`](../output/seed_sweep/bloomglass_seed1976445060_uber_large.glb) |

All three read as dark purple/metallic in the viewer, and stayed dark purple/metallic under
Blender `studio` lighting too (`output/seed_sweep/blender_studio/bloomglass_*_seed*_*.png`).
Viewer screenshot, all three seeds: `output/seed_sweep/bloomglass_all_3_cuda_seeds.jpg`.

## Step 7 — Controller `very_large` seed, run locally on MPS

Ran `965,379,546` (Controller's best-scoring CUDA seed) through our own Mac port.

- Total wall time: 1739.5s (~29 min).
- Output: [`controller_seed965379546_very_large.glb`](../output/seed_sweep_mps/controller_seed965379546_very_large.glb).
- Viewer screenshot next to the CUDA run at the same seed:
  `output/seed_sweep_mps/controller_mps_vs_cuda_same_seed.jpg`. Zoomed crop shows
  reddish/orange tint on one button and a red sliver on another; the other two buttons read
  mostly black. Less button-color coverage than the CUDA run at the same nominal seed number.

## Step 8 — Snag

Attempted the real CUDA repo on the RTX 5090 (32GB), seed `918,955,446`. Failed twice:

1. First attempt: `torch.OutOfMemoryError` during `decode_latent`
   (`sc_vaes` forward / `layer_norm`) — 30.38 GiB in use of 31.36 GiB total.
2. Retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: failed differently —
   `RuntimeError: [CuMesh] CUDA error ... out of memory` inside `cumesh`'s `fill_holes()` /
   `get_edges()` (not a PyTorch tensor allocation, so the allocator flag didn't apply).

Ran the same seed locally on our Mac/MPS port instead (`output/seed_sweep_mps/
snag_seed918955446_very_large.glb`) — [fill in on completion].

## Step 9 — CUDA vs MPS process log diff

Compared the actual startup/backend-selection lines from the two platforms' logs directly
(not just final output stats).

CUDA (any run, e.g. Storm Ram `large`):
```
[SPARSE] Conv backend: flex_gemm; Attention backend: flash_attn
[ATTENTION] Using backend: flash_attn
```

MPS (Controller `very_large`):
```
[SPARSE] Conv backend: flex_gemm; Attention backend: sdpa
[ATTENTION] Using backend: sdpa
UserWarning: The operator 'aten::segment_reduce' is not currently supported on the MPS backend and will fall back to run on the CPU. (trellis2/modules/sparse/basic.py:285)
```

Two concrete differences visible in the logs, not just an inferred "RNG differs":

1. CUDA uses `flash_attn`; MPS uses PyTorch's native `sdpa`. Different attention kernel
   implementations with different floating-point reduction orders — not the same computation
   run on different hardware.
2. MPS silently falls back to CPU for `aten::segment_reduce` (unsupported on MPS) — part of
   the computation graph runs on a third device (CPU) with no CUDA equivalent fallback at all.

Conv backend (`flex_gemm`) is named the same on both, though the compiled implementation
differs (CUDA build vs. this port's Metal-compiled variant).

## Step 10 — Snag, local MPS run timing (in progress)

Second shape-SLat sampling stage (the high-resolution cascade pass) logged **437s for a
single step** — dramatically slower than any other asset in this session (5-75s/step
elsewhere). At 12 steps for this stage alone, plus Stage 3, total runtime is estimated at
1.5-2+ hours. [Fill in final numbers on completion.]

## Step 11 — Snag, raw decode visual inspection (mesh damage, separate from texture darkening)

Baked GLBs of Snag (seed 918955446, both `--pre-cap 1000000` and `--pre-cap 2000000`, same
cached decode) showed shattered/broken-looking geometry. Lowering the pre-cap target did not
change the damage — nearly identical between 1M and 2M. This raised the question of whether
the damage originates in the CPU pre-cap/decimation step, or was already present in the raw,
un-decimated 33.4M-face decode.

Exported the raw decode geometry directly from the cached bundle
(`output/seed_sweep_mps/snag_seed918955446_very_large_decode.pt`) — vertices + faces only, no
color/UV, no decimation: `output/seed_sweep_mps/snag_raw_decode_geometry_only.obj` (1.48GB,
33.4M faces).

Rendered flat-grey in Blender headless from 8 angles (45° yaw increments), and separately
opened directly by the user in Blender (Select > Non-Manifold):

- Damage (holes, non-manifold edges) is concentrated on one half of the mesh; the other half
  is clean, confirmed both in the headless renders and in the user's own Blender inspection
  (Select > Non-Manifold highlighted verts almost entirely on one side).
- The 270°-yaw headless render shows an actual hole with mesh interior visible — not a
  shading artifact.
- The 90-135°-yaw headless renders show a speckled/checkerboard pattern consistent with
  inconsistent face winding / flipped normals, a different defect class from the hole.

Files:
- Renders: `output/seed_sweep_mps/raw_decode_check/snag_sweep_{000,045,090,135,180,225,270,315}.png`
- Raw OBJ: `output/seed_sweep_mps/snag_raw_decode_geometry_only.obj`
- (Unreliable, do not use) raw GLB export: `output/seed_sweep_mps/snag_raw_decode_geometry_only.glb` — Blender's glTF importer crashes on this 593MB file; the OBJ above loads fine.

Code trace (`vendor/trellis-space-mac/deps/trellis2-apple/trellis2/pipelines/trellis2_image_to_3d.py:477-507`,
`decode_latent`): mesh geometry (vertices/faces) comes entirely from
`decode_shape_slat(shape_slat, resolution)`; `tex_slat` (Stage 3) only supplies voxel color
attributes (`v.feats`) and never touches vertex/face topology. `decode_latent` also calls
`m.fill_holes()` (`mesh/base.py:55`, `max_hole_perimeter=3e-2`) before the mesh is cached — the
raw OBJ above is already post-repair. `fill_holes` only closes manifold boundary loops under
that perimeter cap; it does not address general non-manifold topology (edges shared by the
wrong number of faces, overlapping fragments).

## Step 12 — Snag, Stage 1 + LR shape-SLat probe

To narrow down whether the damage originates in Stage 1 (sparse structure) / the LR (512-res)
shape-SLat pass, vs. the HR (1024-res) pass or the final `decode_shape_slat` mesh extraction:
reran the pipeline with the same seed (918955446) and same input image
(`assets_to_test/3-4th-snag-roots-alpha.png`), stopping immediately after the LR shape-SLat
sampling step — no HR pass, no Stage 3, no bake. Script:
`scripts/snag_lr_stage_probe.py` was run from the scratchpad copy; it mirrors `pipeline.run()`'s
call order (`preprocess_image` → `torch.manual_seed(seed)` → `get_cond(512)` → `get_cond(1024)`
→ `sample_sparse_structure` → the LR-only section of `sample_shape_slat_cascade`) then exits.

Result:

| | |
|---|---|
| Stage 1 occupied coarse voxels (32³ grid) | 5,401 |
| Coarse voxel coord range | x [1,30], y [0,31], z [6,25] |
| LR shape-SLat feature norm | min 19.46, max 54.64, mean 34.41, std 3.95 |
| Stage 1 sampling time | 105.6s |
| LR shape-SLat sampling time | 179.1s |

Visualized the 5,401 coarse-occupancy voxels as three orthographic projections (xy/xz/yz),
colored by LR feature norm:

- `output/seed_sweep_mps/lr_stage_probe/snag_lr_proj_xy.png`
- `output/seed_sweep_mps/lr_stage_probe/snag_lr_proj_xz.png`
- `output/seed_sweep_mps/lr_stage_probe/snag_lr_proj_yz.png`
- Raw point cloud (importable in Blender, vertices only, no faces): `output/seed_sweep_mps/lr_stage_probe/snag_lr_coarse_occupancy.obj`
- Raw arrays: `output/seed_sweep_mps/lr_stage_probe/snag_lr_coords.npy`, `output/seed_sweep_mps/lr_stage_probe/snag_lr_feat_norms.npy`

All three projections show dense, complete coverage across the full silhouette — no gap, no
empty half. Feature-norm outliers (brighter points in the projections) appear scattered, not
clustered to one side.

**Not yet run**: the HR (1024-res) shape-SLat pass and the final `decode_shape_slat` mesh
extraction. One of those two is where the half/half split from Step 11 must be getting
introduced, since Step 12 rules out Stage 1 and the LR pass. Next session: extend
`snag_lr_stage_probe.py` (or copy it) to also run `upsample(...)` + the HR sampling pass,
stop before `decode_shape_slat`, and check whether the HR coordinate set / HR feature norms
already show the asymmetry — that would isolate it to the HR sampling pass rather than the
final geometry-extraction step.

## Two code fixes made along the way (independent of the above)

1. `scripts/trellis_stage3.py` — was calling `decode_shape_slat()` + `decode_tex_slat()` as
   two separate steps; this OOM'd deterministically at ~42.4GB on MPS (reproduced twice,
   including after freeing other memory). `trellis_space_generate.py` uses the fused
   `decode_latent()` instead and never hits this. Fixed to match.
2. `scripts/image_gallery.py` — subdirectory image/folder hrefs were built root-relative
   (`"/" + filename`), ignoring the current view path, so any file inside a subfolder 404'd.
   Fixed; regression test added in `tests/test_image_gallery.py`.

## Infra notes

Full RunPod setup process, every blocker hit, and the fixes: `RUNPOD-CUDA-DIAGNOSTIC-2026-08-16.md`.
