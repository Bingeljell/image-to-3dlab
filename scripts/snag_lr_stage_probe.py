#!/usr/bin/env python3
"""Stop right after the LR (coarse, 512-res) shape SLat pass for the Snag run and dump the
occupied voxel coordinates as a point cloud. No HR shape-SLat pass, no Stage 3, no bake --
this mirrors trellis_space_generate.py's pipeline.run() exactly up through
sample_shape_slat_cascade's LR section, then stops.

Question we're answering: is the left/right split we saw in the raw decode already present
in the Stage-1 sparse structure / LR shape-SLat coords, or does it only appear later (HR pass,
or the final decode_shape_slat mesh extraction)?

Same seed (918955446) and same input image as the real Snag run, so this reproduces that
run's RNG trajectory exactly up to this point (torch.manual_seed is called once, before
get_cond/sample_sparse_structure/sample_shape_slat_cascade, same as run() does).
"""
import os
import sys
import time
from pathlib import Path

REPO = Path("/Users/nikhilshahane/projects/image-to-3dlab")
VENDOR = REPO / "vendor" / "trellis-space-mac"
IMAGE_PATH = REPO / "assets_to_test" / "3-4th-snag-roots-alpha.png"
SEED = 918955446
OUT_DIR = REPO / "output" / "seed_sweep_mps" / "lr_stage_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(VENDOR / "cache" / "flex_gemm_autotune.json")
)

sys.path.insert(0, str(VENDOR / "TRELLIS.2"))
stubs = VENDOR / "stubs"
if stubs.is_dir():
    sys.path.append(str(stubs))

import torch
from PIL import Image

import flex_gemm  # noqa: F401  (must import before pipeline construction)
os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor

DEMO_SS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
DEMO_SHAPE = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}

print("Loading TRELLIS.2 pipeline...", flush=True)
t0 = time.time()
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B", load_rembg=False)
pipeline.to(torch.device("mps"))
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

image = Image.open(IMAGE_PATH).convert("RGBA")
image = pipeline.preprocess_image(image)

torch.manual_seed(SEED)
cond_512 = pipeline.get_cond([image], 512)
cond_1024 = pipeline.get_cond([image], 1024)  # called for RNG-trajectory parity with run(), unused below

print("Stage 1: sampling sparse structure...", flush=True)
t0 = time.time()
coords = pipeline.sample_sparse_structure(cond_512, 32, 1, DEMO_SS)
print(f"  done in {time.time() - t0:.1f}s -- {coords.shape[0]} occupied coarse voxels", flush=True)

print("Stage 2 (LR only): sampling coarse shape SLat...", flush=True)
t0 = time.time()
flow_model_lr = pipeline.models["shape_slat_flow_model_512"]
noise = SparseTensor(
    feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(pipeline.device),
    coords=coords,
)
sampler_params = {**pipeline.shape_slat_sampler_params, **DEMO_SHAPE}
if pipeline.low_vram:
    flow_model_lr.to(pipeline.device)
slat = pipeline.shape_slat_sampler.sample(
    flow_model_lr,
    noise,
    **cond_512,
    **sampler_params,
    verbose=True,
    tqdm_desc="Sampling shape SLat (LR)",
).samples
if pipeline.low_vram:
    flow_model_lr.cpu()
print(f"  done in {time.time() - t0:.1f}s", flush=True)

std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(slat.device)
mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(slat.device)
slat = slat * std + mean

# --- Dump the coarse (Stage-1) occupied voxel coords as a point cloud for inspection. ---
# coords columns: [batch, x, y, z] in a 32^3 grid. slat.coords should match 1:1.
pts = coords[:, 1:].float().cpu().numpy()
print(f"coarse occupancy: {pts.shape[0]} voxels, "
      f"x range [{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
      f"y range [{pts[:,1].min():.0f},{pts[:,1].max():.0f}] "
      f"z range [{pts[:,2].min():.0f},{pts[:,2].max():.0f}]", flush=True)

obj_path = OUT_DIR / "snag_lr_coarse_occupancy.obj"
with open(obj_path, "w") as f:
    for x, y, z in pts:
        f.write(f"v {x} {y} {z}\n")
print(f"Saved {obj_path} ({pts.shape[0]} points)", flush=True)

# Also dump the LR shape-SLat feature norm per voxel -- a degenerate/collapsed region should
# show as near-zero or wildly outlier norms relative to the rest.
import numpy as np
feat_norms = slat.feats.float().norm(dim=1).cpu().numpy()
np.save(OUT_DIR / "snag_lr_feat_norms.npy", feat_norms)
np.save(OUT_DIR / "snag_lr_coords.npy", coords.cpu().numpy())
print(f"feat norm stats: min={feat_norms.min():.3f} max={feat_norms.max():.3f} "
      f"mean={feat_norms.mean():.3f} std={feat_norms.std():.3f}", flush=True)
print("done", flush=True)
