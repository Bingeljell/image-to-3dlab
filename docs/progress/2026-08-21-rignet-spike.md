# RigNet auto-rigging spike — 2026-08-21

**Status: paused mid-spike, not concluded.** Pipeline runs end-to-end on Apple Silicon
for free, but the one real result so far (Storm Ram) is not a usable rig. Resume by
re-running with untuned defaults (see "Next step" at the bottom) before judging RigNet
either way.

## Goal

Automate quadruped rigging — specifically the one manual step in our existing pipeline
(`blender_joint_markers.py` + `blender_build_rig.py` + `blender_voxel_weights.py`,
documented in `journal/legacy/pipeline-vs-manual.md`): ~20 minutes of hand-placed joint
markers per character. Looking for a free, open-source model that predicts skeleton +
skinning weights directly from the mesh.

Target test assets, both quadrupeds:

- **Storm Ram** — `output/runpod_seed_sweep/storm_ram_seed965379546_very_large.glb`
  (TRELLIS, generated on a RunPod 5090/CUDA — clean generation)
- **Forest Flicker** — `output/forest_flicker_1024/forest_flicker_1024.glb`
  (TRELLIS, generated locally) — **not yet tried against RigNet**

## Constraints set by the user this session

- **Free/open-source only.** Ruled out Tripo AI (commercial).
- **No RunPod, no CUDA rental — explicitly ruled out.** This eliminated UniRig (MIT,
  best-in-class per SIGGRAPH 2025, but its `spconv` dependency has no Apple Silicon
  build and needs 8GB+ VRAM). Went with **RigNet** (GPLv3, `github.com/zhan-xu/RigNet`,
  SIGGRAPH 2020) instead, since its dependency stack is plausibly CPU/MPS-portable.
- **Blender was in active use throughout — never touched it.** Everything here is
  pure Python, no Blender involved (RigNet's own inference doesn't need Blender at
  all; only final binding into our existing pipeline would, and we never got there).

## What's set up (all free, all local, zero paid infra)

- `vendor/rignet/` — clean clone of `github.com/zhan-xu/RigNet` (git-ignored, matches
  the existing `trellis-mac` / `hunyuan-mlx` vendoring convention).
- `vendor/rignet/.venv/` — **isolated venv, Python 3.11** (not 3.14 — matches this
  repo's standard reasoning about native-dependency wheel availability). Installed:
  torch 2.13 (MPS-capable), `torch_geometric==2.4.0` **(pinned — see below, do not
  upgrade)**, `torch_scatter`/`torch_sparse`/`torch_cluster` (all compiled from source
  successfully via `--no-build-isolation`, no CUDA needed), open3d 0.19, rtree,
  opencv-python, tensorboard, gdown.
- `vendor/rignet/checkpoints/` — RigNet's trained weights (196MB, from the Google
  Drive link in their README), unzipped into place. Git-ignored like the rest of
  `vendor/`; re-download via `gdown` if this vendor dir is ever recreated (the exact
  command is in shell history / can be regenerated from
  `https://drive.google.com/uc?id=1gM2Lerk7a2R0g9DwlK3IvCfp8c2aFVXs`).
- **`scripts/patch_rignet_macos_compat.py`** — the durable, re-appliable fix for
  everything RigNet's 2020-era Ubuntu+CUDA code doesn't know about macOS. Tested in
  `tests/test_patch_rignet_macos_compat.py` (9 passing tests, idempotency + missing-
  anchor checks for every patch). **Must be re-run after any fresh `git clone` of
  `vendor/rignet`, same as every other `patch_*.py` in this repo** — `vendor/` is
  git-ignored, so none of this survives a clone on its own. Five fixes, in order
  discovered:
  1. Bundled `binvox` binary is Linux x86-64 only (`binvox.exe` is Windows-only);
     `quick_start.py` raised on darwin. Replaced with a macOS voxelizer in the new
     `vendor/rignet/utils/voxelize_compat.py` (written by the patch script).
     **Learned the hard way:** don't use `trimesh.Trimesh.contains()` on a large
     point batch — it used ~9GB RSS for just 20,000 of the 681,472 grid points this
     needs and killed the process twice (`SIGKILL`, exit 137) before this was caught.
     `Trimesh.voxelized(pitch).fill()` + `VoxelGrid.is_filled(points)` does the same
     job in <1s and ~600MB.
  2. `torch_geometric.nn.PointConv` renamed to `PointNetConv` upstream — aliased back.
  3. `np.int` / `np.bool` removed (not deprecated, gone) in modern numpy; two were
     import-time-evaluated function defaults in `binvox_rw.py`, so that whole module
     failed to import before this was found.
  4. `create_single_data()`'s own parameter is misspelled `mesh_filaname` everywhere
     in its body except one line, which references the correctly-spelled
     `mesh_filename` — only "works" in RigNet's own script because that name happens
     to exist as a global inside `if __name__ == '__main__':`. Fixed the typo.
  5. `predict_skeleton()` unconditionally opens an **interactive Open3D window** and
     blocks on `vis.run()` waiting for a human to close it. Cost a real 36-minute hang
     here (20s of actual CPU time against 36 minutes of wall clock) before the cause
     was found. Commented out; nothing downstream used its return value.
- **`scripts/rignet_infer.py`** — our own driver (tracked, not vendored) that reuses
  `quick_start.py`'s functions against an arbitrary `model_id` instead of RigNet's
  bundled examples. Also patches around one more caller-contract issue that didn't
  belong in the vendored-code patch script: `predict_skinning()` does `global device`
  expecting `__main__` to have set it, so the driver sets `quick_start.device`
  directly after import.

  Usage: `vendor/rignet/.venv/bin/python scripts/rignet_infer.py <model_id>` with
  `vendor/rignet/quick_start/<model_id>_ori.obj` already in place (any mesh format
  trimesh can load and re-export as obj works — decimation to RigNet's 1K–5K vertex
  target happens automatically inside `quick_start.py` via open3d).

### Why `torch_geometric` is pinned to 2.4.0, specifically

2.8.0 (latest) moved `fps`/`radius`/`knn_interpolate` to hard-require `pyg-lib`, which
**has no macOS wheel at all** (`pip install pyg_lib` finds nothing). 2.4.0 still has the
`torch_cluster` fallback these ops originally used, and `torch_cluster` builds fine
from source here. If `torch_geometric` ever gets upgraded in this venv, RigNet's
`ROOT_GCN` / `PairCls_GCN` models will fail at the `fps()` call with
`ImportError: 'fps' requires 'pyg-lib>=0.6.0'`.

## What actually happened: Storm Ram run

Ran the full pipeline end-to-end (after killing two earlier runs that died — see
patch #1 and #5 above for why). Real result, no crash:

`vendor/rignet/quick_start/storm_ram_ori_rig.txt` — joints + hierarchy + per-vertex
skinning weights, transferred back onto the full 141,438-vertex original mesh.

**But the rig itself is not usable.** Parsed the output:

- **206 total joints, 125 unique** (rest are RigNet's own symmetry-duplicate joints).
  A quadruped should need roughly 20–30 bones — our hand-built fox rig used 21–25.
  5x over-segmented.
- Rendered it (matplotlib, front + side view, mesh points in gray, skeleton in
  red/blue) — see `storm_ram_skeleton_v1_oversegmented.png` next to this file.
  Visually confirms the joint count problem: instead of clean branching limbs, the
  horns/chest/hooves are dense triangulated tangles, not a usable animation skeleton.

**Likely cause, not yet tested:** ran with the `bandwidth=0.045, threshold=0.75e-5`
pair copied from RigNet's bundled example character (`model_id="17872"`,
`quick_start.py`'s default in `rignet_infer.py`'s `--bandwidth`/`--threshold` args).
These control how aggressively the meanshift joint-clustering step merges nearby
candidate joints, and RigNet's own README is explicit that this needs per-character
tuning — copying the example's values was the wrong first move. Their documented
fallback for a genuinely new character is `bandwidth=None` (let the network's own
learned bandwidth apply) and `threshold=1e-5`.

## Next step (not started)

Re-run with the untuned defaults:

```
vendor/rignet/.venv/bin/python scripts/rignet_infer.py storm_ram --threshold 1e-5
```

(`--bandwidth` already defaults to `None` in `rignet_infer.py`.) Remove the stale
intermediate files first (`rm vendor/rignet/quick_start/storm_ram_remesh.obj
vendor/rignet/quick_start/storm_ram_normalized.obj`) so `create_single_data` re-runs
cleanly rather than reusing the old decimation.

If the joint count comes back sane (~20–30), move on to Forest Flicker, then decide
whether RigNet's output is worth feeding into our own `blender_voxel_weights.py`
verification step or binding directly. If it's still over-segmented, the next lever
is sweeping `bandwidth` explicitly rather than relying on the learned default.

## Also discussed, not acted on

- **Rigify** (Blender's built-in auto-rig addon) is not a substitute for this —
  it automates *rig construction* from a skeleton you place by hand, but does nothing
  for *weighting*. Our `blender_voxel_weights.py` step would still be needed
  regardless of whether the armature comes from Rigify or RigNet's predicted skeleton.
- Earlier in this session: surveyed `output/repo_tests/` mesh health across TRELLIS
  vs. Hunyuan-mlx-xiong backends (unrelated to rigging, but same session) — Hunyuan
  outputs were watertight/winding-consistent 4/4 times, TRELLIS outputs were
  fragmented and winding-inconsistent 4/4 times despite `fix_winding=True` being on
  by default; root cause traced to `trimesh.fix_normals()`'s BFS-based algorithm not
  converging on meshes this fragmented (many disconnected components + non-manifold
  edges). Not yet fixed, not part of the rigging work, noted here only because it
  happened in the same session.

## Loose ends, unrelated to this spike, noted so they aren't lost

- `scripts/blender_build_rig.py`, `scripts/blender_joint_markers.py`,
  `scripts/blender_walk_cycle.py` showed as locally modified (`git status`) partway
  through this session, while Blender was in active use. Flagged to the user, not
  investigated, not touched.
- `feat/job-status-polling` branch: 3 commits pushed, no PR opened yet, CHANGELOG
  entry still pending. Fully separate, paused thread from earlier in this same
  session — see that conversation's summary if picking it back up.
