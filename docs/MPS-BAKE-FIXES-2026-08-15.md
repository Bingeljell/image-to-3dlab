# The MPS bake fixes — 2026-08-15 session

**What this is.** The clean TRELLIS.2 port (`vendor/trellis-space-mac`) could sample latents
(the 78-minute Lucian run was banked) but was **blocked at decode → GLB packaging**: cumesh's
Metal `simplify` crashed on ~20M-face meshes. This session unblocked it. Every fix below was
found by validating from first principles — several assumptions in the older docs were wrong,
and the corrections are recorded here so the next session does not re-learn them.

---

## The fixes

### 1. Dropped the nvdiffrast pre-simplify (the original crash)

`app.py` calls `mesh.simplify(16_777_216)` — nvdiffrast's 2²⁴ index cap — before `to_glb`.
On MPS that call is pointless (no nvdiffrast) and crashes: cumesh's Metal `simplify_step` has
an internal off-by-one around 20.5M elements (`AcceleratorError: index 20549376 is out of
bounds`). The clean-port wrapper no longer calls it.

### 2. CPU pre-cap via `fast_simplification`, before the Metal `to_glb`

Replaced with the old port's proven recipe: CPU `fast_simplification` decimation to
`--pre-cap` (default 4,000,000 faces — the old port's `I2L_PRE_CAP` threshold that keeps
mtlbvh/cumesh stable). Verified safe because `to_glb` takes the voxel `attrs`/`coords`
separately from mesh `verts`/`faces`; decimating the surface does not touch the PBR field,
and `remesh=True` rebuilds topology regardless.

### 3. `to_glb` must receive CPU tensors

o_voxel's metal backend sets `device = 'cpu'` internally and builds `aabb`/`voxel_size` on
`coords.device` (`postprocess.py:134-165`). MPS inputs would device-mismatch in the grid math.
All four tensors (`vertices`, `faces`, `attrs`, `coords`) are `.cpu()`'d before the call.

### 4. `fast_simplification` runs in a subprocess

**The first wrong assumption.** We suspected the crash was memory pressure (the 4B pipeline
loaded). A standalone pre-cap on the same 26.8M mesh succeeded 3/3 — but crashed the moment
`o_voxel` was imported first. `fast_simplification` is **not robust on >~20M input faces in
any process that has imported o_voxel's deps** (cumesh/cv2/Metal reserve address space and
shift the heap). The same call in a clean interpreter is fine. Fix: `_precap_subprocess`
spawns a fresh interpreter per attempt.

### 5. Verify-and-retry the pre-cap

**The second wrong assumption.** `fast_simplification` output is **nondeterministically
corrupt above ~20M input faces**: usually a few dozen out-of-range face indices (0.001%,
filterable), rarely millions, sometimes a crash. The corrupt indices are *exactly* what made
mtlbvh's BVH build segfault (`MtlBVHImpl::MtlBVHImpl`) — the second distinct crash of the
session. Fix: verify the output for out-of-range indices, retry up to 5 attempts (each a
fresh subprocess so a crash is just a failed attempt), and post-filter the residual few.

### 6. Free the 4B pipeline before the bake

The bake (pre-cap + `to_glb`) does not need the pipeline. Keeping the ~10 GB model resident
while `fast_simplification` allocates its working set contributed to the first crash. Fix:
decode → save the CPU bundle → `del pipeline` → `gc.collect()` + `torch.mps.empty_cache()` →
bake. **Gotcha:** `del pipeline` must happen in the *caller* — deleting the parameter inside a
callee only drops the callee's local reference.

### 7. Decode cache + `--from-decode`

The decoded mesh is saved to `<out>_decode.pt` (verts/faces/attrs/coords + attr_layout —
the exact schema `--from-decode` loads), so re-baking skips both the decode and the model
load: ~1 minute of setup instead of ~12. Each GLB-packaging iteration is now cheap.

---

## The failure chain (how each crash was isolated)

| Crash | Where | Root cause found |
|---|---|---|
| `SIGSEGV` in `_simplify.so` | pre-cap, model loaded | fast_simplification × o_voxel-imported heap (not just memory) |
| `SIGBUS` in `_simplify.so` | pre-cap, `--from-decode` | same trigger, no model needed |
| `SIGSEGV` in `MtlBVHImpl` | `to_glb` BVH build | corrupt out-of-range indices in pre-cap output |

Key isolation result: the same 26.8M-face pre-cap **works in a clean interpreter** (proven
repeatedly) and **crashes in any o_voxel-imported process** — the trigger is the import
context × input size, not memory alone. And at 20.2M input (Snag) it never misfires; at
26.8M it does, nondeterministically.

---

## Assumptions made, and how they held up

| Assumption | Verdict |
|---|---|
| The old port's 4M pre-cap is safe for the Metal `to_glb` | ✅ Confirmed — Snag & Lucian bakes completed |
| `to_glb` wants CPU tensors | ✅ Confirmed — source + working bakes |
| The crash was memory pressure | ❌ Partly wrong — o_voxel import context is the trigger; memory was secondary |
| `fast_simplification` output can be trusted | ❌ Wrong above ~20M input faces — verify-and-retry added |
| The doc's "CPU pre-cap" plan was sufficient | ❌ Needed subprocess isolation + verification the plan didn't anticipate |

---

## Results (measured)

| Asset | Sampling | Decode | Bake | Total | Faces | Holes | Volume |
|---|---|---|---|---|---|---|---|
| Lucian | 78 min (banked) | 189 s | 443 s | ~12 min from cache | 281,520 | 6,733 (86% pinholes) | 0.0128 (= HF demo) |
| Controller | 23.5 min | 38 s | 141 s | 27.9 min | 278,399 | 591 | 0.0037 |
| Flicker | 10.6 min | 24 s | 85 s | 13.8 min | 278,289 | 561 | 0.0021 (HF: 0.0020) |

All three assets live in `output/space_baseline/`. Flicker's clean-port output is
geometrically near-identical to `assets_to_test/trellis-flicker-huggingface.glb` (same face
budget, volume within 5%).

---

## Still open

- **Holes** (561–6,733 vs the HF demo's 1–14): mostly pinholes, but the pre-cap + DC remesh
  path loses some fidelity vs running `to_glb` on the full mesh (which CUDA can). Worth a
  targeted look if pinholes show up in game-engine backface culling.
- **Texture drift** vs the HF demo: Stage 3 runs at guidance 1.0 (pure sampling), so the
  texture is seed-sensitive; MPS RNG ≠ CUDA RNG at the same seed. The frozen-shape seed
  search (`scripts/trellis_stage3.py`) is the lever.
- **Speed**: 22k-token subjects (Lucian, 78 min) are attention-bound on SDPA. The real lever
  is extending Pedro's fused `mtlgemm` kernel from head-dim 64 → 128; the multiplier grows
  with n².

---

## Files changed (audit worktree)

- `scripts/trellis_space_generate.py` — `_precap_subprocess`, `filter_out_of_range_faces`,
  `_decode_mesh`, `_decode_and_cache`, `_bake_export`, `generate_from_decode`, and the
  `--from-decode` / `--pre-cap` / `--no-save-decode` flags.
- `tests/test_trellis_space_generate.py` — `precap_ratio` and `filter_out_of_range_faces`
  tests (24 passing).
