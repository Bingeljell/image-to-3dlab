# The MPS bake fixes — 2026-08-15 session

**What this is.** The clean TRELLIS.2 port (`vendor/trellis-space-mac`) could sample latents
(the 78-minute Lucian run was banked) but was **blocked at decode → GLB packaging**: cumesh's
Metal `simplify` crashed on ~20M-face meshes. This session unblocked it and wired it into the
web viewer. Every fix below was found by validating from first principles — several
assumptions in the older docs were wrong, and the corrections are recorded here so the next
session does not re-learn them.

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
| The built venv can be moved between locations | ❌ Wrong — compiled kernels embed absolute rpaths; rebuild in place |
| `setdefault` env is safe against stale inherited values | ❌ Wrong — the generator must own backend-selection vars |

---

## Follow-up findings (2026-08-15 evening → 2026-08-16)

The five fixes above got the clean port producing GLBs. Wiring that same engine into the web
viewer and relocating the built venv surfaced four more issues, each again found by
measurement rather than assumption.

### 8. A moved venv breaks `flex_gemm` via a stale `LC_RPATH` (the conv_none chain)

The built venv was moved (audit worktree → repo root). `flex_gemm`'s compiled `_C.so`
embeds an absolute `LC_RPATH` pointing at torch's `lib` directory **at the old path**, so
`import flex_gemm` fails after a move. The generator's
`setdefault("SPARSE_CONV_BACKEND", "flex_gemm")` then kept whatever the environment said —
`"none"` — and trellis2 imports `conv_<CONV>`, so it died on `ModuleNotFoundError: no module
named conv_none`, after which the model loader tried to re-download the checkpoint with a
mangled repo id (the `RepositoryNotFoundError` 404). Three stacked symptoms, one root cause.
Fixes:

- `install_name_tool -delete_rpath <old> -add_rpath <new>` on the `.so` (delete-then-add in
  one call — the load-command table has no spare headerpad for a plain add).
- The generator now **hard-pins** `SPARSE_CONV_BACKEND=flex_gemm` and fails loudly with the
  rebuild command if flex_gemm cannot import — `"none"` is not a real backend, so a silent
  fallback only produces a confusing crash.

**The real lesson: do not relocate a built venv — rebuild it in place.** A fresh clone's
bootstrap builds at the final location and is unaffected.

### 9. Stale environment variables silently select the wrong backend

The web viewer's job subprocess inherits the server's environment; a stale
`SPARSE_CONV_BACKEND` (or `ATTN_BACKEND`) exported in some earlier shell survived into the
generator, and `setdefault` honoured it. Fixes: the generator hard-assigns
`ATTN_BACKEND`/`SPARSE_ATTN_BACKEND`/`SPARSE_CONV_BACKEND` in `configure_environment`
(`require_flex_gemm`), and the viewer strips backend-selection vars from the job subprocess
env (`_job_env`).

### 10. "Run setup" can re-break a moved build

The web UI's Run-setup endpoint spawned a bootstrap even on a machine with a working build
(the button is hidden in that case, but the endpoint was callable) — and a bootstrap rebuild
over a **moved** venv re-bakes the stale torch rpath into `flex_gemm`'s `_C.so` (mtlgemm's
`setup.py` adds `-Wl,-rpath` from the resolved torch path). The endpoint now refuses (409)
when the build already exists, and setup runs are mutually exclusive with generation jobs.

### 11. Silent death during shape-fine — instrumented, not yet explained

A web-UI run died mid shape-fine with **no traceback, no crash report, no jetsam event and
plenty of free RAM** — a silent exit. The exit code existed only in the in-memory SSE
stream, so nothing was written to disk. The viewer now appends `generator exited with code
<n>` (with the signal name for negative codes, e.g. SIGKILL/SIGTERM) and a `[rss X GB]`
sample every 30 s to the job log, so a future silent death leaves a paper trail — the RSS
trajectory distinguishes an OOM-style climb from a flat-then-vanished kill. At the time of
writing the death remains unexplained.

### 12. `fast_simplification` is flaky below the "20M" note too

During a web-UI controller run the pre-cap **SIGBUS'd in a clean subprocess at 9.97M input
faces** — well under the ~20M note in fixes 4/5. Two attempts crashed, attempt 3 succeeded
(the verify-and-retry doing exactly what it was built for). The reliability is not a clean
size cutoff; it is content-and-size-dependent (the controller's dense small features create
many simultaneous collapse candidates). The retry loop is a safety net, not a cure — a
deterministic decimator would be the long-term replacement.

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

## Files changed

- `scripts/trellis_space_generate.py` — `_precap_subprocess`, `filter_out_of_range_faces`,
  `_decode_mesh`, `_decode_and_cache`, `_bake_export`, `generate_from_decode`, the
  `--from-decode` / `--pre-cap` / `--no-save-decode` flags, `require_flex_gemm` (hard-pins
  the conv backend, fails loudly), and hardened `configure_environment`.
- `tests/test_trellis_space_generate.py` — `precap_ratio`, `filter_out_of_range_faces` and
  `require_flex_gemm` tests.
- `viewer/generate_api.py` — repo-root wrapper paths, sanitized job env (`_job_env`),
  weights-on-disk + setup status, Run-setup endpoint with the build-present guard, and the
  exit-code / RSS instrumentation.
- `viewer/index.html` — Generate tab Setup card (build + weights checks, Run-setup button
  with live log).
- `tests/test_viewer_generate_api.py` — env hygiene, weights-on-disk, setup availability,
  signal-hint and RSS-probe tests.
