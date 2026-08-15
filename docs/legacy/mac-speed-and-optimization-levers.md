# Mac speed & optimization levers

**Written 2026-08-14.** A measured, honest map of *why the Mac pipeline is slow* and *what can
actually make it faster* — written because several real levers kept surviving only as fuzzy
memory ("you said fp16 would 2×", "wasn't there a 10×?"). Numbers here are measured where it
says *measured* and estimated where it says *estimated*. Don't cite an estimate as a result.

Related memory/notes: `stage3-metal-flash-attention-is-the-2x-lever`,
`mps-no-flash-kernel-speed-and-decode-memory`, `guidance-is-the-moss-separation-lever`,
`runpod-4090-may-replace-the-mac-port`.

## TL;DR

- **fp16/bf16 alone is ~1.1×, not 2× (measured).** MPS attention is unfused/bandwidth-bound, so
  shrinking the data barely helps. Precision's real value here is *fidelity* and *memory*, not speed.
- **The real speed lever is a *fused* attention kernel — and Pedro already wrote one.** It exists in
  `mtlgemm` but its tiled path supports head dim **≤64**; TRELLIS.2-4B uses **128**, so the 4B model
  falls back to a serial loop that didn't finish one Snag step in 29m44s. **The lever = extend
  Pedro's kernel head-dim 64→128** (bounded work on real code, not from scratch). This is the
  "64→128" that kept coming up.
- **The "10×" is a slowdown we already avoid** (`flex_gemm` import), not an available speedup.
- **The cheapest real Mac speedups are parameter-level** (CFG interval, sampler steps) and cost
  nothing to try — but they trade the exact quality we tune.
- **RunPod is out of scope as a direction** — the goal is a good *local* pipeline; it's noted below
  only as a search convenience, not a product path.

## Where the time actually goes

Measured on the fresh Flicker run (`flicker-fresh-demo-pbr.json`, 1024_cascade, pbr, 3M target):

| Phase | Time | What it is |
|---|---:|---|
| Generation (diffusion) | 313 s | shape stages + Stage-3 texture; **attention-heavy** |
| Bake (`o_voxel`) | 137 s | QEM simplify + xatlas UV unwrap + Metal texture rasterize; **not** attention |
| Overhead | ~146 s | rembg, model load, winding repair, 98 MB export |
| **Wall clock** | **596 s (9m56s)** | |

Snag's ~54 min "just for Stage 3" is the same *generation* phase blown up: its field is far
larger (more attention tokens) **and** the guidance sweep ran with CFG (`g>1`), which runs the
transformer **twice per step**. Attention dominates there in a way it does not on Flicker.

**Consequence for testing:** Flicker (9 min) is the right fast *harness*, but attention is only a
*slice* of its 313 s. A perfect 2×-attention kernel might shave ~15% off Flicker while nearly
halving Snag. Profile attention's real share before committing to kernel work.

## Measured: precision is not the speed lever

MPS attention microbench (`torch.nn.functional.scaled_dot_product_attention`, 16 heads, dim 64,
the backend the Mac actually runs — confirmed `[ATTENTION] Using backend: sdpa`):

| seq len | fp32 | fp16 | bf16 | fp16 speedup |
|---:|---:|---:|---:|---:|
| 4096 | 24.4 ms | 21.5 ms | 21.5 ms | 1.14× |
| 8192 | 95.7 ms | 85.6 ms | 87.3 ms | 1.12× |
| 16384 | 394 ms | 364 ms | 360 ms | 1.08× |

fp16 and bf16 are within noise of each other and ~1.1× over fp32. **Precision is a fidelity and
memory knob, not a speed knob**, on this unfused path. (bf16 is preferred over fp16 *if* it is the
model's native training precision, because it preserves fp32's exponent range and therefore the
guidance/moss behavior; the stored weight dtype was not confirmed here.)

## The lever table

| Lever | Effort | Speedup | Confidence | Notes |
|---|---|---|---|---|
| **Extend Pedro's fused attention kernel head-dim 64→128** | bounded (existing kernel) | ~2×+ (grows at scale) | estimated/unmeasured | **The main lever.** Kernel exists in `mtlgemm`; caps at head-dim 64, model needs 128 → 4B falls back to a serial loop that didn't finish one Snag step in 29m44s. Extending it lets the 4B model actually use fused attention. Audit-worktree only; must not drift guidance/moss behavior |
| **Faster sampler / step distillation** | research | 4–8× fewer steps | estimated | DPM++/distilled schedules cut step count; real research effort, quality validation |
| **CFG interval / fewer steps** | config | up to ~2× on guided runs | estimated | CFG runs the model 2×/step; narrowing `guidance_interval` or cutting `steps` is free but trades the quality we tune |
| **`pack_options` bake flag** | 1 patch | **~11%** | measured (patch) | Bake only (atlas packing 52.9%→58.8%), no geometry change |
| **`torch.compile` / graph capture** | small | unknown | untested | Sometimes 1.2–1.5× free; often broken on MPS. Worth a spike. |
| **fp16 / bf16 precision** | config | **~1.1×** | measured | ❌ not a speed win; value is fidelity + halved attention memory (helps the OOM) |
| RunPod A6000 (search convenience only) | none | ~100× | measured elsewhere | Out of scope as a *direction* — the goal is a local pipeline. Listed only as an iteration convenience, not a product path. |

### Not levers (already handled or constraints)

- **The "~10× on MPS" is a slowdown avoided, not a speedup.** `generate.py:301` deliberately skips
  installing `flex_gemm` because *its import* slows the diffusion hot path ~10×. Already stepped
  around; nothing to gain.
- **32-bit indexing / Metal buffer size — a ceiling, not a lever** (see below).

## The 32-bit / Metal buffer constraint (why we can't just crank quality)

GPU kernels locate data with integer **indices**. Many of these Metal kernels use **32-bit**
indices, which count only to ~4.29 billion (2³²). Two walls follow:

- **Index overflow** — a buffer with more entries than a 32-bit index can address wraps and reads
  garbage.
- **Buffer-size cap** — Metal limits a *single* `MTLBuffer` (a few GB on Apple Silicon;
  `maxBufferLength` was not read on this machine — the Metal Python binding isn't installed).

Receipts from our own code:

- **The hollow/cage bug *was* this.** A 32-bit hashmap lookup returned `0xFFFFFFFF` on a miss and
  the kernel read `udf[0xFFFFFFFF]` — 4 billion entries past the buffer; Metal returns 0 for
  out-of-bounds reads → lattice. Fixed by `patch_metal_hashmap_miss.py`.
- **The face cap `16,777,216` is exactly 2²⁴** (`generate.py:339`) — a power-of-two ceiling, i.e.
  an index/packing limit, not an arbitrary number.
- **The port already splits u32/u64 kernels** by volume (`patch_mtlbvh_production_traversal.py`) —
  it switches to 64-bit indices for large volumes because 32-bit runs out.

**Texture size ties in directly.** The bake builds GPU buffers sized by texel count: 3072²≈9.4M,
4096²≈16.8M, 8192²≈67M texels. Push it high enough and a buffer either fails to allocate or a
32-bit index overflows → crash/corruption. This is the "our texture size would cross the buffer
limit" constraint. **Important:** this caps how *big* we can go; the fix for going bigger (64-bit
indexing) is *slower and more memory*, not faster. It is orthogonal to "go faster."

## Recommendation

1. **Iterate on RunPod** (sweeps, guidance search, experiments) — ~100×, available now.
2. **On the Mac, try the free knobs first:** `guidance_interval` / `steps` (measure the
   quality cost), and the `pack_options` bake patch (~11%). Consider a `torch.compile` spike.
3. **Only then consider the kernel.** Before writing a fused attention kernel, run the
   **generation profile** to learn attention's real share for normal assets vs Snag, and decide
   dense-vs-sparse. Build and validate it **only in the upstream-audit worktree**, proven against
   the parity target and the Fox/Forest/Snag contracts, with a check that fp16 does not drift the
   guidance/moss result.
4. **Do not** treat any estimate above as a delivered number until it is measured.

## Open measurements (cheap, honest next steps)

- Profile Flicker's 313 s generation → attention vs conv vs other (decides if the kernel is worth
  it for normal assets).
- Read this Mac's `maxBufferLength` / `recommendedMaxWorkingSetSize` → the exact texture-size
  ceiling.
- At Snag-scale sequence length, measure where the current sdpa/sparse path starts chunking/OOM →
  that gap is where a fused kernel's real multiplier lives.
