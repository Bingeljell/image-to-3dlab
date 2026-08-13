# Hunyuan3D, evaluated from first principles

**Date:** 2026-08-13. **Status:** weights downloaded, no port adopted, no code written yet.

This document exists because the previous Hunyuan note ([hunyuan-paint-plan.md](hunyuan-paint-plan.md))
was built out of other people's README claims and was wrong within a week of being written.
This one is built the other way round: measurements first, claims last, and every claim
labelled with how much we should trust it.

---

## 1. The frame

**Goal.** An image → 3D pipeline that reproduces our source art. Not "a working pipeline" —
we have one of those. One whose output looks like the drawing we fed it.

**Resources.**

- TRELLIS.2 4B, vendored as a Mac port, running locally, now free of the two defects we
  inflicted on it.
- Hunyuan3D-2.1 — shape DiT 3.3B + Paint PBR 2B — **weights now on disk** (14 GB, ungated,
  in the shared HF cache). Not installed, not running.
- Two hosted reference implementations we can pay per-run to use as controls.
- Other people's ports: three for Hunyuan on Apple Silicon, all of varying quality and age.

**Constraints.**

- **Apple Silicon.** No CUDA. Every rasteriser in this space was written for nvdiffrast.
- **Nobody is maintaining any of it.** This is the constraint we keep underweighting, and
  §4 makes the case that it is the dominant one.
- **We are the maintainer of anything we adopt.** Permanently. There is no upstream to
  send a fix to and no release that will fix it for us.

**Method.** Everything below is graded:

| Tag | Meaning |
|---|---|
| **[MEASURED]** | We ran it against an artifact on this disk. Believe it. |
| **[READ]** | We read the source. Believe the code, not the prose around it. |
| **[CLAIMED]** | Someone wrote it in a README. This is a **hypothesis**, not a fact. |

The repo has been burned three times now by treating **[CLAIMED]** as **[MEASURED]** —
the 200k face cap, the "remesh destroys leafy meshes" finding, and the Hunyuan paint
blocker. The tags are not decoration.

---

## 2. What the control actually is — [MEASURED]

Before spending anything on new controls, we measured the two Hugging Face TRELLIS GLBs
already sitting in `assets_to_test/`, against our own post-fix output for the same subjects.
Reproduce with `python scripts/glb_forensics.py <file.glb>`.

| | HF Flicker | **our Flicker** | HF moss fox | **our fox** |
|---|---|---|---|---|
| faces | 281,889 | 293,488 | 283,149 | 282,610 |
| vertices | 180,610 | 193,930 | 274,113 | **340,015** |
| boundary edges | 75,673 | 88,890 | 224,057 | 295,892 |
| **non-manifold edges** | — | — | **1** | **1,233** |
| winding consistent | yes | yes | yes | **no** |
| `doubleSided` | **false** | **true** | **false** | **true** |
| metallicRoughness map | **3072²** | **none** | **3072²** | **none** |
| metallicFactor / roughnessFactor | 1.0 / 1.0 | 0.0 / 1.0 | 1.0 / 1.0 | 0.0 / 1.0 |
| baseColor map | 3072² | 3072² | 3072² | 3072² |
| face area max/median | 7.1 | 5.9 | 20.0 | **80.9** |
| texture compression | `EXT_texture_webp` | none | `EXT_texture_webp` | none |

**Most of this confirms what 2026-08-12 already established.** Being explicit about which
is which, because conflating them is how a repo talks itself into re-doing settled work:

| | Status |
|---|---|
| Boundary edges are not a quality metric | **already known** (`holes-are-sheets-not-tears`, `check-what-the-metric-measures`) — new here is only that it is now anchored to a control |
| We ship no surface material | **already known** (`surface-finishing-lane`, `matte-mode-kills-the-eye`) — new is the specific channel, its size, and that TRELLIS already emits it |
| Face counts now match the reference | **new** — first control-anchored confirmation the cap fix landed |
| Our fox is still winding-inconsistent | **new** — a live defect on a hero asset |
| We still declare `doubleSided: true`; the reference does not | **new** |
| Our mesh is measurably sliverier (80.9 vs 20.0) | **new** |
| Non-manifold edges: 1 vs 1,233 | **new** — and the candidate to replace the retired hole gate |

The two "already known" rows are the ones that matter least. Read §2.3 and §2.4 first.

### 2.1 Face count is no longer the gap — the cap fix worked

~282k against ~283k and ~293k. We are in the same geometric budget as the reference. The
200k-cap repair landed and it is done. **Stop tuning face counts.**

### 2.2 The control is full of holes, and it looks great anyway — *confirms 2026-08-12*

The reference moss fox has **224,077 boundary edges**. Merging by position changes that by
20 edges, so these are genuine open boundaries, not UV seam splits
([mesh-topology-measurement-trap](../CLAUDE.md) applies and was checked).

This is the output the user describes as reproducing the source "really well."

Our fox has 308,132. Same order of magnitude. **Boundary-edge count does not separate us
from the reference**, which means every hour spent driving that number down was spent on a
metric that does not track the goal. This is the quantitative confirmation of
`holes-are-sheets-not-tears`: foliage *is* sheets, sheets *have* boundaries, and a
watertight moss fox would be a worse moss fox.

The gate was already retired yesterday. What this adds is the number on the *other* side of
the comparison — until now we only knew our own meshes had many boundary edges, not that a
reference asset the user rates highly has just as many. It is a diagnostic for a specific
question ("is this surface closed where I expect it to be closed?"), never a score.

### 2.3 We are still shipping the double-sided crutch, and one asset is still inside-out

The reference declares `doubleSided: false` — it is confident enough in its winding to turn
off glTF's forgiving default. We declare `true`, and **our fox is still
`is_winding_consistent: false`.** The winding fix did not fully take on that asset.

This matters more than it looks: SceneKit and RealityKit both backface-cull by default
(`game-integration-swiftui`), so a double-sided crutch masks in preview exactly what the
game engine will expose. This is the same failure that hid the inside-out meshes for weeks.

### 2.4 The reference ships a PBR material. We ship flat paint. — *sharpens 2026-08-12*

The control has a **3072² metallicRoughness texture** with `metallicFactor: 1.0` and
`roughnessFactor: 1.0` — the map drives the material. We ship **no MR map** and hard-code
`metallic 0.0, roughness 1.0`: mathematically flat, no specular response, under any light.

So part of "the demo reproduces my art really well" is not shape and not albedo at all —
it is that the demo's asset *responds to light* and ours cannot.

`surface-finishing-lane` already recorded "our assets have no surface", and
`matte-mode-kills-the-eye` already found one symptom on Snag's eye. What is new is that
the cause is now located exactly, in our code — **[READ]**,
`image_to_3dlab/trellis_backend.py:118-124`:

```python
if mode == "matte":
    if pbr.get("metallicFactor") != 0.0:
        pbr["metallicFactor"] = 0.0
    pbr["roughnessFactor"] = 1.0
    if pbr.pop("metallicRoughnessTexture", None) is not None:   # <-- the map, deleted
```

`matte` is the **default** (`cli.py:86`). TRELLIS.2 emits the metallicRoughness texture and
we delete it on every organic subject, then set the factors to mathematically flat.

**And the function's own docstring explains how we got here** — TRELLIS exports
`alphaMode=BLEND` *and* `metallicFactor=1` *and* an MR texture, which together rendered as
"transparent, mirror-like shards". The real culprit was `alphaMode=BLEND`, and **both modes
already fix that**. Stripping metalness was bundled into the same commit as the actual fix
and has been riding along ever since.

The control settles it: the reference ships `alphaMode: OPAQUE` **and** `metallicFactor:
1.0` **and** the 3072² MR map. That is exactly what our existing `--material-mode pbr`
produces. So this is not a lane, not new code, and not a Hunyuan problem — **it is a wrong
default**, and the fix is a flag we shipped weeks ago and then defaulted away from.

### 2.5 Non-manifold edges: 1 versus 1,233 — the replacement metric

Retiring boundary-edge count left us without a geometric gate. This is the candidate,
and it fell out of the same measurement.

A **boundary** edge (one adjacent face) is legitimate — it is what a leaf or a moss frond
*is*. A **non-manifold** edge (three or more adjacent faces) is not geometry any renderer,
physics engine or subdivision scheme can interpret; it is always damage.

The reference moss fox has **1**. Ours has **1,233**.

That is the cleanest separation in the entire table: three orders of magnitude, on a metric
that — unlike boundary count — is not confounded by legitimate sheet geometry. It is also
consistent with §2.3's winding inconsistency and §2.5's slivers, all three being signatures
of the same post-processing.

### 2.6 Our fox mesh is measurably messier

Face area max/median of **80.9** against the control's 20.0, and 340k vertices for 283k
faces where the control needs 274k. More degenerate slivers, more attribute splits. That is
a decimation signature, and it is the surviving trace of our own post-processing.

---

## 3. Hunyuan3D: what it is, and what it is for here

**[READ]** Upstream separates shape (`Hunyuan3D-DiT`) from texture (`Hunyuan3D-Paint`), and
Paint accepts a mesh you hand it — upstream's own words are "texture generation for
handcrafted mesh". The API is one call:

```python
paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_path, image_path='ref.png')
```

**This is the whole reason Hunyuan is interesting to us.** Not as a replacement pipeline —
as a *stage*. TRELLIS.2 makes the geometry we now know is in the right budget; Hunyuan
paints it with real PBR channels. It is additive, it keeps the rigs bound to existing
meshes, and it isolates the variable the way `hunyuan-paint-plan.md` originally specified.

Note what §2.4 does to this argument, though: **the gap Hunyuan-Paint would close is
partly a gap we opened ourselves by discarding TRELLIS's own MR map.** Fix that first, then
ask whether Hunyuan still wins. Otherwise we will "prove" Hunyuan's superiority against a
handicapped baseline — the exact error that made every pre-2026-08-12 measurement useless.

### Licence

`provenance.py` already classifies Hunyuan as `territory-restricted` and refuses
`use_case: game` + `distribution: worldwide` — no EU, UK or South Korea. Research manifests
(`showcase` / `private`) pass cleanly. The `LICENSE` and `Notice.txt` are now on disk and
should be read against that classification before any asset ships.

---

## 4. The maintenance picture — [MEASURED], and it is the decisive fact

Queried from the GitHub API on 2026-08-13:

| Repo | Last push | Stars | Open issues | Verdict |
|---|---|---|---|---|
| Tencent-Hunyuan/Hunyuan3D-2 | 2025-10-28 | 14,476 | 244 | **abandoned** |
| Tencent-Hunyuan/Hunyuan3D-2.1 | 2025-10-17 | 3,830 | 153 | **abandoned** |
| kijai/ComfyUI-Hunyuan3DWrapper | 2026-03-16 | 1,033 | — | alive, **CUDA-only work** |
| dgrauet/Hunyuan3D-2.1-mlx | 2026-07-18 | 9 | 0 | **the only live Apple Silicon effort** |
| Brainkeys/Hunyuan3D-2.1-mac | 2025-08-10 | 5 | — | a year cold |
| Maxim-Lanskoy/Hunyuan3D-2-Mac | 2025-02-26 | 0 | — | dead |

Tencent shipped 2.1 in June 2025, pushed until October 2025, and moved to HunyuanWorld.
**Ten months cold with 397 open issues between the two repos.**

Two consequences we should state plainly:

1. **Nothing we fix gets upstreamed.** The two TRELLIS bugs we drafted at least have a
   maintainer to receive them. Here there is no one. Anything we adopt, we own forever —
   and `vendor/` is git-ignored, so we already know what that costs.
2. **`hunyuan-paint-plan.md` cited Brainkeys as the state of the art on 2026-08-07. That
   fork had already been untouched for a year.** We treated a stale README as current
   because we did not check a timestamp. Checking timestamps is now part of RTFM.

---

## 5. The dev's port notes, evaluated — [CLAIMED], April 2026

The 11-item list in [hunyuan-port-notes.md](hunyuan-port-notes.md), assessed. It is a
genuinely useful document, and it is not a patch set we can apply.

**What it gets right and what it costs:**

- **Item 1 kills our recorded blocker.** `custom_rasterizer` "already had CPU code, just
  couldn't build". Our doc said paint was CUDA-blocked; it was a *build system* problem.
  That is the seventh "documented blocker" this repo has disproved.
- **Item 2 prices it.** MPS tensors are copied **to CPU** for the rasteriser. CUDA-free
  correctness, not GPU speed. Same performance class as the CPU fallback fork.
- **Item 6 is real engineering** — chunked attention (Rabe & Staats) to avoid a 170 GB
  O(n²) allocation. Also tells us the vanilla path hard-crashes on Mac above 8192 tokens.
- **Item 10 is a genuine upstream bug** (`enable_model_cpu_offload()` called on a model
  rather than a pipeline) with, per §4, nowhere to send it.

**Item 4 is a red flag we should not adopt under any circumstances.** "Copy local fixed
Python files over HF-downloaded ones before model loading" mutates the **shared** Hugging
Face cache — the same cache now holding `TRELLIS.2-4B`, `TRELLIS-image-large` and the
Hunyuan weights we just pulled. Patches that live outside version control and rewrite a
global cache are strictly worse than `vendor/`, and `vendor/` cost us a week.

**What is missing is the tell.** No validation against a CUDA reference. No timings. No
memory figures. No statement of which Hunyuan version. Items 8 and 9 *change the sampler
and the attention handling*, so output equivalence is not merely unproven — it is unasked.
That is precisely the mistake that cost this repo a week: a port that deviates, never
compared against a control.

### Time decay — the user asked, and it is worse than "somewhat stale"

The notes pin **diffusers 0.37.1** (released 2026-03-25 — current the month they were
written). Since then: **0.38.0** (2026-05-01) and **0.39.0** (2026-07-03). Two releases.
And the two most output-affecting items are precisely the version-coupled ones.

- **Item 8 is not stale — it is permanent, and that is worse.** `rescale_betas_zero_snr=True`
  forces the final `alpha_cumprod` to zero, so `sigma = sqrt((1-α)/α)` diverges at the first
  step. Euler variants consume sigmas directly and blow up; DDIM's parameterisation does
  not. That is a *structural* incompatibility, not a regression — so it was probably never
  "fixed" and never will be. The DDIM substitution is therefore a **permanent sampler change
  whose effect on output nobody has measured.** Same shape as the 200k cap: a workaround
  that outlived any check on what it cost.
- **Item 9 is the opposite risk** — an attention-output-shape patch keyed to one diffusers
  version, exactly the kind of thing 0.38/0.39 could have silently re-broken.

**Verdict: the notes are a map, not a patch.** The 11 categories are the right categories
and would save days of rediscovery. The specific fixes in items 8 and 9 must be re-derived
against whatever diffusers we actually install, and item 4 must be replaced with a patch
script under `scripts/` in the manner CLAUDE.md already prescribes.

---

## 6. The three routes, and what each really costs

| | ComfyUI + MPS port (the notes) | dgrauet MLX | Rent an NVIDIA box |
|---|---|---|---|
| Rasteriser | CPU | **Metal (GPU)** | CUDA |
| Paint existing mesh | via workflow graph | **one call** | yes |
| Our integration cost | **zero** — `comfyui_backend.py` already speaks this | new backend module | new backend module |
| Port discipline | unvalidated (§5) | claims 1e-5 vs PyTorch | reference |
| Bus factor | 1, notes are 4 months old | 1, 9 stars | n/a |
| Landmine | HF-cache mutation, DDIM swap | **auto-remesh to 40k faces** | cost per hour |

**The MLX port's landmine deserves its own line**, because we have seen this exact bug:
it **automatically remeshes to ~40k faces before the texture bake**, because its Metal
rasteriser cannot take stage 1's ~500k. That is structurally identical to the 200k face cap
— a silent pre-simplification upstream of the stage you are trying to evaluate. It is also
trivially defused: hand it a mesh already under 40k so the remesh is a no-op, and **assert
faces-in == faces-out** before believing any comparison it produces.

---

## 7. The path

Ordered by evidence-per-unit-cost, not by ambition.

1. **Flip the material default to `pbr`** (§2.4). No new code — the mode exists. It stops
   deleting a 3072² map TRELLIS already hands us, and it matches the reference exactly.
   Ship `doubleSided: false` alongside it and let culling tell the truth. Re-render Snag's
   eye afterwards; `matte-mode-kills-the-eye` predicts it fixes itself.
2. **Fix the fox's winding** (§2.3) — `is_winding_consistent: false` is a live defect on a
   hero asset, and it is our own.
3. **Adopt non-manifold edge count as the geometric gate** (§2.5), alongside
   `scripts/compare_to_source.py` and a culled render, per `closeness-to-source-is-the-goal`
   and `acceptance-is-what-you-can-see`. Then find which post-processing step introduces
   1,233 of them — winding, slivers and non-manifold edges are almost certainly one bug.
4. **Then, and only then, Hunyuan-Paint on TRELLIS geometry** — via the MLX port, because
   the experiment is one API call and needs no ComfyUI install. Feed it a ≤40k mesh and
   assert the face count. If it wins on a *fair* baseline, promote it into
   `comfyui_backend.py`'s world, which is the better production home.

Steps 1–3 are all ours, all cheap, and all target differences we have now **measured**
rather than read. Step 4 is the only one that needs someone else's code.

---

## 8. What the paid controls are for

The user is buying reference GLBs from both hosted demos. Their value is **not** "does it
work" — that is asserted and not in dispute. Their value is forensic: they are the only
uncontaminated evidence of what a correct pipeline emits for *our* art. Run every one
through `scripts/glb_forensics.py` on arrival and diff against ours, exactly as §2 did.

Specifically, they should answer:

- Does Hunyuan's Paint ship channels TRELLIS does not — normal, occlusion, emissive?
- What face count does Hunyuan's own shape stage land on, and is it remeshed
  (low edge-length CV) or marching-cubes raw?
- Does either reference ever ship `doubleSided: true`, or is single-sided universal?
- Is the 3072² texture size a ceiling or a default?
