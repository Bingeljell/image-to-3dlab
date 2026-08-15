# Resume here — end of 2026-08-12

The day two pipeline bugs were found, most of the previous week's conclusions were
withdrawn, and the project went from "TRELLIS can't do this" to "we were breaking it."
Start here tomorrow.

---

## The one-paragraph version

Our vendored Apple Silicon port was crushing every decoded mesh from 3–27 million
triangles down to 200,000 with a crude decimator **before** any real processing ran, and
shipping meshes inside-out. Both are fixed. Every subject improved from those two changes
alone, with no per-subject tuning. The remaining gaps are subject-specific and smaller.
We still cannot match the official demo exactly, and the leading suspect is that
**remeshing — upstream's route to a closed surface — is broken on this port.**

---

## What is fixed, and stays fixed

| Fix | Where | Status |
|-----|-------|--------|
| The 200k face cap | `scripts/patch_trellis_face_cap.py` | Applied. **Re-run after every bootstrap** — `vendor/` is git-ignored |
| Inside-out winding | `scripts/fix_winding.py`, wired into `trellis_backend` | Automatic on every run |

Both are systemic: they affected 100% of assets, every subject, every run. That is why
Flicker, the fox and the Forest Variant all improved without per-subject work.

**Verified working configuration** (no remesh, crude decimation, winding repaired):

```bash
python scripts/patch_trellis_face_cap.py          # once, after bootstrap
python pipeline.py --run-manifest manifests/flicker-all4s-uncapped.json
```

`texture_size: 3072`, `bake_target_faces: 300000`. Runs in 5–8 minutes for most subjects.

---

## Where each subject stands

| Subject | State | Remaining defect | Control available? |
|---------|-------|------------------|--------------------|
| **Flicker** (old, ceramic) | Good — "infinitely better" | HF control still cleaner on eyes and flank lines. Unexplained | ✅ `assets_to_test/trellis-flicker-huggingface.glb` |
| **Fox** (moss) | Good — same class as control | Was inside-out; now repaired automatically | ✅ `assets_to_test/trellis-mossfox-huggingface.glb` |
| **Forest Variant** (new, low-poly) | Promising — 1.78% holes | Holes where thin blades meet the body (zero-thickness sheets) | ❌ **none — get one** |
| **Snag** (thorn knot) | Barely moved | Colour drift (model property). Decodes at 27.6M so 300k keeps only 1.1% | ⏳ user running it on HF, credits permitting |

**The single highest-value thing to do first tomorrow:** run the Forest Variant through
the [official Space](https://huggingface.co/spaces/microsoft/TRELLIS.2). If its blades come
out solid, our blade holes are our bug and worth chasing. If they are holed too, it is the
model and we should fix it downstream with Solidify instead of hunting a cause.

---

## The remesh bug — evaluated, not yet fixed

`--remesh` produces a **wireframe lattice instead of a surface**, at any `remesh_project`
(tested 0 and 0.9 — identical results). Enclosed volume collapses from +0.02830 to
+0.00013; roughly 90% of the culled view is see-through.

This matters because remesh is how upstream produces a **closed** surface before UV
unwrapping — its own help text says it *"targets boundary edges at source"*. Without it we
have no route to a watertight mesh and must repair holes after the fact.

### What the investigation established

- `cumesh/metal_remeshing.py` is a **faithful line-for-line port** of `cumesh/remeshing.py`.
  The `project_back` block is byte-identical, which explains why that parameter changes
  nothing.
- Only the compiled kernel names differ: `hashmap_lookup_3d_cuda` → `hashmap_lookup_3d`,
  `cuBVH` → `MtlBVH`.
- So the defect is in a **compiled Metal kernel**, one of four: `simple_dual_contour`,
  `hashmap_insert_3d_idx_as_val`, `hashmap_lookup_3d`, or `MtlBVH.unsigned_distance`.

### The leading hypothesis, and why

A quad survives only if all four neighbouring voxels are found in the hashmap:

```python
connected_voxel_indices = _C.hashmap_lookup_3d(...)
connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)
quad_indices = connected_voxel_indices[connected_voxel_valid]   # survivors only
```

**If the Metal hashmap lookup wrongly reports "not found", most quads are dropped and what
remains is a sparse skeleton — exactly a lattice.** That makes `hashmap_lookup_3d` the
prime suspect.

### Where the attempt stopped

A ~15-line harness — insert 512 known coords, look the same ones back up, measure hit rate
— **hung and was killed after 10 minutes.** Unknown whether the kernel deadlocks, whether
`mps` is the wrong device for it, or whether the calling convention was wrong. If a trivial
round-trip really does hang, that is itself the finding.

### Tomorrow, in this order

1. **Check whether the Metal kernel source ships at all**, or only a compiled `.metallib`
   and `.so`. Five minutes, and it decides everything: if source is not available we can
   write an excellent bug report but not a patch.
2. Re-run the hashmap harness **backgrounded with a hard timeout**, not in the foreground.
   Try CPU tensors as well as `mps`.
3. If the hashmap is clean, move to `simple_dual_contour` — check whether `intersected` is
   mostly zero (few edges crossing) or whether the quads are being dropped later.

**Effort:** isolating the kernel, 1–2 hours, high confidence. Fixing it — unknown, gated on
step 1.

---

## What was tried and did not work

| Attempt | Result |
|---------|--------|
| Raise `max_hole_perimeter` 10× (3e-2 → 0.3) | **Worse** (4.57% → 6.07%). A sheet has one long open boundary, not a small loop, so a size threshold never reaches it |
| `--remesh` at `project 0` and `0.9` | Lattice both times |
| QuadriFlow retopology | Refuses these meshes — "needs manifold with consistent normals" — even after a voxel remesh |
| Blender Smart UV re-unwrap | Worse than TRELLIS's own atlas (10,943 islands vs 6,763) |
| Solidify (whole mesh, on a damaged mesh) | Closed holes, **crazed the entire surface** |
| Marking softening / `--protect` / reprojection | Sound machinery, but built to treat a symptom of the face cap. Its necessity is now unproven |

**Not yet tried: Solidify on the blade region only.** This is the next experiment for the
Forest Variant — the earlier failure applied it to a whole, already-damaged mesh; here the
target is a small, genuinely sheet-like region.

---

## Two upstream contributions, drafted and unsent

`docs/upstream-contributions.md` — the 200k pre-simplification, and the remesh lattice.
Each has a pre-filing checklist. The user's instruction: **not to be submitted yet**,
contribute later in the week after more experiments.

---

## How to look at anything

```bash
python scripts/blender_stage.py A.glb B.glb --labels "old" "new"
```

Stages assets side by side in the running Blender, textured, **backface culling on**,
nothing selected. Always use this rather than a render helper — `compare_to_source.py`
and friends **wipe the scene and replace materials with grey**, which repeatedly left the
user looking at grey blobs.

**Judge by eye, culled.** Every metric in this repo has misled us at least once; the tear
and hole percentages were substantially counting flipped faces for weeks.

---

## Standing rules earned the hard way

1. **Get a control group before theorising.** One HuggingFace GLB invalidated a week of
   analysis in ten minutes. Every experiment before it compared our output against our
   other output, so a defect present in all of them was invisible.
2. **RTFM and diff against upstream first.** The README example differs from our call in
   five parameters. Reading it took two minutes.
3. **A documented blocker is a hypothesis.** Every one we actually tested — the mtlbvh
   crash, the "weak texturing", the confetti atlas — failed to block.
4. **Distrust any measurement taken before 2026-08-12.** It was made on damaged meshes.
5. **n=1 is not a result.** Confirm on a second subject before concluding.
