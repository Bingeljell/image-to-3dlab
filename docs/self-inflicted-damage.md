# Two defects we added ourselves, and how a control group found them

> **Read this before diagnosing any mesh quality problem in this repo.** Between 2026-08-05
> and 2026-08-12 a great deal of work went into explaining why TRELLIS produced shattered,
> see-through, softly-textured meshes. Almost all of those explanations were wrong. The
> meshes were damaged by **our own pipeline**, in two places.

## How we found out

On 2026-08-12 the user asked *"how come others are getting MUCH better results with
TRELLIS 2 than we are?"*, ran their own artwork through the official demo at
[huggingface.co/spaces/microsoft/TRELLIS.2](https://huggingface.co/spaces/microsoft/TRELLIS.2),
and handed over the GLB. It is in `assets_to_test/trellis-flicker-huggingface.glb`.

| | Faces | Winding consistent | Signed volume | See-through when culled | Grooves in grey render |
|---|-------|--------------------|---------------|-------------------------|------------------------|
| Ours (100k baseline) | 100,291 | **no** | +0.01230 | yes, badly | forehead V is a torn trench |
| **Official demo** | **281,889** | **yes** | +0.00199 | **no** | **none — smooth** |

Same input image, markings and all. That single reference invalidated more conclusions in
ten minutes than a week of internal experiments produced.

**The methodological failure:** every experiment we had run compared our output against our
other output. A defect present in all of them was invisible and looked like a property of
TRELLIS. **We had no control group.** When a tool ships a public demo, run the real input
through it and keep the artefact, before theorising about the tool's limits.

---

## Defect 1 — a 200,000-face cap that destroyed 94% of the decode

`vendor/trellis-mac/generate.py`, before o_voxel's postprocess runs at all:

```python
# Pre-simplify mesh to avoid mtlbvh crash on large meshes.
# Target ~200K faces — keeps detail, avoids Metal BVH issues.
target_faces = min(args.bake_target_faces, 200000, len(faces_np))
```

From a real run:

```
Mesh: 1,601,340 vertices, 3,207,582 triangles
  Simplifying mesh: 3,207,582 -> ~200,000 faces
```

The decode is **3.2 million triangles**. We crushed it with `fast_simplification` — a
cruder decimator than o_voxel's QEM — and only then ran hole filling, non-manifold repair,
the weld patch, simplification, UV unwrapping and the texture bake. All of it on wreckage.
The visible result is a surface crazed with cracks across the entire body.

**Consequences:**

- **`bake_target_faces` was inert above 200,000.** Requesting 300,000 and 3,000,000 both
  produced ~197k faces. Every sweep of that parameter in this repo's history measured a
  clamped value, including the commit concluding "do not raise `bake_target_faces`".
- Lifting the cap: **290,662 faces, no crash, crazing gone.**

**Fix:** `scripts/patch_trellis_face_cap.py` (idempotent, tested). Default ceiling
1,000,000, matching the TRELLIS.2 README's own example `decimation_target`. Verified at
300,000. The `mtlbvh` crash it guarded against is real but was never quantified — if a run
crashes, lower the ceiling rather than restoring a blanket 200k clamp.

## Defect 2 — meshes shipped inside-out

Our assets had **inconsistent face winding**, and often pointed inward: a Flicker run
measured a signed volume of **-0.02369**. Both demo assets are winding-consistent with
positive volume.

glTF materials are double-sided by default, so a textured preview looks perfect. Turn on
backface culling — which SceneKit, RealityKit and every game engine do — and the asset is
hollow: you see straight through the chest to the inside of the far side. The user spotted
it immediately in Blender: *"the front is not there at all — we can see the inside."*

**This corrupted our own instruments.** Every "see-through hole" percentage measured on a
culled render, and the tear metric this repo gated on for weeks, were substantially
counting *flipped faces* rather than missing geometry. That is why repairs never moved the
number, and why "Recalculate Outside" kept appearing to fix assets that were never torn.

**Fix:** `scripts/fix_winding.py`, and wired into `trellis_backend.generate_trellis` so
every asset ships outward-facing (`TrellisOptions.fix_winding`, on by default). Two steps,
both required:

1. `fix_normals()` makes winding *consistent* — but can leave the mesh uniformly
   inside-out, which is exactly what happened here.
2. If the signed volume is then negative, `invert()`.

---

## What these two defects invalidate

Treat the following as **withdrawn or unverified** until re-measured on a repaired mesh:

| Claim | Status |
|-------|--------|
| "Painted markings become geometry" | **Withdrawn.** The demo carves no grooves from the same artwork. Our decimation did. |
| Softening markings pre-generation, `--protect` ear masks, marking reprojection | Built to treat a symptom of defect 1. The machinery is sound and tested; the *need* for it is unproven. |
| "Fewer faces is a quality win" / "do not raise bake_target_faces" | **Withdrawn.** Measured on a clamped parameter. |
| UV atlas fragmentation tracks face count | **Unverified.** Measured downstream of defect 1. |
| Texture-space painting is impossible on these atlases | **Unverified.** Same reason. |
| Tear metric percentages, hole percentages | **Unreliable.** Largely counted flipped faces. |
| "TRELLIS's texturing is its weak point" | Still true *in the literature*, but not demonstrated by our own outputs — ours were damaged. |

`docs/baseline.md` was measured entirely on damaged meshes and needs re-running.

## The reproducible recipe that works

```bash
python scripts/patch_trellis_face_cap.py --ceiling 1000000   # once, after bootstrap
python pipeline.py --run-manifest manifests/flicker-all4s-uncapped.json
```

with `texture_size: 3072`, `bake_target_faces: 300000`, and winding repair applied
automatically by the backend. Judge the result **backface-culled**, never on a
double-sided preview.
