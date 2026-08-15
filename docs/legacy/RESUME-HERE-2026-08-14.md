# Resume here — Snag material handover for 2026-08-14

This is the current operational handover. Start here before running anything expensive.

## Short version

The catastrophic hollow/cage failure is fixed. Moss Fox and Forest Variant pass the visual
backface-culling test, and the 27.9-million-triangle Snag stress case now completes the same
corrected path without becoming a cage. Do **not** regenerate or remesh these assets tomorrow.

Snag's generated material was a separate outlier: it was almost black, effectively metallic,
and read as dead bark. A fast, deterministic material-only treatment produced
`output/snag_validation/snag_living_v1.glb`. The user reviewed it in Blender and called it
“much better.” This is the accepted starting point, not a finished source match. The next job is
to make the brown vine and green moss read as separate materials and refine the amber eye.

The product direction is now a complete Image-to-3D workbench, documented in
[PRODUCT-PLAN-image-to-3d-workbench.md](PRODUCT-PLAN-image-to-3d-workbench.md).

## Accepted state and artifacts

| Item | Path | Status |
|---|---|---|
| Cached Snag decode | `output/snag_validation/snag_decode.pt` | Reuse; generation is complete |
| Corrected geometry/material master | `output/snag_validation/snag_fixed_opaque.glb` | Geometry starting point; do not overwrite |
| Living-organic candidate | `output/snag_validation/snag_living_v1.glb` | **Accepted starting point for tomorrow** |
| Reproducible recipe | `output/snag_validation/material/snag_living_v1.recipe.json` | Exact v1 parameters and input hashes |
| Current eye UV mask | `output/snag_validation/material/eye_mask_r040.png` | Useful coarse mask; replace/refine by hand |
| Generated v1 maps | `output/snag_validation/material/v1/` | Base colour, metallic/roughness, moss and eye masks |
| Blender review renders | `output/snag_validation/blender_living_v1/` | `profile_yneg` is the intended eye-facing view |

Hashes:

- corrected Snag master: `a8bcaf6538e9df72265d3bc6225e164c7f49a3cc68633e8b5e5142c1b3ea9797`
- coarse eye mask: `84c62b617c28c30ee55956e85d11f373a9a1e78ab8ba1e12009ca1407094734f`
- living-organic v1 GLB: `5fd02f68582461a9651a2d6000808d2ee5ce6ce25419a47bc66efeef30ffdc73`
- living-organic v1 recipe: `bc21afbddda6ea3846d8e589a045acffcc5fc044b361d52f0416f9231778566f`

Blender on port `9876` was left with `snag_living_v1.glb` staged under the same studio camera
and lights used for the review.

## What is fixed, and what is not

Fixed and verified:

- the Mac Metal BVH traversal was silently losing branches on production-sized inputs;
- the exporter was performing a destructive pre-simplification before remeshing;
- generated glTF materials could be left in `BLEND` mode, making opaque rear shells expose the
  front-facing interior under backface culling;
- the fixed Fox and Forest results no longer show the reported cage/false-front failure;
- Snag survives a much larger production run without the catastrophic cage;
- material experimentation now takes seconds and preserves the expensive geometry master.

Still open:

- Snag has small localized boundary/non-manifold defects. It is visually coherent but is not
  formally watertight: 440 boundary edges and 4,257 non-manifold edges were measured;
- Snag v1's moss mask is based mostly on upward-facing normals, so green and brown regions are
  broad and softly blended rather than distinct organic growth;
- the current spherical eye mask is too coarse. The eye can become too large, red-orange, and
  bright; it needs a hand-painted mask and probably separate iris, pupil, and cornea treatment;
- the finishing UI and screen-mask projection path described in the product plan are not built.

The full causal evidence and precise Mac fixes are in
[ROOT-CAUSE-trellis-mac-hollow-fox.md](ROOT-CAUSE-trellis-mac-hollow-fox.md).

## Scale and timing record

| Asset | Raw decode faces | Corrected remesh faces | Export faces | Generation | Cached rebake |
|---|---:|---:|---:|---:|---:|
| Moss Fox | 12,829,054 | 25,592,972 | 484,826 | 1,228.7 s / 20m29s | 412.3 s / 6m52s |
| Forest Variant | 4,956,432 | 9,276,400 | 487,087 | 471.2 s / 7m51s | 136.9 s / 2m17s |
| Snag | 27,876,032 | 41,738,080 | 465,726 | 2,788.4 s / 46m28s | 920.9 s / 15m21s |

Snag's raw generation plus corrected rebake was about 61m49s. The material-only v1 operation
does not invoke diffusion, decode, remesh, UV unwrap, or texture baking and is therefore the
correct loop for tomorrow's visual work.

## Why Snag originally looked wrong

This was not merely dark viewer lighting. The baked Snag field was a per-generation material
outlier:

- base-colour mean luminance was `0.136`, versus source foreground `0.277`;
- median metallic was `1.0`, versus `0.004`–`0.384` in the hosted controls;
- Fox and Forest did not have the same extreme distribution.

The first global-lightness experiments proved that simply lifting the atlas is not the answer.
They made the object washed out while retaining the wrong dead-bark material intent. Snag needs
spatial treatment: warm rough wood/vine, green moss growth, and a localized reflective amber eye.

## Exact accepted v1 recipe

The implementation is [living_organic_material.py](../scripts/living_organic_material.py). It
changes only textures and material parameters; the original corrected GLB remains the master.

```bash
.venv/bin/python scripts/living_organic_material.py \
  output/snag_validation/snag_fixed_opaque.glb \
  output/snag_validation/snag_living_v1.glb \
  --eye-mask output/snag_validation/material/eye_mask_r040.png \
  --body-colour '#6f4521' \
  --body-strength 0.52 \
  --moss-colour '#55742e' \
  --moss-strength 0.72 \
  --moss-up-threshold 0.15 \
  --moss-up-softness 0.45 \
  --moss-blur 2.0 \
  --eye-colour '#d98208' \
  --eye-strength 0.96 \
  --eye-lightness 22 \
  --eye-roughness 0.18 \
  --lightness-gamma 0.94 \
  --rough-low 0.62 \
  --rough-high 0.94 \
  --dump-dir output/snag_validation/material/v1 \
  --recipe output/snag_validation/material/snag_living_v1.recipe.json
```

Do not treat those numbers as universal defaults. They are the accepted Snag v1 checkpoint and
the reason the exported recipe exists.

## Tomorrow's first moves

1. **Do not rerun TRELLIS.** Work from `snag_fixed_opaque.glb` and compare every candidate to
   `snag_living_v1.glb` using the same studio views.
2. **Add screen-mask projection.** Let the user paint a black/white mask over the 900×900
   `profile_yneg` review render, then ray-project painted pixels onto UV space. Keep the current
   spherical UV mask only as a fallback.
3. **Refine the eye.** Use a smaller, accurate iris mask; preserve or explicitly mask the dark
   pupil; use a less red amber (starting near `#c88708` or `#d79a12`); keep roughness around
   `0.12`–`0.22`; avoid whitening the whole eye region.
4. **Separate moss from vine.** Replace the pure broad up-normal mask with a combination of
   upward normal, original green/chroma evidence, and small-scale breakup. Start with a higher
   up threshold (`0.30`–`0.45`) and narrower softness (`0.20`–`0.30`), then A/B the masks before
   exporting a candidate.
5. **Keep the body stable.** The warm-brown v1 body is already much closer. Do not apply another
   global brightness lift.
6. **Get one visual approval** on brown/green separation and the eye before turning the scripts
   into the workbench API/UI described in the product plan.

After the Snag material is approved, the workbench build order begins with job/project folders,
the versioned recipe schema, generation/cache APIs, viewer modularization, profile controls,
mask painting, validation gates, and Fox/Forest/Snag acceptance fixtures.

## 2026-08-14 quick material A/B

Three material-only candidates were generated from the same corrected master. None reran
TRELLIS, remeshing, UV unwrap, or texture baking.

| Candidate | Change | Finding |
|---|---|---|
| `snag_living_v2a.glb` | Warmer brown; narrower up-normal moss; smaller amber-eye mask | Better brown/green division than v1, but green still reads as a continuous painted top |
| `snag_living_v2b.glb` | Strongly intersect moss with green chroma from the original atlas | Proves the atlas-evidence mask works, but suppresses too much visible moss |
| `snag_living_v2c.glb` | Middle strength between v2a and v2b | **Current review candidate.** Retains brown vine and confines green more convincingly |

Current artifact and recipe:

- `output/snag_validation/snag_living_v2c.glb`
  (`c3ed6afd5f704fed5b9539e4c0d86d7b89137c8c5c469d97bcf812b437a09a7b`)
- `output/snag_validation/material/snag_living_v2c.recipe.json`
  (`3331d655f6fd6edcf8f98cee9755667d658bdb01e592e4c975882691f108d509`)
- renders: `output/snag_validation/blender_living_v2c/`

`living_organic_material.py` now exposes `--moss-chroma-strength`. `0` preserves the old
direction-only behavior; larger values intersect the directional mask with green hue and
saturation evidence already present in the generated atlas. This makes the operation
deterministic and reusable without pretending the atlas contains semantic segmentation.

The honest limitation remains: texture finishing can make moss patchier, but it cannot invent
the source image's raised moss-growth geometry. A hand-painted/projected mask is still the next
high-value refinement if v2c is not sufficiently close.

## Pixel-to-mask bridge

[blender_pick_pixel.py](../scripts/blender_pick_pixel.py) raycasts a pixel from a known studio
view and prints both Blender-local and glTF-local coordinates. Only `gltf_local` should be passed
to `feature_mask.py --centre`; Blender changes axes when importing glTF.

The coarse v1 mask was built around the profile-view eye using approximately:

```bash
.venv/bin/python scripts/feature_mask.py \
  output/snag_validation/snag_fixed_opaque.glb \
  output/snag_validation/material/eye_mask_r040.png \
  --centre 0.020049 0.071551 0.169925 \
  --radius 0.040 --size 1024 --dilate 3 \
  --front-only --facing 0 0 1
```

That selected 146 faces and about `0.095%` of the texture atlas. It was enough to prove the
material lane but not precise enough for final art direction.

## Rejected paths and warnings

- `snag_fixed_matte_lift3.glb`: rejected as washed out.
- `snag_fixed_review.glb`: rejected as washed out/lifeless.
- global lightness matching: rejected; it does not create distinct moss, vine, and eye materials.
- blanket normal recalculation: rejected; it visibly made Snag worse.
- double-sided rendering or disabling backface culling: not a geometry fix and not an acceptance
  criterion. Final assets must read correctly with ordinary culling.
- old eye coordinates from another mesh/export: unsafe; raycast the exact staged artifact.
- another diffusion run: unnecessary for the current material task and costs roughly 46 minutes
  on Snag before rebaking.

## Validation commands

Focused material tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_living_organic_material.py \
  tests/test_feature_mask.py \
  tests/test_lift_lightness.py
```

The focused result at handover was `20 passed`, with one harmless scikit-image gamut warning.
The latest full-suite result was `433 passed, 4 warnings` in `0.68s`. The warnings are three
Pillow deprecations and one expected scikit-image gamut-clipping warning.

A direct array comparison between the corrected master and living-organic v1 also confirmed:

- vertices byte-for-byte equal: `true` (`434,435` vertices);
- triangle indices byte-for-byte equal: `true` (`465,726` faces).

Run the full suite again before extracting or publishing a standalone Mac-port repository.

## Later: standalone open-source Mac port

The current repository is broader than a TRELLIS Mac port. When publishing, create a clean
standalone repository with upstream attribution, a pinned upstream/port lineage, the Metal and
export fixes, reproducible fixtures, topology/material metrics, and the Fox/Forest/Snag
acceptance tests. Do not extract it until the worktree is audited; this workspace also contains
unrelated experiments and user work.
