# Finishing a generated asset

> **The tear-metric gate in this document is retired — see [baseline.md](baseline.md)
> (2026-08-12).** It measures the mesh against itself, so it cannot see a dead texture,
> a missing sheen, or a marking that came out thin. It had already collected three
> caveats; Flicker added a fourth when its tear score halved while the mesh visibly got
> worse. Keep printing it as a diagnostic; judge with the four-panel source comparison
> (`scripts/compare_to_source.py`) and your eye. Everything else here still stands.

A generated asset arrives with a correct silhouette, roughly correct colour, and **no
surface**. This document covers what happens after generation: the four post-processing
steps that take an asset from "correct shape, lifeless" to something you would put in a
game, and the judgement calls each one needs.

Generation is one step. Everything here is the other six.

## Why finishing exists

TRELLIS emits a base colour texture and a metallic-roughness texture that is
**near-constant** — measured on the thorn-knot Snag, its red channel is identically 0 and
roughness sits at ~238/255 everywhere, standard deviation 13.6. It also sets
`metallicFactor=1` and `alphaMode=BLEND`, which together render a dense mesh as
transparent mirror-like shards; `_normalize_glb_material` in `trellis_backend.py`
therefore strips the whole thing in `matte` mode.

Either way, **the model never says which parts are wet and which are dry**. Every texel
is equally matte, so nothing catches light, and the asset reads as one undifferentiated
lump. A user's description of the result was "it looks like a turd rather than bark",
and that is exactly the defect: correct colour, no surface.


## Gate the whole thing on how torn the decode is

**Run `scripts/ribbon_metric.py` before spending a minute on anything below.** It reports
the share of faces touching an open edge. A closed mesh is 0%, a surface with a few tears
is 1-3%, and above ~10% no amount of finishing helps.

Set empirically from our own assets: Flicker 3.1% (the asset judged best by eye), moss fox
14.7%, thorn-knot Snag **40.9%** before the fix below. At 40% the average patch is two or
three triangles wide -- a mesh of ribbons, not a surface with holes.

**Three limits, all learned the hard way:**

1. **Necessary, not sufficient.** Both SF3D assets score a perfect 0.0% and are unusable.
   Use it to reject, never to accept.
2. **Not comparable across face counts.** 392K faces scored 8.8% and 98K scored 8.9% for
   meshes that look nothing alike: the same *proportion* over 4x the faces means many small
   holes rather than few consolidated ones.
3. **Judge smooth-shaded, or both.** Flat shading gives each face its own normal so tears
   read as mild creases -- it flatters a torn mesh. Smooth shading interpolates across the
   tear and glitters. Flat is right for diagnosis and wrong for shippability, and a full
   round of conclusions here inverted when re-rendered smooth.

### The defect the gate was catching

`o_voxel/postprocess.py` calls `repair_non_manifold_edges()` immediately before every
`simplify()`. That repair splits vertices -- cumesh's docstring: *"This creates duplicate
vertices with the same coordinates."* QEM edge collapse cannot collapse across a duplicate
pair, so the simplifier tears the surface open at every seam it just made: **7.8% torn
entering that step, 44.7% leaving it.**

Fix with `scripts/patch_ovoxel_weld_before_simplify.py`, which welds exactly-coincident
vertices first. **Do not raise `bake_target_faces` to compensate** -- decimation is doing
essential cleanup here, and more faces preserves more damage. Keep ~100K, and note that
the raw decode is not a better master to finish from.

## The steps

| # | Step | Script | Input it needs |
|---|------|--------|----------------|
| 1 | Generate | `pipeline.py --run-manifest` | source image + manifest |
| 2 | Close holes | `blender_solidify.py` | any mesh |
| 3 | Grade colour | `colour_match_albedo.py` | GLB + source image |
| 4 | Bake AO | `blender_bake_ao.py` | GLB with UVs |
| 5 | Mask a feature | `feature_mask.py` | GLB with UVs + a 3D point |
| 6 | Build the surface | `surface_detail.py` | GLB with a base colour texture |
| 7 | Render and judge | Blender, headless | — |

**Backend choice is per-purpose, and it is not a finishing decision.** Measured on
Flicker 2026-08-11: SF3D produced 12,980 faces with **no eye sockets, no muzzle and no
mouth** — a smooth lump with smudges where the features should be — against TRELLIS's
97,045 faces with all of them intact. SF3D's geometric resolution is fixed by its
architecture; `remesh`/`target_vertices` can only *reduce* it, so there is no quality
knob. Use TRELLIS for anything with a face, SF3D for props, where it wins on being
watertight, fast, and hole-free.

Notably SF3D *does* ship a real normal map (only 0.7% neutral texels) and a predicted
roughness of 0.405 — both of which TRELLIS lacks — and it still looks worse. **A normal
map adds surface to a correct shape; it cannot add shape.** That is the cleanest
statement of what finishing can and cannot do.

**Only step 1 is backend-bound.** Steps 2–7 need nothing but *a GLB with UVs and a base
colour texture*, so they work on SF3D or Hunyuan output too. Three caveats: the *numbers*
do not transfer (normal strength depends on texel density, and SF3D ships ~23k faces
against TRELLIS's 100k–370k); step 2 is a no-op on watertight SF3D output; and a backend
emitting **vertex colours instead of a UV atlas** would break steps 3–6 entirely.

### Worked example — the thorn-knot Snag

```bash
# 2. close the holes (Solidify takes hole size 125.20 -> 0.00)
blender -b -P scripts/blender_solidify.py -- <generated>.glb output/regions/thorn_solid.glb

# 3. grade chroma toward the source art; 0.6-0.8 is the usable range
python scripts/colour_match_albedo.py output/regions/thorn_solid.glb \
    assets_to_test/3-4th-snag-roots-alpha.png output/regions/thorn_graded_0.6.glb \
    --strength 0.6

# 4. bake contact-scale ambient occlusion
blender -b -P scripts/blender_bake_ao.py -- \
    output/regions/thorn_graded_0.6.glb output/regions/thorn_ao.png 24 2048 0.035

# 5. mask the eye so it can be made glossy (see "Locating a feature" below)
python scripts/feature_mask.py output/regions/thorn_graded_0.6.glb \
    output/regions/thorn_eye_mask.png --centre 0.0501 -0.1638 0.0918 \
    --radius 0.042 --front-only

# 6. derive the surface and wire everything into the material
python scripts/surface_detail.py output/regions/thorn_graded_0.6.glb \
    output/regions/thorn_final.glb \
    --normal-strength 6.0 --rough-low 0.55 --rough-high 0.95 \
    --ao output/regions/thorn_ao.png --ao-strength 0.35 \
    --gloss-mask output/regions/thorn_eye_mask.png --gloss-roughness 0.3
```

## The recipe is per-subject, and a human picks it

This is the single most important thing in this document. **There is no universal
recipe.** Two subjects finished on 2026-08-11 needed opposite settings, and the deciding
input was a person looking at the source art and judging what the material is.

| | Snag — bark | Flicker — glazed ceramic |
|---|---|---|
| roughness | 0.55–0.95 (matte) | **0.12–0.40 (glossy)** |
| normal map | 6.0 — helps | **0 — actively harmful** |
| eye | needs its own gloss mask | **free** — global gloss lights it |
| AO | helps | helps |

**Why the normal map flips sign.** `surface_detail.py` derives relief from the albedo's
luminance, so every dark mark becomes a groove. On bark that is right — the dark marks
really are cracks. On porcelain the markings are *paint*, and at strength 6.0 Flicker's
ears and flank went visibly crusty with its stripes embossed into dents.

**AO was the only step that transferred unchanged.** That is not a coincidence: AO is
*measured* from geometry, while the normal and roughness maps are *inferred* from paint.
Measured steps generalise; inferred ones need a human.

Suggested starting points, to be checked by eye and not trusted:

| material read | roughness | normal strength |
|---|---|---|
| bark, stone, fur, foliage | 0.55–0.95 | 4–8 |
| glazed ceramic, polished shell, wet skin | 0.12–0.40 | 0 |
| rough stone with deep relief | 0.65–0.95 | 6–10 |

## Locating a feature

To make one part of an asset behave differently — a glossy eye, an emissive slot — you
need to know which texels it occupies. **Colour will not tell you.** On the thorn-knot,
grading warms the bark toward the same amber as the eye, so a hue threshold shatters into
155 disconnected fragments, the largest just 21 texels.

Geometry will. The route that works:

1. Render the asset and find the feature in the **render**, where it is a clean
   contiguous blob (the thorn's eye: 1133 pixels, 47x64, unmistakable).
2. Cast rays back through those pixels in Blender to get the hit faces.
3. Read those faces' UVs and rasterise them into a mask.
4. **Verify before using it** — paint the masked texels magenta and render. Do not trust
   a mask you have not looked at.

`feature_mask.py` covers steps 3–4 given a centre and radius. Steps 1–2 are still a
hand-run script; folding them in as a `--from-render` mode is an open task.

Note the mask will be scattered across the atlas rather than contiguous — the eye's 176
faces spanned nearly the whole 2048 map. That is fine for masking; it is only a problem
for operations that need to dilate, which is what blocked painting the Monolith's eyes.

## Traps

**Blender does not render glTF `occlusionTexture`.** It imports it into an inert
"glTF Settings" group that affects export only. A three-way comparison of "no AO / AO /
more AO" will show no difference at all, and it is tempting to conclude AO does not help.
To *evaluate* AO in Blender, multiply it into the base colour; keep that as a viewing
copy only. The shipped asset keeps AO as a separate `occlusionTexture`, which SceneKit
and RealityKit both honour — baking it in would double-darken once the engine adds its
own.

**AO ray distance is the whole ballgame.** At Blender's default the rays reach across the
entire subject, so a coiled mass answers "am I buried inside the pile?" — yes, nearly
everywhere — instead of "am I in a crevice?". The first bake came back 68% occluded and
rendered as mud. `blender_bake_ao.py` now derives the distance from the subject's own
bounding box; 3.5% of the diagonal gives contact shading (47% occluded).

**Judge at the right zoom.** At full-body distance the normal map is nearly invisible and
a comparison sheet will look like four identical images. The relief operates at
bark-crack scale. Render a crop, and orbit the model — normal maps and gloss only prove
themselves when the light angle moves across them.

**Scale the model up for viewing.** A lineup normalised to 2 Blender units is too small
to navigate comfortably; use ~10.

**Painted markings can become geometry.** Verified on Flicker: with every texture
stripped, the forehead V and shoulder chevrons are still there as physical cracks with
ragged lips, and the eye rims are torn. A hard dark line on a light body reads as a
shadow, and a shadow implies a crease, so the generator carves one. **No texture work
can fix this** — see `conditioning-images.md`. The likely fix is to soften marking
contrast in the conditioning image; the fallback is masked geometry smoothing.

## Colour grading is per-subject too

`colour_match_albedo.py` grades chroma in CIE LAB and deliberately leaves lightness
alone. Above strength ~0.8 it over-saturates: rescaling chroma *spread* to match the
source stretches the body's hue toward the eye's, and the eye stops reading as an eye
because nothing separates it from the body.

The hoped-for win — one global inverse applied to every asset ever made — **is dead.**
Measured ratio of baked to source brightness, by percentile:

| percentile | thorn | flicker |
|---|---|---|
| 1 (shadows) | 1.03 | 0.45 |
| 50 | 0.50 | 0.77 |
| 95 (highlights) | 0.45 | 0.82 |

Opposite directions. The thorn's highlights are crushed, which is why its amber eye came
out pale; Flicker's shadows are crushed while its white body survives. Grade strength is
per-subject, picked by eye off a rendered lineup.

**Brightness is probably not a texture defect at all.** Rendering the same asset under
brighter lights versus with a brightened texture reached the same overall brightness, but
the lit version kept the crevices dark while the lifted texture washed them out and
flattened the form. A de-lit bark albedo sitting near 0.12 mean is *correct*. Prefer
fixing the lighting; `lift_lightness.py` exists for when the lighting is not yours to
change.

## Not done yet

- **None of this is in the manifest.** These are loose scripts run by hand, so a finished
  asset is reproducible only by reading a chat transcript. See the proposed `material`
  block in the README.
- `feature_mask.py` has no `--from-render` mode; locating a feature is still manual.
- Emissive materials are untouched — the Monolith's glowing eye bar needs
  `emissiveTexture`, which nothing here writes yet.
- Geometry repair for carved-in markings is unbuilt.
