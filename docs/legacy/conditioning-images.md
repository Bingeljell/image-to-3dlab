# Authoring the conditioning image

The single image you feed the pipeline decides most of the outcome. This is what to ask
an image model for, and why — every rule below traces to a measurement in
`docs/open-questions.md` §1c–§1d, not to taste.

> **The conditioning image is a different artifact from the game sprite.** Same
> character, same design language, a different render. Trying to make one image serve
> both jobs is where this goes wrong.

---

## 1. Hard requirements

Non-negotiable — the pipeline fails or silently degrades without them.

| Requirement | Why |
|---|---|
| **Transparent background, real alpha** | The TRELLIS backend refuses to run without the BRIA-disable patch (a licence guardrail we do not bypass), so background removal is unavailable. The alpha must come from the generator. |
| **Square canvas, subject centred** | Preprocessing crops to a square centred on the subject's alpha bounding box. |
| **Full body, nothing cropped** | Anything outside the frame is invented. |
| **Even soft lighting, no cast shadow, no ground plane** | Baked shadow becomes baked *geometry* and *texture*. |
| **Three-quarter front view** | Holds framing constant across characters so results are comparable. |

Crop overflow past the canvas edge is harmless — the crop pads with transparent, and
100% of subject pixels survive. Verify before a run rather than assuming.

---

## 2. Design rules, with the evidence

Measured on one code state, identical parameters, five subjects. "Hole size" is total
boundary-loop perimeter relative to the mesh diagonal.

| subject | surface | form | hole size |
|---|---|---|---|
| Snag (stone brute) | smooth | chunky | **0.48** |
| Flicker (ceramic) | smooth | **thin** | **1.07** |
| Monolith | **carved relief** | chunky | **44.78** |
| pangolin | **armour scales** | chunky | **97.82** |
| moss fox | **fur + moss** | thin | **126.58** |

### Rule 1 — surface detail is the dominant variable

Hold the surface smooth and change the form from chunky to thin: 0.48 → 1.07, barely
moves. Hold the form chunky and change the surface from smooth to carved: 0.48 → 44.78,
**93× worse**.

Ask for **large shapes rather than fine texture**. Fur, scales, carved relief and dense
pattern are all reconstructed as separate zero-thickness sheets.

This does not forbid detailed designs — a Solidify pass recovers them (§1d) — but it
predicts how much repair a subject needs, and how long it takes to generate. The
pangolin ran **3.4× slower per step** than the fox: TRELLIS is sparse, so detail means
more occupied voxels.

### Rule 2 — thin is fine, thin *and* fine-textured is not

Flicker has paper-thin ears, a tapering tail and slender limbs, and scored 1.07. Thin
geometry costs a little. It is the combination with surface detail that ruins a mesh.

### Rule 3 — negative space does not survive

A design whose identity is the gaps between its parts — a snarl, a lattice, a tangle —
is the hardest case, for two compounding reasons:

- Deep interior gaps are **unseeable from the conditioning view**, so they are invented.
- Solidify, the fix for everything else, **thickens sheets and therefore narrows gaps**.
  The repair works against the design.

Prefer "one fused mass" over "a loose tangle" when the silhouette allows it.

### Rule 4 — put important detail on the surface, not buried

The Monolith's glowing eye slot survived as geometry but lost its glow entirely, because
the recess was only partly visible from the conditioning view. Painting it back required
projecting texels through the same camera, which could only reach the part that camera
saw.

If a feature matters — an eye, an emblem, a gem — ask for it **on the surface, clearly
visible**, not recessed or half-hidden in shadow.

### Rule 5 — smooth, matte, unbroken

Say it explicitly. "Polished ceramic", "smooth matte stone", "surface smooth and
unbroken with no scales, no fur" all produced clean meshes. The negative phrasing does
real work.

---

## 3. Template

```
<one-sentence subject: what it is, its overall mass and stance>.
<surface: material, finish — say "smooth matte", say what it is NOT>.
<two or three large features, each described as a solid volume with a thickness>.
<colour: two or three, muted>.
<mood in one or two words>.
Full body, three-quarter front view, even soft studio lighting, no cast shadow,
no ground plane, no props. Transparent background, PNG with alpha, centred,
square canvas.
```

The last two lines are fixed. Everything above them is the character.

---

## 4. Pixel art is not a conditioning image

Pixel-art sprites are a legitimate output path and the repo keeps prompts for them. They
are a poor **input**:

- Dithering is high-frequency noise, which is the thing Rule 1 warns about.
- Quantised colour and hard aliased edges give the model no continuous shading to read
  depth from.
- It is circular: the 3D pipeline exists to supersede the sprites, so conditioning on a
  sprite feeds the old output back into the new input.

**Untested claim.** No pixel-art image has actually been run through this pipeline. The
reasoning above follows from what was measured on rendered inputs; it is not itself a
result. If it matters to a decision, it is one 6–16 minute run to settle.

---

## 5. Checklist before spending a run

1. Alpha present, and not fully opaque.
2. Square canvas; the subject's square crop fits or overflows only into transparency.
3. No matting fringe — check semi-transparent edge pixels for a colour cast. The stone
   Snag arrived with a green screen fringe on 95.9% of its edge pixels.
4. Surface reads smooth at a glance.
5. Nothing important is buried in a recess or behind another part.
6. Framing matches the other characters, so results stay comparable.

## High-contrast painted markings can become geometry

> **WITHDRAWN 2026-08-12.** The official TRELLIS.2 demo, given this exact artwork with its
> markings intact, produces a mesh whose culled grey render has **no forehead groove at
> all**. TRELLIS does not carve painted markings. The grooves and tearing were caused by a
> 200,000-face cap in our own `generate.py` that destroyed 94% of the decode before any
> real processing ran. See [self-inflicted-damage.md](self-inflicted-damage.md).
>
> The observation below — that the grooves survive stripping the texture — was real. The
> *attribution* to the input artwork was wrong. `soften_markings.py` and its `--protect`
> mask remain useful tools, but do not reach for them before checking that the face cap is
> lifted and the winding repaired.

**Verified on Flicker (2026-08-11).** Its crisp dark markings — the forehead V, the
shoulder chevrons — came out of TRELLIS as *physical cracks in the mesh*, with ragged
lips, and the eye rims were torn open. Rendering the asset with every texture stripped
and a plain grey material shows all of it still there, which is the decisive test: if a
defect survives losing its textures, no texture work will fix it.

The mechanism, **now supported by a regeneration test (2026-08-12)**: a hard dark line on
a light body reads as a shadow, a shadow implies a crease, so the generator carves one.
The irony is that the cleaner and more graphic the artwork's lines, the more likely this
is.

Two fixes:

1. **Soften the markings in the conditioning image** so they read as paint rather than
   shadow — `scripts/soften_markings.py`. Attacks the cause and improves every future
   asset. **Tested:** at `--lighten 0.5`, see-through holes across eight camera angles
   fell from 2.67% to 1.43% of body area.
2. **Repair the geometry afterwards** by smoothing only the creased vertices. Blanket
   smoothing would also soften eyelids and nose, which are meant to be sharp — but on a
   subject like Flicker the markings are dark on a light body, so a colour threshold
   isolates them cleanly and `scripts/feature_mask.py` can turn that into a face
   selection. Per-asset repair rather than a fix.

### Soften paint. Never soften shading.

The 0.5 run improved seven of eight angles and made the **dead-front view worse** —
holes 1.00% → 2.52% — with the whole regression in the ears.

Softening treats every dark region as flat paint. Flicker's forehead V *is* flat paint.
Its **ear interiors are not**: they are dark because they are a real hollow, and that
darkness is a depth cue. Lightening it told the generator the ear was flatter than it is,
so it built a thin membrane that tore.

We tried to separate the two automatically by local variance — flat paint should be
uniform, real shading should have a gradient. **It does not work.** Measured on Flicker,
ear interior local standard deviation is 24–47 and painted markings 37–43; the ranges
overlap, so no threshold separates them.

So the region has to be *named*, not inferred. `--protect <mask.png>` takes a greyscale
mask where white means "leave alone". Hand-painting it is entirely reasonable, and for
Flicker one is derived from the image itself (the ear regions are the large dark connected
components whose centre sits above the forehead marks) in
`assets_to_test/flicker-ear-protect-mask.png`.

**Apply the rule to any new subject:** before softening, look at every dark region and ask
whether it is *paint on a surface* or *a hollow you are looking into*. Protect the
hollows — ear cups, nostrils, open mouths, under-chin, deep-set eye sockets.
