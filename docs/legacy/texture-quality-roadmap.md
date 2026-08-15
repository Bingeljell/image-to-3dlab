# Getting to a lovely-looking model

Written 2026-08-07, after the moss fox reached a rigged, trotting, breathing state with
texture as the remaining weak spot. The question: how do we get output that stands next
to Meshy?

## First, the thing that is *not* the answer

**Swapping generators will not get us there.** Current comparisons put TRELLIS.2-4B as the
best open-source model for PBR assets, Hunyuan3D 2.1 as best for texture, Hi3DGen as best
for geometry, and Stable Fast 3D as fastest. We are already on TRELLIS.2; the only model
that beats it on texture is Hunyuan, and its paint stage is
[hardware-blocked on Apple Silicon](hunyuan-paint-plan.md). SF3D is already vendored and
is a speed play, not a quality one.

So the local generator ceiling is roughly where we are. **The gap to Meshy is not a model
gap — it is a post-processing gap.** No studio ships a raw generator output either.

## The ceiling we are actually hitting

`texture_size` caps at **2048**, enforced by the vendored port. One material means one
2048 atlas for the entire animal — body, tail, head, ears, eyes. At 101k faces that is
~41 texels per triangle, and the face gets a small share of a fixed budget.

Every idea below is a way to either **raise that ceiling** or **stop needing it**.

## 1. Region splitting + per-region textures — the biggest lever

**The insight: 2048 is the cap *per material*, not per model.** Game characters normally
ship with separate texture sets — head, body, accessories — each with its own map. Split
the fox into three material slots and the texture budget triples without touching a single
model setting.

This is also what makes a **head-crop generation pass** worth doing. The point is not to
replace the model; it is to *source a high-detail face texture* that gets baked onto the
head region's own 2048 material. The face is where detail matters most and where we are
weakest (eyes dark and soft, ear foliage gone).

**Half of this already exists.** The foliage labelling lane —
[labelling-pipeline.md](labelling-pipeline.md) — takes a painted 2D mask, projects it onto
the mesh, snaps to labels, and fills the unseen side from nearest 3D neighbours. It reached
100% coverage from one painted view. That is exactly the segmentation this needs, and it
has been sitting unused since it shipped.

**Steps:** paint a region mask (head / body / tail) → project with `project_labels.py` →
split into material slots → generate a head-crop → bake its texture onto the head region →
export with three texture sets.

**Open question:** whether a separately generated head can be baked onto the existing head
geometry cleanly, or whether it only serves as a detail reference. Blender can bake
surface-to-surface by proximity, so this is testable rather than theoretical.

## 2. Normal maps — cheap, and completely untouched

We bake albedo only. Baking surface detail into a **normal map** makes the lighting engine
render bumps that do not exist in the geometry — it is *the* standard technique for making
a low-poly mesh read as high-poly.

Nothing in this repo has tried it. It buys perceived detail everywhere at once, it costs no
texture budget from the albedo, and it would let the face count drop further without losing
apparent detail. Highest payoff per hour of the four.

## 3. Texture refinement with depth-conditioned Stable Diffusion

This is the genuinely novel piece, and it is **Hunyuan's paint stage rebuilt from parts
that run on Apple Silicon**:

1. Render depth and normal passes from N viewpoints in Blender.
2. Run depth-conditioned Stable Diffusion (ControlNet) with the concept art as reference.
3. Project the results back onto the UV atlas.
4. Blend across views, resolving seams.

Published approaches doing exactly this: **SyncMVD**, **Text2Tex**, **Paint3D**, **TEXTure**.

**Why it works here when Hunyuan does not:** Hunyuan's blocker is a custom CUDA rasteriser.
This pipeline needs no such thing — Stable Diffusion runs well on MPS, and Blender does the
rendering and projection. Every component is already viable on this hardware.

Also the only approach on this list that can *invent* detail rather than redistribute it —
which is what the flowers and ear foliage actually need.

## 4. Quad remesh — topology, not texture

Meshy outputs clean quad topology; ours is triangle soup with 226 components. Blender ships
**QuadriFlow** built in. Quads unwrap better, deform better, and subdivide cleanly.

Worth doing, but note it invalidates the rig binding — the weights are bound to the current
mesh. Sequence it before a re-rig, not after.

## Recommended order

1. **Region splitting + head-crop.** The only lever that raises the actual texture ceiling
   rather than redistributing a fixed budget, and the segmentation half is already built.
2. **Normal maps.** Cheap, untouched, buys detail everywhere.
3. **SD texture pass.** The novel piece, and the closest local equivalent to what Meshy
   does. Do it once the region split means each pass has a full 2048 to work into.
4. **Quad remesh**, only alongside a planned re-rig.

## Carried over from the current backlog

Independent of the above, and cheap:

- Adopt `output/hero/moss_fox_hero_101k_filled.glb` — hole filling took boundary edges
  16,467 → 7,905, and the tearing around the ears and muzzle is what makes the texture read
  badly there. Needs a re-bind (markers unchanged, so two commands).
- Luminance-weight the colour grade so near-black pixels stop turning blue.
- The unrun 50k/20k face sweep.

## Standing caution

Every quality claim in this repo has needed checking against a render rather than a metric,
and several confident diagnoses were wrong. Judge each of these on the textured, lit render
next to the concept art — the standard set in [hero-asset-fidelity-bar]: do the flowers
survive, does the ear foliage read as foliage, is the hue right without correction, are the
eyes amber.

## Region splitting: built, measured, and a REGRESSION (2026-08-07)

`scripts/blender_split_regions.py` works mechanically — it splits the fox into head, body
and tail material slots, bakes a 2048 albedo per region, and exports a 3-primitive GLB.
The face counts land exactly where predicted:

    head   35,824    body   36,827    tail   28,647

**It makes density worse**, and getting to that answer required correcting the
measurement twice.

### Measure UV coverage, not pixel brightness

The first metric used was "fraction of atlas pixels brighter than near-black". That is
**wrong**, and it inflated every number by about 3x. The bake dilates colour outward from
each island by its margin to prevent seams, so a third of the canvas *looks* filled while
only a tenth carries triangles.

The correct metric is **summed UV triangle area** — add up the area of every triangle in
UV space. It measures coverage directly and cannot be fooled by bleed:

    area = sum over faces of |(b-a) x (c-a)| / 2   for UV coords a, b, c

### The real numbers

| | UV coverage | TRUE texels/face |
|---|---|---|
| original, one shared atlas | **53.0%** | **21.9** |
| head, own 2048 | 10.1% | 11.8 |
| body, own 2048 | 9.4% | 10.7 |
| tail, own 2048 | 11.6% | 17.0 |

So the split roughly **halves** real density. Three atlases provide 3x the raw budget and
the repacking gives back more than all of it.

### What was wrong, and how it was caught

- **"The original atlas is 97-99% filled"** — no, it is 53%. Brightness was measuring
  margin bleed.
- **"41 texels per triangle"**, quoted throughout earlier work — that is
  `2048^2 / faces`, which assumes the whole canvas carries triangles. The true figure for
  the 101k fox is **21.9**. Relative comparisons between runs are probably still sound
  because coverage was similar, but absolute figures were ~2x optimistic.
- **"Island count is the blocker"** — contradicted by the data: the tail has half the
  islands of the head and they are nearly twice as large, yet it packed no better. The
  user spotted this hole. The actual problem is that packing shrinks the islands instead
  of growing them.

Island counts, for the record: head 4,683 / body 4,630 / tail 2,221, against 11,340 for
the whole mesh — the split creates 194 extra islands where one straddled a boundary.

### Still worth trying: xatlas

The generator uses xatlas and achieved 53% coverage on this mesh. Per-region packing is a
strictly easier problem — fewer islands into the same canvas — so there is real headroom.
But the earlier claim that xatlas "demonstrably hits 97%" was based on the broken metric
and should not be repeated.

Second-order option if xatlas repacking also disappoints: reduce the island count itself.
11,340 islands for 101k faces is high, and fewer, larger islands would pack better under
any packer.

---

## The albedo is darkened and desaturated at bake (2026-08-11)

**Every lever in this document is about *sharpness*. The user's actual complaint was
*colour*, and no amount of texel budget addresses it.**

They gave two examples of the thorn-knot Snag against its source: *"the eye was brighter
and amber, more alive and menacing — the model texture is flatter and pale"*, and *"there
are nice mossy green patches on the tentacles in the image; it's just barky brown with
brown patches in the model."*

Measured — source subject pixels against the used texels of the baked 2048 map:

| | source | baked texture |
|---|---|---|
| mean brightness | 63.2 | **31.8** |
| saturation | 0.701 | **0.429** (−39%) |
| **peak brightness** | **252** | **129** |
| amber pixels | 7.60% | **0.00%** |

**Peak 129/255 is the finding.** The texture never gets brighter than mid-grey, so a
bright amber eye is not rendered dim — it is *unrepresentable*. Amber goes from 7.6% of
the source to exactly zero. The moss is the same effect: take 39% of the saturation off
olive-green and it becomes drab bark.

### What we got wrong on the way

The first diagnosis was **texel density**, and the numbers supported it: the thorn-knot
runs at **27 texels per triangle** (97,707 faces on one 2048 map, 63.1% of the atlas
used), against the moss fox's 41. That is a real problem and everything above still
applies to it — but it is a *sharpness* problem, and it would not have recovered the
amber eye or the green moss. Both of the user's examples survive at any resolution,
because the transform happens after the detail is placed.

The lesson is the one this repo keeps relearning: **a plausible cause that predicts the
right direction is not the cause.** "Texture looks worse than the source" is consistent
with both low resolution and a global grade, and only a measurement separates them.

### What follows

- It is a **global transform, so it should invert as a post-process** — no regeneration.
  `scripts/colour_match_albedo.py` exists for this, and `docs/roadmap.md` already records
  the moss fox's albedo running cool by +17 red, which looks like the same phenomenon at
  smaller magnitude.
- **If the transform is constant across subjects, it becomes a standard pipeline step**
  and every asset already generated improves at once. Worth checking on two or three.
- **Cause is not established, and should not be guessed.** Candidates: shading or ambient
  occlusion baked into the albedo, a colour-space mismatch at bake time, or a
  deliberately darkened unlit base that expects lighting on top. The measurement stands
  whichever it is.
- **Ordering changes.** Fix the grade first — minutes, no compute, affects everything.
  Then the sharpness levers in this document.
