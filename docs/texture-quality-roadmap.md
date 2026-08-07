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
