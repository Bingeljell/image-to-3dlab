# Labelling pipeline: painted masks → part-aware meshes

How to give a generated mesh knowledge of its own parts, and what that unlocks.
Written after the moss-fox foliage session; the wind demo is the proof it works.

## The problem it solves

A generated mesh is **one undifferentiated blob**. Nothing in it says "these triangles
are the tail." That is not a bug in the export — the generator never had the concept.
TRELLIS predicts *which cells of a 3D grid are occupied*, fills each with a latent, and
decodes a surface. It works in occupancy-per-point-of-space. There is no object
hierarchy at any stage to lose.

This matters because **part information is the gate**. Foliage wind, cloth, hair, and
multi-material effects are all blocked by the same missing thing:

- You cannot simulate a cape you cannot select.
- You cannot assign a wind material to leaves you cannot identify.
- You cannot give a piece its own physics if it is welded into one mesh.

Multi-view input does **not** fix this. More views improve the *evidence about shape*;
they add no semantics. No number of photographs teaches the model what a leaf is.

## The method

Paint the labels in 2D and project them onto the mesh.

1. Paint flat colours over the **source image** — foliage green, body red, flowers blue.
2. Generate as normal. The generator is untouched.
3. For every vertex, work out where it appears in that image and **sample the colour
   painted there**.
4. Snap to the nearest label, fill what one view cannot see, export.

The projection mirrors TRELLIS's own preprocessing, which is simple and deterministic:
remove the background, take the subject's alpha bounding box, and **crop to a square
centred on it**. So the mesh's bounding square maps onto the image's subject square.
The view is treated as orthographic, which is close enough — validated below.

`scripts/project_labels.py`. Pure Python (numpy, trimesh, PIL); **no Blender**.

### Solving the view angle

A reference is rarely a dead-on view. The moss fox is a three-quarter view while the
mesh sits in canonical orientation, and projecting straight down an axis put vertices
off the subject — visible as ghost silhouettes offset from the ears.

`--auto-yaw` sweeps rotations and keeps the angle whose projected silhouette best
overlaps the image subject (intersection-over-union). On the moss fox it finds -30
degrees, IoU 0.82, lifting coverage from 48% to 70% of vertices.

**Consequence: draw your reference at any angle.** The tool works it out.

### Validate before painting

Project the **source photo itself** and render the result. If the projection is right,
the mesh comes out looking like its own front view — moss on the back, cream on the
chest. If it is wrong, colour smears across the wrong parts, which is exactly how the
yaw problem announced itself.

This costs no painting and catches every alignment error. Always do it first.

## Authoring a mask

Two separate flat PNGs: the source, and the paint alone on transparency.

| Rule | Why |
|---|---|
| **Opaque paint, no translucency** | The raw pixel colour *is* the label. Green at 50% over cream fur is a muddy blend that classifies ambiguously, and varies with whatever is underneath. |
| **Same canvas size and alignment** | The tool errors out on a mismatch rather than silently misaligning. |
| **Flat, saturated, distinct colours** | Anti-aliased edges get snapped to the nearest label; wildly different hues make that unambiguous. |
| **Leave the background transparent** | Unpainted means unlabelled. |
| **Over-cover rather than under-cover** | Overflow past the silhouette is free — no vertices project there. Gaps leave vertices unlabelled. |

Workflow: paint on a layer above the subject so you can see what you are doing, then
**hide the subject layer and export just the paint**. Precision is not required; the
boundary quality comes from the projection, not the brushwork.

## Three bugs worth not rediscovering

Each of these produces a *plausible-looking* wrong answer rather than an error.

1. **Deriving the crop from the mask.** The crop must come from the **source image**. A
   mask's own bounding box is defined by where paint happens to land, so overflow or a
   missed edge shifts the square crop and silently offsets *every* label. Hence
   `--source`. This is why painting can be rough.

2. **Unpainted pixels snapping to a random label.** A transparent mask pixel reads as
   black in RGB, and black is equidistant from red, green, and blue — so snapping
   picks whichever label is first. Gaps in the painting become *confident wrong
   answers*. Unpainted must mean unlabelled.

3. **Defaulting unlabelled vertices to "rigid".** The tempting default, and it tears
   geometry apart. One view cannot see the far side, so the far half of a swaying tail
   would freeze while the near half moved — the tail splits down the middle. Worse than
   not animating at all.

   Fix: `--fill` gives each unlabelled vertex the label of its **nearest labelled
   vertex in 3D**. A far-side tail vertex sits millimetres from a labelled tail vertex,
   so it correctly inherits foliage. This took the moss fox from 69% to **100%
   labelled from a single painted view**, which downgrades multi-view masks from a
   correctness requirement to a quality improvement.

## Stiffness is derived, not painted

Wind needs more than "is this foliage" — it needs **how floppy**, rigid at the base and
loose at the tips.

This does not need painting. It falls out of the labels: a foliage vertex's **distance
to the nearest body vertex**, normalised and eased. Zero where the tail meets the hip,
one at the tips.

So labelling is the *only* manual input. Everything downstream is derived.

## The wind itself

Foliage in a game is neither rigged nor simulated. It is moved by a **vertex shader** —
a wave applied per point, scaled by stiffness. Trunk rigid, leaf tips loose. No bones,
no solver, effectively free at runtime.

`scripts/blender_wind_demo.py` runs the same arithmetic locally so it can be rendered
without a game engine. Two waves at different rates plus a slow gust, all multiples of
the loop length so the last frame meets the first.

Result: `output/video/moss_fox_wind.mp4`. 71,899 of 256,712 vertices move; the body
stays planted.

## Known limitations

**The motion reads flatter than the geometry is.** Measured, the tail's thinnest
principal axis is 0.50 of its thickest — flattened, but not a cutout. So the flat
*feel* is the animation, not the mesh. The current wind **translates** every vertex
along one direction scaled by stiffness. That is a shear, not a bend: the tail slides
rather than pivoting about its base, so the tip traces no arc and the volume never
rotates to reveal its depth. Phase also varies smoothly with position, so neighbouring
leaves move in lockstep and read as one sheet.

Fixes, in order of expected payoff:
1. **Rotate about the anchor instead of translating.** Bending about the tail base
   produces an arc and preserves length, which is what reads as volume.
2. **Per-cluster random phase**, so fronds flutter independently instead of together.
3. **More motion along the view depth axis**, so the tail sweeps toward and away from
   camera rather than only across it.

**Labels inherit the confetti-mesh problem.** Interior shards poke through and never
get labelled cleanly. Harmless for a wind shader, which only needs the foliage set. It
will matter when splitting geometry into genuinely separate pieces with their own
materials, because the shards do not form clean boundaries. See
`docs/open-questions.md`.

**Resolution settings do not transfer between subjects.** This fox was generated at
`pipeline_type: 512` because 512 beat 1024 on the *Nikita* asset. That finding did not
transfer — earlier fox runs at `1024` and `1024_cascade` have crisper leaf relief. The
detail-versus-artefact tradeoff is **subject-dependent**; re-test per subject rather
than carrying a winner across.

## Still to build

1. **Split** labelled regions into separate mesh pieces
2. **Material slots** per piece, so an engine can attach a wind shader
3. **Export the stiffness** as vertex colours so a real engine can read it
4. **A second painted view** for far-side quality
