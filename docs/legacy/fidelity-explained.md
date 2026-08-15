# Understanding character fidelity: base mesh, materials, and VFX

A teaching write-up from the session where we compared our image→3D output to hosted
services (Meshy) and to the game's approved art direction. Written for someone new
to 3D: concepts are explained from scratch, with a glossary at the end. Read this
before judging whether the pipeline "can make" a given concept — because the answer
almost always depends on *which layer* of the character you're looking at.

---

## 1. Where this started

We generated a mossy fox and it looked good from the front but the foliage smeared,
especially on the sides. Hosted services like Meshy captured that kind of detail
better. The question: **why, and can we close the gap locally on a Mac?**

That question turned out to have two very different answers depending on what
"detail" means — and the second answer reframes the whole project.

---

## 2. Why hosted services look better (the honest four)

1. **They synthesize texture; we (mostly) bake it.** More on this in §4.
2. **Cloud GPUs vs. Apple MPS.** They run datacenter GPUs with tens of GB of memory,
   so high polygon counts and high texture resolution are routine. We work within a
   unified-memory budget.
3. **Bigger / proprietary models, more training data.** Pure capital; not matchable.
4. **Multi-view conditioning.** They can consume or synthesize several views. We
   feed a single image.

Points 2 and 4 are partly ours to close; point 1 is the real texture gap. But none
of these is the *main* reason the hummingbird concept looks out of reach — that's §6.

---

## 3. The multi-stage pipeline, and what our own experiments showed

Good image→3D isn't one model. It's roughly: **coarse shape → refine shape →
synthesize texture.** TRELLIS.2 already works this way internally. Two things we
verified in code and by running them:

- **We had been defaulting to the lowest-quality mode.** `pipeline_type="512"`,
  while the port also supports `1024` and `1024_cascade`. Our showcase manifests
  were already maxed on texture (2048) and polygon budget (200k faces) — the *only*
  untapped lever was this pipeline resolution.

- **We ran the ablation.** Same fox, same everything, three modes:

  | Mode | Wall time | Result |
  |---|---|---|
  | Old 512 (early run) | — | Broken "confetti" blob (also a bad run, unfair baseline) |
  | plain `1024` | ~18.8 min | Smooth, polished, lovely amber eye, clean silhouette |
  | `1024_cascade` | ~19.5 min | Crisper individual-leaf relief, slightly busier/noisier |

  **The surprise:** plain 1024 and the cascade cost *almost the same time* (~40s
  apart). Going straight to 1024 shape sampling is nearly as expensive as
  512-then-refine. So there is **no cheap middle gear** — 1024 quality costs ~19 min
  either way, and the choice between them is purely aesthetic, not cost.

**Takeaway:** we already unlocked a real, visible quality jump (512 → 1024) for free
(the knob existed). But past that, both 1024 modes still show *garbled fine detail*
up close — which pointed us at the deeper question.

---

## 4. Why fine detail (leaves, scales) looks garbled — two separate causes

This is the crux. "Garbled leaves" is actually two independent problems:

### Cause A — the geometry-resolution ceiling
TRELLIS builds the entire creature as **one continuous surface** derived from a 3D
grid (think of it as sculpting from a block of fixed-size clay voxels). A real leaf
edge or flower petal is *thinner* than the smallest feature that grid can represent.
So the leaves are never modeled as leaves — they're a lumpy surface with leaf-shaped
bumps. 512→1024 shrank the clay grain (more detail), but 1024 is the finest grain
this port offers. **More of this knob won't turn blobs into individual leaves.**

### Cause B — single-view texture
The color painted onto that surface comes from your **one** input image, projected
on. The front looks right; anything the camera couldn't see (the back, the far side)
is smeared or invented. This is a *missing information* problem, not a resolution one.

Two different causes → two different fixes. Confusing them is the trap.

---

## 5. What multi-view input actually fixes (and what it doesn't)

Multi-view — feeding the model several views of the same subject — is real and
supported by the architecture (the conditioning layer already accepts a list of
images; the entry point just isn't wired yet). **Important constraint:** the views
must be the *same pose with the camera orbited* (front/back/left/right of one pose),
NOT different poses (a sitting view + a standing view would break reconstruction —
a moved limb reads as geometry existing in two places at once).

- **Multi-view fixes Cause B.** The back and sides get *real* information instead of
  guesses → consistent shape all the way around, no smeared far side.
- **Multi-view does NOT fix Cause A.** It doesn't raise the geometry-resolution
  ceiling. The leaves become more *correct and consistent*, but still lumpy-surface
  leaves, not crisp individual ones.

So the earlier excitement that "multi-view will fix the leaves" was half-right:
it fixes *consistency*, not *sharpness*. Worth doing — for the right reason.

---

## 6. The big reframe: a character is THREE layers, not one asset

Now the part that changes everything. Look at the approved hummingbird concept:
crystalline scaled body, glowing purple/blue veins, translucent crystal wings, and
wingtips/tail **dissolving into floating pixel cubes** (a glitch effect).

**No image→3D tool makes that image — Meshy wouldn't either — because it isn't one
thing. It's three layers of work stacked together:**

| What you see in the concept | Layer | Who makes it | Pipeline today? |
|---|---|---|---|
| Body + wing **shape** | **1 — Base mesh** | image→3D (this repo) | ✅ Yes |
| Fine **scales / facets** | 1b — Surface detail via **normal maps** | texture-synthesis stage | ⚠️ *Look* achievable (Hunyuan paint), not literal geometry |
| Glowing **veins** | **2 — Emissive material** | material/shader authoring | ❌ Not yet — but addable |
| Translucent **crystal wings** | **2 — Transmission material** | material/shader authoring | ❌ Not yet — but addable |
| The **glitch pixel-dissolve** | **3 — Runtime VFX** | game engine (shader + particles) | ❌ Never, by nature |

### Layer 1 — Base mesh + PBR textures
The shape and its base color/roughness/metalness maps. **This is all image→3D
produces**, and it's what our whole "512 vs 1024 vs paint lane" discussion is about.

### Layer 2 — Material properties
How the surface *responds to light*: does it glow (emissive), is it see-through
(transmission), is it iridescent? These are **not geometry and not base color** —
they're extra material channels. Image→3D bakes everything as opaque, non-glowing
color, so these are missing by default. But they can be **authored**, and some can
be **derived from the concept art** (see §7). We already dipped into Layer 2 with the
`pbr` material mode for the metallic pangolin.

### Layer 3 — Runtime VFX
The glitch dissolve is the clearest example. In a game engine it's a **dissolve
shader** (a noise pattern eats away the mesh edges, with a glowing rim at the
boundary) plus a **particle system** spawning little cubes that detach and drift. It
is **animated at runtime** — which is exactly why no static mesh can contain it, and
why you wouldn't *want* the reconstructor to try. The "cute" fallback hummingbird
*still has the glitch pixels* — because even there, it's a shader, not the mesh.

### The consequence
**The magic — glow, translucency, glitch — lives in Layers 2 and 3, authored in your
game engine, on top of a base mesh.** This is true for professional game art
universally. Meshy would hand you a nicely-textured *opaque, non-glowing,
non-dissolving* hummingbird; you'd build the glow and glitch yourself in
Unity/Unreal/Godot.

---

## 7. What this means for the project

**You do not have to "downgrade" to the low-poly cute version.** That version is a
*style choice*, not the pipeline's ceiling. The gap between the gorgeous concept and
the cute one is almost entirely Layers 2 + 3 (materials + VFX) — **not** base-mesh
quality. So:

- Want the detailed look → pipeline gives a detailed base mesh, you author glow/glitch
  in-engine.
- Want the stylized look → pipeline gives a lower-poly base mesh, *same* glow/glitch
  in-engine.

Either way, **the concept is achievable** — just not as one generate-and-done button.

### Where this repo can genuinely reach
Today the pipeline owns only Layer 1. Natural, in-spirit extensions that climb into
Layer 2 — and would move us toward these creatures automatically:

- **Emissive-map extraction:** pull the bright glowing regions *from the concept art*
  and bake them as an emissive channel on the GLB → the glowing veins, straight from
  the source image. This is the most novel and useful research direction here.
- **A `translucent` material mode:** flag a material as transmissive (glass-like) →
  the crystal/bubble look. Cheap, mirrors the existing `matte`/`pbr` modes.
- **Normal-map bake:** make existing geometry *read* as far more detailed under
  light without changing the mesh — the cheap half of the "scales/leaves" problem.
- **The glitch dissolve stays a documented in-engine recipe**, not a pipeline feature.

### The sharpened research question
> How far up the material stack (Layer 2) can an image→3D pipeline reach
> *automatically from the concept art* — before a human must author in-engine?

That's a genuinely novel question, and it's exactly where the glow/translucency of
this art direction lives. It, not multi-view, is the interesting frontier for these
particular characters.

---

## 8. Practical roadmap implied by all this

1. **Base mesh (Layer 1):** promote `1024` (plain or cascade — taste call, same cost)
   as the showcase default. For organic detail, the **Hunyuan paint lane** is the
   real texture upgrade (adds synthesized high-frequency detail + normal relief).
2. **Materials (Layer 2):** prototype emissive-map extraction and a translucent
   material mode. These directly target the glow/crystal look of the roster.
3. **VFX (Layer 3):** document engine-side recipes (dissolve shader + particles) —
   this is where the glitch effect belongs; the pipeline just feeds it a clean mesh.
4. **Multi-view:** worth building for *shape consistency* on unseen sides — not as a
   fix for fine-detail sharpness.

None of this blocks rigging; for rigging, a draft-quality base mesh is enough.

---

## Glossary (plain language)

- **Mesh** — the 3D shape itself: a surface made of many flat triangles (**polygons**).
  Their corner points are **vertices**. "Higher poly" = more triangles = more detail.
- **Topology** — *how* those triangles are arranged. Good topology deforms nicely when
  animated; TRELLIS meshes have messy topology (see the rigging notes).
- **Texture** — a 2D image wrapped onto the mesh's surface, like a sticker/skin.
- **UV / UV bake** — the process of unwrapping the 3D surface flat so a 2D texture can
  be painted onto it. "Baking" = computing and saving that texture image.
- **Albedo (base color)** — the plain surface color, with no lighting or shine baked in.
- **PBR (physically based rendering)** — a material system describing a surface by
  physical properties so it reacts to light realistically. Key channels:
  - **Metalness** — is it metal (0 = not, 1 = fully metallic)?
  - **Roughness** — how matte (1) vs. mirror-shiny (0)?
  - **Normal map** — a texture that fakes tiny bumps/relief by tilting how light hits
    the surface, *without* adding real geometry. This is how "scales" and "leaf
    detail" are faked cheaply.
  - **Emissive** — a channel that makes parts *glow* (emit light) — the veins.
  - **Transmission** — how see-through/glass-like the surface is — the crystal wings.
- **alphaMode** — how transparency is handled: `OPAQUE` (solid), `BLEND`
  (see-through). The TRELLIS "glass shards" bug was a wrong `BLEND` + metallic combo.
- **Shader** — a small program that decides how each pixel of a surface looks; where
  glow, translucency, and dissolve effects are actually implemented in an engine.
- **Dissolve shader** — a shader that uses a noise pattern to progressively hide
  (clip) parts of a mesh, usually with a glowing edge — the "disintegrating" look.
- **Particle system** — an engine feature that spawns many small objects (here, the
  floating pixel cubes) with their own motion — the drifting shards.
- **VFX** — visual effects; runtime, animated, engine-side. Layer 3 above.
- **Rig / armature / skinning** — the skeleton inside a mesh and the rules binding
  skin to bones so it can be posed/animated (covered in `rigging-plan.md`).
- **Feed-forward image→3D** — models like TRELLIS/SF3D that produce a 3D asset in one
  forward pass from an image, versus a human sculpting it.
