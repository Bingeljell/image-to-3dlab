# Output fidelity plan (closing the gap to Meshy)

Working notes on *why* a hosted service like Meshy.ai produces higher-detail
image→3D output than our current runs, and which of those levers we can actually
pull locally. Written as an eval before committing engineering time — nothing here
is built yet.

## The question

Meshy's models capture fine detail (e.g. the leaves and flowers in the moss fox's
fur) that our TRELLIS/SF3D runs smear or lose. Why, and what can we do about it
without a cloud GPU farm?

## Why hosted services look better (four stacked advantages)

1. **They synthesize texture; we bake it.** This is the biggest one. Meshy runs a
   *separate* texture-diffusion model that renders the bare mesh from many angles
   and paints each view with a 2D diffusion model trained on millions of assets,
   then projects it back. It **hallucinates** consistent new detail on the sides
   the camera never saw. Our current path takes the *single input photo* and
   projects it onto the surface — the front looks fine, but the unseen sides and
   fiddly foliage have no real pixels to draw from, so they smear.
2. **Cloud GPUs vs. Apple MPS.** H100-class hardware with tens of GB of VRAM lets
   them run high face counts and high texture resolution routinely. Our budget
   caps (50k-face bake default, modest texture size) are a direct consequence of
   running on unified memory.
3. **Bigger / proprietary models + more training data.** Pure capital; not
   matchable.
4. **Multi-view conditioning.** Their pipeline consumes or synthesizes several
   views for consistency. We feed a single image (see the multi-view section).

Points 2 and 4 are partly ours to close. Point 1 is the real texture gap and needs
a texture-synthesis stage (Hunyuan's paint lane), not just a bigger number.

## The multi-stage insight — we already have it, we just default it off

The user's instinct ("coarse shape → refine → texture, inspect before paying for
texture") is exactly how the good open models work, and **TRELLIS.2 already works
this way internally.** Two concrete findings from the vendored port:

- **We default to the lowest-quality mode.** `TrellisOptions.pipeline_type` is
  `"512"` (`image_to_3dlab/trellis_backend.py`). The port also supports `1024`,
  `1024_cascade`, and `1536_cascade` — the cascade modes are the high-fidelity
  path (coarse 512 structure → refine up to 1024/1536). This is **already plumbed
  end to end** (backend flag → `generate.py --pipeline-type` → pipeline `run()`).
  Trying it is nearly free. This is the first thing to test.
- **The geometry-preview gate is half-built.** `generate.py` already has a
  `--no-texture` flag that emits the bare mesh without the expensive bake. So
  "inspect geometry before spending on texture" is a small wiring job (expose the
  flag → render a turntable → gate the texture pass on approval), not new ML.

Other knobs already exposed: `texture_size`, `bake_target_faces`, `steps`, and
`max_num_tokens` (the sparse-structure token budget, a shape-detail lever).

## Multi-view: the eval the user asked for

**Verdict: high-value, genuinely supported by the architecture, but not a flag —
it's a Medium build with one honest catch.**

What the code shows:

- TRELLIS.2's public `run()` takes a single `image: Image.Image`.
- But one layer down, `get_cond()` already accepts `list[Image.Image]` — the
  conditioning backbone *can* fuse multiple views. The port simply never exposed a
  multi-image entry point (the original TRELLIS shipped a `run_multi_image`; this
  port dropped it).

So enabling multi-view = (a) patch the pipeline to expose a multi-image `run`
path, and (b) thread a list of images through our generator wrapper → backend →
CLI/manifest. That part is tractable.

**The catch — where do the extra views come from?** Multi-view input assumes you
*have* several consistent views of the object. Two cases:

- **Real object, real photos:** if the user shoots or has front/back/left/right of
  a physical thing, feed them directly. This is the ideal case and gives the
  biggest quality jump. Worth supporting first because it's pure upside.
- **AI concept art (the moss fox):** there's only one image, and it's imagined —
  no other "true" views exist. To go multi-view you must first *synthesize*
  consistent novel views with a multi-view diffusion model (Zero123++, MVDream, or
  Hunyuan3D's own MV variant), then feed those in. That adds a whole model to the
  pipeline and its own consistency failure modes (the synthesized back may not
  agree with the front). Medium–Large, and the quality ceiling is set by how good
  the view-synthesizer is.

**Recommendation:** support **real multi-view input** as a first-class path (clean
win, matches the "I don't mind 5–10 min for awesome quality" appetite), and treat
**synthetic multi-view for single concept images** as a later, separate experiment
once the real-input path proves out.

## The two-path structure (draft vs. showcase)

Maps cleanly onto manifests:

- **Draft path** — fast, for iteration/testing: SF3D or TRELLIS `512`, small
  texture, 50k faces. Seconds to a minute. All we need to check "is the pose
  riggable." Keep this as the default for the rigging lane.
- **Showcase path** — slow, for finals, budget ~5–10 min: TRELLIS `1024_cascade`
  (or `1536_cascade`) + higher `steps` + 50k–100k face bake + larger texture,
  **with the geometry-preview gate in the middle**, and optionally multi-view
  input when the user has real extra views. For texture-heavy organic subjects
  (foliage/fur), route to the **Hunyuan3D paint lane** — its dedicated multi-view
  PBR texture model is the closest open thing to Meshy's painter, and it's the
  real fix for the moss fox's leaves specifically.

## Lever list, ranked by fidelity-per-effort

| Lever | Fixes | Effort | Status in code |
|---|---|---|---|
| Switch `pipeline_type` to `1024_cascade` / `1536_cascade` | Shape + overall detail | **S** | Already plumbed; just change the default/manifest and confirm the higher models are present |
| Raise `bake_target_faces` (→ 50k–100k) + `texture_size` | Sharpness, foliage smear | **S** | Both already exposed |
| Raise `steps` / `max_num_tokens` | Shape cleanliness | **S** | `steps` exposed; `max_num_tokens` not yet surfaced |
| Geometry-preview gate (`--no-texture` → turntable → approve → bake) | "Inspect before paying" | **M** | `--no-texture` exists in `generate.py`; needs wiring + render + gate |
| Real multi-view input path | Consistency, unseen sides | **M** | `get_cond` accepts a list; `run()` must be patched to expose it |
| Hunyuan3D paint lane for organic subjects | Texture (leaves/flowers) | **M–L** | Our `--quality` ComfyUI lane; needs the paint workflow |
| Synthetic multi-view (Zero123++/MVDream) for single concept images | Consistency w/o real photos | **L** | New model + consistency risk |

## Recommended sequence (when we pick this up)

1. **Cheap spike first:** run the existing moss fox at `1024_cascade` + 100k faces
   + larger texture and eyeball the gain. This is ~free and tells us how much of
   the gap is "we defaulted to 512" vs. "we're missing texture synthesis."
2. If shape is now good but texture still lags → build the **Hunyuan paint lane**
   (real fix for the foliage) and/or the **geometry-preview gate**.
3. Add the **real multi-view input** path as a first-class manifest option.
4. Only then consider **synthetic multi-view** for single-image concept art.

## Notes

- 50k–100k faces is the agreed sweet spot — we don't need more.
- 5–10 min per showcase run is acceptable to the user; draft runs stay fast.
- None of this blocks the rigging lane; for rigging the draft path is sufficient
  (we only need separated limbs, not pretty fur). See `docs/rigging-plan.md`.
