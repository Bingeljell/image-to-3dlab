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

**The critical constraint — same pose, orbiting camera.** Multi-view
reconstruction assumes every view is the *identical frozen object* with only the
camera moved (front / back / left / right of ONE pose). It fuses views by assuming
a body point sits at the same 3D location in all of them.

- ✅ Works: "the standing A-pose fox seen from back / left / right." Same instant,
  rotated camera.
- ❌ Breaks: a sitting view + a standing view. A moved limb reads as geometry in
  two places at once; the reconstructor fights itself and blurs/duplicates the
  mesh. Different *poses* are great design references but cannot be fused into one
  reconstruction.

**Where the extra views come from (updated — the user can generate them):** the
user makes the concept art themselves and can generate N views at authoring time.
So we are *not* blocked on a separate view-synthesis model (Zero123++/MVDream)
after the fact. The requirement is just that they generate **one chosen pose,
orbited** — and the A-pose is the ideal pick because it's also the riggable pose,
folding the fidelity win and the rigging win into one image set.

- **User-generated multi-view (primary path now):** generate 3–4 consistent orbits
  of one pose up front, feed them in. Medium build (expose a multi-image `run`
  path + thread a list through generator/backend/CLI/manifest).
- **Real photos of a physical object:** same code path; pure upside when available.
- **Remaining risk = consistency.** A 2D generator inventing the *back* may not
  match the front (leaf placement, tail curl). Tools with a reference/consistency
  mode usually hold identity well enough; TRELLIS multi-image also tolerates *some*
  disagreement by treating extra views as soft guidance. Minor mismatch → fine; a
  back that's clearly a different fox → degrades. **Test front+back first** before
  investing in a full 4-view set.

**Recommendation:** build the **multi-image input path** as first-class (manifest
takes a list of view images), and have the user generate a same-pose orbit set of
the A-pose fox. This is now a near-term experiment, not a "later, separate" one.

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
| Multi-view input (user generates a same-pose orbit set) | Consistency, unseen sides | **M** | `get_cond` accepts a list; `run()` must be patched to expose it. User can author the views → no separate synthesis model needed |
| Hunyuan3D paint lane for organic subjects | Texture (leaves/flowers) | **M–L** | Our `--quality` ComfyUI lane; needs the paint workflow |
| Synthetic multi-view (Zero123++/MVDream) for a single existing image | Consistency w/o authoring extra views | **L** | Fallback only — not needed while the user can generate views up front |

## Recommended sequence (when we pick this up)

1. **Cheap spike first:** run the existing moss fox at `1024_cascade` + 100k faces
   + larger texture and eyeball the gain. This is ~free and tells us how much of
   the gap is "we defaulted to 512" vs. "we're missing texture synthesis."
2. If shape is now good but texture still lags → build the **Hunyuan paint lane**
   (real fix for the foliage) and/or the **geometry-preview gate**.
3. Add the **multi-image input** path as a first-class manifest option, and have
   the user generate a **same-pose orbit set** of the A-pose fox (test front+back
   before a full 4-view set). This doubles as riggable geometry.
4. Only if working from a single pre-existing image with no way to author views,
   consider **synthetic multi-view** (Zero123++/MVDream) as a fallback.

## Notes

- 50k–100k faces is the agreed sweet spot — we don't need more.
- 5–10 min per showcase run is acceptable to the user; draft runs stay fast.
- None of this blocks the rigging lane; for rigging the draft path is sufficient
  (we only need separated limbs, not pretty fur). See `docs/rigging-plan.md`.

---

## Multi-view result (2026-08-06): partial, and the failure is diagnostic

Ran a clean A/B on the same moss fox source, same seed and settings, only view count
differing. `scripts/patch_trellis_multiview.py` concatenates the per-view feature
tokens into one conditioning sequence.

**Outcome: the back is markedly better, the front is wrecked.** The back of the head
went from a hollow, eaten shell to a solid head; the face fell apart.

**Why.** TRELLIS.2 has **no multi-image support**: only `run()`, no camera or view
embeddings anywhere in the conditioning path (every camera reference lives in the
renderers). Token concatenation therefore hands the model a bag of tokens with no way
to tell which view each came from, so it must reconcile them as a single observation.
Front and back are the *maximally contradictory* pair — a face and a back-of-skull
share almost no content — so the averaging damages the region where they disagree
most. Which is exactly what we see.

Cost: 1731s for two views against 848s for one, roughly linear in view count.

Measurements moved the wrong way and should not be read as a verdict on their own:
boundary edges 21,630 -> 47,385, components 84 -> 449 (though non-manifold edges fell
2,991 -> 448). The visual is the finding.

**Two ways forward:**

1. **Overlapping views instead of opposed ones.** Two 3/4 views share content, so
   confused averaging is far less destructive. Predicts a better front at some cost to
   the back improvement — a sweet spot nearer 90-120 degrees apart than 180. Cheap to
   test, needs no code change.
2. **Stop concatenating tokens (the real fix).** Run the denoiser with each view's
   conditioning separately at every step and combine the predictions — averaged, or one
   picked at random per step. That treats each view as its own observation rather than
   mashing them into one, so maximally-separated views stop hurting. Inference-time
   only, no retraining. TRELLIS v1 shipped a multi-image path with fusion modes of this
   kind; v2 dropped it.

Note the generated mesh's canonical orientation flipped 180 degrees relative to the
single-view run, so like-for-like renders need the azimuth adjusting.

### Pending experiment: separate the code fix from the view choice

Two things changed between the failed front/back run and the 3/4 run: the fusion was
corrected (token concatenation -> per-view prediction averaging) **and** the view angles
changed. A good 3/4 result therefore cannot be attributed to either alone.

**The experiment:** re-run the original straight-on front/back pair
(`moss-fox-mv-front.png` + `moss-fox-mv-back.png`) through the corrected fusion, same
seed and settings. Compare against both the broken-fusion front/back run
(`425b70769bd7`) and the 3/4 result.

**What each outcome would mean:**

- *Front/back improves a lot and matches 3/4* — the fusion was the whole story, and
  view geometry matters less than we think.
- *Front/back improves but stays behind 3/4* — both mattered, and the expected result:
  the fix was necessary but straight-on views are genuinely worse for this subject.
- *Front/back stays broken* — the fusion fix is not sufficient, and something else is
  wrong.

**Why straight-on views are suspected regardless of fusion:** a frontal view is
**depth-ambiguous** for anything extending along the view axis. A bushy tail seen head
on appears as two masses flanking the body, with nothing in the image saying whether
they sit behind the fox, beside it, or attached to its chest. The model has no depth
channel, so it guesses. A 3/4 view shows the attachment point and the length in the
same frame. This is the user's observation and it is a better argument than the
original one about overlapping views contradicting each other less -- that reasoning
was specific to the broken token-concatenation fusion and mostly evaporates now that
views are kept separate.

Low urgency, since we already have reason to think straight-on is a poor pair for a
bushy-tailed quadruped. Worth running to close the loop honestly: we hypothesised,
were wrong about the mechanism, fixed it, and should record whether the fix alone was
enough.
