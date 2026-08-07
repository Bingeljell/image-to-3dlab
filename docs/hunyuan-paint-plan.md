# Hunyuan3D paint lane — BLOCKED on Apple Silicon

> **Outcome (2026-08-07): do not attempt this on Apple Silicon.** Hunyuan's paint stage
> depends on a custom CUDA rasteriser. The community Mac fork
> (`Brainkeys/Hunyuan3D-2.1-mac`) replaces it with a CPU software rasteriser and
> documents texture generation as *"Limited/Disabled"*, *"No CUDA Support: Custom
> rasterizer uses CPU fallback"*. **Shape generation works on MPS; texture does not.**
> Since texture was the entire point, the lane yields nothing here. Verified by reading
> the fork's `README_macOS.md` before installing anything — fifteen minutes instead of an
> afternoon of downloads.
>
> Revised options are at the end of this document.

**The question this lane answers:** how good can the *texture* get? Geometry is no longer
the weak spot — the 101k moss fox has a clean silhouette, a defragmented UV atlas and a
working rig. Texture is what still fails: flowers and ear foliage do not survive, fine
detail smears, and the hue needed a post-hoc correction.

## The experiment, stated precisely

**Do not regenerate the fox with Hunyuan.** Hunyuan3D 2.1 separates *shape generation*
from *texture painting*, and the paint stage can texture a mesh handed to it. So:

> Keep the existing TRELLIS geometry and run **only Hunyuan's paint stage** on it.

Two reasons this is the right shape for the experiment:

1. **It isolates the variable.** Regenerating shape too would confound "is Hunyuan's
   texture better?" with "is Hunyuan's geometry different?". We only care about the first.
2. **The rig is bound to this exact mesh.** New geometry means re-placing 27 markers and
   re-weighting. Painting the existing mesh keeps the rig intact.

Fallback if paint-on-arbitrary-mesh does not work: run the full Hunyuan pipeline and
compare textures on its own geometry, accepting the confound.

## The real risk: Apple Silicon

`ComfyUI-3D-Pack` and several Hunyuan3D wrapper nodes assume CUDA. Some of their
rasterisation dependencies have no MPS path at all — this is exactly the class of problem
the vendored `trellis-mac` port exists to solve for TRELLIS.

**So check viability before downloading weights.** Model downloads are several GB; the
import check is seconds. Order matters:

1. Install ComfyUI, confirm it starts on MPS.
2. Install the Hunyuan3D node pack.
3. **Import-check the nodes** — do they load without CUDA?
4. Only then download weights.

A "this needs CUDA" result at step 3 is a legitimate outcome and should be written up.
Dead ends are deliverables in this repo.

## Licence position

`provenance.py` classifies Hunyuan as `territory-restricted` and `validate_run_policy`
hard-refuses `use_case: "game"` + `distribution: "worldwide"` — outputs are not licensed
in the EU, UK or South Korea.

**This does not block the experiment.** A research manifest (`use_case: "showcase"`,
`distribution: "private"`) passes the gate cleanly. The restriction bites only at
publication, and the user's position is that this lane is establishing what is possible,
not producing shippable assets. Keep the manifests at `showcase`/`private` so the gate
stays meaningful.

## Workflow contract

`workflows/README.md` already specifies what the exported graph must contain: a
`LoadImage` node, the shape and paint stages, and a node that saves a `.glb`. The
`comfyui-backend` uploads an image, patches it into the `LoadImage` node, submits the
prompt, and pulls the resulting asset. Nothing in our code needs to change — only the
missing workflow JSON.

## Free experiments to run alongside

Both are unattended, both bear on the same question, both already have committed
manifests or proven scripts:

1. **Face-count sweep**, `manifests/fox-34-multiview-50000faces.json` and
   `-20000faces.json`. Prediction on record: 50k holds or improves on 101k; 20k visibly
   loses tail relief and ear tufts.
2. **`scripts/fill_holes.py`** on the hero asset. Proven — took an earlier fox from
   34,789 to 18,736 boundary edges. Directly targets the tearing visible around the ears
   and muzzle.

## Success criteria

Compare against `output/hero/moss_fox_hero_101k_grade07.glb`, same subject, same
geometry, judged on the standard textured render:

- Do the **flowers** survive? They do not today.
- Does the **ear foliage** read as foliage?
- Is the **hue** right without a post-hoc colour match?
- Are the **eyes** amber rather than dark and soft?

Those are the four failures recorded in `hero-asset-fidelity-bar`. Any of them fixed is a
real result.

## Revised options, now that Hunyuan paint is unavailable locally

Ranked by cost against the same goal — better texture, especially flowers, ear foliage
and eyes.

1. **Head-crop generation pass** (roadmap item 8), and it is *more* attractive now.
   Texture resolution is hard-capped at 2048 for the whole model, so the face gets a small
   share of a fixed budget. Generating the head as its own subject spends a full 2048 atlas
   on it. Open question: whether the result can be recombined with the body, or whether it
   only serves as a reference for what detail is achievable.
2. **Normal maps — still completely unexplored, and probably the biggest lever left.**
   We bake albedo only. Baking a high-poly sculpt's detail into a normal map is the
   standard way to get high-poly surface detail onto a low-poly mesh, and it would let the
   face budget drop further without losing perceived detail. Nothing in this repo has
   touched it.
3. **Fill the holes.** `scripts/fill_holes.py` is proven and unrun on this asset. The
   tearing visible around the ears and muzzle is geometry, not texture, and it damages how
   the texture reads.
4. **The face-count sweep.** 50k and 20k manifests are committed and unrun.
5. **Rent an NVIDIA box** if Hunyuan's texture quality is worth establishing. This is the
   only way to answer the original question, and it is a cost decision rather than a
   technical one. Worth doing only if the local levers above prove insufficient.

**Note the licence position is unchanged and does not block anything here** — the gate
fires only on `use_case: "game"` plus worldwide distribution, and research manifests pass.
The blocker is purely hardware.
