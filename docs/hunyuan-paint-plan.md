# Hunyuan3D paint lane — plan

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
