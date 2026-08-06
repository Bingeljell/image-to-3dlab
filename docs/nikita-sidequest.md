# Nikita Bier farewell model — session notes

A side quest: turn `assets_to_test/nikita_holding_beer.png` (Nikita Bier, T-pose,
holding a beer — the name is the joke) into a 3D model, then two videos: a looping
360 turntable and a "cheers" animation.

## Status

- **Turntable: done.** `output/video/nikita_turntable.mp4` — 4s, 1080x1080, seamless
  loop, built from the 512 model. Reusable helper: `scripts/blender_turntable.py`.
- **Model: not locked yet.** The face-vs-body tradeoff below is unresolved.
- **Rig / cheers animation: not started.**

## Runs so far

All TRELLIS.2, `material_mode: matte`, texture 2048, 200k bake target unless noted.
Generation is far faster on this subject than on the fox: 512 ~107s, 1024 ~310s.

| pipeline_type | seed | outcome |
|---|---|---|
| 512 (tex 1024) | 42 | clean, but soft face — the first draft/geometry gate |
| **512** | 42 | **clean sweater/mug/back; dead smeared eyes** |
| **1024** | 42 | **good face (real eyes, brows, nose); holes, skin showing through** |
| **1024** | **7** | **cleanest body/back of any run — no holes; dead hollow eyes** |
| 1024 | 123 | clean but washed-out grey skin; dead eyes |
| 1024_cascade | 42 | best eyes (visible irises); messy back, face-through-skull returns |

GLB hashes: 512 s42 `1f1c37ff0d95`, 1024 s42 `05062424f4c0`, **1024 s7 `480361379a16`**,
1024 s123 `e759c567512b`, cascade s42 `dcf005e81336`.

**Recommended hero: 1024 seed 7** (`480361379a16`). It resolves the holes, the back,
and the body texture, leaving exactly one defect (eyes) instead of several. The seed
hunt did not break the face-vs-body tradeoff — no variant wins on both.

Committed manifest: `manifests/nikita-showcase-trellis-512.json`.

## What the source image bought us

The T-pose input worked exactly as the fidelity/rigging plans predicted: **legs and
arms came out cleanly separated**, so the model is riggable. This is the lever that
failed on the sitting fox and the pangolin.

## Open defects (user-reported, in priority order)

1. **The beer mug has three handles.** Single-view hallucination: TRELLIS sees one
   handle and invents plausible ones on the unseen sides. This is the clearest
   evidence yet for the multi-view path — resolution will not fix it.
2. **Eyes and lips are botched** at 512; noticeably better at 1024.
3. **Holes in the back of the head** — you see the inside of his face through the skull.

## Diagnosis (the useful part)

Defects 2 and 3 share a root cause. A TRELLIS mesh is not a surface but a **shard
soup**: ~26k disconnected fragments, ~155k open boundary edges (measured after
`merge_vertices`; note glTF splits verts at UV seams, so raw `is_watertight` on a
round-tripped GLB is always False and means nothing).

Two things follow, both confirmed experimentally:

- **Voxel remesh does not rescue it.** The shards only fuse into one body at voxel
  ~0.012, which leaves 7k faces and destroys the face. At detail-preserving sizes
  (0.004) you get 258 disconnected components — and the largest one is just the legs.
  `scripts/blender_solidify_bake.py` implements remesh + component filter + albedo
  bake; it works mechanically but the quality is not there. Do not reach for it as-is.
- **Large regions of the mesh have inverted normals.** Shading backfaces near-black
  (to make gaps read as shadow) blackened the face, jeans, and mug instead — they are
  facing *inward*. This is probably why the interior is so visible, and is the most
  promising unexplored lead. `--darken-backfaces` on the turntable script is left in,
  opt-in and off by default, as a diagnostic.

## Plan when resuming

1. Pick the best face from the four 1024-family variants (renders land in
   `output/preview_textures/`, compare head crops side by side).
2. **Replace the mug.** It is a rigid prop welded to the hand — swapping it for a clean
   modelled mug kills the 3-handle bug outright. Best effort-to-payoff item.
3. Rig for the cheers. Only the mug arm needs to move, so bad weights have far less
   surface to show up on than a walk cycle. Bone-heat weighting will still fail on the
   shard mesh — expect to fall back to the rigging plan's weighting workaround.
4. **Multi-view input is the real fix** for the mug, the invented back, and face
   sharpness — the user can author front/back/profile views of the same T-pose at
   authoring time. Medium build (`run()` takes one image today). Belongs in a proper
   session, not a deadline sprint. See `docs/fidelity-plan.md`.

## Environment gotchas

- Use `.venv/bin/python`, not system `python3` (3.14, unsupported here).
- Blender must be running with the MCP addon on port 9876 for every render script.
- The preview renderer's camera labels do not match this model's axes: `profile_yneg`
  is the **front** view. The model's forward axis is -Y.
