# Rigging & animation plan (the "animatable" lane)

Working notes for turning a generated 3D model into a rigged, animatable character.
Written after the first exploration session; pick up here next time.

## Goal

Take a TRELLIS/SF3D output (a static 3D model) and make it **animatable** — give it a
skeleton and make it walk — as a lane separate from the existing showcase/render track.

## Where we got to (status)

- We produced a **basic walk cycle** on the A-pose fox (gif + mp4). It works and loops,
  but the deformation is rough — legs move like stiff poles and unrelated parts (the face)
  move with the legs. This is a *known, diagnosed* problem, not a mystery. See below.
- The rigging code so far lives in scratchpad (exploratory). The committed repo artifacts
  are the A-pose source image and the run manifests.

## What we learned (the diagnosis)

Three findings, in order of discovery:

1. **Fused limbs can't be rigged.** The sitting fox and the pangolin came out as one blob
   with legs merged into the body. There's no separate leg geometry to attach a bone to.
2. **The source image pose is the biggest lever.** Regenerating the fox from an **A-pose**
   reference (standing, legs clearly apart, tail extended) produced geometry with
   **separated legs and tail** — riggable. Same model, different input pose → different
   result. This was the breakthrough.
3. **The mesh is "confetti."** A TRELLIS mesh is tens of thousands of tiny disconnected
   fragments. Blender's normal automatic weighting ("bone heat") needs a *connected*
   surface, so it failed completely and assigned **zero** influence — the skeleton moved
   but the skin didn't. We worked around it with crude distance-based ("proximity")
   weights, which is why the current walk looks stiff/wrong.

## Why the current walk looks "off" (two root causes)

- **Wrong joints.** Bones were auto-placed from noisy geometry and mislabeled (couldn't tell
  front-left from back-right). A bone's line runs through places it shouldn't, so it drags
  the wrong parts.
- **Dumb weights.** Proximity weighting binds each skin point to its *physically nearest*
  bone by straight-line distance. It doesn't understand the body, so a leg bone grabs the
  nearby face. → the "pole from ear to paw."

Both are fixable. The plan below fixes both.

## ELI5: what "weighting" (skinning) is

The model is a **jelly animal**; the bones are **sticks pushed inside it**. *Weighting* is
the rule for **how much jelly each stick pulls when it moves**. Good weighting = the knee
skin is smoothly shared between thigh and shin, so it bends naturally. Bad weighting = the
wrong stick grabs the wrong jelly (our leg-grabs-face bug).

**The proper fix — "heat" weighting:** imagine each stick glows warm and the heat spreads
*along the body's surface*; a leg stick only warms the leg, not the face across the gap.
But heat needs a *connected* surface, and our mesh is confetti. So: temporarily **"dip the
fox in wax"** to fill the cracks into one solid surface (*voxel remesh*), compute clean heat
weights on that, then **copy the weights back** onto the original detailed fox. Clean
weights + original detail.

## The decided approach

- **Joint marking: done by hand in Blender first** (the user wants to learn Blender doing
  it). A click-to-mark **HTML tool** is a **future QoL feature** — useful for this repo and
  for others who want to rig gen-models remotely — but not the starting point.
- **Weighting: the full fix** (voxel-remesh → heat → transfer back), because the lighter
  falloff fix won't remove the face-moving artifact.

## How the Blender-guided marking works (data flow)

The agent (Claude) sees rendered images (it opens and looks at every PNG) but **not** the
live Blender viewport. So the loop is:

1. Agent loads the fox into Blender in a known orientation and spawns **pre-named marker
   dots** ("empties"), one per joint, parked at the origin — nothing to create or name.
2. User drags each dot onto its joint in the 3D view (orbiting for depth), guided by a
   checklist. ~14 markers.
3. Agent reads every marker's XYZ back over the socket.
4. Agent renders a snapshot with the markers shown as dots, **looks at it**, and flags any
   that landed wrong ("nudge the front-left knee down"). Loop until right.
5. Agent builds the skeleton from those exact positions, applies the heat-transfer weights,
   and re-runs the walk.

## Joint template (quadruped, ~14 markers)

- Body: `pelvis` (hips center), `chest` (shoulders center)
- Head: `neck_base`, `head`
- Tail: `tail_base`, `tail_tip`
- Each leg (×4: front-left, front-right, back-left, back-right): `<leg>_top` (hip/shoulder),
  `<leg>_knee`, `<leg>_paw`

Optional shortcut: for a symmetric character, mark only the near-side legs and mirror the
far side. Marking all four by orbiting is more accurate and better for learning.

## Effort (T-shirt sizes)

| Task | Size |
|---|---|
| Joint marking — Blender, guided (chosen) | Medium (mostly agent scripting + ~15 min user) |
| Rig builder (joints → skeleton) | Small–Medium |
| Weighting fix — full (voxel-remesh heat transfer) | Medium (some headless-iteration risk) |
| Re-run the walk | Small |
| HTML click-tool (deferred QoL) | Large |

Path to a genuinely good walking fox: **Medium total**, ~one focused session (+ maybe a
little weight polish).

## Next session — start here

1. Confirm Blender is up with the MCP server on port 9876 and the A-pose fox GLB handy
   (`output/conditional/moss-fox-apose__trellis2__*.glb`).
2. Agent loads the fox + spawns the named joint markers.
3. Walk the user through placing the ~14 markers (doubles as a first Blender lesson:
   orbiting, selecting, grab/move).
4. Read back, verify with a marker snapshot render, correct.
5. Build skeleton → voxel-remesh heat weights → transfer → re-run the walk → export gif/mp4.

## Also queued

The user is new to 3D and wants a guided walkthrough of Blender + 3D fundamentals
(vertices, polygons, meshes, topology) and a recap of the bugs we solved here. Weave that
teaching into the hands-on marking session.
