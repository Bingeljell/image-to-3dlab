# Roadmap

What to build next and why, ranked. Detail lives in the linked docs; this is the
ordering and the reasoning about what blocks what.

Last updated after the moss-fox foliage session.

## Tier 1 — finish what is started

Small, high payoff, and they complete stories already 90% done.

1. **Export stiffness as vertex colours.** Stiffness is currently derived at render
   time and thrown away. Baking it into the GLB is what turns this from a local demo
   into a pipeline *output* — SceneKit/RealityKit read it and run the same wave live.
   Small change. See [labelling-pipeline.md](labelling-pipeline.md).
2. **Split labelled regions into separate mesh pieces with their own material slots.**
   The proper engine handoff: a piece an engine can attach a wind shader to. Previously
   flagged as blocked by a "confetti" mesh with no clean boundaries — that was a
   measurement artefact, and the geometry is a connected surface, so this should be
   more tractable than assumed.
3. **USDZ export.** Required, not optional, for the SwiftUI target: SceneKit and
   RealityKit do not load GLB natively. Blender's USD exporter or Reality Converter.

## Tier 2 — the research question blocking everything else

4. **Fix the inconsistent winding.** *(Now the top candidate — see
   [open-questions.md](open-questions.md) question 2.)* `mesh_health.py` measures
   `winding_consistent: False`. With the shard-soup theory withdrawn, wrongly-facing
   surfaces are the leading explanation for the see-through interior and the blackface
   render. `trimesh.repair.fix_winding` + `fix_normals` on a connected surface, then
   re-render. Cheap, and it would improve **every model already on disk** with no
   regeneration.
5. **Fill the remaining holes.** Our port disables upstream's `fill_holes()` (question
   1b), and `cumesh` is not installed at all, so re-enabling the call alone will not
   work — it needs a Python/trimesh implementation. Far smaller in scale than first
   believed: 4,786 boundary edges on the Nikita hero, not 155,000.

Both are written up in [open-questions.md](open-questions.md).

**Superseded:** the former top item — "is the confetti topology inherent or our bug?" —
was withdrawn. There is no confetti; that measurement counted UV islands. See
question 1.

## Tier 3 — quality

6. **Multi-view input.** The real fix for single-view hallucination — the three-handled
   mug, invented backs, soft faces. Medium build: `run()` takes one image today.
   See [fidelity-plan.md](fidelity-plan.md).
7. **Second painted view** for far-side label quality. Downgraded from *required* to
   *nice* once nearest-neighbour fill reached 100% coverage from one view.
8. **Face resolution.** Faces are consistently weakest; likely a
   resolution-allocation problem. Test a head-crop pass against the full-frame run.
9. **Rigging beyond a known pose.** Analytic weighting works for an axis-aligned
   T-pose; the quadruped case still needs the guided marking in
   [rigging-plan.md](rigging-plan.md).

## Standing practice

- **Re-test resolution settings per subject.** `512` beat `1024` on the Nikita human
  (higher resolution exposed the shell interior) but loses badly on the moss fox,
  where cascade gives crisp leaves. The detail-versus-artefact tradeoff does not
  transfer between subjects.
- **`scripts/mesh_health.py` is the shared diagnostic.** Use it rather than ad-hoc
  measurement — it merges by position only, which is precisely the trap that produced a
  wrong diagnosis for several sessions. Open design question: should it *gate* a run,
  the way `validate_run_policy` gates on licensing, rather than only report?
- **Verify a tool does what your own caveat says.** The UV-seam warning was written
  down, then `trimesh.merge_vertices` was trusted to honour it. It does not.

## Why this order

Tier 2 is cheap and improves assets already on disk, which is why it comes before more
building. The former justification for this ordering — that the mesh was a shard soup
with no usable boundaries — turned out to be a measurement artefact (question 1), so
tier 1's geometry-splitting work is **less blocked than previously thought** and could
be attempted sooner.

Worth keeping in mind: vertex colours, stiffness, wind, and analytic rigging never
cared about topology at all, which is why the foliage lane ran to completion without
ever hitting the supposed wall. That should have been a hint the wall was not there.
