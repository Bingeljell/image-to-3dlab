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
   The proper engine handoff: a piece an engine can attach a wind shader to. Warning —
   this runs straight into the confetti problem, because shards do not form clean
   boundaries. Partly blocked by tier 2.
3. **USDZ export.** Required, not optional, for the SwiftUI target: SceneKit and
   RealityKit do not load GLB natively. Blender's USD exporter or Reality Converter.

## Tier 2 — the research question blocking everything else

4. **Is the confetti topology inherent to TRELLIS, or a bug in our Mac port?**
   The single highest-value experiment in the repo. Every output measures ~26k
   disconnected components and ~155k open boundary edges, and that one fact causes the
   see-through holes, the interior speckle, and the rigging failures. If it is our
   port, fixing it improves *everything* downstream at once.
   **How to settle it:** run the same image through a reference TRELLIS (Colab/CUDA)
   and measure component count and boundary edges identically. Also worth auditing
   `vendor/trellis-mac/patches/` for anything touching mesh extraction.
5. **Inverted normals.** Large regions face inward — shading backfaces near-black
   blackened the face and jeans instead of the gaps. May account for much of the
   visible "interior". Cheap to test (per-component recalculate-outside), and if it
   works it improves every model already on disk with no regeneration.

Both are written up in [open-questions.md](open-questions.md).

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
- **A mesh-health diagnostic** (components, boundary edges, normal consistency) would
  have surfaced most of this session's problems immediately. Open design question:
  report only, or *gate* like `validate_run_policy` does for licensing.

## Why this order

Tiers 1 and 3 keep bumping into the same wall: the mesh is not a surface. Splitting
geometry, exporting clean parts, and simulating cloth all need boundaries that a shard
soup does not have. So it is worth finding out whether that foundation is fixable
(tier 2) before investing further in routing around it.

The exception is anything that does *not* care about topology — vertex colours,
stiffness, wind, and analytic rigging all work fine on a shard soup, which is why the
foliage lane got as far as it did.
