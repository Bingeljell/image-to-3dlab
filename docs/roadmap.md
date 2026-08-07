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

4. **Normals are inverted, but DO NOT fix them yet — fix holes first.** Meshes are
   substantially inside-out (60% of area on one run) and this does matter for SceneKit
   and RealityKit, which cull backfaces. But applying *Recalculate Outside* to a mesh
   with 8,146 hole edges makes the backface-culled render **worse** — the ray heuristic
   needs a closed surface. `scripts/blender_fix_normals.py` exists but is a diagnostic
   until holes are closed. Meanwhile `doubleSided` materials are the correct
   mitigation. See [open-questions.md](open-questions.md) question 2.
5. **Fill holes post-export — WORKS, do it.** `scripts/fill_holes.py` takes the moss
   fox from 34,789 to 18,736 boundary edges, and in a backface-culled render the hind
   legs go from near-invisible wisps to solid limbs. Cheap and visible. Candidate for
   the pipeline. It cannot be complete — a third of the boundary is dangling ends
   rather than rims, and there are 24k non-manifold edges — so **decode-time repair**
   (on unsimplified geometry, where upstream does it) and **screened Poisson
   reconstruction** remain the routes to go further. See
   [open-questions.md](open-questions.md) question 1b.

Both are written up in [open-questions.md](open-questions.md).

**Superseded:** the former top item — "is the confetti topology inherent or our bug?" —
was withdrawn. There is no confetti; that measurement counted UV islands. See
question 1.

## Tier 3 — quality

**DECIDED (2026-08-07): texture fidelity is the lane.** Multi-view is done. Every session
until now worked on geometry while the visibly wrong thing was texture. The three defects
are now separated — hue, atlas fragmentation, and eye colour — see
[fidelity-plan.md](fidelity-plan.md).

**NEXT: colour-match the albedo.** The source concept leads red by +17; every render leads
green by 7-20. The shift sits in the baked albedo, so it is not a material or lighting
artefact and needs no regeneration. Cheapest remaining win by a wide margin.

6. **Multi-view input. NEXT UP.** With winding disproven, the
   see-through back of the head is best explained by the model never having seen the
   back. A hole-filler would stretch a flat membrane over the skull; multi-view gives it
   an actual back of a head. Also fixes the three-handled mug and soft faces, and
   should reduce hole count generally: the extractor drops quads where the voxel set is
   sparse, and sparseness follows uncertainty.

   **The build is smaller than "Medium" suggested.** `run()` already calls
   `get_cond([image], 512)` — the conditioning path is list-shaped throughout, and
   `image_feature_extractor.__call__` accepts a list of PIL images and returns
   `(B, N, D)`. Only `run()`'s signature is single-image.

   **The one real piece of work:** with B views the encoder returns a *batch* of B
   feature sets, while `sample_sparse_structure` builds `noise` of shape
   `(num_samples, ...)` and expects a single conditioning. The B view features must be
   concatenated along the **token** axis into `(1, B*N, D)` — one longer conditioning
   sequence — rather than left as a batch of B, which would ask for B separate objects.

   Constraint (from [fidelity-plan.md](fidelity-plan.md)): views must be the **same
   pose** from an orbiting camera. Different poses break reconstruction.

   **DONE 2026-08-07.** Token concatenation was the wrong technique; the fix is per-view
   denoiser prediction averaging, restored from TRELLIS v1 in
   `scripts/patch_trellis_multiview.py` and verified firing on all three samplers.
   Geometry measurably improves (components 516 -> 106, boundary edges 48,246 ->
   33,864) but appearance in a normal textured render barely changes — the payoff is
   downstream (engine culling, part splitting, collision), not turntable beauty. See
   [fidelity-plan.md](fidelity-plan.md).
7. **Second painted view** for far-side label quality. Downgraded from *required* to
   *nice* once nearest-neighbour fill reached 100% coverage from one view.
8. **Face resolution.** Faces are consistently weakest; likely a
   resolution-allocation problem. Test a head-crop pass against the full-frame run.
9. **Rigging beyond a known pose.** Analytic weighting works for an axis-aligned
   T-pose; the quadruped case still needs the guided marking in
   [rigging-plan.md](rigging-plan.md).

## Texture-fidelity backlog (parked 2026-08-07)

Deliberately parked to keep the critical path clear. **The first milestone is a rigged
quadruped doing funky animations**; these compete with it and are secondary. Manifests and
tooling for all of them already exist, so each is cheap to resume.

- **Face-count sweep, 50k and 20k.** `manifests/fox-34-multiview-50000faces.json` and
  `-20000faces.json` are committed and ready. Open question: fewer faces raises texel
  density and defragments the atlas, but below some threshold *geometric* detail — the
  tail's leaf relief, ear tufts, paw toes — starts dissolving. Where is the optimum?
  Prediction on record: 50k holds or improves on 101k; 20k visibly loses tail relief.
- **Colour-grade strength.** `--strength 1.0` is too mustard on the fox; the user judged
  the cream good and the green over-warm. 0.6-0.7 is the likely sweet spot. A refinement
  worth trying: grade the green foliage without pulling the cream, rather than scaling
  both uniformly.
- **Normal maps — probably the biggest untouched lever.** We bake albedo but no normal
  map. Baking a high-poly sculpt's detail into a normal map is *the* standard way to get
  high-poly surface detail onto a low-poly mesh, and would let the face budget drop
  further without losing perceived detail. Entirely unexplored here.
- **Head-crop generation pass** (roadmap item 8). The eyes are brown rather than amber
  *at generation*, so no UV, density or grading work will fix them.
- **Atlas padding / re-unwrap.** Islands are packed with no gutter, so filtering bleeds
  neighbouring colour across every seam. Only worth it if lower face counts prove
  insufficient for the flowers and ear foliage.
- **Provenance records only the first view path**, so a multi-view run is
  indistinguishable from a single-view one in the sidecar. A real gap in a repo built
  around provenance.
- **`pytest -q` as documented in `CLAUDE.md` fails collection**; it needs `PYTHONPATH=.`.
  Either fix the docs or add the setting to `pyproject.toml`.

## Also proposed

10. **Subject profiles** — one pipeline, a `subject.class` + `subject.features` block
    in the manifest supplying defaults and switching optional stages on. Motivated by
    measured findings: `512` won on the human and lost on the fox, hole counts differ
    7x by subject, and rigging strategy differs by body plan. See
    [subject-profiles.md](subject-profiles.md). Deliberately *not* parallel pipelines —
    generation, bake, provenance, labelling and export are identical across subjects.

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
- **Fewer faces is not a quality tradeoff here — it is a quality *win*.** Halving the
  budget (192k -> 101k) doubles texels per triangle (21.8 -> 41.4) and defragments the UV
  atlas: an eye that spanned two islands plus scattered fragments lands in one. Appearance
  is equal or slightly better at half the geometry. Do not reflexively raise face counts.
- **Read the flag's own help string.** `--bake-target-faces` said "Triangle budget for the
  CPU UV/texture fallback" and was still assumed to be general. It was inert on the Metal
  path — that is, on every Apple Silicon run — for the repo's entire history.
- **Never judge mesh repair by the centroid "inward area" metric.** It is meaningless
  for concave and off-centre geometry and has produced three wrong conclusions. Render
  with **backface culling** — the same test an engine applies.

## Why this order

Tier 2 is cheap and improves assets already on disk, which is why it comes before more
building. The former justification for this ordering — that the mesh was a shard soup
with no usable boundaries — turned out to be a measurement artefact (question 1), so
tier 1's geometry-splitting work is **less blocked than previously thought** and could
be attempted sooner.

Worth keeping in mind: vertex colours, stiffness, wind, and analytic rigging never
cared about topology at all, which is why the foliage lane ran to completion without
ever hitting the supposed wall. That should have been a hint the wall was not there.

**And the holes have not actually blocked a deliverable.** The foliage lane shipped,
the rig shipped, and every render looks fine because glTF materials are `doubleSided`,
which hides the problem entirely. The defects that genuinely cost output quality — the
three-handled mug, soft faces, the invented back of the head — are all single-view
hallucination. Hence multi-view first, and hole filling as a cleanup pass run once when
preparing an asset for an engine.

Note also that hole count tracks **thin geometry**, not just uncertainty: the human is
2.5% boundary edges, the fur-and-leaves fox 17.5%, on the same pipeline and settings.
Multi-view will help the solid parts far more than the fuzzy ones.
