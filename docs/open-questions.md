# Open questions

Things this project has hit but not understood. Written after the Nikita side quest
(see `docs/nikita-sidequest.md`), which surfaced most of them in one afternoon.

Each entry: what we observed, what we think is going on, and what would settle it.
Ordered by how much answering them would unblock.

---

## 1. Why is a TRELLIS mesh "confetti" at all?

**Observed.** Every TRELLIS output measures the same way: ~26,000 disconnected pieces,
~155,000 open boundary edges, not watertight. It is not a surface, it is a soup of
overlapping shards that happens to *look* solid from outside.

**ELI5.** A proper 3D model is like a balloon: one continuous skin with a defined
inside and outside. Ours is more like a pile of leaves raked into the shape of a
person. From a distance you see a person. Up close there are gaps between the leaves,
and you can see the leaves on the far side through them.

**Why it matters.** This one fact caused most of our problems: the see-through holes,
the skin-coloured speckles, and the rigging failures on the fox.

**What we don't know.** Is this inherent to how TRELLIS decodes geometry (a mesh
extracted from a sparse voxel/latent grid, where each active cell contributes its own
patch), or is it an artefact of *our* vendored Mac port and its export path? Nobody
has looked at whether the reference implementation produces the same topology.

**How to settle it.** Run the same input through a reference TRELLIS (Colab/CUDA) and
measure component count and boundary edges the same way. If it is equally fragmented,
this is inherent and every consumer must plan around it. If it is not, we have a bug
in the port worth finding. This is the single highest-value experiment on this list.

---

## 2. Why are large regions of the mesh inside-out?

**Observed.** Shading backfacing polygons near-black — expecting gaps to darken —
instead blackened his face, jeans, and the mug. Those surfaces are facing *inward*.

**ELI5.** Every face of a 3D model has a front and a back, like a sheet of paper with
a printed side. The renderer needs the printed side pointing out. On big patches of
our model, the paper is in backwards.

**Why it matters.** This is the most promising unexplored lead, because it may be
*half the reason* the interior is so visible. If wrongly-facing patches are being
drawn over correctly-facing ones, some of what we called "holes" may not be holes at
all — just surfaces rendered from the wrong side.

**What we don't know.** Whether normals can be recovered. `trimesh` reported winding
as *consistent*, which suggests each shard is internally coherent but individual
shards disagree with each other about which way is out.

**How to settle it.** Per-connected-component "recalculate normals outside" (Blender
can do this, or `trimesh.repair.fix_normals`), then re-render. Cheap to try, and if it
works it may improve every model we have already generated — no regeneration needed.

---

## 3. Why did higher resolution make the interior *more* visible?

**Observed.** `pipeline_type: 1024` gave a clearly better face (real eyes, brows,
nose) but far more visible interior — speckles on the sweater, a face showing through
the back of the skull. `512` was cleaner but had dead, smeared eyes. The seed hunt
(4 variants) never broke this tradeoff.

**Hypothesis.** Higher resolution means smaller, thinner shards, so the gaps between
them are proportionally larger and easier to see through. Detail and watertightness
may be directly in tension.

**How to settle it.** Measure component count, boundary edges, and mean shard area at
512 vs 1024 vs cascade on the same seed. If shard size shrinks faster than gap size,
the hypothesis holds and "just use higher resolution" is not a free win.

---

## 4. Why didn't voxel remeshing fix it?

**Observed.** Voxel remesh should wrap a shard soup in one clean skin. It didn't. At
detail-preserving sizes (0.004) we got 258 disconnected components — and the largest
was *just the legs*. Only at 0.012 did it fuse into 3 bodies, by which point the mesh
was 7,000 faces and the face was destroyed.

**ELI5.** Voxel remeshing is "dip the model in wax and keep the wax shell." It works
if the wax is thick enough to bridge the gaps. Ours had to be so thick it filled in
his eyes and nose too.

**What we don't know.** Whether a *better tool* solves this. Blender's voxel remesh is
one option, but **screened Poisson surface reconstruction** (Open3D, PyMeshLab) is
built precisely for turning noisy, gappy point/shard data into a watertight surface
and may bridge the gaps without the same detail cost. We never tried it.

**How to settle it.** Sample points + normals from the shard soup, run Poisson
reconstruction at a few depths, compare face detail and watertightness against the
voxel results. Note this depends on question 2 — Poisson needs *correct* normals, so
inside-out patches would poison it.

---

## 5. Can single-view hallucination be fixed by authoring views?

**Observed.** The beer mug came out with **three handles**. TRELLIS saw one handle
from the front and invented plausible ones on the sides it could not see. Same root
cause as the invented back of the head.

**Why it matters.** This is the clearest evidence yet that resolution is not the
lever — no seed and no `pipeline_type` will fix a handle the model never saw. It is
the strongest argument for the multi-view path in `docs/fidelity-plan.md`.

**Known constraint** (from the fidelity work): views must be the *same pose* from an
orbiting camera. Different poses break reconstruction.

**Open sub-questions.** How many views are actually needed — is front+back enough, or
do we need 4? How consistent must they be before the model averages them into mush
instead of fusing them? And does multi-view also reduce the fragmentation in
question 1, or only fix the invented geometry?

---

## 6. Why is the face the weakest part, and is cropping the fix?

**Observed.** Faces are consistently the weakest region across every model we have
made. Eyes especially — usually dark hollow sockets.

**Hypothesis.** The face occupies a tiny fraction of the source image, so it gets a
tiny fraction of the model's representational budget. It is a resolution-allocation
problem, not a model-quality problem.

**How to settle it.** Generate from a head-only crop of the same image and compare the
face against the full-body run. If the cropped head is dramatically better, the fix is
a two-pass approach (body at full frame, head from a crop, combined) — and that is a
real feature, not a workaround.

---

## 7. Should the pipeline diagnose mesh health automatically?

Not a mystery — a proposal. Every problem above was invisible until measured, and we
only measured because things looked wrong.

A `mesh health` step could report, per run: connected components, boundary edges,
watertightness, normal consistency, face count. Cheap to compute, and it turns "the
back of his head looks weird" into "this mesh has 26,000 components." Would have
saved most of an afternoon.

Open question: should it just report, or should it *gate* — refuse to promote an
output that fails a threshold, the way `validate_run_policy` gates on licence?

---

## 8. When does analytic weighting generalise?

**Observed.** The rigging plan assumed hand-marked joints, because auto-placement
failed on the fox. But for the T-posed human we skipped marking entirely: landmarks
were measured straight off the mesh (shoulder 0.12, elbow 0.215, wrist 0.30), and
weights were assigned from vertex position rather than inferred from the mesh.

**Why it matters.** This completely sidestepped the confetti problem. No voxel remesh,
no heat diffusion, no weight transfer. Bone-heat weighting was never the obstacle for
a single-limb gesture.

**What we don't know.** Where the boundary is. It clearly works when the pose is known
and the limbs are axis-aligned (T-pose). It clearly fails for a quadruped in an
arbitrary pose. Does it extend to a full walk cycle on a human? To legs? The honest
answer is we only proved it for one arm on one figure.

**Related.** A held prop must be bound to its bone *before* any distance-based test,
not after — the mug is taller than the arm is thick, so a band test around the arm
clipped its top and bottom and stretched it into taffy. Probably a general rule for
props.
