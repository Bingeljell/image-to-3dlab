# Open questions

Things this project has hit but not understood. Written after the Nikita side quest
(see `docs/nikita-sidequest.md`), which surfaced most of them in one afternoon.

Each entry: what we observed, what we think is going on, and what would settle it.
Ordered by how much answering them would unblock.

---

## 1. ~~Why is a TRELLIS mesh "confetti" at all?~~ — WITHDRAWN, the premise was false

**This question was based on a measurement error. Corrected 2026-08-06.**

The claim was that every output is ~26,000 disconnected pieces with ~155,000 open
boundary edges — a shard soup rather than a surface. It is not.

**The trap.** glTF splits a vertex at every **UV seam**, and the texture bake produces
~17,851 xatlas charts, so there are tens of thousands of seams.
**`trimesh.merge_vertices()` does not merge vertices whose UVs differ**, so it leaves
every seam split in place. Counting connected components after it measures *UV
islands*, not geometry.

Always merge by **position only** before measuring topology:

```python
q = np.round(v / (scale * 1e-6)).astype(np.int64)
_, inv = np.unique(q, axis=0, return_inverse=True)   # then reindex faces by inv
```

**Corrected measurements:**

| mesh | claimed components | actual | largest holds | boundary edges |
|---|---|---|---|---|
| Nikita s7 (the rigged one) | ~26,000 | **1** | 100% | 4,786 |
| moss fox cascade | 21,247 | 171 | 99.2% | 34,789 |
| moss fox A-pose (old) | — | 63 | 99.3% | 45,609 |

The meshes are **essentially single connected surfaces** with real holes and a little
debris. `scripts/mesh_health.py` measures this correctly.

**What this invalidates:** the shard-soup framing throughout these docs; the claim that
bone-heat weighting could not work for lack of a connected surface; and the reasoning
behind question 4 below.

**What survives:** the holes are real, and **winding is inconsistent** (measured, not
inferred) — see question 2, which is now the leading explanation for the see-through
interior.

**The lesson worth keeping:** a measurement artefact produced a confident, coherent,
wrong diagnosis that survived several sessions and shaped real design decisions. The
warning about UV seams was even written down — and then the tool was trusted to honour
it without checking. Verify that a tool does what your caveat says before building on
its numbers.

---

## 1b. Our Mac port disables upstream's mesh repair

*(Separate finding, surfaced while investigating question 1. Still live — the holes
are real even though the shard-soup framing was not.)*

Two findings from auditing `vendor/trellis-mac/patches/mps_compat.py`:

1. **`patch_mesh_base()` unconditionally disables `fill_holes()`.** Upstream TRELLIS.2
   calls it during decode; our port returns immediately, because the Metal build of
   `cumesh` segfaults on the full ~400K-vertex decode mesh. `remove_faces()` and
   `simplify()` are skipped for the same reason. Confirmed live in the installed
   source at `TRELLIS.2/trellis2/representations/mesh/base.py:43`.

2. **The extraction itself drops geometry at every boundary.**
   `install_mesh_extract()` replaces the CUDA `o_voxel.convert` with a pure-Python
   dual-grid extractor (`backends/mesh_extract.py`). Each intersected edge becomes a
   quad spanning **four** neighbouring voxels, and the quad is emitted only if all
   four exist:

   ```python
   connected_voxel_valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
   ```

   Wherever the sparse active voxel set has a boundary, the quad is silently dropped.
   Holes are therefore *expected* output of this stage — which is precisely why
   upstream repairs them afterwards.

So the causal chain is: extraction leaves holes by design → upstream fills them →
**we skip the fill** → holes survive into the output. This is a deliberate, documented
workaround whose downstream cost was never measured. Note the scale is far smaller than
first believed (question 1): 4,786 boundary edges on the Nikita hero, not 155,000. Also
note `cumesh` is not installed at all, so the guarded import always took the fallback
path — re-enabling the call alone would not work.

**Still to prove.** Re-enable the repair and measure. `cumesh` segfaults on Metal at
decode size, so the options are: run `fill_holes` on CPU, run it post-decode after
simplification (when the mesh is ~200K rather than ~400K), or implement hole-filling
in Python/trimesh. Any of these is a **local** experiment — no CUDA needed. If
component count and boundary edges collapse, this question is closed.

---

## 2. ~~Why are large regions of the mesh inside-out?~~ — DISPROVEN by visual test

**Observed.** Shading backfacing polygons near-black — expecting gaps to darken —
instead blackened his face, jeans, and the mug. Those surfaces are facing *inward*.

**ELI5.** Every face of a 3D model has a front and a back, like a sheet of paper with
a printed side. The renderer needs the printed side pointing out. On big patches of
our model, the paper is in backwards.

**Why it matters.** This is the most promising unexplored lead, because it may be
*half the reason* the interior is so visible. If wrongly-facing patches are being
drawn over correctly-facing ones, some of what we called "holes" may not be holes at
all — just surfaces rendered from the wrong side.

**TESTED AND DISPROVEN (2026-08-06).** Winding is measurably imperfect — 2.13% of
edges disagree — but it is **not** the cause of the visible artefacts.

The test: render the artefact-heavy Nikita mesh (`05062424f4c0`) from behind, apply
Blender's *Recalculate Outside* (`mesh.normals_make_consistent`), render again.
**The two renders are identical.** `trimesh.repair.fix_winding`/`fix_normals` likewise
moved the inward-facing area not at all (60.0% before, 60.0% after — trimesh bails on
meshes with this many boundaries).

**What the render actually shows.** Close up from behind, the artefact is his *face*,
seen through a large opening where the back of the skull should be. The dark speckles
are the fragments of hair that did generate. The geometry is simply **missing** — this
was never a shading problem.

Note the centroid-based "23-27% of faces point inward" figure quoted earlier is a poor
proxy on a concave body (armpits, between the legs legitimately face inward) and should
not be read as flipped area.

**Where this leaves the artefact.** Two live explanations, both in question 1b and
question 5: geometry dropped by the extractor and never repaired, and — more likely for
a hole this size — the model never having seen the back of the head at all. A
hole-filler would stretch a flat membrane across the skull; multi-view input would give
it an actual back of a head. **This raises the priority of multi-view.**

---

## 3. Why did higher resolution make the interior *more* visible?

**Observed.** `pipeline_type: 1024` gave a clearly better face (real eyes, brows,
nose) but far more visible interior — speckles on the sweater, a face showing through
the back of the skull. `512` was cleaner but had dead, smeared eyes. The seed hunt
(4 variants) never broke this tradeoff.

**Hypothesis (revised).** Higher resolution means thinner walls and finer features, so
the dual-grid extractor drops more boundary quads (question 1b) and the resulting holes
are more numerous. Detail and watertightness may be in tension.

**How to settle it.** Run `scripts/mesh_health.py` at 512 vs 1024 vs cascade on the
same seed and compare boundary-edge counts, **measured position-only**. Note the
related finding that the tradeoff is subject-dependent: 512 beat 1024 on the human but
loses badly on the fox.

---

## 4. Why didn't voxel remeshing fix it?

**Observed.** Voxel remesh was tried as a way to wrap the geometry in one clean skin.
It didn't work. At detail-preserving sizes (0.004) it produced 258 disconnected
components — the largest being *just the legs*. Only at 0.012 did it fuse into 3
bodies, by which point the mesh was 7,000 faces and the face was destroyed.

**Note (2026-08-06):** this was motivated by question 1's false premise. The input was
already a connected surface, so remeshing was solving a problem that did not exist —
and the component counts quoted here were themselves measured with the UV-seam bug.
Worth re-deriving before drawing conclusions from it.

**ELI5.** Voxel remeshing is "dip the model in wax and keep the wax shell." It works
if the wax is thick enough to bridge the gaps. Ours had to be so thick it filled in
his eyes and nose too.

**What we don't know.** Whether a *better tool* solves this. Blender's voxel remesh is
one option, but **screened Poisson surface reconstruction** (Open3D, PyMeshLab) is
built precisely for turning noisy, gappy point/shard data into a watertight surface
and may bridge the gaps without the same detail cost. We never tried it.

**How to settle it.** Sample points + normals, run Poisson reconstruction at a few
depths, compare detail and watertightness against the voxel results. Lower priority now
that the input is known to be a connected surface — ordinary hole filling is the
cheaper first thing to try. Note this depends on question 2 — Poisson needs *correct* normals, so
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
back of his head looks weird" into a number. Built as `scripts/mesh_health.py`. It must
merge by **position only** — the UV-seam trap in question 1 is exactly the kind of
error a shared diagnostic prevents from spreading.

Open question: should it just report, or should it *gate* — refuse to promote an
output that fails a threshold, the way `validate_run_policy` gates on licence?

---

## 8. When does analytic weighting generalise?

**Observed.** The rigging plan assumed hand-marked joints, because auto-placement
failed on the fox. But for the T-posed human we skipped marking entirely: landmarks
were measured straight off the mesh (shoulder 0.12, elbow 0.215, wrist 0.30), and
weights were assigned from vertex position rather than inferred from the mesh.

**Why it matters.** It is predictable regardless of mesh quality — no voxel remesh, no
heat diffusion, no weight transfer. **Corrected:** the original justification (the mesh
being too fragmented for bone-heat weighting) was false, see question 1. The Nikita mesh
is a single connected surface, so standard weighting might have worked too. The approach
still stands on its own for a known pose.

**What we don't know.** Where the boundary is. It clearly works when the pose is known
and the limbs are axis-aligned (T-pose). It clearly fails for a quadruped in an
arbitrary pose. Does it extend to a full walk cycle on a human? To legs? The honest
answer is we only proved it for one arm on one figure.

**Related.** A held prop must be bound to its bone *before* any distance-based test,
not after — the mug is taller than the arm is thick, so a band test around the arm
clipped its top and bottom and stretched it into taffy. Probably a general rule for
props.
