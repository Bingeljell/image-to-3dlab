# The decode-time mesh cleanup is disabled, and that is the root defect

Written 2026-08-09, after the user pushed back that the hero mesh "has gaps and holes
especially if you zoom in, and if you export it that way there might be holes you can see
through". They were right, and chasing it upstream found something bigger than the holes.

## The finding

`o_voxel/postprocess.py::to_glb` — the prescribed entry point — cleans the mesh **four
times** during decode, and every one of those calls happens **before UV unwrapping**:

| line | call |
|---|---|
| 191 | `mesh.fill_holes(max_hole_perimeter=3e-2)` |
| 217 | `mesh.simplify(decimation_target * 3)` |
| 225 | `mesh.fill_holes(max_hole_perimeter=3e-2)` |
| 230 | `mesh.simplify(decimation_target)` |
| 238 | `mesh.fill_holes(max_hole_perimeter=3e-2)` |
| 269 | `mesh.fill_holes(max_hole_perimeter=3e-2)` |
| 274 | `mesh.simplify(decimation_target)` |
| **290** | **`mesh.uv_unwrap(...)`** |

`vendor/trellis-mac/patches/mps_compat.py:263-294` makes three of those methods return
immediately:

```python
def fill_holes(self, max_hole_perimeter=3e-2):
    return  # Skip — Metal cumesh segfaults on large decode meshes
def remove_faces(self, face_mask):
    return
def simplify(self, target=1000000, ...):
    return
```

The stated reason is sound — TRELLIS.2 calls these on the full ~400K-vertex decode mesh
and the Metal port of `cumesh` segfaults at that size. But the consequence was never
traced: **every asset this repo has produced was UV-unwrapped from an uncleaned mesh.**

## What this explains

- **889 holes / 16,467 boundary edges** on the hero — `fill_holes` never ran, four times.
- **688 loose components, 6.4% of faces** — `remove_faces` never ran.
- **Plausibly the 11,340 UV islands and 53% atlas coverage.** xatlas cannot run a chart
  across a hole, so every one of those boundary loops forces a chart boundary. This is a
  *hypothesis*, tested in [the experiment below](#experiment-1).

## How this compares with what we already wrote down

This is the third time a session has circled this drain, so the corrections matter more
than the finding.

**`scripts/fill_holes.py` already knew half of it.** Its docstring says: *"Upstream repairs
this during decode with `fill_holes()`; our Mac port disables that call... so they survive
into the export."* Correct, and written months before this session. What it did not do was
follow the consequence: it treated the disabled call as a reason to patch holes
**downstream**, and never asked what else the same patch disabled or what ran *after* the
missing step. The answer to both is: a lot.

**`docs/trellis-prescribed-flow.md` is wrong on one point.** It states the mesh "is
simplified **twice**: `fast_simplification` in the port before the call, then
`mesh.simplify(decimation_target * 3)` and `mesh.simplify(decimation_target)` inside
`to_glb`." Both of those inner calls are no-ops. Only the port's own
`fast_simplification` runs. That doc audited the *parameters* `to_glb` accepts and never
checked whether the methods it calls still had bodies — the same class of error it was
written to warn about ("we optimised inside a flow without reading its contract").

**`docs/texture-quality-roadmap.md` attributes UV fragmentation to the wrong cause.** It
reasons entirely about packing and clustering — proposing xatlas repacking, region
splitting, reducing island count — and never considers that the surface handed to the
unwrapper is full of holes. Region splitting was then measured as a regression, and
cluster refinement made coverage *worse* (53% → 38.3%). Both were attempts to fix the
atlas without fixing its input.

**`docs/roadmap.md` deprioritised holes on a reasonable but circular argument** — "the
holes have not actually blocked a deliverable... every render looks fine because glTF
materials are `doubleSided`, which hides the problem entirely." The renders that proved
holes were harmless were structurally incapable of showing holes. Once rendered with
backface culling, the mesh is visibly perforated.

**What is genuinely new here:** that `remove_faces` and `simplify` are *also* stubbed
(not just `fill_holes`); that there are four cleanup calls, not one; that all of them
precede `uv_unwrap`, which links the hole problem to the texture problem we had treated as
independent; and that an unused CUDA-free implementation already exists in the tree.

## The unused CPU path

`o_voxel/postprocess_cpu.py` implements the same `to_glb` with `trimesh.repair.fill_holes`
(lines 279, 298), `fast_simplification` and `xatlas` — no CUDA anywhere. `generate.py`
imports `o_voxel.postprocess` and never the CPU variant.

Caveat before treating it as a drop-in: `trimesh.repair.fill_holes` only closes triangular
and quad holes — roughly 20% of the loops on our mesh. `scripts/fill_holes.py` traces
arbitrary loops and fans them from a centroid, so **our implementation is the stronger
one** and is the better thing to inject.

Note also the upstream perimeter limit is `3e-2`, five times stricter than our script's
`0.15` default.

## Ruled out: issue #140, duplicate inner walls

[TRELLIS.2 issue #140](https://github.com/microsoft/TRELLIS.2/issues/140) reports most
generations carrying a duplicate inner shell. **It does not affect our fox.** Firing 400
rays through the torso, 86% cross exactly twice and the median crossing count is 2 — a
single-walled surface. Component analysis agrees: one body holds 99.1% of faces and the
rest are sub-250-face specks, none shell-shaped.

## Experiments

### Experiment 1 — RUN 2026-08-09. **The ordering hypothesis is WRONG.**

Three meshes through identical xatlas calls, so the only variable is the geometry:

| mesh | faces | bnd edges | islands | coverage | texels/face |
|---|---|---|---|---|---|
| *shipped atlas (reference)* | 101,298 | 16,467 | 11,321 | **53.0%** | 21.9 |
| hero, re-unwrapped | 101,298 | 16,467 | 12,714 | 53.8% | 22.3 |
| pruned | 95,689 | 14,818 | **10,683** | 53.3% | 23.4 |
| repaired (holes filled) | 103,630 | **6,877** | 13,556 | 53.2% | 21.5 |

Cutting boundary edges by 58% moved coverage from 53.8% to 53.2%, and *raised* the island
count. **Coverage is invariant near 53% across every geometry change.** Filling holes
before unwrapping does not improve the atlas, and the section above claiming it plausibly
would is superseded by this table.

One thing did help: pruning loose specks cut islands 12,714 → 10,683. So the disabled
`remove_faces` has a real, if modest, effect on fragmentation — just not on coverage.

### Experiment 1b — the packer, not the mesh

Coverage being geometry-invariant pointed at xatlas itself:

| config | charts | coverage | texels/face |
|---|---|---|---|
| default | 9,403 | 53.3% | 23.4 |
| **`bruteForce=True`** | 9,403 | **59.9%** | **26.3** |
| `padding=0`, `blockAlign=False` | 9,403 | no effect | — |
| relaxed chart merging | 8,314 | 53.5% | 23.5 |
| aggressive merge + `bruteForce` | 8,354 | 59.1% | 25.9 |
| maximal merge, all weights 0 | 10,331 | 59.4% | 26.0 |

**`bruteForce` packing is a free 12% relative gain** — one flag, no geometry change, no
regeneration. Everything else is noise. Setting every chart-segmentation weight to zero
*increased* chart count to 10,331, so chart count is not controllable by those knobs and
does not drive coverage regardless.

**~60% is the practical ceiling**, and the chart count is intrinsic: a mesh of thousands
of separate leaf blades genuinely needs thousands of charts. This closes the
"island count is the blocker" theory for good — the repo's own data had already
contradicted it once, when the tail had half the head's islands and packed no better.

### Experiment 2 — NOT RUN, and the case for it has weakened

Proposed: replace the three `return` stubs with pure-Python equivalents so cleanup runs
before `uv_unwrap`. Experiment 1 removed the atlas justification. The geometry
justification also needs qualifying:

**Upstream's `max_hole_perimeter=3e-2` is five times stricter than our `0.15` default**, so
restoring it would fill *fewer* holes than the downstream pass already does. It would fill
them at 400K vertices before decimation, which yields better patches than patching
afterwards — real, but modest. It would **not** close the 96 large tears in the shoulder
and flank, which exceed every one of these limits by design because they are missing
evidence, not artefacts.

So the remaining visible defect needs *information* — multi-view, better input art, or the
labelling split that makes only the solid body watertight — not more cleanup.
