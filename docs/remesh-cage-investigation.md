# The remesh cage: what it is, and four things it is not

**Date:** 2026-08-13. **Status:** cause narrowed to dual-contouring vertex placement; not yet
proven. Everything below is measured on `output/branch_test/fox_decode.pt` (the cached
12.8M-face decode of `3-4th-fox-alpha-front.png`), re-bakeable in ~2 minutes with
`scripts/trellis_rebake.py --remesh`.

## The symptom

`to_glb(remesh=True)` — the branch the reference implementation always takes — produces a
wireframe cage. Correct silhouette, recognisable fox, no surface.

| | ours (Branch 2) | reference |
|---|---|---|
| faces | 468,238 | 283,149 |
| **total surface area** | **0.878** | **4.989** |
| median face area | 1.58e-06 | 1.16e-05 |
| signed volume | **-0.000303** | **+0.005202** |
| rays hitting the mesh | 112/300 (37%) | 205/300 (68%) |
| median ray crossings | **1** | 3 |

**65% more faces covering 5.7x less area.** The faces exist and none are degenerate; they
simply do not tile a surface. A median of 1 ray crossing is impossible for a closed solid.

## Four eliminated causes

Each was a working theory. Each is now excluded by measurement, which is worth recording so
nobody re-runs them.

### 1. Hashmap lookups — CLEARED

`metal_remeshing.py:169` emits a quad only when all four neighbouring voxels resolve, so a
faulty `hashmap_lookup_3d` looked like the obvious culprit. Instrumented:

```
lookups 8,848,368   misses 376,077 (4.25%)
quads surviving 1,935,382 of 2,212,092 (87.49%)
miss rate per corner: ['0.00%', '5.31%', '6.37%', '5.33%']
```

**Corner 0 is the voxel itself and misses 0.00%**, as it must — it is in the map by
construction. The other three are neighbours, and 5-6% is correct behaviour at the edge of a
narrow band where neighbours legitimately fall outside it. 87.5% quad survival is not a cage.

`1,935,382 quads x 2 triangles = 3,870,764` — exactly the reported post-remesh face count.

### 2. The macOS GPU watchdog — CLEARED

`generate.py`'s own `_watchdog_help_message` documents a watchdog that kills long Metal
dispatches *without raising a Python exception*, leaving empty tensors. It explained every
symptom, including a machine crash mid-experiment (three WindowServer userspace watchdog
timeouts, `IOGPU`/`AGX` in the report).

It is real, and it is not this. `MTL_CAPTURE_ENABLED=1`, which extends the watchdog, changed
nothing: volume -0.000280 → -0.000303, faces 467,010 → 468,238. The result is deterministic
to within 0.3% across three runs on a quiet machine.

**The watchdog is a genuine hazard for anything else in this port** — it caused the crash and
two silent harness hangs (0% CPU, no output, no error). It is simply not the cage.

### 3. `simplify` — CLEARED

Branch 2 is `remesh → simplify`, and the cage could have been decimation damage. Probed the
signed volume either side:

```
volume AFTER REMESH:   -0.000817  (3,870,764 faces)
volume AFTER SIMPLIFY: -0.000620  (  495,038 faces)
```

Already a cage before simplify runs. It faithfully decimated something that was never solid.

### 4. Winding / face orientation — CLEARED

3.87M faces enclosing ~0 volume is the classic signature of inverted faces cancelling out,
and Branch 2 notably **never calls `unify_face_orientations`** (Branch 1 does, as its step 5).
But repair does not recover it:

```
as shipped        volume -0.000303   winding_consistent False
after fix_winding volume -0.000328   winding_consistent False
after fix_normals volume -0.000328   winding_consistent False
```

The volume is absent because the surface is absent, not because it is inside-out.

## SOLVED — the narrow band is one voxel thick

`eps = band * scale / resolution`, and with the demo's `band=1` at resolution 1024:

```
band=1.0  scale=1.0029296875  resolution=1024  ->  eps=0.00097942
voxel size = scale / resolution                  =  0.00098
```

**`eps` is exactly one voxel.** `bvh.unsigned_distance` returns an *unsigned* field, so
`distances_vert -= eps` makes negative only the points within `eps` of the surface — a shell
straddling it, not the interior. The UDF minimum is exactly `-eps` (-0.00098), which is the
signature of an unsigned field: its minimum is 0, on the surface.

A shell one voxel thick is at the sampling limit. Most voxel edges step clean over it, so few
register a sign change, and voxels that register none fall back to their grid centre — which
is the lattice.

Thickening the band fixes it:

| band | eps | UDF negative | fallback to centre | **volume after remesh** |
|---|---|---|---|---|
| 1 | 0.00098 | 13.98% | 29.16% | **-0.000817** |
| 2 | 0.00196 | 27.68% | 23.66% | **+0.001046** |
| **3** | **0.00296** | **37.63%** | 22.47% | **+0.005824** |

Control: **+0.005202**. Band 3 is within 12% — a solid mesh, and the first one this pipeline
has produced on the remesh path.

**The metric that tracks it is the negative fraction, not the fallback rate.** Fallback
plateaus (29% → 23.7% → 22.5%) while the result goes from inverted to correct; predicting
from fallback alone would have called band 3 a failure.

Cost: faces after remesh go 3.9M → 9.6M → **15.3M**, so band 3 is ~4x the geometry to
simplify down, and correspondingly slower.

**Open question for upstream:** the reference demo runs `band=1` and gets a solid mesh. If
one voxel of band suffices on CUDA and not here, either `MtlBVH.unsigned_distance` differs
from `cuBVH`'s, or there is a half-voxel offset between where distances are sampled and where
crossings are tested. Thickening the band works but is treating the symptom.

## The original hypothesis (confirmed)

`simple_dual_contour_u32_kernel` (`src/metal/remesh.metal:291`) places one vertex per voxel
at the mean of its edge-surface intersections, and **falls back to the voxel centre when it
finds none**:

```c
} else {
    out_vertices[tid * 3 + 0] = float(vx) + 0.5f;   // fallback: voxel center
```

Widespread fallback would put vertices on a regular grid, connect them into small regular
quads, and produce exactly what is measured: correct silhouette, many tiny faces, negligible
surface area, no enclosed volume. Only 2,212,092 of 9,483,753 possible edges (23.3%) are
marked as crossing the surface, which is consistent with the sign test failing widely.

The sign test is `(val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)` over a field built by
`distances_vert -= eps` — an **unsigned** distance field shifted to make a thin negative band.
If `eps` or the shift behaves differently here than in the CUDA path, few edges register a
crossing.

**Next measurement:** count how many dual-contour vertices land exactly on voxel centres.
That distinguishes "the kernel found no intersections" from "it found them and placed them
badly", and needs one instrumented rebake.

## One unexplained deviation

`metal_remeshing.py:117` adds `coords = torch.unique(coords, dim=0)` inside the octree
refinement loop; the CUDA reference has no such line. Children of distinct parents cannot
collide, so it only sorts — but sorting reorders every index, and
`hashmap_insert_3d_idx_as_val` stores index-as-value. Probably harmless. Still unexplained.

## Reproducing

Instrumentation is applied to the **installed** copies, not the `deps/` sources — the
`site-packages` copy is what executes, and patching `deps/mtlmesh/cumesh/` does nothing.
Confirm with `python -c "import cumesh.metal_remeshing as m; print(m.__file__)"` before
editing anything.

```bash
I2L_REMESH_DIAG=1 vendor/trellis-mac/.venv/bin/python scripts/trellis_rebake.py \
    output/branch_test/fox_decode.pt /tmp/out.glb --remesh --texture-size 1024
```
