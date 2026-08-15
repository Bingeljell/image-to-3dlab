# Handover: the remesh cage

**Date:** 2026-08-13, end of a full day on it. **Status:** NOT FIXED. A real defect was found
and patched in the Metal kernel; the patch made **no difference**. Details below — read
"The patch, and why it did not work" before doing anything.

Everything here is measured on `output/branch_test/fox_decode.pt` — the cached 12.8M-face
decode of `3-4th-fox-alpha-front.png` — re-bakeable in ~2-5 minutes with
`scripts/trellis_rebake.py`, no re-sampling.

---

## The defect found (real, but not the cause)

`vendor/trellis-mac/deps/mtlmesh/src/metal/remesh.metal`:

```c
inline float get_vertex_val_u32(...) {
    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];                    // no check that the lookup hit
}
```

`linear_probing_lookup_u32` returns `0xFFFFFFFF` on a miss (same file, line 36). So a missed
lookup evaluates `udf[0xFFFFFFFF]` — four billion elements past the end of the buffer.

**Metal defines out-of-bounds buffer reads as returning zero.** The dual-contouring crossing
test is

```c
if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0))
```

and `0.0` is not `< 0`, so a missed lookup reads as *outside the surface*, deterministically.
Two misses on an edge agree, no crossing is registered, and the kernel strands that voxel's
vertex at its grid centre:

```c
} else {
    out_vertices[tid * 3 + 0] = float(vx) + 0.5f;   // fallback: voxel center
```

Quads then connect grid centres to grid centres, producing axis-aligned squares. **That is
the lattice.** Measured: 22-29% of vertices stranded on exact voxel centres.

Zero is the worst possible substitute — it means *exactly on the surface*. A vertex outside
the narrow band is far outside it, so the honest answer is a large positive distance.

### Why CUDA does not show it

Identical source; different out-of-bounds semantics. CUDA leaves OOB reads undefined, and in
practice they land in adjacent allocated memory returning values of varied sign, which
produce crossings often enough to hide the defect. Metal's defined zero makes it systematic.

### The patch, and why it did not work

`scripts/patch_metal_hashmap_miss.py`, then `scripts/rebuild_metallib.sh`:

```c
if (idx == 0xFFFFFFFFu) return 1.0e30f;
return udf[idx];
```

Both `u32` and `u64` variants have the same defect. The rebuild compiles with `xcrun metal`
(needs `DEVELOPER_DIR=/Applications/Xcode.app/...`; Command Line Tools has no Metal compiler)
and installs over the **site-packages** copy, which is the one that loads.

**It changed nothing.** The rebuilt library was definitely installed — md5 went
`2a9e9108…` → `8ee41b4f…`, size 530,150 → 530,454 — and the run produced *byte-identical*
numbers: volume −0.000817, fallback 29.16%, area 1.097.

**The likely reason: the guarded path is never taken.** The 4.25% miss rate we measured was
on `hashmap_vox` (voxel lookups for stitching quads). `get_vertex_val_*` reads `hashmap_vert`,
which is built from `grid_verts` — *all* active vertices of the sparse grid — so every vertex
the kernel asks about is in the map by construction. The out-of-bounds read is real and worth
fixing defensively, but it does not fire here.

**Keep the patch.** It is correct, it costs nothing, and it removes a genuine OOB read. It is
simply not the cause.

---

## What it is NOT — all eliminated by measurement

Recorded so nobody re-runs them. Each was a working theory.

| suspect | how it died |
|---|---|
| **Voxel hashmap lookups** (`hashmap_vox`) | Direct test: of 376,077 missed lookups, **0** were voxels present in `coords`. All genuinely absent. The map returns every key it was given. |
| **macOS GPU watchdog** | Real — it crashed the machine and produced two silent 0%-CPU hangs — but `MTL_CAPTURE_ENABLED=1` changed nothing and the result is deterministic to 0.3% across runs. |
| **`simplify`** | Signed volume was already ~0 *before* it ran (-0.000817 post-remesh). It faithfully decimated a cage. |
| **Winding / orientation** | `fix_winding` and `fix_normals` both fail to recover volume. Turning backface culling **off** in the viewer leaves it a cage — the geometry genuinely is not there. |
| **`MtlBVH` precision** | Measured against analytic ground truth on a sphere: p99 error **2.68e-05** against an eps of **9.79e-04** — a 37x margin, `sign_flip_risk` exactly 0.0. Distances are essentially exact. |
| **Band thickness** | Swept 1, 2, 3. Coverage improves (area 0.88 → 4.88) and it stays a lattice. |
| **Subdivision threshold** | Swept 0.87, 1.0, 1.5. Quad survival 87.5% → 95.5% and it stays a lattice. |
| **`project_back`** | Swept 0, 0.5, 0.9, 1.0. Area rises to 4.19 — best of any run — and it stays a lattice. |
| **Pre-simplified vs full decode** | Both. Lattice. |

**Tuning cannot fix this.** Every parameter moved its own metric in the right direction and
none changed what the render shows, which is the signature of a defect below the parameters.

---

## Where I would go next

The single most informative number left, and it is cheap:

0. **Why do 29% of voxels find no crossing when the UDF is correct?** Only **13.98%** of grid
   vertices are negative at band 1 (37.63% at band 3). A voxel needs corners of both signs; if
   most voxels have all eight corners positive, no crossing exists to find and the fallback is
   *correct behaviour on a wrong field*. That reframes the whole problem: the question is not
   "why is the crossing test failing" but **"why is so little of the field negative?"**

   `distances_vert = bvh.unsigned_distance(pts_vert) - eps` makes negative only points within
   `eps` of the surface. `MtlBVH` is precise (p99 error 2.7e-05, verified). So the arithmetic
   is right and the field is *genuinely* mostly positive — meaning the narrow band as
   constructed simply does not straddle the surface the way DC needs.

   **Instrument `hashmap_vert` misses directly** (the same present-vs-absent test that cleared
   `hashmap_vox`) to confirm the kernel is reading the values we think it is. If those are
   clean too, the defect is in how `grid_verts` / `distances_vert` are built, not in any
   lookup.

2. **`torch.unique(coords, dim=0)` at `metal_remeshing.py:117`** — present in the Metal port,
   absent from the CUDA reference, and still unexplained. It cannot create duplicates
   (children of distinct parents cannot collide) so it only *sorts* — but sorting reorders
   every index, and `hashmap_insert_3d_idx_as_val` stores index-as-value. Probably benign.
   Cheap to test: remove it and re-run.

3. **`hash32` collision behaviour.** The map is allocated at capacity `2 * N` with linear
   probing. If `hash32` clusters, probe chains lengthen and lookups can walk into a
   `0xFFFFFFFF` slot belonging to a different chain and report a false miss. Worth checking
   the load factor and the hash's distribution.

4. **The u32/u64 kernel split.** `VOL = resolution³` decides which variant runs; at
   resolution 1024 that is 1.07e9, just under the 2^32 boundary, so the u32 path is used and
   `flat_idx` is computed in `ulong` then truncated to `uint`. Near that boundary the
   truncation is worth verifying.

---

## What is genuinely fixed today (independent of the remesh)

- **`--material-mode` now defaults to `pbr`.** `matte` was deleting TRELLIS's 3072²
  metallicRoughness map on every organic subject and pinning the factors flat. Our maps match
  the reference's closely; we were discarding them on export. Judged culled on Flicker: eye
  reflections return, body gains surface variation.
- **Decode cleanup re-enabled** (`patch_trellis_enable_cleanup.py`). `fill_holes`,
  `remove_faces` and `simplify` were stubbed to `return` with a "Metal cumesh segfaults"
  comment. They do not segfault — verified on a 4M-face mesh.
- **Two-stage split** (`--dump-decode` + `trellis_rebake.py`). Sampling and baking are now
  separate, so a `to_glb` experiment costs minutes instead of 20.
- **Browser viewer** (`viewer/`), backface-culled by default, which is the only instrument
  that has not misled us.
- **Verdict register** (`mark_asset.py`) and **forensics** (`glb_forensics.py`).

## Instruments that lied, and how

Worth internalising — five separate metrics moved the right way while the render did not.

| metric | blind to |
|---|---|
| signed volume | a cage with the correct silhouette encloses roughly the right volume |
| surface area | a dense lattice has enormous area — struts have sides |
| boundary edges | a lattice of *closed tubes* has almost none (1,620 vs the control's 1) |
| fallback rate | plateaus at ~22% while output quality changes a lot |
| thin-slab cross-section | slicing by face centroid makes any closed surface look dotted |

`glb_forensics.py` also had a real bug for half the day: `merge_vertices()` preserves UV and
normal seams, so hole counts were inflated 20,000x on the reference assets. Always pass
`merge_tex=True, merge_norm=True`.

**The culled render in `viewer/` is the only reliable judge.** Every conclusion today that
came from a number alone was wrong.
