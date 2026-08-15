# The prescribed TRELLIS flow, and how far we deviate

Written 2026-08-07 after a session spent fighting UV fragmentation, texture density and
holes downstream — then discovering that the upstream entry point exposes controls for all
three that we never passed.

## The prescribed entry point

`o_voxel.postprocess.to_glb()` — documented as performing *"cleaning, optional remeshing,
UV unwrapping, and texture baking from a volume."* We are on this path (provenance records
`texture_backend: metal-o-voxel`). The CPU fallback in `backends/texture_baker` is a
different route the port uses when Metal is unavailable.

## Its full parameter set, versus what we pass

| Parameter | Default | We pass | Controls |
|---|---|---|---|
| `decimation_target` | `1_000_000` | 100,000 | Target **vertices** for simplification |
| `texture_size` | `2048` | 2048 | Bake resolution |
| `remesh` | `False` | never | Narrow-band DC remeshing |
| `remesh_band` | `1` | never | Remesh band size |
| `remesh_project` | `0.9` | never | Snaps vertices back to the original surface |
| `mesh_cluster_threshold_cone_half_angle_rad` | `90 deg` | never | Cone threshold for UV clustering |
| `mesh_cluster_refine_iterations` | **`0`** | never | Cluster refinement in UV unwrapping |
| `mesh_cluster_global_iterations` | `1` | never | Global clustering iterations |
| `mesh_cluster_smooth_strength` | `1` | never | Cluster smoothing |

**Two of roughly ten quality parameters.** The Mac port passes `decimation_target`,
`texture_size` and `verbose`, and nothing else.

## Three deviations that matter

### 1. decimation_target is vertices, and we treat it as faces

The docstring is explicit: *"target number of vertices for mesh simplification."* The Mac
port computes a **face** budget (`min(args.bake_target_faces, 200000, len(faces_np))`) and
passes it straight into that vertex parameter. Our flag inherited the name
`--bake-target-faces`, so the wrong unit propagated all the way into our manifests and our
reasoning.

Note also the mesh is simplified **twice**: `fast_simplification` in the port before the
call, then `mesh.simplify(decimation_target * 3)` and `mesh.simplify(decimation_target)`
inside `to_glb`.

> **CORRECTION (2026-08-09): the two inner `simplify` calls never run.**
> `patches/mps_compat.py` stubs `simplify`, `fill_holes` and `remove_faces` to `return`
> immediately, because Metal `cumesh` segfaults on the 400K-vertex decode mesh. Only the
> port's own `fast_simplification` executes. This document audited the *parameters*
> `to_glb` accepts without checking whether the methods it calls still had bodies — the
> same mistake it was written to warn about. See
> [decode-cleanup-disabled.md](decode-cleanup-disabled.md), which also shows that the
> four disabled `fill_holes` calls all precede `uv_unwrap`, making this a likely cause of
> the UV fragmentation this document treats as a separate problem.

### 2. texture_size has no cap inside to_glb

There is no assert, clamp, `min()` or `max()` on `texture_size` in the function; it is used
to allocate buffers of that size. **The 2048 ceiling is the port's argparse
`choices=[512, 1024, 2048]` and nothing more.** Earlier work in this repo recorded 2048 as
a hard model limit — that is wrong, and 4096 should be tested.

### 3. UV clustering is unrefined by default

`mesh_cluster_refine_iterations` defaults to **0**, so chart refinement is off, and the
cone threshold sits at a very permissive 90 degrees. These parameters explicitly govern
*"clustering in uv unwrapping"* — which is exactly what produces this asset's **11,340 UV
islands and 53% atlas coverage**, the problem that made region splitting a regression.

## Why this reframes the texture roadmap

Every downstream fix considered so far — xatlas repacking, region splitting, SD texture
refinement — is compensating for output the prescribed flow may simply produce better if
asked. **Try the knobs before building the workarounds.**

Suggested order, cheapest first, measuring **summed UV triangle area** rather than pixel
brightness (see [texture-quality-roadmap.md](texture-quality-roadmap.md) for why):

1. `mesh_cluster_refine_iterations` above 0, and a tighter cone threshold. Directly targets
   island count and atlas coverage.
2. `texture_size=4096`, now that the cap is known to be a wrapper restriction.
3. `remesh=True` with `remesh_project`. Targets the 16,467 boundary edges at source rather
   than patching them with `fill_holes.py`.
4. A `decimation_target` set in the units it actually wants.

None of these need new code beyond threading parameters through `trellis_backend.py` and
the port.

## Standing lesson

We optimised inside a flow without reading its contract. The signature took one command to
inspect and answered questions we had spent hours working around.
