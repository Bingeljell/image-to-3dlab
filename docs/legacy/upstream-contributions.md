# Draft contributions to `shivampkumar/trellis-mac`

> **Status: drafts, NOT submitted. Do not send without working the checklists.** Written
> 2026-08-12 against the Apple Silicon TRELLIS.2 port
> (`https://github.com/shivampkumar/trellis-mac`). Self-contained: assumes no knowledge of
> this repo. Every number needs re-verifying from a clean checkout first.
>
> Two independent findings:
>
> 1. **[The 200,000-face pre-simplification](#finding-1--the-200000-face-pre-simplification)**
>    — destroys 94–99% of every decode before the real pipeline runs. High confidence,
>    affects every user of the port.
> 2. **[Remeshing produces a wireframe lattice](#finding-2--remeshing-produces-a-lattice-not-a-surface)**
>    — `--remesh` yields a cage of struts rather than a surface, at any `remesh_project`.
>    Needs a minimal reproduction before filing.

---

# Finding 1 — the 200,000-face pre-simplification

## Summary

`generate.py` pre-simplifies the decoded mesh to **200,000 faces** using
`fast_simplification` *before* handing it to `o_voxel.postprocess.to_glb`. On a typical
subject the decode is **3–27 million triangles**, so this discards 94–99% of the geometry
with a crude decimator, and every subsequent stage — hole filling, non-manifold repair,
QEM simplification, UV unwrapping and texture baking — then runs on the reduced mesh.

Removing the pre-simplification produces meshes visually comparable to the official
[TRELLIS.2 HuggingFace Space](https://huggingface.co/spaces/microsoft/TRELLIS.2), with **no
`mtlbvh` crash** at 3.2M and 4.9M faces on an M-series Mac.

## The code

`generate.py`, in the Metal baking branch:

```python
# Pre-simplify mesh to avoid mtlbvh crash on large meshes.
# Target ~200K faces — keeps detail, avoids Metal BVH issues.
import fast_simplification
target_faces = min(args.bake_target_faces, 200000, len(faces_np))
if len(faces_np) > target_faces:
    ratio = 1.0 - (target_faces / len(faces_np))
    simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
    ...
```

`target_faces` is then also passed as `decimation_target` to `to_glb`.

## Three consequences

**1. Surface damage.** The mesh handed to `to_glb` has already lost most of its geometry to
a decimator with no quality metric. Output surfaces come out crazed with fine cracks. The
official Space, given the same input image, produces a clean surface.

**2. `--bake-target-faces` is silently clamped above 200,000.** Requesting 300,000 and
3,000,000 both yield ~197k faces, because the `min()` caps it. Any parameter sweep above
200k measures nothing.

**3. Inconsistent / inverted winding (intermittent).** Meshes through this path frequently
come out with inconsistent face winding and negative signed volume — i.e. inside-out. glTF
materials are double-sided by default so previews look fine, but under backface culling
(SceneKit, RealityKit, most engines) the asset is hollow. Removing the pre-simplification
produced winding-consistent, positive-volume meshes on the subjects tested; leaving it in
inverted some and not others.

*Note: (3) is correlational. We observed it consistently across subjects but have not
isolated `fast_simplification` as the mechanism.*

## Evidence

Same input image, an M-series Mac, TRELLIS.2-4B, `1024_cascade`:

| | Faces | Winding consistent | Signed volume | Surface |
|---|-------|--------------------|---------------|---------|
| Port, as shipped | 100,291 | no | +0.01230 | crazed with cracks |
| Port, pre-simplify removed | 293,488 | **yes** | **+0.02368** | clean |
| Official HF Space (reference) | 281,889 | yes | +0.00199 | clean |

Stage log with the pre-simplification removed, showing the pipeline working as designed:

```
handing 3,207,582 faces to o_voxel (decimation_target 300,000)
After filling holes:          3,229,568
After initial simplification:   866,044
After initial cleanup:          861,438
After final simplification:     292,799
After final cleanup:            293,492
```

Total runtime 324s — **faster** than the capped path on the same machine, because the
crude simplification of 3.2M faces is itself expensive.

## Proposed fix

Keep a safety net, but make it a genuine ceiling rather than a working target:

```python
# Pre-simplify only as an mtlbvh safety net, not as a quality target.
# o_voxel's own QEM decimation (decimation_target) is quality-aware; this is not.
PRE_SIMPLIFY_CEILING = 4_000_000
if len(faces_np) > PRE_SIMPLIFY_CEILING:
    ratio = 1.0 - (PRE_SIMPLIFY_CEILING / len(faces_np))
    simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
    ...
else:
    simp_verts_t, simp_faces_t = mesh_out.vertices, mesh_out.faces
```

and pass `decimation_target=args.bake_target_faces` unclamped.

## On the `mtlbvh` crash

The comment cites a crash on large meshes. We could not reproduce it: runs at **3,207,582**
and **4,929,482** faces completed normally, including a 4.9M-face subject with remeshing
enabled. We have not tested beyond ~5M — a subject here decodes at 27.6M and has not yet
been run uncapped — so a ceiling is still worth keeping. If maintainers know the failing
size, that number should replace our arbitrary 4M.

## Reproduction

```bash
python generate.py INPUT.png --seed 42 --output out_capped \
  --pipeline-type 1024_cascade --texture-size 3072 --bake-target-faces 300000

# then remove/raise the 200000 clamp and repeat
python generate.py INPUT.png --seed 42 --output out_uncapped \
  --pipeline-type 1024_cascade --texture-size 3072 --bake-target-faces 300000
```

Compare **with backface culling enabled** — a double-sided preview hides the winding
problem entirely. Check signed volume and winding consistency, e.g. via `trimesh`:

```python
m = trimesh.load("out.glb", force="mesh", process=False)
print(m.is_winding_consistent, m.volume)   # want True and positive
```

## Before submitting

- [ ] Re-run both paths from a clean checkout to confirm the numbers
- [ ] Test one very dense subject (~27M decode) uncapped to find the real `mtlbvh` limit
- [ ] Confirm whether winding inversion tracks the pre-simplification or something else
- [ ] Check whether the maintainers already know; the crash comment suggests a real incident
- [ ] Note that `--remesh` defaults off here while the upstream README example uses
      `remesh=True, remesh_project=0`; on this port `remesh_project=0` produced an
      unusable lattice, which may be worth a separate report once understood

---

# Finding 2 — remeshing produces a lattice, not a surface

## Summary

Enabling `--remesh` (narrow-band dual-contouring remeshing, run before UV unwrapping)
produces a **wireframe cage of thin struts instead of a closed surface**. The silhouette is
correct — the subject is recognisable — but the skin is missing between the struts, so the
asset is unusable.

`remesh_project` has **no effect on this**: `0` and `0.9` produce visually identical
lattices. Since that parameter controls how far rebuilt vertices are pulled back onto the
original surface, a result that ignores it points at a defect *upstream* of the projection
step — in the narrow-band construction or the dual-contouring kernel.

This matters because the upstream TRELLIS.2 README's own example runs with
`remesh=True, remesh_band=1, remesh_project=0`. Anyone following upstream's documented
settings on this port gets an unusable mesh.

## Evidence

Same subject, same seed (261852270), `1024_cascade`, `texture_size=4096`,
`bake_target_faces=1000000`, pre-simplification removed. Signed volume is the tell: a
closed surface encloses volume, a cage encloses almost nothing.

| Run | Faces out | Signed volume | See-through (culled, worst angle) |
|-----|-----------|---------------|-----------------------------------|
| no remesh | 282,882 | **+0.02830** | 4.6% |
| `--remesh --remesh-project 0` | 948,910 | **+0.00013** | ~90% |
| `--remesh --remesh-project 0.9` | 935,103 | **+0.00031** | ~90% |

A ~90x collapse in enclosed volume, with the parameter that should change the result
making no difference.

Stage log (project 0.9), showing remeshing itself completing without error:

```
After filling holes: 2428063 vertices, 4956432 faces
After remeshing:     1648841 vertices, 3108294 faces
After cleanup:       1528229 vertices, 3021046 faces
After simplifying:    468658 vertices,  954799 faces
```

Note `cumesh/metal_remeshing.py:188` also emits
`UserWarning: Using torch.cross without specifying the dim arg is deprecated` during the
run — probably unrelated, but it indicates this path is not exercised often.

## Where the defect is likely to be

`cumesh/metal_remeshing.py` (226 lines) is a near-parallel port of `cumesh/remeshing.py`
(251 lines). Diffing them, the `project_back` block is **byte-for-byte identical**, which
is consistent with `remesh_project` having no effect on the failure. The divergence is
therefore earlier: either the narrow-band voxel construction, or the compiled call

```python
dual_verts, intersected = _C.simple_dual_contour(...)
```

which on this port resolves to a Metal kernel rather than the CUDA one. That kernel is
compiled, so it cannot be inspected from Python.

## Before filing

- [ ] **Build a minimal reproduction** — call `remesh_narrow_band_dc` directly on a simple
      watertight mesh (a sphere or a cube), with no TRELLIS pipeline involved. If a sphere
      remeshes into a cage, the report is airtight and tiny. This is the single most
      valuable next step.
- [ ] Compare the intermediate `dual_verts` / `intersected` tensors against expectations —
      are vertices missing, or are the quads that connect them missing?
- [ ] Check whether `resolution` (taken from `grid_size.max()` in `o_voxel/postprocess.py`)
      is sane for the subject; a too-coarse grid could plausibly produce sparse output
- [ ] Confirm on a second machine / Metal version before blaming the kernel
- [ ] Search the port's issues — the deprecation warning suggests this path is rarely run,
      so it may simply be unreported rather than known-broken

## Impact if confirmed

Remeshing is how upstream produces a closed surface before UV unwrapping — the step whose
own help text says it *"targets boundary edges at source"*. On this port it is effectively
unavailable, which means Apple Silicon users have no working route to a watertight mesh
and must rely on post-hoc hole repair instead.
