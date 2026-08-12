# Draft contribution to `shivampkumar/trellis-mac`

> **Status: draft, not yet submitted.** Written 2026-08-12. Intended for an issue or PR
> against the Apple Silicon TRELLIS.2 port. Self-contained: assumes no knowledge of this
> repo. Verify every number against a fresh run before submitting.

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
