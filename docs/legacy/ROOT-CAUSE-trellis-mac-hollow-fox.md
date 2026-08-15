# Root cause report: hollow-looking TRELLIS Fox on macOS

**Status:** the hollow/cage defect is fixed and visually validated for the Moss Fox and
Forest Variant on 2026-08-13. Snag also completes the corrected path without the cage, but
retains small localized rear openings and needs a per-asset material correction; see the
stress-test result below.

## Executive summary

TRELLIS did not simply hand us a bad Fox. The cached raw decode contains a convincing shape,
and three defects in the Mac `to_glb` path damaged or misrepresented it:

1. the production Metal BVH traversal silently lost branches on meshes this large, producing
   incorrect surface distances during remeshing;
2. the Metal cleanup order called `fill_holes` on coincident repair seams and created tens of
   thousands of geometric openings;
3. the exporter treated alpha-zero UV padding as intentional transparency and emitted a
   blended, double-sided material.

The third defect was the decisive explanation for the reported view: from behind the Fox,
the eyes and mouth on the far side remained visible through the rear of the head, making its
inside look like a second front-facing Fox. With the **same geometry and rear camera**, changing
only the material to opaque and single-sided makes the real back of the head render normally.

The correct conclusion is therefore not “TRELLIS constructs poorly,” nor merely “OBJ to GLB
loses quality.” The raw neural decode is an intermediate representation, and our Mac port's
remesh, cleanup, and glTF material export introduced real downstream defects.

## Acceptance test

The Hugging Face GLBs are capability controls, not pixel-perfect targets. A different seed may
change pose, proportions, colour, or small details. A result passes when it:

- authentically represents the source;
- remains visually solid from every angle with backface culling enabled;
- has no visible skull/body holes, exposed interior shell, cage, or large inverted regions.

Formal watertightness is useful diagnostic evidence, but it is not the visual acceptance gate.
The Hugging Face control itself is not formally watertight.

## Where these defects sit in the pipeline

The relevant flow is:

`source image -> TRELLIS sampling/decoding -> sparse mesh + attributes -> to_glb -> GLB`

The raw cached decode is not a finished PLY or OBJ. `to_glb` still performs substantial work:
remeshing, topology cleanup, simplification, UV generation, texture baking, material selection,
and glTF export. The official demo also uses the remesh branch, with `remesh_band=1` and
`remesh_project=0`. That made the Mac implementation of this final stage the correct place to
investigate.

## Root causes and fixes

| Defect | Direct evidence | Fix | Effect |
|---|---|---|---|
| 24-entry Metal BVH stack overflow | On the 12.8M-face Fox, exact triangle centroids had unsigned-distance p50 `0.0030036`; `87.86%` exceeded remesh epsilon `0.00097942`. A four-way BVH needed about 34 pending entries, while the stack silently dropped entries after 24. | Use escape-index stackless traversal for unsigned distance, `FixedStack48` for other queries, chunk queries, and wrap CPU/unified-memory dispatches in an autorelease pool. | The full-Fox centroid invariant became p50 `4.06e-9`, p99 `2.65e-8`, max `4.74e-8`, with `0%` over epsilon. |
| Unsafe Metal cleanup order | The remesh shell had `0` geometric boundary edges. The old pre-simplify `fill_holes` call changed that to `77,358` and introduced overlapping/duplicate caps around coincident repair seams. | Skip both Metal `fill_holes` calls. Before and after simplification, remove duplicate faces, repair non-manifold edges, and remove small components. | Final geometry fell to `408` boundary edges, with no geometric duplicate faces and consistent winding. |
| Incorrect glTF transparency | Local material was `alphaMode=BLEND`, `doubleSided=true`; HF control was `OPAQUE`, `false`. Both textures contain alpha-zero texels, proving minimum alpha does not distinguish intentional transparency from UV padding. | Always export this baked surface as opaque and single-sided, matching the official control. Existing GLBs can be repaired by changing only their material JSON. | At the identical true-rear camera, the opaque copy shows the back of the head; the old blended copy shows eyes and mouth through it. |

## Geometry checkpoint evidence

Exact tensor checkpoints localized the topology damage rather than inferring it from a render.
Counts below use exact-position welding, so coincident split vertices do not masquerade as holes.

| Stage | Boundary/open edges | Finding |
|---|---:|---|
| Immediately after remesh | 0 | Dual contouring produced a geometrically closed shell. |
| After `repair_non_manifold_edges` | 0 | Vertex splitting preserved geometric closure. |
| After `remove_small_connected_components` | 0 | Still closed. |
| After old pre-simplify `fill_holes` | 77,358 | This call created the major damage. |
| Simplified without that fill | 3,407 | Simplification left smaller torn fragments. |
| Post-simplify cleanup, without fill | 408 | Better boundary count than the measured HF control's 690. |
| After UV/export duplication | 408 | UV/export did not introduce more openings. |

The final fixed Fox contains 484,826 faces, 408 boundary edges, 3,960 non-manifold edges,
zero geometric duplicate faces, and consistent winding. It is not formally watertight. The HF
control measured 690 boundary and 1,380 non-manifold edges, so the local result is better only
on boundary count—not on every topology metric.

## Production-scale comparison

All three subjects used the 1024 cascade and a 500,000-face export target. Generation time
includes sampling and creation of the cached raw decode. Corrected rebake time starts from that
cache and includes remesh, simplification, UVs, texture bake, and GLB export. These times are
not expected to scale linearly with the raw triangle count: active sparse-grid occupancy is the
dominant cost in the final remesh level.

| Subject | Raw decode triangles | Corrected remesh triangles | Exported triangles | Generation | Corrected rebake |
|---|---:|---:|---:|---:|---:|
| Moss Fox | 12,829,054 | 25,592,972 | 484,826 | 1,228.7 s (20m29s) | 412.3 s (6m52s) |
| Forest Variant | 4,956,432 | 9,276,400 | 487,087 | 471.2 s (7m51s) | 136.9 s (2m17s) |
| Snag | 27,876,032 | 41,738,080 | 465,726 | 2,788.4 s (46m28s) | 920.9 s (15m21s) |

Snag is the useful upper-end datapoint: its raw decode was 2.17 times the Fox's triangle count,
and its corrected rebake took 2.23 times as long. The 41.7M-triangle remesh is an intermediate
detail-preserving shell, not a shipping mesh; simplification reduces it by about 98.9% before
export.

## Final result

Use:

`output/final_cleanup_fix/fox_fixed_opaque.glb`

- SHA-256: `c14bd672abf19b9f3d98f6d646ae34ca2bb7bfe4c8417af06a7f028671ce816d`
- material: `alphaMode=OPAQUE`, `doubleSided=false`
- stored mesh: 530,469 vertices / 484,826 faces
- texture: 1024 x 1024

Do **not** use `output/final_cleanup_fix/fox_fixed.glb` as the result. It deliberately retains
the old blended/double-sided flags as the A/B control, although its geometry is corrected.
Also do not use `output/checkpoints/fox_checkpointed.glb`; it came from the older bad-fill run.

The browser A/B comparison is:

`http://127.0.0.1:8777/viewer/index.html?a=output/final_cleanup_fix/fox_fixed_opaque.glb&b=output/final_cleanup_fix/fox_fixed.glb&la=opaque%20fix&lb=old%20transparent`

At the same rear camera, the left model should show the solid back of the head while the right
model shows the misleading face-through-head result.

## Reproducibility

The runtime dependencies under `vendor/` are git-ignored. These checked-in patch scripts are
therefore the durable form of the fixes:

- `scripts/patch_mtlbvh_production_traversal.py`
- `scripts/rebuild_mtlbvh_metallib.sh`
- `scripts/rebuild_mtlbvh_native.sh`
- `scripts/measure_bvh_on_surface.py`
- `scripts/patch_ovoxel_skip_spurious_fill.py`
- `scripts/patch_ovoxel_opaque_material.py`
- `scripts/fix_glb_opaque_material.py`
- `scripts/patch_ovoxel_remesh_checkpoints.py`
- `scripts/mesh_checkpoint_forensics.py`
- `scripts/replay_cleanup_checkpoints.py`

After modifying or rebuilding MtlBVH's native extension, stop all Python/TRELLIS workers and
explicitly ad-hoc sign both `.so` copies. A linker-only signature passed verification but macOS
still killed the import with `CODESIGNING: Invalid Page`; `rebuild_mtlbvh_native.sh` performs
the required copy, signing, verification, and import check.

Apply and rebuild with:

```sh
vendor/trellis-mac/.venv/bin/python scripts/patch_mtlbvh_production_traversal.py
scripts/rebuild_mtlbvh_metallib.sh
scripts/rebuild_mtlbvh_native.sh
vendor/trellis-mac/.venv/bin/python scripts/patch_ovoxel_skip_spurious_fill.py
vendor/trellis-mac/.venv/bin/python scripts/patch_ovoxel_opaque_material.py
```

Rebake from the cached decode without running diffusion again:

```sh
vendor/trellis-mac/.venv/bin/python scripts/trellis_rebake.py \
  output/branch_test/fox_decode.pt output/final_cleanup_fix/fox_fixed_opaque.glb \
  --remesh --texture-size 1024
```

Or repair only the material flags in an already-generated GLB, with no GPU rebake:

```sh
.venv/bin/python scripts/fix_glb_opaque_material.py input.glb --output output.glb
```

The repository test suite passed with `432 passed` using:

```sh
.venv/bin/python -m pytest -q
```

## Generalization results

### Forest Variant

The Forest Variant was regenerated at its original seed (`261852270`) and the raw decode was
cached before post-processing. The corrected remesh produced 9,276,400 faces rather than the
old lattice, then simplified and exported successfully in 136.9 seconds.

Use `output/forest_validation/forest_fixed_opaque.glb` (SHA-256
`0a68ef9ae8ab47edda68e91d6a072539c8cde1ec522a2f6acb6f12284ecf89bd`). It has 487,087
exported faces, 7 boundary edges, 606 non-manifold edges, consistent winding, positive volume,
and an opaque single-sided material. Under backface culling, its rear and side views remain
solid and retain the source's thin blades. The previous same-seed GLB becomes almost entirely
dark/culled from the rear in the same viewer comparison. This is a successful generalization
test, not a Fox-only tuning result.

### Snag stress test

Snag was generated at seed `42`. Its cached decode is
`output/snag_validation/snag_decode.pt`; the unmodified corrected PBR export is
`output/snag_validation/snag_fixed_opaque.glb` (SHA-256
`a8bcaf6538e9df72265d3bc6225e164c7f49a3cc68633e8b5e5142c1b3ea9797`). It has 465,726
exported faces, 440 boundary edges, 4,257 non-manifold edges, consistent winding, positive
volume, and an opaque single-sided material.

The corrected result no longer becomes a hollow lattice under backface culling. Its front and
major coils remain coherent in the viewer and in five neutral-grey Blender views. However,
extreme rear/profile views reveal a small number of localized openings and underside fragments.
Recalculating normals makes those regions substantially worse and is not an acceptable repair.
This is therefore strong evidence that the production BVH/cleanup fix generalizes to a 27.9M-
triangle decode, but it is not evidence that every generated subject becomes formally or
visually perfect.

Snag also produced an outlier material field. Its baked base-colour luminance is `0.136`, versus
about `0.51` for both Hugging Face controls, and its metallic texture has median metalness
`1.0`, versus `0.004`–`0.384` for the controls. Fox and Forest match the control distributions,
so this is a per-generation material result rather than a common channel/export defect. The
reversible review copy `output/snag_validation/snag_fixed_review.glb` keeps the same geometry,
sets the organic body non-metallic, and applies a 2x linear-light albedo lift. It is a finishing
variant, not evidence that TRELLIS regenerated a better texture.

## What remains

Remaining work:

1. Turn the reproducible patch scripts into a standalone, credited Mac-port repository with
   pinned dependency commits, fixtures, regression tests, and before/after metrics.
2. Decide whether the small residual Snag openings warrant a separate, conservative final-mesh
   repair. Do not restore the unsafe pre-simplify `fill_holes` call or recalculate normals
   globally.
3. Make material outlier handling explicit: retain PBR when the generated map matches the
   controls, but flag near-unity metalness on organic subjects for review or a matte fallback.
4. Treat formal watertightness as a separate improvement unless it creates a visible acceptance
   failure; the HF controls themselves are not formally watertight.

No further Fox diffusion run is needed to establish this finding.
