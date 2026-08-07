# What is pipeline, what is manual, what is just this fox

Written after taking the 3/4 moss fox from concept art to a rigged, trotting asset
(2026-08-07). The question this answers: of everything done to that asset, how much
generalises to the *next* character, and how much was one-off tuning?

Three buckets: **general** (works on any subject, belongs in the pipeline), **per-asset**
(a real step every character needs, but its value must be chosen by eye), and
**this-fox-only** (specific numbers with no reuse value).

## 1. Before generation — image preparation

| Step | Where | Bucket |
|---|---|---|
| Pre-matted RGBA input | `trellis_backend._prepare_rgba` | **General.** Required: the BRIA background remover is disabled as a licence guardrail, so inputs must already carry alpha. |
| Authoring multiple views | Manual, at concept-art time | **Per-asset.** The constraint is general — same pose, orbiting camera, never different poses — but the views themselves are drawn per character. 3/4 opposing views beat straight-on front/back, because a frontal view is depth-ambiguous for anything extending along the view axis. |

## 2. Generation — TRELLIS settings

| Setting | Bucket | Note |
|---|---|---|
| `multi_image_mode` fusion | **General** | `multidiffusion` averages per-view denoiser predictions each step. Restored from TRELLIS v1 by `scripts/patch_trellis_multiview.py`; v2 has no multi-image support at all. |
| `bake_target_faces` | **General**, with a per-asset value | The Metal path ignored this for the repo's entire history (fixed in `40aaf9f`). **Lower is better here**: 192k → 101k doubled texel density *and* defragmented the UV atlas. |
| `pipeline_type` | **Per-asset** | Does not transfer. `512` beat `1024` on the Nikita human and loses badly on the fox, where `1024_cascade` gives crisp leaves. Re-test per subject. |
| `texture_size` | **General** | Always max it. The ceiling is **2048**, enforced by the vendored `generate.py` argparse choices. |
| `seed` | Per-asset | Only matters for reproducibility. |

## 3. After generation — GLB post-processing, no Blender

All pure Python on the exported GLB, all re-runnable, all **general algorithms**:

| Step | Script | Bucket |
|---|---|---|
| Material normalisation | `trellis_backend` | **General.** Forces `alphaMode` OPAQUE; `matte` mode also drops metalness. Fixes the transparent/mirror-shard look. |
| Albedo colour match | `scripts/colour_match_albedo.py` | **General algorithm, per-asset strength.** Matches CIE LAB *chroma only*, leaving lightness alone — the concept art is lit and the albedo is not, so matching lightness would flatten the cream-vs-green structure. The fox needed `--strength 0.7`; 1.0 read as mustard. |
| Mesh health report | `scripts/mesh_health.py` | **General.** Merges by position only — the trap that produced several wrong diagnoses. |
| Hole filling | `scripts/fill_holes.py` | **General**, not used on this asset. |

## 4. Blender — rigging and animation

| Step | Script | Bucket |
|---|---|---|
| Spawn/read joint markers | `blender_joint_markers.py` | **General for quadrupeds.** Markers are pre-placed at fractions of the bounding box, so the template adapts to any similarly-proportioned subject. |
| **Placing the markers** | Manual, ~20 min | **Per-asset, and unavoidable.** This is the only genuinely manual step in the whole chain. |
| Marker validation | inline checks | **General.** Catches crossed sides, inverted chains, broken L/R convention — all errors hand placement reliably produces. |
| Build armature | `blender_build_rig.py` | **General for quadrupeds.** The 21-bone hierarchy is a body-plan template; a biped needs a different one. |
| **Voxel-proxy weighting** | `blender_voxel_weights.py` | **General, and the most reusable thing built here.** Generated meshes are never watertight, so bone heat weighting fails and leaves every weight zero. Remeshing a throwaway copy, weighting *that*, and transferring back solves the problem for any generated mesh, from any backend. |
| Gait cycles | `blender_walk_cycle.py` | **General for quadrupeds**, per-asset values. Trot and walk phase tables are anatomy, not tuning. |
| Ground planting | `--crouch` / auto-plant | **General.** Re-plants paws on the rest floor after folding the legs. |

## 5. Genuinely this-fox-only

- The 27 marker positions.
- `--front-fold-sign -1`. **Which rotation direction closes a joint depends on rest
  geometry.** Once the elbow sat behind the shoulder, positive rotation *straightened*
  the leg. Any new character needs this checked, not copied.
- Posture numbers (`--chest-drop`, `--crouch`, `--head-yaw -32`). The yaw exists only
  because this fox was generated with its head turned.
- `--strength 0.7` on the colour grade.

## What this means for the next character

Roughly **80% of the chain is reusable**. The genuinely manual work is:

1. Author 2+ orbiting views of one pose.
2. Test `pipeline_type` for this subject — it does not transfer.
3. Place ~27 markers in Blender (~20 min).
4. Choose a colour-grade strength by eye.
5. Check the fold sign per limb, and tune posture.

Everything else runs unchanged. Three scripts take a placed set of markers to a trotting
character in about a minute.

**The single most valuable general result is the voxel-proxy weighting.** It is not
fox-specific, not even TRELLIS-specific — it makes *any* non-watertight generated mesh
riggable, which was previously the hard blocker on this whole lane.

## Standing traps

- **Measure the mechanism, not the parameters.** Six rounds of tuning failed to make the
  front legs bend; one measurement of the joint angle showed the rest pose was collinear
  and no rotation could ever have fixed it.
- **`hide_viewport` is not "hide".** It removes an object from the viewport dependency
  graph, so hiding an armature that way silently freezes the mesh while renders keep
  animating. Use `hide_set()` (the eye icon) instead.
- **Building the rig leaves it unweighted.** `blender_build_rig.py` binds with plain heat
  weighting, which is *expected* to fail here. Always follow it with
  `blender_voxel_weights.py`.
- **Re-runnable scripts must anchor to absolute references.** Reading the current
  armature Z as a baseline made every regeneration sink the rig further.

## Known unfixed: the toe flex points the wrong way (2026-08-07)

With the heel pivot and toe segment in place, the trot reads as "a prissy princess fox
lifting up its toes" — the toes rise during push-off instead of pressing down. The heel
pivot itself is verified correct (rotating the paw bone 25 degrees drops the toe 0.036
against the heel's 0.012), so the geometry is right and only the *drive* is wrong.

**Most likely cause: an inverted sign, the same class of bug as `--front-fold-sign`.**
Which rotation direction closes a joint depends on rest geometry, and the toe bones were
added without checking. Test before tuning amplitude: rotate one toe bone +20 and -20 in
Pose Mode and measure which direction lowers the toe tip. `--toe` is already a flag, so
a negative value may be the whole fix.

Do not chase this with amplitude — that mistake cost six rounds on the leg fold.
