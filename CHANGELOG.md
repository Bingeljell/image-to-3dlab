# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Generate page now drives all three backends** (TRELLIS.2 clean port, SF3D, Hunyuan3D-MLX)
  instead of only TRELLIS. `viewer/generate_api.py` gained a `BackendSpec` registry
  (interpreter, wrapper, settings validation, progress parsing, and readiness are all
  per-backend) so the job runner, SSE progress stream, and setup-status check dispatch by
  backend id instead of being hardcoded to one. New `scripts/hunyuan_mlx_generate.py` chains
  dgrauet's shape stage, a `fast_simplification` remesh, and ZimengXiong's paint stage
  (bridging their two separate venvs) into one CLI, mirroring `trellis_space_generate.py`'s
  shape — the exact recipe (octree_resolution=512, ≤500k-face decimation) validated
  end-to-end on 2026-08-18. The paint stage's seed is now configurable (`PAINT_SEED` env var)
  instead of hardcoded. Job output folders are now named `<image>__<backend>__<timestamp>`
  (or a user-supplied label) instead of an opaque UUID, so a batch of generate-page runs
  stays legible without manual renaming. The ComfyUI Hunyuan path was evaluated and dropped —
  all three "Mac ComfyUI" candidates found still ship Tencent's unmodified CUDA-only
  `custom_rasterizer`, which cannot build on macOS at all.
- **`scripts/image_gallery.py`** — a thumbnail-grid gallery server that binds to a Tailscale
  IP so generated images in `output/` can be browsed from a phone. Responsive dark-theme
  grid with lazy loading, breadcrumbs, folder cards, image cards, and canonical trailing-
  slash redirects. Pure-function renderer (`render_listing`, `redirect_location`) extracted
  for unit testing (`tests/test_image_gallery.py`). Served live at
  `http://REDACTED-TAILSCALE-IP:8000/`.
- **`docs/IMAGE-GALLERY.md`** — documents the gallery server: purpose, the live instance,
  run/stop recipe, flags, design notes, and a directory of what it serves.
- **`scripts/mark_asset.py`** — an append-only register of human verdicts on generated
  assets (`output/verdicts.jsonl`). The provenance sidecar records source art, settings and
  output but never whether the result was any good, so a render that looked right could not
  be traced back to the GLB that produced it. Keyed by content hash because the interesting
  derived assets have no sidecar and get renamed; snapshots the forensic measurements beside
  the verdict so it accumulates into the fine-tuning dataset described in
  `docs/training-trellis.md`; and records whether the judgement was made backface-culled,
  since a double-sided verdict cannot distinguish a solid mesh from a hollow one.
- **`scripts/restore_pbr_material.py`** — re-attaches the metallicRoughness map that
  `--material-mode matte` orphaned, on assets already on disk. `matte` only ever rewrote the
  GLB's JSON chunk, so the 3072² map is still in every file we shipped, merely unreferenced;
  restoring it is a JSON edit rather than a regeneration. Also restores `metallicFactor: 1.0`
  (in glTF the factor multiplies the texture — restoring the map while leaving the factor at
  0.0 changes nothing) and turns `doubleSided` off to match the reference. Verified on the
  moss fox: the orphaned texture has an all-zero red channel and G/B distributions matching
  the Hugging Face reference's own MR map.
- **`scripts/glb_forensics.py`** — dumps what a GLB actually contains (PBR channels present,
  `doubleSided`, texture sizes, boundary vs non-manifold edges, winding, signed volume,
  edge-length CV) so a reference asset from a hosted demo can be diffed against ours
  instead of judged by eye. Reads the glTF JSON chunk directly rather than through a mesh
  library, because loaders normalise materials — the very thing being inspected.
- **`docs/hunyuan-eval-2026-08-13.md`** — Hunyuan3D evaluated from first principles, with
  every claim graded MEASURED / READ / CLAIMED. Records that upstream Hunyuan3D has been
  unmaintained since October 2025, that the "CUDA-blocked paint stage" was a build-system
  problem, and that a dev's April port notes are pinned to a diffusers release that is now
  two versions stale with the two most output-affecting fixes coupled to it.

### Changed
- **`viewer/` model panes now use image-based studio lighting instead of a 3-point rig.**
  The three directional lights (borrowed from `scripts/blender_stage.py`'s diagnostic rig)
  gave asymmetric, harshly-shadowed results that didn't match how Blender's Material
  Preview/LookDev viewport actually lights an object — that viewport uses a studio HDRI
  environment map, not lamps. Added `vendor/environments/RoomEnvironment.js` (ported from
  three.js's own example) and bake it per-renderer with `PMREMGenerator` into
  `scene.environment`, with only a faint hemisphere light left for ambient fill.
- **`--material-mode` now defaults to `pbr`, not `matte`.** `matte` discarded TRELLIS's
  metallic-roughness map and pinned the factors flat, so every organic asset shipped with no
  specular response under any light. Measuring our maps against the Hugging Face reference
  shows they match closely — moss fox roughness 0.765 / metallic 0.412 against the
  reference's 0.784 / 0.384, Flicker 0.396 / 0.000 against 0.404 / 0.004 — so TRELLIS was
  producing exactly what the reference implementation ships and we were deleting it on
  export. Judged backface-culled on Flicker: eye reflections return and the body gains
  surface variation. `matte` remains available.
- `restore_pbr_material.py --roughness-scale` multiplies the roughness map for subjects that
  still read duller than their source art.

### Fixed
- **Clean TRELLIS.2 port now produces GLBs end-to-end on Apple Silicon.** The decode→GLB
  bake was blocked by cumesh Metal simplify crashing on ~20M-face meshes. The clean-port
  wrapper (`vendor/upstream-audit-worktree/scripts/trellis_space_generate.py`) now CPU
  pre-caps the decoded mesh with `fast_simplification` — in a subprocess (the C extension
  crashes in any process that imported o_voxel's Metal/OpenCV deps), with verify-and-retry
  (its output is nondeterministically corrupt above ~20M input faces) and a post-filter for
  the residual corrupt indices that segfaulted mtlbvh's BVH build — hands `to_glb` CPU
  tensors, frees the 4B pipeline before the bake, and caches the decoded mesh so
  `--from-decode` re-bakes without the model or a re-decode. First assets: Lucian,
  controller, Flicker (Flicker geometrically matches the HF demo control). See
  `docs/MPS-BAKE-FIXES-2026-08-15.md`.
- **`viewer/serve.py` imports `generate_api` robustly.** The sibling import only resolved
  when run as a script; importlib-loaded by the test suite it broke collection.
- **The 200,000-face cap that was destroying 94% of every decode**
  (`scripts/patch_trellis_face_cap.py`). The Mac port pre-simplified the decoded mesh —
  ~3.2 million triangles — down to 200k with a crude decimator *before* o_voxel's
  postprocess, so hole filling, non-manifold repair, simplification, UV unwrapping and the
  texture bake all ran on wreckage. This was the cause of the crazed, cracked surfaces. It
  also made `bake_target_faces` **inert above 200k**: 300,000 and 3,000,000 both produced
  ~197k faces. Lifting it gives 290,662 faces, no crash, and no crazing. Re-apply after
  every bootstrap; `vendor/` is git-ignored.
- **Meshes shipped inside-out** (`scripts/fix_winding.py`, wired into the TRELLIS backend
  via `TrellisOptions.fix_winding`). Generated assets had inconsistent face winding and
  frequently negative signed volume (-0.02369 on Flicker). glTF materials are double-sided
  by default so previews looked fine, but backface-culled — as every game engine renders —
  the asset was hollow. This also means previously reported "see-through hole" and tear
  percentages were substantially counting flipped faces, not missing geometry.

### Changed
- **`docs/self-inflicted-damage.md` is the new entry point** for mesh-quality work. It
  documents both defects above, how the official TRELLIS.2 HuggingFace demo exposed them as
  a control group, and which earlier conclusions are withdrawn — notably "painted markings
  become geometry" (the demo carves no grooves from the same artwork) and "do not raise
  `bake_target_faces`". `docs/baseline.md` carries a banner: its method stands, its numbers
  were measured on damaged meshes.

### Added
- **A source-vs-render comparison** (`scripts/compare_to_source.py`). Renders an asset from
  a camera matched to its source image and lays out source / textured / culled grey /
  silhouette overlay. Every other metric here measures the mesh against itself and so
  cannot see a dead texture or a thin marking. Angles are fixed per subject: Flicker 130,
  Snag 95, fox 210. See `docs/baseline.md`.
- **`soften_markings.py --protect`** — a mask of regions to leave alone. Softening treats
  every dark region as flat paint, which is wrong for darkness that is *shading of real
  geometry*: lightening Flicker's ear hollows made the generator build a membrane that
  tore. With the ears protected, see-through holes fall from 2.67% to 1.07% of body area
  across eight angles, with no angle worse than baseline.
- **Marking projection** (`scripts/project_markings.py`). Paints the source's markings back
  onto a generated texture after they have been softened out of the conditioning image.
  Samples per texel from interpolated projected coordinates, derives the mask by comparing
  the two conditioning images, and transfers the marking as a ratio so the artwork's own
  lighting is not baked in.
- **Re-unwrap and retopology bakes** (`scripts/blender_reunwrap_bake.py`,
  `scripts/blender_retopo_bake.py`). Both work; both are negative results, kept so the
  measurements are not repeated.

### Changed
- **The tear metric is a diagnostic, not a gate.** It cannot see a dead texture, a missing
  sheen or a thin marking, and Flicker's score halved while the mesh visibly got worse.
  Judge with the four-panel comparison instead. `docs/finishing.md` carries a banner.
- **`docs/baseline.md` is the current state of Flicker, Snag and the fox**, all measured the
  same way on 2026-08-12, and supersedes older per-experiment notes where they disagree.

### Fixed
- **The Snag's flat eye was self-inflicted.** `material_mode: matte` strips metalness
  entirely, so a wet eyeball has no material to sit on; grading cannot restore it. Restore
  gloss with the existing eye mask first, then grade.
- **Most of the Snag's apparent tearing is flipped faces**, not missing geometry —
  Recalculate Outside removes nearly all of it.

### Added
- **A tear metric that gates post-processing** (`scripts/ribbon_metric.py`). The share of
  faces touching an open edge: 0% is closed, 1-3% is a surface with tears, and the
  thorn-knot Snag measures **40.9%** — a mesh of ribbons two or three triangles wide, which
  no repair can fix. Measured across the library to set the gate empirically: Flicker 3.1%
  (the asset judged best by eye), moss fox 14.7%, Snag 40.9%. **Necessary but not
  sufficient** — both SF3D assets score a perfect 0.0% and are unusable, so it may reject
  an asset but never accept one.
- **A finishing layer for generated assets** (`scripts/surface_detail.py`,
  `scripts/blender_bake_ao.py`, `scripts/feature_mask.py`). Generated assets arrive with no
  surface at all: TRELLIS emits one flat roughness for the whole subject. This derives a
  normal and roughness map from the albedo, bakes contact-scale ambient occlusion from the
  geometry, and can mask a single feature — an eye — to make it glossy. See
  `docs/finishing.md`.
- **Feature masking from a render** — locate a feature in a rendered image, raycast back
  onto the mesh, and rasterise the hit faces' UVs into a mask. Colour cannot do this: after
  grading, a hue threshold for the Snag's eye shatters into 155 fragments.
- `scripts/visibility_cull.py` — deletes faces never observed from outside. Works as
  specified and does **not** fix the shattering, for a reason worth keeping: anything
  visible in a render is by definition seen by the cull.
- `scripts/soften_markings.py` — reduces the contrast of flat painted markings in a
  conditioning image, protecting genuinely dark features such as eyes.
- `scripts/lift_lightness.py` — brightens an albedo without shifting hue or saturation.
  Prefer fixing the lighting; this is the second choice.
- `docs/finishing.md`, plus a "How to use this repo" section in the README covering all
  seven pipeline steps and a proposed schema-v2 `finishing` manifest block.
- `docs/second-opinion-snag-mesh.md` and `docs/handoff-to-worklings-coder.md` — two
  independent reviews of the shattered mesh and the evidence exchanged with them.

### Fixed
- The visibility cull no longer destroys the material. It had replaced the mesh's materials
  with the face-ID shader and exported without restoring them, shipping correct UVs and no
  albedo.
- `lift_lightness` scales linear RGB rather than LAB's L channel. The first version claimed
  to preserve hue and saturation while doing the opposite — holding a/b fixed while raising
  L desaturates, measured at 13% loss. A test caught it.

### Changed
- **The recipe is per-subject and human-judged.** Bark wants matte with strong derived
  relief; glazed ceramic wants gloss and *no* relief, because an albedo-derived normal map
  turns painted markings into dents. Ambient occlusion is the only step that transferred
  between subjects unchanged — it is measured from geometry, while normal and roughness are
  inferred from paint.
- **The albedo transform is not constant across subjects**, so there is no single global
  inverse. The Snag's highlights are crushed; Flicker's shadows are. Grade strength is
  chosen per subject off a rendered lineup.


### Fixed
- Documented that the see-through holes on detailed subjects are **zero-thickness
  sheets, not tears**, and that a Blender Solidify pass closes every one of them:
  pangolin 97.82 → 0.00, moss fox 126.58 → 0.00, monolith 44.78 → 0.00 (hole perimeter
  relative to the mesh diagonal). Costs roughly 4x the faces. This supersedes the
  art-direction rule recorded earlier the same day, which said detailed surfaces should
  be avoided; they need thickening, not avoiding. See `docs/open-questions.md` §1d.

### Added
- Provenance now records `software.pipeline_revision` — this repository's commit and
  whether the working tree was dirty — alongside the existing `backend_revision`. The
  patches that change TRELLIS's behaviour live in this repo, so the backend SHA alone
  never identified the code that produced an asset. Two runs with identical recorded
  parameters could behave differently with nothing in the sidecar to show it: the
  clockwork pangolin generated 2026-08-02 declares `bake_target_faces: 200000`, but the
  commit that made that value take effect on Metal landed five days later.
- Repository guide (`CLAUDE.md`) with layout, commit conventions, and changelog rules.
- `docs/` folder with an index and an architecture overview.
- This changelog.
- TRELLIS material normalization: exported GLBs are rewritten to render as an
  opaque, matte surface (`alphaMode` → `OPAQUE`, `metallicFactor` → 0, the
  metallic-roughness texture dropped), fixing the transparent/mirror-shard look
  while leaving geometry and the baked albedo untouched. On by default; opt out
  with `--trellis-raw-material` (or `"normalize_material": false` in a manifest).
  Recorded in provenance as `material_normalized`.
- TRELLIS material mode (`--trellis-material-mode {matte,pbr}`, or `"material_mode"`
  in a manifest). Both modes force `alphaMode` to `OPAQUE`; `matte` (default) also
  drops metalness for organic subjects, while `pbr` keeps the baked
  metallic-roughness so genuinely metallic subjects (brass, chrome) keep their
  sheen. Recorded in provenance as `material_mode`.
- Blender preview script `--env {dark,studio}` option. `dark` (default) is the
  near-black world that flatters matte assets; `studio` lifts the world and enables
  ray-traced reflections so metallic (`pbr`) assets preview with real sheen.

- `scripts/colour_match_albedo.py`, which grades a generated GLB's baked albedo
  toward the colour of its source concept art. TRELLIS renders the moss fox a cool
  grass green where the concept is a warm yellow-olive (the source leads red over
  green by +17, the bake by -20). The correction runs in CIE LAB and touches only
  the a/b chroma channels, leaving lightness alone, so hue moves without flattening
  the cream-versus-green structure — the concept art is lit and the albedo is not,
  so their lightness legitimately differs. `--strength` scales the correction.

- `scripts/remove_loose_parts.py`, which drops disconnected junk from a generated
  mesh while preserving UVs and the baked texture. On the hero fox it removes 688
  components totalling 5,604 faces, taking connected components from 226 to 3.
- `scripts/classify_thickness.py`, which measures local thickness by ray casting.
  Recorded as a **failed** approach to deriving solid-vs-foliage labels without a
  painted mask: on the moss fox the tail measures thicker than the legs, so no
  threshold separates them. Kept for the negative result.
- `scripts/blender_render_asset.py --culled` and `--recalc-normals`. `--culled`
  renders plain grey with backface culling — what SceneKit and RealityKit actually
  show, where a textured `doubleSided` render hides holes entirely.
- Brute-force UV packing, on by default. `scripts/patch_ovoxel_pack_options.py`
  teaches `o_voxel.postprocess.to_glb` to forward `xatlas_pack_charts_kwargs` (which
  `cumesh`'s `uv_unwrap` already accepted), and `scripts/patch_trellis_quality.py`
  adds `--uv-brute-force-packing` / `--no-uv-brute-force-packing`. Measured on the
  hero fox through the real Metal path, atlas coverage goes 52.90% → 58.76% for one
  extra second on 101k faces — no geometry change, no regeneration. The generator
  inspects `to_glb`'s signature first, so an unpatched or reinstalled `o_voxel`
  warns and packs the old way rather than failing at bake time.

### Fixed
- **Hole filling no longer destroys the texture.** `scripts/fill_holes.py` welded by
  position and exported the welded mesh, collapsing the vertices glTF splits at every
  UV seam, so its output had no UVs and no material. It now welds only to locate
  boundaries and appends patches against the original vertex indices, leaving existing
  geometry and the baked texture untouched. Its own boundary-edge report is also fixed;
  counting on raw indices measured UV islands, not geometry.
- `bake_target_faces` is now honoured on the Metal bake path. The patch that
  introduced the option only rewrote the CPU fallback's budget line, so every
  Metal-accelerated run (that is, every run on Apple Silicon) silently pinned the
  mesh at a hardcoded 200,000 faces and ignored the manifest. The 200,000 remains
  as a ceiling — it guards against an `mtlbvh` crash on large meshes — so only
  lower requests are honoured.
- Blender preview script (`scripts/blender_render_asset.py`) no longer double-rotates
  imported glTF assets (the importer already converts Y-up to Z-up), which had laid
  meshes face-down, and now clears the default startup Cube/Light/Camera so they
  cannot occlude the asset or hijack the active camera.
- Blender preview script now starts each render from a clean slate, removing every
  object and collection left in a long-lived session (by an earlier render or other
  tooling, regardless of naming) so nothing interpenetrates or occludes the new asset.

## [0.1.0] - 2026-08-02

### Added
- CLI (`pipeline.py`) with three backends: SF3D (`--fast`), Hunyuan3D via ComfyUI
  (`--quality`), and TRELLIS.2 (`--trellis`).
- Manifest-driven runs (schema v1) with pre-generation license policy validation.
- Provenance sidecars recording input/output hashes, license classification,
  component licenses, package versions, and backend revision.
- License-class output foldering and the TRELLIS BRIA-disable guardrail.
- Bootstrap and patch scripts for the vendored backends; Blender render helper.
- pytest coverage for provenance and the ComfyUI client.
