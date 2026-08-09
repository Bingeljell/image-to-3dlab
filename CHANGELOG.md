# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
