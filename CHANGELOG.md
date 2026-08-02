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

### Fixed
- Blender preview script (`scripts/blender_render_asset.py`) no longer double-rotates
  imported glTF assets (the importer already converts Y-up to Z-up), which had laid
  meshes face-down, and now clears the default startup Cube/Light/Camera so they
  cannot occlude the asset or hijack the active camera.
- Blender preview script now purges assets imported by earlier runs in the same
  long-lived session, so a leftover mesh no longer interpenetrates the new asset.

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
