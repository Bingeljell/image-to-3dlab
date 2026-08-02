# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository guide (`CLAUDE.md`) with layout, commit conventions, and changelog rules.
- `docs/` folder with an index and an architecture overview.
- This changelog.

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
