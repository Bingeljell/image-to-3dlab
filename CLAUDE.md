# Repository Guide

Local Apple Silicon **image → 3D** pipeline wrapping three backends (SF3D, Hunyuan3D
via ComfyUI, TRELLIS.2) behind one CLI, with license provenance as a first-class concern.

## Layout

| Path | What lives here |
|------|-----------------|
| `pipeline.py` | CLI entry point (sets MPS env, delegates to `image_to_3dlab.cli`) |
| `image_to_3dlab/` | The package: `cli.py`, `provenance.py`, and one `*_backend.py` per backend |
| `manifests/` | Versioned run manifests (schema v1) — the preferred, traceable way to run |
| `scripts/` | Bootstrap + patch scripts, Blender render helper |
| `workflows/` | ComfyUI API-format workflow JSON for the Hunyuan `--quality` path |
| `tests/` | pytest suite (currently `provenance`, `comfyui_backend`) |
| `docs/` | Design and reference docs — see `docs/README.md` |
| `vendor/` | Vendored backend checkouts (git-ignored, bootstrapped locally) |
| `output/` | Generated assets + `.provenance.json` sidecars (git-ignored) |

## Commit conventions

Use **[Conventional Commits](https://www.conventionalcommits.org)**, one logical change per commit:

```
<type>: <imperative summary>

<optional body: what and why, wrapped ~72 cols>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `chore`.

- Keep commits small and self-contained — a commit should build and pass tests.
- Separate refactors from behavior changes.
- **No LLM co-author or attribution trailers.** Plain, human-authored messages only.

## Changelog

Follow **[Keep a Changelog](https://keepachangelog.com)** in `CHANGELOG.md`.

- Every user-facing change adds a line under `## [Unreleased]`, grouped by
  `Added` / `Changed` / `Fixed` / `Removed` / `Security`.
- On release, rename `[Unreleased]` to the version + date and open a fresh `[Unreleased]`.
- Versioning is **SemVer**. Provenance schema and manifest schema bumps are breaking.

## Development

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
ruff check .
python pipeline.py --help
```

- Target Python 3.10/3.11 (not 3.14 — SF3D's native deps don't fit yet).
- Backends that load real models are exercised manually; keep them out of the unit suite.
- Prefer running via a manifest (`--run-manifest manifests/...json`) so runs stay traceable.

## Testing (write the test first — one minute now saves fifteen)

**Every code addition gets a test.** Generation runs cost 15-20 minutes and Blender steps
are interactive, so a defect found by running the real thing is expensive; the same defect
found by a unit test is nearly free. A missing one-line import once cost a full 16-minute
generation run that crashed on its very last statement.

Three rules, each learned the hard way:

1. **Test the real artifact, never a re-derived copy.** A test that re-extracts code from
   a file, or re-implements the logic it is checking, tests something that is not what
   ships. One such test reported a failure that did not exist, costing more time than no
   test at all. Import the function; do not `exec` a copy of it.
2. **Extract logic so it can be imported.** Anything embedded in a string that is sent to
   Blender, or injected by a patch script, is unreachable by tests. Pull the pure parts
   (geometry maths, file writers, validation) into module-level functions and test those;
   leave only the thin `bpy` calls in the string.
3. **Verify the environment before the expensive step, not after.** Check that an operator
   exists, an import resolves, a flag is honoured — these take seconds. Discovering them
   after a 16-minute generation is a self-inflicted wound.

For patch scripts specifically: assert the anchor is present, assert re-running is
idempotent, and **never assume the host file's imports** — patched code must import what
it uses.

## Provenance & licensing (do not weaken)

- Every run emits a `.provenance.json` sidecar; outputs are sorted into license-class folders.
- `validate_run_policy` gates generation on declared intent — keep it ahead of model work.
- The TRELLIS backend **refuses to run** unless the BRIA-disable patch is present. This is a
  license guardrail; never bypass it. BRIA RMBG-2.0 must stay unloaded.
