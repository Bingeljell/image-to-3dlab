# Documentation

**Baseline:** 2026-08-16 — the clean TRELLIS.2 port now produces GLBs end-to-end on Apple
Silicon (Lucian, controller and Flicker all baked). Old docs are preserved in
[`legacy/`](legacy/README.md); everything here describes the current state.

## Read this first

| Doc | For |
|---|---|
| [STATE-OF-REPO-2026-08-16.md](STATE-OF-REPO-2026-08-16.md) | **Start here.** The two ports, how to run image→GLB, measured timings, what's next |
| [MPS-BAKE-FIXES-2026-08-15.md](MPS-BAKE-FIXES-2026-08-15.md) | The full fix chain: bake path, venv relocation, env hygiene, instrumentation |
| [legacy/](legacy/) | Everything written before the 2026-08-16 baseline (kept for history) |

## Current docs

| Doc | For |
|---|---|
| `STATE-OF-REPO-2026-08-16.md` | Two ports, run recipes, timings, open threads |
| `MPS-BAKE-FIXES-2026-08-15.md` | Session fixes: pre-cap subprocess, verify/retry, CPU tensors, `--from-decode`, viewer wiring |

## Conventions

- **Conventional Commits**, one logical change per commit.
- **Keep a Changelog** in `CHANGELOG.md`; every user-facing change lands under `[Unreleased]`.
- **Test first.** Generation runs cost 15–80 minutes; unit tests are nearly free. Import the
  real module; never re-derive a copy.
- **Judge assets backface-culled, by eye** — glTF is double-sided, so a hollow mesh looks fine
  in preview and fails only in a game engine.
- **Measure holes with a position-only merge** (`merge_vertices(merge_tex=True, merge_norm=True)`);
  UV/normal seams otherwise read as fake holes.
