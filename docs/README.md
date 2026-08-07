# Docs

Design and reference documentation for the image-to-3D lab.

| Doc | What it covers |
|-----|----------------|
| [architecture.md](architecture.md) | How the CLI, backends, and provenance layer fit together |
| [rigging-plan.md](rigging-plan.md) | Plan for the "animatable" lane: rigging a generated model and making it walk |
| [fidelity-plan.md](fidelity-plan.md) | Eval of how to close the output-quality gap to hosted services (cascade modes, geometry gate, multi-view) |
| [fidelity-explained.md](fidelity-explained.md) | Teaching write-up: why fine detail looks garbled, and the base-mesh / materials / VFX three-layer model of a game character |
| [labelling-pipeline.md](labelling-pipeline.md) | Painted masks → part-aware meshes: how to tell a generated blob which bits are leaves, and the wind demo it unlocks |
| [subject-profiles.md](subject-profiles.md) | Proposal: per-subject-class defaults and optional stages, since settings measurably do not transfer between subjects |
| [open-questions.md](open-questions.md) | Roadblocks we hit but do not yet understand, and the experiment that would settle each |
| [nikita-sidequest.md](nikita-sidequest.md) | Session log: T-pose human → turntable + rigged "cheers" animation, including the dead ends |

## What goes here

- **Design notes** — why a piece works the way it does (e.g. why Hunyuan is ComfyUI-only).
- **Reference** — schemas (run manifest, provenance sidecar), backend setup deep-dives.
- **Runbooks** — reproducible steps for a run or a bootstrap that outgrows the README.

Keep the top-level `README.md` as the quickstart; move anything longer-form here and
link to it.
- [pipeline-vs-manual.md](pipeline-vs-manual.md) — what generalises to the next character, what is manual per asset, and what was only ever this fox
- [hunyuan-paint-plan.md](hunyuan-paint-plan.md) — Hunyuan paint on Apple Silicon: CPU-rasteriser fallback, disabled by default; degraded rather than impossible
- [texture-quality-roadmap.md](texture-quality-roadmap.md) — the path to Meshy-grade output: region splitting, normal maps, SD texture refinement, quad remesh
