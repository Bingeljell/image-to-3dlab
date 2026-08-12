# Docs

Design and reference documentation for the image-to-3D lab.

| Doc | What it covers |
|-----|----------------|
| [RESUME-HERE-2026-08-12.md](RESUME-HERE-2026-08-12.md) | **Start every session here.** Full state, what was tried and failed, the ordered plan, and the standing rules |
| [how-it-works-and-where-we-broke-it.md](how-it-works-and-where-we-broke-it.md) | **Plain-language walkthrough, no 3D knowledge assumed.** What each generation stage does, the post-processing the demo page doesn't show, and the two places our pipeline broke it |
| [upstream-contributions.md](upstream-contributions.md) | **Drafts, unsent.** Two bugs found in the Mac port: the 200k pre-simplification, and remeshing producing a lattice. Evidence, proposed fixes, pre-submission checklists |
| [self-inflicted-damage.md](self-inflicted-damage.md) | The same two defects for someone who already knows the pipeline: exact code, measurements, and the conclusions they invalidate. **Read first** Two defects our own pipeline added — a 200k face cap that destroyed 94% of the decode, and meshes shipped inside-out — and the list of earlier conclusions they invalidate |
| [baseline.md](baseline.md) | Was **start here**, but every figure in it was measured on damaged meshes; needs re-running. Where Flicker, Snag and the Fox actually stand as of 2026-08-12, all measured the same way. Supersedes older per-experiment notes where they disagree |
| [training-trellis.md](training-trellis.md) | Fine-tuning TRELLIS.2 on our own art — trainable components, hardware and data requirements, why it's parked, and what to start collecting now |
| [architecture.md](architecture.md) | How the CLI, backends, and provenance layer fit together |
| [finishing.md](finishing.md) | **The other six steps:** AO, normal/roughness, feature masks and gloss. Why the recipe is per-subject, and the traps |
| [rigging-plan.md](rigging-plan.md) | Plan for the "animatable" lane: rigging a generated model and making it walk |
| [fidelity-plan.md](fidelity-plan.md) | Eval of how to close the output-quality gap to hosted services (cascade modes, geometry gate, multi-view) |
| [fidelity-explained.md](fidelity-explained.md) | Teaching write-up: why fine detail looks garbled, and the base-mesh / materials / VFX three-layer model of a game character |
| [labelling-pipeline.md](labelling-pipeline.md) | Painted masks → part-aware meshes: how to tell a generated blob which bits are leaves, and the wind demo it unlocks |
| [subject-profiles.md](subject-profiles.md) | Proposal: per-subject-class defaults and optional stages, since settings measurably do not transfer between subjects |
| [decode-cleanup-disabled.md](decode-cleanup-disabled.md) | **Root defect:** the port stubs out `fill_holes`/`remove_faces`/`simplify`, so every asset was UV-unwrapped from an uncleaned mesh. Corrects three earlier docs |
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
- [trellis-prescribed-flow.md](trellis-prescribed-flow.md) — the upstream contract and how far our usage deviates: unused remesh and UV-clustering controls, and a texture_size cap that is only in the wrapper
