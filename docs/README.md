# Docs

Design and reference documentation for the image-to-3D lab.

| Doc | What it covers |
|-----|----------------|
| [architecture.md](architecture.md) | How the CLI, backends, and provenance layer fit together |
| [rigging-plan.md](rigging-plan.md) | Plan for the "animatable" lane: rigging a generated model and making it walk |
| [fidelity-plan.md](fidelity-plan.md) | Eval of how to close the output-quality gap to hosted services (cascade modes, geometry gate, multi-view) |
| [fidelity-explained.md](fidelity-explained.md) | Teaching write-up: why fine detail looks garbled, and the base-mesh / materials / VFX three-layer model of a game character |
| [open-questions.md](open-questions.md) | Roadblocks we hit but do not yet understand, and the experiment that would settle each |
| [nikita-sidequest.md](nikita-sidequest.md) | Session log: T-pose human → turntable + rigged "cheers" animation, including the dead ends |

## What goes here

- **Design notes** — why a piece works the way it does (e.g. why Hunyuan is ComfyUI-only).
- **Reference** — schemas (run manifest, provenance sidecar), backend setup deep-dives.
- **Runbooks** — reproducible steps for a run or a bootstrap that outgrows the README.

Keep the top-level `README.md` as the quickstart; move anything longer-form here and
link to it.
