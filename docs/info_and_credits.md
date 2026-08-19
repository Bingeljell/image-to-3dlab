# Info & Credits

Markdown counterpart to the Generate page's in-app "Credits & Info" tab
(`viewer/index.html`). The in-app version is the terse, always-current summary; this is
the place for the longer version — more context per pipeline, more room to explain the
tradeoffs. Draft as of 2026-08-18 — expand freely.

## Credits, by pipeline

This repo wraps other people's models and ports. It doesn't train or fine-tune anything
itself (yet — see the fine-tuning notes if that's changed).

### TRELLIS.2 (clean port)

- **Mac/Metal port foundation:** [pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple),
  plus Pedro Naugusto's `mtlbvh`, `mtldiffrast`, and `mtlgemm` Metal kernel libraries —
  the pieces that make TRELLIS.2 run on Apple Silicon at all.
- **Upstream model:** Microsoft [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
  (MIT license; the DINOv3 image encoder it depends on carries its own separate license —
  check before redistributing).
- **The original Mac port** ([shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac))
  is retired internally. It's kept only as the historical source of two self-inflicted bugs
  documented in `CLAUDE.md` (a 200k-face decode cap, and inconsistent mesh winding) —
  not as a build foundation for anything current.

### Stable Fast 3D

- [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d)
  (Stability AI Community License). The fast, lower-fidelity option — seconds, not minutes.

### Hunyuan3D-MLX — two variants, two different shape stages

Both variants share the same underlying model (Tencent Hunyuan3D-2.1, Tencent Hunyuan
Community License — **not licensed for use in the EU, UK, or South Korea**) but combine
different people's independent MLX ports of it:

**dgrauet shape + Xiong paint** — the original, more-tested path in this app.
- Shape stage: [dgrauet](https://github.com/dgrauet)'s MLX port
  (`dgrauet/hunyuan3d-2.1-mlx`). Evaluated as genuinely excellent — clean, watertight
  geometry, ~5 minutes.
- Paint stage: [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX)'s
  paint module. dgrauet's own paint stage produces a shattered, non-coherent UV atlas —
  that's why paint is sourced from a different repo entirely rather than staying
  single-author.
- Wired up via `scripts/hunyuan_mlx_generate.py`.

**Xiong, full pipeline** — both shape and paint from the same repo, one author, end to end.
- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) — `hy3d shape`
  and `hy3d paint` under a shared codebase (`python/shape/hy3dmlx` +
  `python/paint`), parity-tested against the original PyTorch reference and against a
  native Swift port in the same repo.
- Newer to this app. Independently verified clean (Blender Face Orientation overlay, no
  flipped winding) but **unbenchmarked for speed** — see Known shortcomings below.
- Wired up via `scripts/hunyuan_mlx_xiong_generate.py`.

### Evaluated, not shipped

- [RobertBeckebans/AI_trellis2cpp](https://github.com/RobertBeckebans/AI_trellis2cpp)
  (C++/ggml Metal port). A real upstream `purego` ARM64 bug was found and reported while
  testing it (mis-packed stack-spilled arguments on Apple's tight per-type ABI packing —
  matches `ebitengine/purego#352`/`#353`, fixed upstream in v0.10.0+).

## Known shortcomings

As of 2026-08-18. This list is honest-and-incomplete on purpose — update it as things
change rather than letting it go stale.

- **Hunyuan3D-MLX has two open defects** found on real assets: a paint-stage mouth
  artifact (diffusion-origin) and a shape-stage geometry fusion defect. Seen on the
  dgrauet+Xiong path; not yet checked against the Xiong-full path.
- **Hunyuan's paint stage has a hard face-count wall.** The `xatlas` UV-unwrap step goes
  from ~3 minutes to 37+ minutes between 500k and 700k faces; 1M faces never completed in
  testing. Keep `decimation_target` at or under 500,000. This applies to *both* Hunyuan
  variants — they share the same paint stage.
- **TRELLIS's own texture generation drifts from the reference image** — a published,
  documented weakness of the model itself, not this port. Hunyuan's paint tracks the
  reference image more closely, since it's directly conditioned on it and TRELLIS's isn't.
- **Hunyuan3D-MLX setup is entirely manual right now** — two or three separate repos,
  separate Python venvs, hand-downloaded weights (multiple GB each). No automated
  bootstrap exists yet, unlike TRELLIS's one-button setup on the Generate page.
- **Xiong, full pipeline is unbenchmarked at speed.** Its own shape stage defaults to
  full-precision inference on the 3.3B-parameter MoE 2.1 model — a real run at those
  settings took roughly 48 minutes just for the shape stage on the dev machine, versus
  ~5 minutes for dgrauet's shape stage. This app defaults `quantize=8` specifically to
  claw that back, but no timed run at that setting exists yet. Treat your first real run
  as the actual benchmark, not this note — and update this section once you have one.
- **The retired TRELLIS Mac port** shipped two self-inflicted bugs for a long time before
  they were caught: a 200,000-face cap that crushed every decode with a crude decimator,
  and inconsistent mesh winding that left assets hollow under backface culling. Full story
  in `CLAUDE.md`. Worth remembering as a cautionary tale even though that port is retired —
  a hollow, backwards-facing mesh looks completely fine in a double-sided glTF preview and
  only fails once something backface-culls it.

## Power-user notes

Peeking behind the curtain without the browser:

- A submitted job's folder appears in `output/` the instant you hit Generate, before any
  compute starts — `ls -lat output/ | head` confirms a job was accepted.
- `tail -f output/<folder>/run.log` streams the exact same progress lines the browser's
  live view shows.
- `ps aux | grep -E "trellis_space_generate|hunyuan_mlx_generate|hunyuan_mlx_xiong_generate|pipeline.py"`
  confirms the generation subprocess is alive and shows its exact arguments.
- `debug` (off by default) keeps only the final `.glb`. Check it to keep the manifest,
  textures, and intermediate meshes a run produces along the way.

## TODO (for tomorrow)

- [ ] Fill in real numbers once the Xiong-full pipeline gets its first timed run.
- [ ] Confirm whether the two open Hunyuan defects (mouth drift, eye/thorn fusion) also
      show up on the Xiong-full path, or are specific to dgrauet's shape stage.
- [ ] Anything else worth promoting out of `journal/` into `docs/progress/`.
