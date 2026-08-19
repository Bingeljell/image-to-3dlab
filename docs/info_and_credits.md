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

Both variants sit on Tencent's Hunyuan3D-2 model family (Tencent Hunyuan Community
License — **not licensed for use in the EU, UK, or South Korea**; verify exact terms per
model before any redistribution-sensitive use) but combine different people's independent
MLX ports of it. **Since 2026-08-19, they also differ in licensing at the code level, not
just weights** — see the licensing note below.

**dgrauet shape + Xiong paint.**
- Shape stage: [dgrauet](https://github.com/dgrauet)'s MLX port
  (`dgrauet/hunyuan3d-2.1-mlx`, vendored at `vendor/hunyuan-mlx`). Re-verified 2026-08-19
  in a direct A/B against Xiong's own 2.0 shape stage: still the cleanest shape we've
  tested — no dents, no dimples, 10/10 — which is why this path is kept despite the extra
  manual setup below.
- Paint stage: [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX)'s
  paint module, now tracked in-repo at `hunyuan_mlx/paint/` (see below). dgrauet's own
  paint stage produces a shattered, non-coherent UV atlas — that's why paint is sourced
  from a different repo entirely rather than staying single-author.
- **Licensing:** dgrauet's shape code carries Tencent's Community License, not a
  permissive one (all three of its `LICENSE` files are Tencent's own text) — the same
  territorial/use restriction that applies to the weights applies to the *code*, too.
  It stays manually vendor-cloned (`vendor/hunyuan-mlx`) rather than brought into this
  repo's tracked tree.
- Wired up via `scripts/hunyuan_mlx_generate.py`.

**Xiong, full pipeline** — both shape and paint from the same repo, one author, end to end.
- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) — `hy3d shape`
  and `hy3d paint` under a shared codebase, parity-tested against the original PyTorch
  reference and against a native Swift port in the same repo. **MIT licensed.**
- **Brought in-repo 2026-08-19**: the code (not weights) moved from
  `vendor/hunyuan-mlx-paint` into this repo's tracked tree at `hunyuan_mlx/shape/` and
  `hunyuan_mlx/paint/` (MIT notice preserved at `hunyuan_mlx/LICENSE`). A clone of this
  repo alone has the code that runs; only `weights/` (multi-GB, git-ignored) needs a
  separate download — `python hunyuan_mlx/download_weights.py` pulls them from Hugging
  Face. `uv sync` in each of `hunyuan_mlx/shape` and `hunyuan_mlx/paint` sets up the venvs.
- **Model choice, benchmarked 2026-08-19** (Flicker, octree=512, quantize=8, 30 steps,
  shape stage only): **2.0** ~167s, cleanest result, Xiong's own recommended pick and now
  this app's default; **2.0-turbo** ~60-105s but shows real distillation-noise dents even
  at 30 steps (its PCM schedule caps out at 100 steps — more steps helps, doesn't fully
  clear it); **2.1** ~450s with `--octree-decode` (~48min without) and not Xiong's
  recommended pick regardless (weaker DINOv2-large conditioner vs 2.0/2.0-turbo's
  DINOv2-giant). Full writeup: `docs/hunyuan-mlx-recipes.md`.
- Independently verified clean (Blender Face Orientation overlay, no flipped winding).
- Wired up via `scripts/hunyuan_mlx_xiong_generate.py`.

### Evaluated, not shipped

- [RobertBeckebans/AI_trellis2cpp](https://github.com/RobertBeckebans/AI_trellis2cpp)
  (C++/ggml Metal port). A real upstream `purego` ARM64 bug was found and reported while
  testing it (mis-packed stack-spilled arguments on Apple's tight per-type ABI packing —
  matches `ebitengine/purego#352`/`#353`, fixed upstream in v0.10.0+).

## Known shortcomings

As of 2026-08-19. This list is honest-and-incomplete on purpose — update it as things
change rather than letting it go stale.

- **Texture tear on concave geometry (inner thigh, armpit, ear folds) — fixed
  2026-08-19.** The paint stage filled texels no camera could see (self-occluded creases)
  by grabbing the nearest already-painted texel in flat 2D UV-atlas space —
  xatlas can and does pack unrelated 3D regions (an eye chart, a leg chart) next to each
  other on that flat sheet, so occluded creases got filled with the wrong, unrelated
  color. Root-caused by measuring true camera occlusion directly: 7.8% of surface texels
  had zero visibility from all 6 fixed views, clustered into ~7 localized regions (a real
  occlusion signature, not rasterizer noise). Fixed by filling occluded-but-in-chart
  texels from their nearest neighbor in actual **3D surface space** instead of 2D atlas
  space. Applies to *both* Hunyuan variants — they share the same paint stage. The
  shape-stage geometry fusion defect noted previously is a separate, still-open issue.
- **Hunyuan's paint stage has a hard face-count wall.** The `xatlas` UV-unwrap step goes
  from ~3 minutes to 37+ minutes between 500k and 700k faces; 1M faces never completed in
  testing. Keep `decimation_target` at or under 500,000. This applies to *both* Hunyuan
  variants — they share the same paint stage.
- **TRELLIS's own texture generation drifts from the reference image** — a published,
  documented weakness of the model itself, not this port. Hunyuan's paint tracks the
  reference image more closely, since it's directly conditioned on it and TRELLIS's isn't.
- **dgrauet's shape stage stays manually vendor-cloned** (`vendor/hunyuan-mlx`) — it's
  Tencent-licensed *code*, not just weights, so it isn't part of the clone-and-go
  simplification below. Xiong's shape+paint is MIT and tracked in-repo at `hunyuan_mlx/`:
  `uv sync` in `hunyuan_mlx/shape` and `hunyuan_mlx/paint`, then
  `python hunyuan_mlx/download_weights.py` (Hugging Face). RealESRGAN super-res weights
  aren't part of the official HF repos, so they're a separate step:
  `hunyuan_mlx/paint/scripts/convert_realesrgan.py` (needs torch, dev-time only) fetches
  the official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) release and
  converts it — bit-identical to what this repo had shipped without a documented source
  before 2026-08-19.
- **Xiong, full pipeline — benchmarked 2026-08-19** (Flicker, octree=512, quantize=8,
  30 steps, shape stage only): 2.0 ~167s (default, cleanest); 2.0-turbo ~60-105s (real
  distillation-noise dents even at 30 steps); 2.1 ~450s with `--octree-decode` (~48min
  without), not Xiong's recommended pick regardless (weaker DINOv2-large conditioner).
  Full writeup: `docs/hunyuan-mlx-recipes.md`.
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
