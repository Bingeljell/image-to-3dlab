# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The project is now licensed under Apache-2.0, with a `NOTICE` file asking forks and
  derived projects to credit image-to-3dlab. Model weights keep their own licences.

## [0.3.1] - 2026-09-24

### Fixed
- Windows: the viewer no longer crashes on start after an interrupted job, and stopping
  a job now works there too (it used a Mac/Linux-only call).
- Pixal3D no longer tells you to "enable rembg" for images without a cut-out. It removes
  the background itself, and the badge now says so.
- Setup: the download progress bar now appears under the model you clicked, not at the
  bottom of the page.

## [0.3.0] - 2026-09-24

**Now runs on NVIDIA Linux.** Text to image to 3D works on a Linux machine with an NVIDIA
card, as well as on Apple Silicon: images via stable-diffusion.cpp's Vulkan build, 3D via
Pixal3D and Stable Fast 3D on CUDA. Tested end to end on an RTX 4090 and an RTX 3090 Ti.
One line installs it, and the same line updates it. Windows is wired up but untested.

### Added
- **Text to image to 3D on NVIDIA (Linux, and Windows untested).** The Generate Image
  tab and the Pixal3D route now run on a Linux or Windows machine with an NVIDIA card as
  well as on Apple Silicon. Nothing was ported: both upstreams publish NVIDIA builds, and
  the installers now choose the right one for the machine. The CUDA runtime is bundled,
  so there is no CUDA toolkit to install.
- **`scripts/bootstrap_pixal3d.py`** replaces `bootstrap_pixal3d_cpp.sh`. Macs still build
  from source with Metal. On Linux with the CUDA toolkit it compiles for the card, which
  ran about twice as fast as upstream's prebuilt on a 4090 (unless the toolkit is newer
  than the driver). Otherwise NVIDIA machines get the prebuilt CUDA 12 build when the
  driver is 575 or newer, or it says which driver to install. `--prebuilt` picks the
  prebuilt anyway, for comparing the two. It names
  the route, the size and the licence and asks before downloading, which the shell
  script never did.
- **Stable Fast 3D on NVIDIA Linux, and `scripts/bootstrap_sf3d.py`.** SF3D now runs on
  CUDA when there is a card. The new installer builds its texture baker with CUDA when
  the CUDA toolkit matches PyTorch, with its CPU kernel when it does not, and with Metal
  on a Mac. It also fetches DINOv2 (1.1 GB) up front, which SF3D used to download
  unannounced on its first run, and explains the gated licence step if Hugging Face refuses.
  Setup & Status can now install SF3D by itself.
- **The viewer says when a new version is out.** While it runs, it asks GitHub at most once
  a day for the newest release (nothing about you is sent) and shows a slim banner with the
  exact line that updates this install. Offline, nothing happens. Dismiss it until the next
  release, turn it off on the About page, or set `I3D_NO_UPDATE_CHECK=1`.
- **A one-line installer, which is also the updater.** `install.sh` (Mac and Linux) and
  `install.ps1` (Windows, untested) check the machine, install the code and Python 3.11,
  and print how to start the viewer. Re-running moves to the newest release and refuses to
  overwrite local edits. They download no model weights. `--yes` and `--dry-run` are
  there for scripts and agents.
- **An About page, which also announces updates.** `Credits & Info` is now `About`. Its top
  says hello, names your machine and lists the routes that run on it, with what changed
  in the latest release read from this changelog. The viewer lands there once on a first
  visit and once after each update. The name in the top bar opens it too, and lives in
  `viewer/brand.json`.
- **No silent CPU runs on NVIDIA.** If stable-diffusion.cpp cannot reach the GPU it quietly
  runs on the CPU instead, which takes many minutes per picture. The installer now checks
  for the GPU before downloading weights, and the viewer stops a CPU-only image job at
  once. Both say what fixes it (on a headless Linux box, `apt install libegl1 libgl1`).

### Fixed
- `image_to_3dlab.__version__` said 0.1.0 through the 0.2.0 release. A test now keeps
  it in step with this changelog.
- **SF3D models came out inside a grey slab** when the input came from Generate Image.
  Those images have a transparency channel that is really just noise, and SF3D took
  that as "already cut out", so the background became geometry. SF3D now checks
  whether anything is actually transparent, and cuts the background out if not.
- SF3D on NVIDIA Linux failed at the texture step when the CUDA toolkit did not match
  PyTorch. The baker falls back to its CPU kernel there, but SF3D still handed it GPU
  data. It now bakes on the CPU and hands the result back to the GPU.
- Generate Image read sd-cli's output in 256-byte blocks, so short lines could sit unseen
  until more arrived. It now reads whatever is there.
- Setup & Status said a finished download was a few bytes ("done · 120 B") with newer
  huggingface_hub, which stores the files outside each model's folder and links to them.
  Sizes now follow the links and count each file once.
- Setup & Status showed "0 B of 8.4 GB" while Pixal3D fetched its 674 MB build, and on a
  slow line would have called it stalled. Until weights arrive it now shows the current
  step, and only claims a stall once they have started.
- Pixal3D's CUDA compile ran one job per CPU with no limit, and on a 96-CPU machine
  with 31 GB of memory the compilers were killed for running out of it. It now caps the
  job count by memory as well, and reads a container's real limits, not the host's.

### Removed
- `scripts/bootstrap_pixal3d_cpp.sh`, superseded by `scripts/bootstrap_pixal3d.py`.

## [0.2.0] - 2026-09-23

### Added
- **A `Generate Image` tab, and `Generate` is now `Generate 3D`.** The pipeline assumed you
  already had a picture; this is the step before that. Type a prompt, get an image, hand it
  to Generate 3D. It runs Qwen-Image 2.1 locally through `stable-diffusion.cpp` on Metal.
  Defaults are the measured fast ones rather than the upstream recipe: the model does not
  use guidance, so `--cfg-scale 6.0` was doing two passes per step for nothing.
- **`scripts/bootstrap_qwen_image.py`** — installs that route: a prebuilt
  `stable-diffusion.cpp` binary and 13.4 GB of weights. It names the backend, the route,
  the size and the licence, then stops until you agree. Setup & Status lists it alongside
  the 3D backends and its download button runs this.
- **A `research-only` licence class in `provenance.py`.** Qwen-Image is non-commercial, and
  unlike the existing classes that permits no commercial use at all. `validate_run_policy`
  allows it only for a private showcase, and `allow_conditional` is not consent to it. The
  restriction sits at the front of the chain, so a mesh made from a generated picture
  inherits it.
- **`scripts/blender_agent.py`** — lets a local model build and light a scene in Blender by
  calling a fixed set of verbs rather than writing Blender Python, which small models get
  wrong. Each round that changes the scene is rendered and the picture handed back, so the
  model sees its own work.
- **A visual README.** The page now opens on three source images above the models
  generated from them, and carries a 360° turntable of one of them — a still cannot show
  that a result is a real model rather than a flattering angle. Also `social-preview.jpg`,
  the 1280x640 card for when the repo is linked on X, Slack or Discord (upload it under
  *Settings → General → Social preview*; GitHub cannot take it from the tree). 543 KB
  added in total, against a documented budget in `docs/images/README.md`.
- **`pixal3d_generate.py --gss/--gsh`** — guidance strength is now reachable from the
  wrapper, and `--gss` defaults to 10 rather than leaving `trellis-cli` on 7.5. That
  default is what dropped the warrior girl's sword blade entirely; 10 recovers it.
- **`scripts/retopo_repaint.py --resume`** — reuses any stage artifact already sitting
  beside the output instead of recomputing it, so a run that died in compression is not
  charged for the five-minute repaint a second time (measured: 8s → 0.1s on a
  retopologise+compress run). A zero-byte artifact counts as a stage that died mid-write
  rather than one that finished, so the stage that actually failed is the one re-run.
- **Pixal3D as a backend** — `scripts/bootstrap_pixal3d_cpp.sh` builds the C++/GGML runtime
  (raven38/pixal3d.cpp) with Metal and fetches the 8.1 GB single-view Q8_0 weights;
  `scripts/pixal3d_generate.py` runs image → textured GLB; the viewer offers it alongside
  the other backends. One pass, no repaint stage, 5m50s on the moss fox.
- **`scripts/export_decode_highpoly.py`** — turns a cached decode into the high-poly PLY a
  normal bake reads (19,172,397 faces on the Snag, 98% of which `generate.py` discards).
  Welds and repairs winding per component; deliberately does not decimate, because
  fast_simplification shatters these meshes (1,072 components to 215,842).
- **`scripts/compress_glb_textures.py`** — re-encodes a GLB's textures in place, leaving
  geometry untouched. The paint stage writes a 4096² albedo and a 4096² metallic-roughness
  map as uncompressed PNG: 30.5 MB of a 32 MB asset. Snag 32.0 → 4.8 MB, fox 31.3 → 4.4 MB,
  a difference measuring below the renderer's own sampling noise (5.48/255 against 6.00 for
  the same file rendered twice). Default is core-glTF JPEG at 2048; `--format webp` is
  smaller where the destination handles the extension.
- **Finishing runs are recoverable from the browser** — the Finish panel lists every run
  under `output/finish/`, with a download link for one that completed and a Resume button
  for one that stopped part-way. The job registry lives in the server's memory and dies
  with it; the run directories do not, so a run whose browser tab was closed is no longer
  lost. Each directory now holds a `settings.json` written before the first stage, which
  is what makes a resume faithful; one without it is listed but not offered for resume.
- **Real progress during the repaint** — the paint stage's own `step 7/15` output is
  parsed into the progress panel, with a per-stage and overall ETA extrapolated from
  measured pace rather than a constant. The overall bar is weighted by measured stage
  cost (retopology 5%, repaint 92%, compress 3%), so it moves continuously through the
  five minutes that used to show one unchanging row.
- **Finishing jobs in the browser** (`POST /api/finish`) — retopologise, repaint and
  compress an asset the viewer already has, with SSE progress and the result and record
  fetched by URL. Refuses to start while a generation is running; every setting is
  bounds-checked before it reaches a subprocess argument.
- **`scripts/retopo_repaint.py`** — runs retopologise → repaint → compress as one command,
  keeping every intermediate and writing a JSON record of the settings used, so assets
  finished in a batch are comparable. Emits `I2L_STAGE::` progress lines.

### Changed
- **Generation timings no longer read as hardware-neutral advice.** "10 steps is enough"
  and "1024 is slowest" were written on one Mac; on a fast GPU the extra step costs under
  a second and the advice inverts. The hints now give the mechanism rather than the
  verdict, and the "timings are from one machine, share yours" line is on the Generate 3D
  tab as well as Generate Image — it was only on the faster of the two steps.

### Fixed
- **A generated image's background came back as geometry.** Pixal3D was handed pictures
  that had never been cut out, so it reconstructed the backdrop as mesh: the flat grey
  studio background behind a low-poly fox returned as two enormous white sheets either
  side of its head, welded to the model. Two faults, one behind the other. `has_alpha()`
  decided "already matted" from the file's *mode*, and Qwen-Image writes RGBA whose alpha
  is opaque noise (219-255, nothing transparent, corners included) -- so background
  removal was skipped on every image the new Generate Image tab produces. Behind that, the
  un-matted branch built a command line `trellis-cli` rejects outright (`--pixal3d-weights`
  requires `--sv-image`), which had never been noticed because the first fault meant it had
  never once run. `has_alpha` now inspects the channel rather than the mode, and the
  subject is cut out with u2net before generation, with `--matte` / `--no-matte` to
  override. Asking `trellis-cli` to matte instead was measured and does nothing: it makes
  the same mode-based mistake, and the mesh came back with identical extents and face
  count.
- **Tests and the code they test now share one copy of each module.** `viewer/` and
  `scripts/` are not on Python's path, so test files hand-loaded their modules with
  `importlib` — and every hand-load makes a *new* copy. Where production code imported the
  same name, the test and the code under test held two different objects, so a
  monkeypatch reached nothing. `tests/conftest.py` puts both directories on the path, the
  two exposed files (`download_api`, `backend_catalog`) use a plain `import`, and
  `tests/test_no_module_forks.py` fails if a test ever hand-loads a name production code
  imports. Files whose module nobody else imports are untouched: a private copy cannot
  diverge from anyone.
- **One catalogue of backends, not two.** The Generate tab kept its own list and the
  Setup & Status page kept another, and they had drifted: Stable Fast 3D and the dgrauet
  Hunyuan route were missing from the catalogue entirely, the Xiong route was spelled two
  ways, and asking the readiness endpoint about the image route answered "unknown
  backend" for something the other page lists. `backend_catalog.py` is now the single
  registry every route is looked up in, and a backend that has a live probe of its own
  layers that detail on top rather than replacing it. Routes the viewer cannot install for
  you (SF3D's shell bootstrap, the Tencent-licensed Hunyuan shape clone) say so and print
  the command, instead of offering a button that throws on click.
- **A test that had stopped testing anything.** `tests/test_backend_catalog.py` loaded a
  second copy of the catalogue module under the same name, so a monkeypatch in one test
  file landed on a different object than the code under test held. The "an unsupported
  machine is refused before anything downloads" guard silently passed without refusing
  anything, visible only when the two files ran in a non-alphabetical order.
- **The Setup & Status page now says when a backend cannot run on your machine**, instead
  of offering a Set up button that failed with a raw `[WinError 2] The system cannot find
  the file specified` (reported from Windows, after a successful Hugging Face login).
  Support is declared per backend, so the NVIDIA routes will switch themselves on by
  naming the platform rather than by unpicking a macOS check. The viewer also refuses such
  a download server-side, so nothing is fetched before the failure.
- **Virtual environment paths are resolved per platform** (`.venv/bin/python` against
  `.venv\Scripts\python.exe`), and a command that is missing is now explained — "run
  `uv sync` in ..." — rather than surfacing the operating system's error number.
- **`pixal3d_generate.py` could not run on a relative path.** `trellis-cli` is launched from
  its own tree so it can find its Metal library, so a relative input or output resolved
  against *that* directory and the run died immediately with `can't fopen`. Both paths are
  resolved before the command is built.
- **A finished Finish run left the browser with no GLB.** The worker's own
  `I2L_STAGE::done` marker reached the browser as a `phase: "done"` event carrying no
  artifact URLs. The page treated it as the job's completion, closed its event stream on
  it, and so never received the real completion event — the only one with `result_url` —
  leaving a download link pointing at `undefined` while the finished asset sat complete
  in `output/finish/`. The worker's marker is now swallowed: only the job API, the one
  thing that knows the URLs, may end a run.
- **The last stage in a progress panel never ticked.** A stage was only marked done when
  a *later* stage started, and the last stage has none, so it sat on "estimating…" for
  good. A terminal event now completes the whole list (and marks the running stage failed
  on an error). Affected Generate as well as Finish.
- **The Finish progress bar never filled.** Its fill was styled by `#generate-overall-bar`
  alone, an id the Finish panel's identically structured bar does not have; the rule is
  now `.progress-track > div`.
- **Stages with no sub-progress claimed to be estimating.** Retopology and compression
  report no percentage of their own, and the panel rendered that as `· ~estimating…`,
  which reads as a stall. They now read `running`.
- **Normal bakes came out as rainbow confetti, and now do not.** The decode is non-manifold,
  so winding repair cannot converge and roughly half the rays returned the hit normal
  reversed — 48.9% of hits more than 90° from the low-poly normal, upper quartile 164°. A
  tangent-space normal cannot point into the surface, so `blender_bake_normals.py` negates
  any texel that does (`--keep-sign` opts out) and reports the fraction, which reads as the
  source's winding quality. Negative-Z texels went from 48.1% to 0.1%.
- `blender_bake_normals.py` no longer ties `max_ray_distance` to the cage extrusion, shade-
  smooths the bake source, and takes `--device`.
- **Retopology face targets were silently doubled.** The Decimate ratio was computed against
  `len(mesh.polygons)`, but COLLAPSE decimation applies its ratio to *triangles* and the
  voxel remesh before it emits *quads*. Asking for 40,000 faces produced 79,991; asking for
  20,000 produced 39,361. Now computed against the triangle count: 40,000 lands at 39,987.
  Face counts recorded before this fix are roughly twice what was requested.
- `blender_retopo_bake.py` welds by position before doing anything else, and accepts a voxel
  fraction of `0` to skip the remesh. A glTF mesh arrives split along every UV seam and
  measures as broken until welded — the shipped Snag reads 237,359 non-manifold edges as
  loaded and 2,671 welded, the same file.

### Removed
- **The repository was slimmed from 244 MB to 8.5 MB and its history rewritten.** This is
  an image → 3D pipeline people clone and run, and 96% of what it carried was not that:
  a promo-video project with its own brief, bug notes, audio masters and renders; 90 MB of
  backend-comparison meshes; screenshots from working sessions; per-creature rigs and
  animations; and 49 run manifests pointing at source art the repository does not contain,
  so none of them could run for anyone else.

  All of it still exists for us — `videos/`, `assets_to_test/`, `manifests/*.json` and
  `characters/<name>/` are git-ignored rather than deleted. Per-creature rigs and
  animations moved into `characters/<name>/` alongside their own tests, so `scripts/` and
  `tests/` now describe the pipeline and nothing else.

  **Two consequences worth knowing.** The history was rewritten with `git-filter-repo` and
  force-pushed, so anyone holding an existing clone or fork needs a fresh clone — an old
  copy cannot be reconciled with the rewritten history. And **entries further down this
  file reference files that are no longer in the repository**; they were accurate when
  written, and nothing in the pipeline depends on them.

### Fixed
- **`blender_retopo_bake.py` silently shipped meshes it had not retopologised.** Blender's
  QuadriFlow declines with a *warning* rather than an exception when it will not run, so the
  `try/except` never fired, the operator left the mesh untouched, and the script wrote the
  intermediate voxel mesh while printing an "out faces" line as though it had worked. On one
  asset that produced 1,324,656 triangles from a request for 20,000 -- 4.7x **larger** than
  the input. It now verifies the face count actually moved, falls back to collapse
  decimation, and fails loudly if nothing reduced the mesh.

  QuadriFlow refuses every mesh this pipeline produces, and its error message is a red
  herring. Ruled out by measurement: manifoldness (zero non-manifold edges and vertices
  after the voxel remesh), inconsistent normals, component count (refused on a single
  separated component), and size (refused at 58k faces). It accepts a primitive sphere.
- Ignore `hunyuan_mlx/*/outputs/`. The paint stage writes its results beside its own code,
  64 MB on the first run, into a repository deliberately cut to 8.5 MB.
- Expose the reduced mesh's surface response as arguments to `blender_retopo_bake.py`
  (`metallic`, `roughness`, `ior`). Only base colour is baked, so the source's
  metallic-roughness *map* does not survive and the material would otherwise ship
  mathematically flat, which reads as dead plastic under any light. The right values depend
  on whether the artwork is wet bark, dry stone or painted metal, so they are tuned per
  asset rather than fixed; the defaults are a neutral organic surface.
- Add `--clay` to `scripts/render_glb_comparison.py`, which strips materials and renders
  neutral grey. Texture and geometry fail in different ways, and a broken UV map makes a
  sound mesh look ruined, so a shape comparison has to take the paint off.

### Fixed
- **Filter degenerate decode faces on the CPU instead of on Metal.** Boolean-mask indexing
  a multi-million-row tensor on MPS returned a garbage index -- observed as
  `index -1097849984 is out of bounds: 0, range 0 to 7419814` while dropping 426 bad faces
  from a 7.4M-face decode. Metal work is queued, so the fault surfaced later at the first
  synchronisation and killed a run whose sampling had already finished. The gather is cheap
  at this size and the result is identical. The fault has not reproduced, so this is a
  precaution against the most likely trigger rather than a confirmed fix.

### Added
- Add an `mlx` sparse-attention backend for TRELLIS.2 on Apple Silicon
  (`scripts/patch_trellis_mlx_attention.py` plus `image_to_3dlab/mlx_attention.py`),
  selected with `--sparse-attn-backend mlx`.

  TRELLIS.2-4B has a head dimension of 128. PyTorch's MPS backend has no fused attention
  kernel, and the vendored Metal kernel supports head dimensions only through 64, so the
  model falls back to unfused SDPA. Measured at the real Stage-3 shape (9,801 tokens,
  12 heads, head dim 128), **attention is 93.2% of sampling time**.

  On a full 1024-cascade Storm Ram run against a recorded baseline with identical seed and
  parameters, sampling went from **1760.7s to 963.6s (1.83x) at fp32**, and to **552.2s
  (3.19x) at fp16**. Two incidental findings: fp16 on torch MPS SDPA is *slower* than fp32,
  so half precision is not a lever on the old path; and the MLX-vs-torch fp32 difference is
  ~1e-3 on unit-scale inputs, arising from MLX's arithmetic on Metal rather than from the
  fused kernel.

  The change is additive. The existing `sdpa` path is untouched and remains the default,
  and the new backend defaults to fp32 so it changes speed without changing precision.
  `I2L_MLX_ATTN_DTYPE=fp16` opts into the faster, lower-precision path.

  Requires `mlx` in the vendor venv:
  `uv pip install --python vendor/trellis-space-mac/.venv/bin/python mlx`.

### Added
- Expose the sparse-attention backend in the web UI's TRELLIS panel
  (`sparse_attn_backend`, validated to `sdpa` or `mlx`, passed straight through to the
  wrapper). It defaults to `sdpa` because `mlx` needs
  `scripts/patch_trellis_mlx_attention.py` applied to the vendored checkout and mlx
  installed in its venv, and a default that fails on a fresh clone is worse than one that
  is merely slower.

### Added
- Offer attention precision as part of the web UI's backend choice: `sdpa`, `mlx`
  (fp32) or `mlx-fp16`. Precision is a property of the fused MLX kernel, so it belongs
  to the same control rather than a second one -- fp16 on the stock `sdpa` path is
  measurably *slower* than fp32 and would be a meaningless combination to offer. The
  chosen precision is pinned into the job's environment rather than inherited, so two
  runs that look identical in the UI cannot compute different things.

### Added
- Surface the MLX attention backend in the browser: a readiness check
  (`mlx_attention_status`) reports whether the vendored checkout carries the dispatch
  branch and whether mlx is installed in its venv, the Setup card names whichever step is
  missing, and the backend's options are disabled until both are satisfied rather than
  left to crash a run partway through. Readiness is advisory: the default `sdpa` path
  needs none of it, so an unready MLX never blocks generation.
- Document the attention backends in the in-app Credits & Info tab and
  `docs/info_and_credits.md`, including Apple MLX's attribution and a Speed section giving
  the measured 34.3 / 22.4 / 14.3 minute comparison and its caveats.

### Fixed
- Record the attention precision in the run manifest (`sparse_attn_dtype`). It travels by
  environment variable rather than by flag, so two runs that computed different things
  produced identical provenance records and a comparison made later could not be
  interpreted. It is `null` for non-MLX backends rather than a default, because a
  plausible-looking value would be a lie in a provenance record.

### Added
- Add `scripts/render_glb_comparison.py`: render several GLBs from one fixed camera,
  headlessly, and lay them out as a single comparison image for documentation. It never
  touches a running Blender session, and it crops every panel with one shared box, because
  per-panel crops rescale subjects independently and manufacture differences between assets
  that are actually identical. Also adds `docs/images/` with a size and naming convention,
  since this repository is deliberately slim.

### Fixed
- Correct the attention-backend claims in the web UI and the Credits & Info tab. They
  advertised a fixed speedup, which is true at 1024 and false at 512: attention cost grows
  with the square of the token count, and at 512 the stock path is already fast enough that
  the fused kernel's advantage is cancelled by the cost of moving tensors into MLX and back
  (measured 176s against 184s, inside noise). The guidance now says where the option is
  worth choosing, and records that the choice does not change the output.

### Fixed
- Release MLX's reserved Metal memory at the sampling/decode boundary, and report how much
  was held. Three 1024 runs failed during decode with MLX resident in the process, each
  with a different symptom -- a garbage negative index, an out-of-range hashmap lookup, and
  a sparse tensor size mismatch. Varied corruption-shaped failures fit memory pressure
  better than one logic bug, and re-decoding the same cached latents in a fresh process has
  succeeded every time. MLX keeps a buffer cache separate from torch's, which stays claimed
  after sampling while decode -- the most memory-hungry stage -- runs without that headroom.
  **This is a hypothesis under test rather than a confirmed fix**, which is why it prints
  what it released instead of acting silently.
- Gate the MLX attention tests per test rather than at module level. A module-level skip
  was silently disabling the pure packing-maths tests in any interpreter without torch and
  mlx, which is the one the suite normally runs under: ten tests reported as skipped when
  six of them needed neither library.

### Added
- Document a second, independent colour effect in TRELLIS.2: at `1024_cascade` an asset can
  come out markedly desaturated compared with the same asset at `512`, and the effect
  follows the pipeline type rather than the seed (two seeds per setting, clean split). Each
  pipeline type selects a *different texture flow model*, so changing resolution changes
  which model paints the asset rather than only how finely it samples. Recorded in
  `docs/trellis2-flat-illustration-colour-drift.md` with a four-panel comparison.

### Added
- Test a third-party report that PyTorch's MPS attention silently returns garbage above
  ~18,000 tokens, and record that **it does not reproduce here**. Measured against a CPU
  reference, torch MPS tracks it to ~1.5e-07 up to 32,768 key/value tokens chunked, and up
  to 12,288 unchunked, with no cliff. The write-up states what that does not establish: the
  unchunked path could not be tested past 12,288 because the score tensor exceeds 16 GiB,
  and it was measured on one torch version. It also notes why this repository may be immune
  — the `sdpa` branch already chunks the query axis for memory reasons — and that MLX's
  fused kernel never materialises the score tensor at all.

### Fixed
- Correct the record on the decode faults. Releasing MLX's Metal cache before decode was
  committed as a hypothesis under test; instrumenting the boundary refuted it, showing MLX
  holding 0.31 GB with a 0.17 GB peak against a decode needing tens of gigabytes. The call
  stays because it is cheap and correct, but it is **not** a fix. The documentation now says
  the cause is unknown, and names the untested control: every clean decode so far was either
  a small mesh or a from-latents run, so the backend is perfectly confounded with sampling
  and decoding in one process at scale.

### Added
- Add `blender_quadruped_pipeline.py`: staged Rigify binding with alignment checks,
  reference-pose capture, profile-tuned walk/trot and standing transitions, audit
  reports, and a documented LLM-assisted tuning workflow. Visible meshes are not remeshed.
- Document the accepted fox trot, rejected neck-deformation experiments and
  portability requirements in [the reusable quadruped gait plan](docs/reusable-gait-quadruped-trot.md).
- **A reusable quadruped gait for the Rigify `basic_quadruped` metarig**:
  `scripts/quadruped_gait.py` holds the maths as pure functions and
  `scripts/blender_quadruped_walk.py` drives Blender. `--gait walk|trot|scamper` with
  cadence, stride, lift, bob, roll and tail motion all derived from *the rig's own*
  proportions -- the ram and the fox are within 5% in height but have inverted limb
  segments, so absolutes copied between creatures over-reach the IK and skate the feet.
  Rotation axes are calibrated by probing each chain rather than assumed, because Rigify
  bone rolls differ between chains and a wrong guess still prints plausible numbers.

  Authored cycles **audit themselves** against the failures this pipeline has actually
  shipped, each bound carrying the measured number that justifies it: a welded body
  (0.00 motion), high-stepping (8.72% of body height), over-striding (3.3x), feet that
  skate (planted 6 frames of 33), and a loop that pops. Bounds scale with the gait, so a
  scamper is not failed for lifting more than a walk.

  Two relationships hold the result together, both measured on the accepted reference
  walk and confirmed independently on a second animal: **foot lift is 0.27x stride** --
  set independently it drifted to 1.18 and the gait marched on the spot -- and lateral
  body roll is **0.35x the vertical bob**, since matching the two puts a hip swing at
  stride frequency that reinforces with the tail into a disco strut.

  Full findings, including the fox's 94.3%-extension foreleg and what it forbids, are in
  `docs/quadruped-gait-2026-09-20.md`.
- **`scripts/blender_bind_rig.py` binds a mesh to an armature headlessly**, driving the
  existing voxel-proxy weight transfer on a saved `.blend` rather than over the live GUI
  socket, where a remesh of a few hundred thousand vertices blocks Blender's handler long
  enough to wedge the session. It applies object scale before solving: `voxel_size` on the
  Remesh modifier is measured in *local* space while `mesh.dimensions` is world space, so
  a mesh scaled 1.5 silently got a proxy 1.5x coarser than requested -- coarse enough to
  fuse a quadruped's legs and bleed weights between them. Mismatched mesh and armature
  scales also distort every later deformation.

  With `--generate-rigify` it runs Rigify generation first (reusing `blender_rebind`'s
  `regenerate_rigify`) and binds to the generated rig's `DEF-` bones instead of to the
  metarig, which is what makes the result posable from the IK/FK controls rather than by
  dragging deform bones. Scale is applied to the metarig *before* generation, since a rig
  generated from a scaled metarig is born scaled and applying scale afterwards has to
  fight the constraints, drivers and widget sizes generation just created. Rigify's `WGT-`
  control widgets are excluded from mesh selection.

### Fixed
- **Bone heat weighting failed on every bone at once when the voxel proxy had loose
  islands.** A voxel remesh of a generated decode routinely leaves a few orphan specks
  floating off the body -- one moss fox produced 13 isolated 8-vertex cubes. Bone heat
  solves a single linear system across the whole surface, so an island with no bone inside
  it makes that system singular: Blender reported "failed to find solution for one or more
  bones" and left *all* 34 vertex groups empty, not just the islands'. `transfer_weights`
  now reduces the throwaway proxy to its largest connected island first, which took that
  fox from 0 of 34 groups weighted to 34 of 34 with no unweighted vertices.

### Added
- **`scripts/README.md` indexes every tool in `scripts/`, and a test keeps it honest.**
  Ninety-odd scripts had no index, so the only way to find out whether a tool already
  existed was to read the directory listing and guess from filenames. The new registry
  groups them by what you are trying to do — generate, pre-flight, repair the mesh,
  texture and material, measure and judge, Blender staging, Blender geometry, rig and
  animate, vendor patches.

  It is generated from the scripts' own docstrings rather than written beside them,
  because a hand-maintained index of that size is wrong within a month and a wrong
  index is worse than none. `tests/test_scripts_registry.py` fails if a script is
  missing from the registry, if the registry names a script that no longer exists, or
  if a summary has drifted from its docstring — so a new script is not finished until
  it is listed. It honours `.gitignore`, so deliberately local tools stay unlisted.

- **`scripts/paint_eyes.py` repaints a generated head's eyes at a usable resolution.**
  No image-to-3D backend models eyes; they paint them into the same atlas that carries
  the whole body, and eyes are small, so they get almost nothing. A clay render of the
  moss fox's head shows a smooth muzzle with a faint mound where each eye belongs, and
  its 1024x1024 atlas spends about a 25x25 patch on each one — no iris edge, no pupil,
  no highlight, and the two sides do not match, because the generator hallucinated each
  independently. Every backend here fails this way; the fox is just where it was measured.

  The script gives each eye its own planar UV projection and a small dedicated material,
  so an eye gets ~300 texels across without growing the shared atlas. The patch starts as
  a resample of the original texture, so the fur around the eye and the boundary with the
  atlas material are unchanged; only the eye is painted over it. Placement is measured,
  not guessed: image moments fit the dark almond the generator painted, so the new eye
  lands where the old one was and stays consistent with the eyelid shading around it.
  That fit is also what makes it work on an unfamiliar creature — the only per-asset
  input is a rough point inside each eye. The roughness map is painted too, so the eye
  is wet and catches a real specular highlight rather than a baked white dot.

  Everything about the eye is a knob (`--iris "#6f9ec4"`, `--pupil-scale`, `--iris-scale`,
  `--sclera`, `--forward`), one iris colour derives its own rim and limbal ring, and a
  material whose roughness is a plain value rather than a packed map is handled by
  synthesising the map. It keeps the original UVs in a backup layer, so `--revert` undoes
  the whole edit and the style can be re-tuned without reimporting, and `--export` writes
  a GLB with that working layer left out.

- **A runnable manifest template.** `manifests/example-trellis2.json` is a copy-and-edit
  starting point and the only tracked manifest, with `manifests/README.md` explaining why
  the rest are ignored. It also documents the detail that is easiest to get wrong: paths
  inside a manifest resolve relative to the manifest file, not the working directory.
- **`docs/browser-workshop.md` is linked from the README.** It is the product and
  architecture boundary for the browser workshop — upload a creature image, generate a 3D
  asset, make it deformable with a known rig, paint it, author an animation, export a GLB
  — and nothing pointed at it.

### Changed
- `blender_joint_markers.send` takes an optional read timeout; it was hard-coded to 300s.


### Added
- **A local, non-blocking TinyCLIP advisor for TRELLIS.2 inputs.** The Generate page now
  warns about flat/vector-style artwork before an expensive run and scores a selected
  image with the pinned MIT-licensed TinyCLIP ViT-8M/16 checkpoint. Conservative
  thresholds catch both known near-black flat-dog inputs while leaving ambiguous images
  as uncertain. The advisor never alters the source or disables generation; manual visual
  inspection remains the fallback.
- **TRELLIS.2 flat-illustration input guidance and investigation.** Controlled MPS/CUDA,
  colour, style, resolution, and material-stage tests traced the near-black dog outputs to
  an input-dependent upstream material-generation failure, strongest on flat/vector art
  without recognizable 3D lighting cues—not to the Metal port. The guide documents the
  current user recommendation, evidence, regression set, and next product steps.
- **Ram headbutt and Rigify pose curves.** A headbutt/charge clip for the custom
  quadruped rig (horns as the weapon, jaw shut, front legs as landing gear that fold
  back at impact), plus walk/trot and headbutt curves retargeted onto a
  Rigify-generated quadruped. The Rigify headbutt drives only the `spine_fk` chain:
  `head`/`neck` pose bones exist on the Basic Quadruped metarig but move zero deform
  bones, so posing them is dead motion.
- **`--hind-lead-l` / `--hind-lead-r` on the walk cycle.** Swing amplitude is
  symmetric, so more forward reach always bought an equally bigger backward kick.
  These add a static per-side bias after the cosine, shifting a leg's swing range
  forward without changing its span.
- **RigNet inference on Apple Silicon**, via a patch to the vendored checkout plus
  glue that drives its five-network pipeline on our own model ids. Spike is paused,
  not concluded — see `docs/progress/2026-08-21-rignet-spike.md`.
- **Reusable Blender scene helpers** for the dungeon-stage lane: append a rigged
  character into the live scene, inspect a scene or unopened `.blend`, frame an orbit
  camera, and bake a named Action's frame sequence at the locked stage angle.

### Fixed
- **TRELLIS recovery checkpoints now exist before decoding starts.** Upstream
  `pipeline.run(return_latent=True)` still decoded before returning its latents, so a
  decoder crash could erase an hour-long sampling run even with Debug enabled. The
  wrapper now postpones that built-in decode, checkpoints the sampled latents first, and
  performs one decode instead of two. Failed web-UI runs retain the checkpoint even when
  Debug is off; successful non-debug runs remove it after the GLB has been written.
- **An image with alpha is not necessarily an image that was cut out.** The TRELLIS
  wrapper gated on "does any pixel have alpha < 255", which two transparent letterbox
  bars satisfy while the subject still sits on an opaque backdrop. `preprocess_image`
  keeps every opaque pixel, so the backdrop was reconstructed as geometry: a ~45-minute
  run at resolution 1024 ending in a slab behind the subject. The wrapper now also
  measures the outer border ring and refuses an uncut image up front, naming the
  measurement and offering `--allow-uncut` for a subject that genuinely reaches the
  frame edge. Across this repo's own assets the split is total — every real cutout
  scores 0.0%, the one uncut image scored 39%.
- **The web UI now rejects an uncut image at upload**, before a job is created, with the
  same measurement in the message. It imports the wrapper's helper rather than restating
  the rule, because the original bug came from the UI and the wrapper each keeping their
  own idea of "has alpha". If the check cannot run (no numpy in the server interpreter) the
  upload proceeds and the wrapper refuses it at run start instead.
- **A failed generation now says why in the browser.** The error banner reported
  `generator exited with code 1` while the real explanation -- an alpha refusal, a missing
  weight, a traceback -- sat unread in the log tail. The job now walks back past the
  progress chatter and surfaces the generator's own last words, keeping the exit code as a
  separate field.
- **Quadruped joint markers were mirrored, and four were missing.** Every `*_L`
  marker sat at x<0.5 and every `*_R` at x>0.5, so the rig's left leg was the
  character's right — caught by eye on the Tempest Ram. `blender_build_rig.py` also
  referenced four `*_toe` markers the table never defined, crashing the rig build the
  first time the pipeline ran end to end on a non-fox character. The hind mid-leg
  joint is now `stifle` rather than `knee`, matching the front limb's naming, and the
  armature is `QuadRig` rather than `FoxRig`.
- **`rigify_walk_pose.sample()` accepted degenerate frame counts**, raising
  `ZeroDivisionError` at zero and returning two frames of nonsense at one. It now
  raises `ValueError`, matching both sibling pose modules.

### Added
- **Rig Review room with deform/fit skeleton separation.** Inspect tapered deform bones,
  search their hierarchy, lower mesh opacity, toggle x-ray rendering, and optionally load a
  fingerprint-verified `.rig.json` fit skeleton mapped in armature-local coordinates. Select
  and drag fit joints in the viewport, edit precise positions, mirror paired changes,
  undo/reset corrections, and download the revised sidecar for Blender rebind. A matching
  prepared `.blend` can be submitted directly to a queued headless Blender rebind; progress
  covers Rigify regeneration, voxel weight transfer, and export, then the replacement GLB
  is loaded automatically with downloadable artifacts.
- **Animate room for rigged GLB/GLTF inspection.** Load a bound character, inspect its
  bone/skinned-mesh/clip inventory, toggle the skeleton, play or scrub embedded animation
  clips, switch back to the bind pose, and control loop playback in the browser.
- **Click-to-inspect deform bones in Animate.** Always-visible joint markers can be selected
  without mistaking an orbit drag for a click; the inspector shows hierarchy plus immutable
  bind and live pose transforms while clips play.
- **Generate-page jobs now survive the server dying.** All job state (subprocess handle,
  event log) lived only in the web server's process memory, so if `viewer/serve.py` itself
  crashed or was closed mid-run — which happened twice on 2026-08-20, once as a silent
  death and once as a false "stuck" alarm that turned out to be a slow-but-healthy run —
  there was nothing on disk to say a job was in flight, and its detached subprocess
  (`start_new_session=True`, so Cancel can `killpg` it independently) could be left running
  as an untracked ghost. Each job now writes `<job-dir>/pid` the moment its subprocess
  starts; on the next server startup, `_reconcile_orphaned_jobs` finds any leftover pid
  files, kills whichever processes are still alive, and annotates that job's `run.log`
  either way so it stops trailing off silently. A clean shutdown (Ctrl-C, `kill`) now also
  kills the active job's process group before exiting, via new `SIGTERM`/`SIGINT` handlers
  in `serve.py` — previously only `SIGINT` was even caught, and neither one touched the
  child.

### Changed
- **Xiong's Hunyuan3D-MLX shape+paint port moved from `vendor/hunyuan-mlx-paint`
  (git-ignored) into this repo's tracked tree at `hunyuan_mlx/`.** It's MIT-licensed code,
  so cloning this repo alone now gets it; only `weights/` (git-ignored) needs a separate
  download, via the new `hunyuan_mlx/download_weights.py` (Hugging Face). The two patch
  scripts that used to reapply fixes to the git-ignored vendor copy
  (`scripts/patch_hunyuan_xiong_shape_deps.py`,
  `scripts/patch_hunyuan_paint_occlusion_fill.py`) are retired — their fixes are just part
  of the tracked source now. dgrauet's shape stage (`vendor/hunyuan-mlx`, used by the
  hybrid backend) stays vendored on purpose: it's Tencent-licensed *code*, not just
  weights, and a fresh A/B against Xiong's 2.0 shape stage (2026-08-19) confirmed it's
  still the cleanest shape available, so the hybrid backend is kept.
- **`hunyuan-mlx-xiong` Generate-page backend now exposes model choice** (2.1 / 2.0 /
  2.0-turbo, default 2.0) instead of hardcoding 2.1. Benchmarked 2026-08-19: 2.0 is
  fastest-to-clean-result and Xiong's own recommended pick; 2.1 was never Xiong's
  recommendation (weaker DINOv2-large conditioner). See `docs/hunyuan-mlx-recipes.md`.

### Fixed
- **Setup panel showed all Hunyuan3D-MLX (Xiong, full pipeline) shape weights as missing,
  even when downloaded.** `_hunyuan_xiong_readiness()`'s `weights` field was a flat
  `{name: bool}` dict; the frontend indexes into it expecting `{label, present, human}`
  objects (the shape every other backend uses). A JS boolean has no `.present`/`.label`
  property, so the panel always took the "missing" branch and rendered the model name as
  the literal string `"undefined"`. Confirmed on this repo's real weights: all three
  models (2.1, 2.0, 2.0-turbo — 13.7 GB / 4.6 GB / 4.6 GB) were actually on disk the whole
  time.
- **Generate mode's alpha-transparency check silently blamed the image when Pillow wasn't
  installed.** `image_has_transparent_alpha()`'s bare `except Exception` caught a missing
  Pillow import the same as a genuinely opaque image, so on any interpreter without Pillow
  every upload failed the check — "no transparent alpha foreground" — regardless of
  whether the image had real transparency. Confirmed on `leather_satchel.png`: alpha range
  0–255, but the server's own interpreter had no PIL at all. Now raises a clear,
  actionable error instead of a wrong 422. Quick start updated to set up a small venv with
  Pillow for exactly this reason — a bare `pip install Pillow` outside a venv refuses to
  run at all on Homebrew Python.
- **Generate page showed no progress at all during Hunyuan's shape stage.**
  `_hunyuan_parse_line` only recognized paint-stage print lines plus a single
  `"shape generated"` completion marker for the *entire* shape stage — the shape
  pipeline's own `[denoise] i/n`, `[vae] grid...`, `[mesh]...` progress lines were logged
  as raw text but never turned into progress events, so the browser sat at 0% for the
  whole shape stage (seconds to tens of minutes depending on model/settings) before
  jumping straight to 100%. Found via a real web-UI test run. Now maps shape-stage denoise
  steps, VAE decode, and mesh extraction to incremental `phase: "shape"` progress.
- **Two "patchy" weight-wiring workarounds in Hunyuan's paint stage, closed properly**
  now that the code is ours to edit rather than an untouchable vendored blob. (1) The
  `weights/dinov2-giant` symlink is gone — `run_paint_pbr.py` and
  `test_pbr_parity.py` now load DINOv2 directly from where it actually ships
  (`weights/hunyuan3d-paintpbr-v2-1/dinov2/`). (2) RealESRGAN super-res weights
  (`weights/realesrgan/rrdbnet.npz`) had no documented source or reproducible conversion;
  `hunyuan_mlx/paint/scripts/convert_realesrgan.py` now fetches the official
  `xinntao/Real-ESRGAN` release and converts it — verified bit-identical to the
  previously-undocumented file already in place.
- **Hunyuan paint-stage texture tear on concave geometry** (inner thigh, armpit, ear
  folds) — present on both Hunyuan backends since they share the same paint code. The old
  `MeshRender.inpaint()` filled camera-occluded texels by grabbing the nearest
  already-painted texel in flat 2D UV-atlas space; xatlas packs unrelated 3D regions next
  to each other on that flat sheet, so occluded creases got filled with an unrelated
  chart's color. Root-caused by measuring true camera occlusion directly (7.8% of surface
  texels invisible from all 6 fixed views, clustered into ~7 localized regions — a real
  occlusion signature, not rasterizer noise). Fixed via `MeshRender.inpaint_occlusion_aware()`,
  which fills occluded-but-in-chart texels from their nearest neighbor in 3D surface space
  instead. Verified end-to-end on a real asset.

### Added
- **New Generate-page backend: Hunyuan3D-MLX (Xiong, full pipeline)**, alongside the existing
  dgrauet-shape + Xiong-paint hybrid (now labeled explicitly as such in the dropdown so the
  two aren't confused). New `scripts/hunyuan_mlx_xiong_generate.py` chains ZimengXiong's own
  shape stage (`vendor/hunyuan-mlx-paint/python/shape`, a separate venv/weights from the
  hybrid's dgrauet shape stage) into the same paint stage the hybrid already uses — one
  author, one repo, end to end. Exposes shape-stage quantization (`quantize=8` by default)
  since ZimengXiong's shape stage runs full-precision by default, which measured ~48 minutes
  for one shape-only run versus dgrauet's ~5 minutes; the quantized default is unbenchmarked
  for speed as of this writing. Details and caveats in `docs/info_and_credits.md` and the
  Generate page's Credits & Info tab.
- **`docs/info_and_credits.md`** — an elaborated, editable Markdown counterpart to the
  in-app Credits & Info tab.
- **`docs/progress/`** — a home for targeted, curated writeups going forward, as opposed to
  the raw investigation logs (see Changed, below).
- **Generate page now drives all three backends** (TRELLIS.2 clean port, SF3D, Hunyuan3D-MLX)
  instead of only TRELLIS. `viewer/generate_api.py` gained a `BackendSpec` registry
  (interpreter, wrapper, settings validation, progress parsing, and readiness are all
  per-backend) so the job runner, SSE progress stream, and setup-status check dispatch by
  backend id instead of being hardcoded to one. New `scripts/hunyuan_mlx_generate.py` chains
  dgrauet's shape stage, a `fast_simplification` remesh, and ZimengXiong's paint stage
  (bridging their two separate venvs) into one CLI, mirroring `trellis_space_generate.py`'s
  shape — the exact recipe (octree_resolution=512, ≤500k-face decimation) validated
  end-to-end on 2026-08-18. The paint stage's seed is now configurable (`PAINT_SEED` env var)
  instead of hardcoded. Job output folders are now named `<image>__<backend>__<timestamp>`
  (or a user-supplied label) instead of an opaque UUID, so a batch of generate-page runs
  stays legible without manual renaming. The ComfyUI Hunyuan path was evaluated and dropped —
  all three "Mac ComfyUI" candidates found still ship Tencent's unmodified CUDA-only
  `custom_rasterizer`, which cannot build on macOS at all.
- **`scripts/mark_asset.py`** — an append-only register of human verdicts on generated
  assets (`output/verdicts.jsonl`). The provenance sidecar records source art, settings and
  output but never whether the result was any good, so a render that looked right could not
  be traced back to the GLB that produced it. Keyed by content hash because the interesting
  derived assets have no sidecar and get renamed; snapshots the forensic measurements beside
  the verdict so it accumulates into the fine-tuning dataset described in
  `docs/training-trellis.md`; and records whether the judgement was made backface-culled,
  since a double-sided verdict cannot distinguish a solid mesh from a hollow one.
- **`scripts/restore_pbr_material.py`** — re-attaches the metallicRoughness map that
  `--material-mode matte` orphaned, on assets already on disk. `matte` only ever rewrote the
  GLB's JSON chunk, so the 3072² map is still in every file we shipped, merely unreferenced;
  restoring it is a JSON edit rather than a regeneration. Also restores `metallicFactor: 1.0`
  (in glTF the factor multiplies the texture — restoring the map while leaving the factor at
  0.0 changes nothing) and turns `doubleSided` off to match the reference. Verified on the
  moss fox: the orphaned texture has an all-zero red channel and G/B distributions matching
  the Hugging Face reference's own MR map.
- **`scripts/glb_forensics.py`** — dumps what a GLB actually contains (PBR channels present,
  `doubleSided`, texture sizes, boundary vs non-manifold edges, winding, signed volume,
  edge-length CV) so a reference asset from a hosted demo can be diffed against ours
  instead of judged by eye. Reads the glTF JSON chunk directly rather than through a mesh
  library, because loaders normalise materials — the very thing being inspected.
- **`docs/hunyuan-eval-2026-08-13.md`** — Hunyuan3D evaluated from first principles, with
  every claim graded MEASURED / READ / CLAIMED. Records that upstream Hunyuan3D has been
  unmaintained since October 2025, that the "CUDA-blocked paint stage" was a build-system
  problem, and that a dev's April port notes are pinned to a diffusers release that is now
  two versions stale with the two most output-affecting fixes coupled to it.

### Changed
- **The old `docs/` tree (60 files: `docs/legacy/` plus every dated session-investigation
  doc) is no longer tracked.** It was our own working notes, not something a user of this
  repo needs, and it had accumulated broken cross-references as things moved. It now lives
  in an untracked, gitignored `journal/` folder (still on disk locally, still in git history
  before this commit — nothing was deleted, only untracked). `docs/` is being rebuilt as a
  smaller, curated, tracked folder going forward (see Added, above); `CLAUDE.md` and
  `README.md` had their now-dead `docs/...` links removed or replaced with inline summaries.
- **`viewer/` model panes now use image-based studio lighting instead of a 3-point rig.**
  The three directional lights (borrowed from `scripts/blender_stage.py`'s diagnostic rig)
  gave asymmetric, harshly-shadowed results that didn't match how Blender's Material
  Preview/LookDev viewport actually lights an object — that viewport uses a studio HDRI
  environment map, not lamps. Added `vendor/environments/RoomEnvironment.js` (ported from
  three.js's own example) and bake it per-renderer with `PMREMGenerator` into
  `scene.environment`, with only a faint hemisphere light left for ambient fill.
- **`--material-mode` now defaults to `pbr`, not `matte`.** `matte` discarded TRELLIS's
  metallic-roughness map and pinned the factors flat, so every organic asset shipped with no
  specular response under any light. Measuring our maps against the Hugging Face reference
  shows they match closely — moss fox roughness 0.765 / metallic 0.412 against the
  reference's 0.784 / 0.384, Flicker 0.396 / 0.000 against 0.404 / 0.004 — so TRELLIS was
  producing exactly what the reference implementation ships and we were deleting it on
  export. Judged backface-culled on Flicker: eye reflections return and the body gains
  surface variation. `matte` remains available.
- `restore_pbr_material.py --roughness-scale` multiplies the roughness map for subjects that
  still read duller than their source art.

### Fixed
- **`hunyuan-mlx-xiong`'s remesh step crashed on any mesh over the 300k-face decimation
  target** (`scripts/patch_hunyuan_xiong_shape_deps.py`). `run_remesh()`'s
  `import fast_simplification` had never actually succeeded — the package isn't in vendor's
  own `pyproject.toml` (upstream never decimates) and nothing installed it into
  `vendor/hunyuan-mlx-paint/python/shape/.venv`, the venv the wrapper is invoked with. Went
  unnoticed because no prior run had crossed the threshold. Re-apply after every bootstrap;
  `vendor/` is git-ignored.
- **Clean TRELLIS.2 port now produces GLBs end-to-end on Apple Silicon.** The decode→GLB
  bake was blocked by cumesh Metal simplify crashing on ~20M-face meshes. The clean-port
  wrapper (`vendor/upstream-audit-worktree/scripts/trellis_space_generate.py`) now CPU
  pre-caps the decoded mesh with `fast_simplification` — in a subprocess (the C extension
  crashes in any process that imported o_voxel's Metal/OpenCV deps), with verify-and-retry
  (its output is nondeterministically corrupt above ~20M input faces) and a post-filter for
  the residual corrupt indices that segfaulted mtlbvh's BVH build — hands `to_glb` CPU
  tensors, frees the 4B pipeline before the bake, and caches the decoded mesh so
  `--from-decode` re-bakes without the model or a re-decode. First assets: Lucian,
  controller, Flicker (Flicker geometrically matches the HF demo control). See
  `docs/MPS-BAKE-FIXES-2026-08-15.md`.
- **`viewer/serve.py` imports `generate_api` robustly.** The sibling import only resolved
  when run as a script; importlib-loaded by the test suite it broke collection.
- **The 200,000-face cap that was destroying 94% of every decode**
  (`scripts/patch_trellis_face_cap.py`). The Mac port pre-simplified the decoded mesh —
  ~3.2 million triangles — down to 200k with a crude decimator *before* o_voxel's
  postprocess, so hole filling, non-manifold repair, simplification, UV unwrapping and the
  texture bake all ran on wreckage. This was the cause of the crazed, cracked surfaces. It
  also made `bake_target_faces` **inert above 200k**: 300,000 and 3,000,000 both produced
  ~197k faces. Lifting it gives 290,662 faces, no crash, and no crazing. Re-apply after
  every bootstrap; `vendor/` is git-ignored.
- **Meshes shipped inside-out** (`scripts/fix_winding.py`, wired into the TRELLIS backend
  via `TrellisOptions.fix_winding`). Generated assets had inconsistent face winding and
  frequently negative signed volume (-0.02369 on Flicker). glTF materials are double-sided
  by default so previews looked fine, but backface-culled — as every game engine renders —
  the asset was hollow. This also means previously reported "see-through hole" and tear
  percentages were substantially counting flipped faces, not missing geometry.

### Changed
- **`docs/self-inflicted-damage.md` is the new entry point** for mesh-quality work. It
  documents both defects above, how the official TRELLIS.2 HuggingFace demo exposed them as
  a control group, and which earlier conclusions are withdrawn — notably "painted markings
  become geometry" (the demo carves no grooves from the same artwork) and "do not raise
  `bake_target_faces`". `docs/baseline.md` carries a banner: its method stands, its numbers
  were measured on damaged meshes.

### Added
- **A source-vs-render comparison** (`scripts/compare_to_source.py`). Renders an asset from
  a camera matched to its source image and lays out source / textured / culled grey /
  silhouette overlay. Every other metric here measures the mesh against itself and so
  cannot see a dead texture or a thin marking. Angles are fixed per subject: Flicker 130,
  Snag 95, fox 210. See `docs/baseline.md`.
- **`soften_markings.py --protect`** — a mask of regions to leave alone. Softening treats
  every dark region as flat paint, which is wrong for darkness that is *shading of real
  geometry*: lightening Flicker's ear hollows made the generator build a membrane that
  tore. With the ears protected, see-through holes fall from 2.67% to 1.07% of body area
  across eight angles, with no angle worse than baseline.
- **Marking projection** (`scripts/project_markings.py`). Paints the source's markings back
  onto a generated texture after they have been softened out of the conditioning image.
  Samples per texel from interpolated projected coordinates, derives the mask by comparing
  the two conditioning images, and transfers the marking as a ratio so the artwork's own
  lighting is not baked in.
- **Re-unwrap and retopology bakes** (`scripts/blender_reunwrap_bake.py`,
  `scripts/blender_retopo_bake.py`). Both work; both are negative results, kept so the
  measurements are not repeated.

### Changed
- **The tear metric is a diagnostic, not a gate.** It cannot see a dead texture, a missing
  sheen or a thin marking, and Flicker's score halved while the mesh visibly got worse.
  Judge with the four-panel comparison instead. `docs/finishing.md` carries a banner.
- **`docs/baseline.md` is the current state of Flicker, Snag and the fox**, all measured the
  same way on 2026-08-12, and supersedes older per-experiment notes where they disagree.

### Fixed
- **The Snag's flat eye was self-inflicted.** `material_mode: matte` strips metalness
  entirely, so a wet eyeball has no material to sit on; grading cannot restore it. Restore
  gloss with the existing eye mask first, then grade.
- **Most of the Snag's apparent tearing is flipped faces**, not missing geometry —
  Recalculate Outside removes nearly all of it.

### Added
- **A tear metric that gates post-processing** (`scripts/ribbon_metric.py`). The share of
  faces touching an open edge: 0% is closed, 1-3% is a surface with tears, and the
  thorn-knot Snag measures **40.9%** — a mesh of ribbons two or three triangles wide, which
  no repair can fix. Measured across the library to set the gate empirically: Flicker 3.1%
  (the asset judged best by eye), moss fox 14.7%, Snag 40.9%. **Necessary but not
  sufficient** — both SF3D assets score a perfect 0.0% and are unusable, so it may reject
  an asset but never accept one.
- **A finishing layer for generated assets** (`scripts/surface_detail.py`,
  `scripts/blender_bake_ao.py`, `scripts/feature_mask.py`). Generated assets arrive with no
  surface at all: TRELLIS emits one flat roughness for the whole subject. This derives a
  normal and roughness map from the albedo, bakes contact-scale ambient occlusion from the
  geometry, and can mask a single feature — an eye — to make it glossy. See
  `docs/finishing.md`.
- **Feature masking from a render** — locate a feature in a rendered image, raycast back
  onto the mesh, and rasterise the hit faces' UVs into a mask. Colour cannot do this: after
  grading, a hue threshold for the Snag's eye shatters into 155 fragments.
- `scripts/visibility_cull.py` — deletes faces never observed from outside. Works as
  specified and does **not** fix the shattering, for a reason worth keeping: anything
  visible in a render is by definition seen by the cull.
- `scripts/soften_markings.py` — reduces the contrast of flat painted markings in a
  conditioning image, protecting genuinely dark features such as eyes.
- `scripts/lift_lightness.py` — brightens an albedo without shifting hue or saturation.
  Prefer fixing the lighting; this is the second choice.
- `docs/finishing.md`, plus a "How to use this repo" section in the README covering all
  seven pipeline steps and a proposed schema-v2 `finishing` manifest block.
- `docs/second-opinion-snag-mesh.md` and `docs/handoff-to-worklings-coder.md` — two
  independent reviews of the shattered mesh and the evidence exchanged with them.

### Fixed
- The visibility cull no longer destroys the material. It had replaced the mesh's materials
  with the face-ID shader and exported without restoring them, shipping correct UVs and no
  albedo.
- `lift_lightness` scales linear RGB rather than LAB's L channel. The first version claimed
  to preserve hue and saturation while doing the opposite — holding a/b fixed while raising
  L desaturates, measured at 13% loss. A test caught it.

### Changed
- **The recipe is per-subject and human-judged.** Bark wants matte with strong derived
  relief; glazed ceramic wants gloss and *no* relief, because an albedo-derived normal map
  turns painted markings into dents. Ambient occlusion is the only step that transferred
  between subjects unchanged — it is measured from geometry, while normal and roughness are
  inferred from paint.
- **The albedo transform is not constant across subjects**, so there is no single global
  inverse. The Snag's highlights are crushed; Flicker's shadows are. Grade strength is
  chosen per subject off a rendered lineup.


### Fixed
- Documented that the see-through holes on detailed subjects are **zero-thickness
  sheets, not tears**, and that a Blender Solidify pass closes every one of them:
  pangolin 97.82 → 0.00, moss fox 126.58 → 0.00, monolith 44.78 → 0.00 (hole perimeter
  relative to the mesh diagonal). Costs roughly 4x the faces. This supersedes the
  art-direction rule recorded earlier the same day, which said detailed surfaces should
  be avoided; they need thickening, not avoiding. See `docs/open-questions.md` §1d.

### Added
- Provenance now records `software.pipeline_revision` — this repository's commit and
  whether the working tree was dirty — alongside the existing `backend_revision`. The
  patches that change TRELLIS's behaviour live in this repo, so the backend SHA alone
  never identified the code that produced an asset. Two runs with identical recorded
  parameters could behave differently with nothing in the sidecar to show it: the
  clockwork pangolin generated 2026-08-02 declares `bake_target_faces: 200000`, but the
  commit that made that value take effect on Metal landed five days later.
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
- Brute-force UV packing, on by default. `scripts/patch_ovoxel_pack_options.py`
  teaches `o_voxel.postprocess.to_glb` to forward `xatlas_pack_charts_kwargs` (which
  `cumesh`'s `uv_unwrap` already accepted), and `scripts/patch_trellis_quality.py`
  adds `--uv-brute-force-packing` / `--no-uv-brute-force-packing`. Measured on the
  hero fox through the real Metal path, atlas coverage goes 52.90% → 58.76% for one
  extra second on 101k faces — no geometry change, no regeneration. The generator
  inspects `to_glb`'s signature first, so an unpatched or reinstalled `o_voxel`
  warns and packs the old way rather than failing at bake time.

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
