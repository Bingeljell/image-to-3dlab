# TRELLIS.2 colour drift with flat illustrations

**Status:** diagnosed on 2026-09-03. Static guidance and a non-blocking TinyCLIP
input advisor are implemented. We do not silently alter or reject a source image.

## Finding

TRELLIS.2 can preserve colour well when the reference looks like a photograph or a
softly lit 3D render, but it can produce extremely dark or nearly black materials from
flat/vector-style illustrations that do not contain convincing 3D lighting and surface
cues.

This is not simply a problem with brown, yellow, dogs, Apple Silicon, or our GLB export.
The strongest comparison used the same kind of blue dog:

- a flat blue illustration produced a nearly black dog;
- a 3D-rendered blue reference preserved the blue;
- a rendered yellow rubber duck also preserved its yellow.

The useful distinction is therefore not **simple versus detailed geometry**. The duck is
geometrically simple. It is **flat illustration versus recognizable 3D appearance**.

## Why it happens

TRELLIS.2 does not project the reference pixels directly onto the mesh. Its material
stage predicts four PBR attribute types:

1. **Base colour** — surface colour before scene lighting.
2. **Metalness** — whether the surface responds like a metal.
3. **Roughness** — how glossy or matte it is.
4. **Opacity** — how transparent or solid it is.

The reference first passes through DINOv3-L, a visual feature extractor. DINOv3 splits
the image into patches and turns them into a spatial grid of learned features describing
visual content such as colour, edges, texture, parts, and layout. These features are not
a pixel-for-pixel copy and they do not invent views of the object.

TRELLIS.2 then predicts material latents jointly from those image features and the
generated geometry. DINOv3 supplies evidence; the downstream TRELLIS material model
makes the material decision.

This matters because a single image does not uniquely reveal which brightness belongs
to the material and which belongs to illumination. Microsoft trained the image-conditioned
model on about 800,000 3D assets, rendering 16 views of each asset in Blender with
randomized fields of view and lighting. That teaches the model to separate intrinsic PBR
materials from illumination, but its training prompts are still rendered 3D views. A
flat illustration can fall outside those learned appearance cues. The model may then
interpret a bright drawn region incorrectly and infer a much darker underlying material.

That explanation is an inference from the paper, implementation, community reports, and
our controlled results. Microsoft does not identify flat illustration colour collapse as
an official limitation.

Sources:

- [TRELLIS.2 paper](https://arxiv.org/html/2512.14692)
- [Official TRELLIS.2 repository](https://github.com/microsoft/TRELLIS.2)
- [Official image-to-3D pipeline](https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py)

Related community reports are not identical to this failure, but establish that dark
materials and weak handling of flat art have been observed outside this project:

- [TRELLIS.2 issue #162: black and reflective spots on light images](https://github.com/microsoft/TRELLIS.2/issues/162)
- [TRELLIS discussion #49: black or extremely dark generated textures](https://huggingface.co/spaces/microsoft/TRELLIS/discussions/49)
- [TRELLIS issue #58: flat anime inputs lacking relief and depth](https://github.com/microsoft/TRELLIS/issues/58)

## Evidence from this investigation

| Test | Result | What it ruled out or supported |
|---|---|---|
| Flat golden cartoon dog, 512, multiple seeds | Nearly black | Seed alone is not a remedy |
| Same failed shape, texture stage rerun | Still dark | The bad colour is predicted in the material stage |
| Exact material test on local MPS and official CUDA | Both nearly black | Not a Metal/MPS-specific drift |
| White-matte preprocessing control | Still dark | Transparent-background compositing was not the cause |
| Flat golden dog at 1024 cascade | Some brown recovered, still too dark and metallic | Resolution affects the prediction but does not reliably solve it |
| More realistically shaded golden dog | Gold preserved | Brown/gold is not intrinsically broken |
| Flat blue version of the failed dog | Nearly black | The failure is not limited to warm colours |
| 3D-rendered blue dog | Blue preserved | 3D lighting/material cues are the strongest observed separator |
| Rendered yellow rubber duck | Yellow preserved | Simple geometry and bright warm colour can work |

One additional control smoothed and quantized the successful detailed golden dog while
holding its generated geometry and material seed fixed. Both versions remained golden.
That control removed fine detail but retained broad fur layers and 3D shading, reinforcing
that the important signal is believable surface form and illumination rather than detail
count alone.

The raw failed material field was already almost black before texture baking and GLB
display. This rules out the viewer, lighting setup, texture bake, and export colour space
as the primary cause of this case.

Numerically, the failed 512 MPS run had median RGB base colour of approximately
`[0.0005, 0.0003, 0.0007]`. The matching official CUDA material test was similarly dark
at `[0.0190, 0.0010, 0.0005]`. The 1024 cascade recovered dark brown
`[0.289, 0.159, 0.078]`, but also incorrectly predicted median metalness `1.0`.

### Local evidence files

These paths are intentionally untracked under `output/` and exist on the investigation
machine rather than in a fresh clone:

- `output/trellis2_cutedog/dog-darkness-512-seed0-debug/`
- `output/trellis2_cutedog/dog_1024cascade/`
- `output/trellis2_cutedog/cute-dog-blue__trellis__20260903-173223/`
- `output/trellis2_cutedog/cute-dog-blue-3d-alpha__trellis__20260903-181610/`
- `output/trellis2_cutedog/yellow_duck/`
- `output/trellis2_cutedog/style_control_512/`

## Guidance for users now

For TRELLIS.2, avoid completely flat or vector-style reference images for now. Prefer a
transparent-background image that resembles a softly lit 3D render or photograph and has:

- visible light-to-shadow gradients across the form;
- enough surface shading to make volume legible;
- recognizable material cues, even if the subject is stylized;
- the intended colours visible across broad surface regions.

Changing seed, decimation target, or baked texture size is not a dependable fix for a
near-black result. The 1024 cascade may change colour materially, but it is slower and did
not fully repair the failed dog.

We will not automatically restyle user artwork at this stage. Doing so can change the
character, silhouette, markings, or art direction without consent.

The Generate page also runs a small local TinyCLIP check after a TRELLIS image is
selected. It compares the image with prompt groups describing flat artwork and
dimensional renders. The resulting “flat-style score” is an **uncalibrated similarity
signal**, not the probability that TRELLIS will fail. It only advises; generation stays
available regardless of the result, and visual inspection remains the fallback.

The checkpoint is pinned to
[`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
revision `a2a8c6eaa2549ad66eb7c31b85022bf58273a26c`. It runs locally in the TRELLIS
environment and does not upload the image.

## What we build next

### 1. Input guidance in Generate — implemented

Add a short TRELLIS.2-specific notice next to image upload:

> Best results come from photographs or softly lit 3D-style renders. Flat/vector artwork
> may produce very dark or incorrect materials.

The notice and TinyCLIP advisory are live. The UI deliberately does not block generation.
Example thumbnails can be added once redistributable source-image licences are confirmed.

### 2. A durable regression set

Create a small, redistributable test set after checking the source-image licences:

- one flat warm-colour illustration that fails;
- the equivalent flat blue illustration;
- one softly lit 3D version that succeeds;
- one simple rendered object such as the yellow duck;
- one normally successful detailed asset from the existing sweep.

Run all examples with a fixed manifest and seed. Record both viewer screenshots and raw
PBR statistics, especially median base-colour luminance and metalness. Keep CUDA as an
occasional upstream parity check; MPS remains the normal local pipeline.

### 3. Input-suitability detection — first pass implemented

The first pass uses TinyCLIP ViT-8M/16 with four flat-art prompts and four dimensional-
render prompts. Conservative thresholds reserve “higher risk” for a score of 85% or
above and “appears dimensional” for 35% or below; everything between is uncertain.

Observed calibration set:

| Input | Known TRELLIS result | TinyCLIP flat-style score | Advice |
|---|---:|---:|---|
| Flat blue dog | Nearly black | 96.9% | Higher risk |
| Flat golden dog | Nearly black | 96.4% | Higher risk |
| 3D-rendered blue dog | Blue preserved | 47.0% | Uncertain |
| Rendered yellow duck | Yellow preserved | 28.3% | Appears dimensional |
| Detailed golden dog | Gold preserved | 74.0% | Uncertain |
| Bloomglass | Broadly successful | 16.7% | Appears dimensional |
| Storm ram | Broadly successful | 30.9% | Appears dimensional |

This detector identifies the particular flat-art risk; it does not predict every way a
TRELLIS run can fail. Keep it advisory and recalibrate only against a larger labelled set.

### 4. Optional 3D-reference conversion, later

If users need flat-art support, add an explicit opt-in preprocessing step that creates a
softly lit 3D-style reference while attempting to preserve silhouette, markings, and
palette. Show the transformed reference before generation, preserve the original, and
record the transformation in provenance. Never make this conversion silently.

## Current product behaviour

- TRELLIS.2 users see the flat/vector warning before starting an expensive run.
- The warning links to examples and the explanation above.
- It remains possible to continue with any valid cutout image.
- Documentation no longer describes this as a Metal-port artifact.
- TinyCLIP failures and uncertain results never disable the Generate button.
- Manual eyeballing remains the fallback.

The remaining next step is a redistributable regression manifest that can reproduce the
known failure and success cases locally.
