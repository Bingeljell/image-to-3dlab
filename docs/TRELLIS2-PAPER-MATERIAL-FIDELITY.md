# TRELLIS.2 paper notes: material fidelity and Mac parity

**Reviewed:** 2026-08-14  
**Primary source:** [Native and Compact Structured Latents for 3D Generation](https://arxiv.org/html/2512.14692)  
**Executable reference:** [Microsoft TRELLIS.2 Hugging Face Space](https://huggingface.co/spaces/microsoft/TRELLIS.2/blob/main/app.py)

## Short finding

The paper supports the current observation that geometry can be good while material is
wrong. Shape and material use separate latent models. Stage 3 predicts a native 3D PBR
material field conditioned jointly on the source image and the generated geometry. GLB
extraction subsequently samples that already-generated field; it cannot invent a semantic
brown-vine/green-moss split that Stage 3 failed to produce.

Therefore the first parity audit should compare image conditioning, Stage-3 attention and
sampler math, dtype/precision, and geometry conditioning. Remesh, UV unwrap, and GLB packing
remain important for preserving an existing result, but they are downstream of the missing
semantic separation.

## What each stage actually predicts

1. **Sparse structure:** occupancy layout of the sparse voxel grid.
2. **Geometry:** shape latents within the active voxels.
3. **Material:** material latents aligned to the generated geometry.

For Stage 3, a sparse DiT is conditioned on both DINOv3-L features from the source image and
the generated geometry latent. Image conditioning enters through cross-attention; geometry
is concatenated with the material model's input channels. This was the highest-risk area when
replacing CUDA FlashAttention with MPS SDPA; the 128-wide self/cross-attention integration
gate now matches the reference equation within `6e-8` on MPS.

## What the material contains

Each active O-Voxel stores six material channels:

- base color RGB;
- metallic;
- roughness;
- opacity.

When an O-Voxel is converted to a textured mesh, a surface/texel query receives those
attributes by trilinear interpolation from neighboring voxels. There is no view-dependent
texture synthesis or semantic repainting during export. The conversion should preserve the
Stage-3 field and package it as a standard PBR asset.

## Why the model can separate material from lighting

The material VAE is first trained with direct L1 supervision on material attributes, then
fine-tuned at high resolution with render-based perceptual supervision. The generative model
was trained on roughly 800,000 assets, augmented with TexVerse, using 16 Blender views per
asset with randomized fields of view and lighting. That training setup is intended to make
the predicted PBR channels intrinsic to the object rather than a literal copy of illumination
in one input render.

Training also progressively raises geometry/material output from 512^3 to 1024^3 while image
conditioning rises from 512 to 1024 pixels. A local run advertised as 1024 parity must prove
that Stage 3 actually receives the same 1024-resolution conditioning path.

## Consequences for the Snag result

The HF control's brown wood and green moss are spatial material decisions. A locally decoded
field that is broadly all brown or all green is not principally a lighting problem, and a
UV bake cannot restore the missing distinction. A renderer can make the result brighter,
darker, glossier, or flatter, but it cannot recover absent spatial color information.

Seed variance can change a generative result, but it is not a sufficient explanation until
the same source image, preprocessing, checkpoint, pipeline type, sampler defaults, and seed
have been shown to produce comparable conditioning and Stage-3 latents on CUDA and MPS.

There is also an important seed-boundary detail. The official demo seeds once before all
three stages. A Stage-3-only experiment that resets that same integer immediately before
material sampling does **not** recreate the demo's texture noise. On the frozen Snag shape,
the full-run and reset-before-Stage-3 texture latents have identical coordinates but only
`0.7411` cosine similarity (`1.5494` mean absolute difference). Treat an independent texture
seed as a new candidate, not a replay of the hosted seed.

## Metal attention audit

Pedro's fused attention is correct on its tested shapes but is not the production backend
for TRELLIS.2-4B. Its optimized tiled kernel supports head dimensions through `64`; the model
uses `128`. The fallback performs a serial key loop for each query and did not finish the
first Snag material step in `29m44s` at `22,894` tokens. The clean port therefore uses
sequence-wise PyTorch SDPA for the 4B model and fails fast if the unsupported fused path is
selected. For a normal one-image job, this makes one direct SDPA call per attention layer—no
padding is introduced.

## Cheap parity gates before another full run

1. **Input contract:** hash the exact RGBA image after preprocessing; record crop, alpha,
   compositing color, resize kernel, value range, and 512/1024 tensor shapes.
2. **DINO contract:** record checkpoint/revision, processor configuration, dtype, token shape,
   token norms, channel statistics, and a deterministic feature digest.
3. **Attention contract:** compare FlashAttention and MPS SDPA on saved Stage-3 Q/K/V fixtures,
   including cross-attention, masks, QK normalization, RoPE, and padded sparse batches.
4. **Sampler contract:** freeze initial noise and compare every Stage-3 step's latent summary,
   guidance calculation, timestep/rescale, normalization mean/std, and final latent digest.
5. **Geometry-conditioning contract:** verify the exact shape latent coordinates/features fed
   into Stage 3, not merely the appearance of the decoded mesh.
6. **Decoder contract:** compare decoded PBR voxel-channel distributions and spatial colour
   clusters before remesh or texture baking.
7. **Export preservation:** from one frozen decoded PBR field, verify that GLB base-color,
   metallic, roughness, and alpha samples match the source volume within tolerance.

Only after gates 1-6 pass should a fresh full-resolution GPU run be used as evidence. Gate 7
can use the cached Snag material output and is independent of stochastic generation.

## Release architecture implication

The parity core should be a minimally patched copy of the pinned official Space source. Mac
backend substitutions should be isolated and tested against saved CUDA fixtures. Optional
features such as alternate background removal, multi-image conditioning, viewers, heuristics,
and material controls belong above that core so they cannot silently alter the baseline.

## Sources

- [TRELLIS.2 paper](https://arxiv.org/html/2512.14692)
- [Official Microsoft Hugging Face Space app](https://huggingface.co/spaces/microsoft/TRELLIS.2/blob/main/app.py)
- [Official Microsoft TRELLIS.2 repository](https://github.com/microsoft/TRELLIS.2)
