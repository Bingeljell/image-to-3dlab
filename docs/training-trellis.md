# Fine-tuning TRELLIS.2 on our own art

> **Not started. Deliberately parked** until the pipeline work is finished — see
> "Sequencing" below. Recorded now so the option is on the map, with the facts gathered
> 2026-08-12 from the upstream README.

## Why this is interesting for us

Some of our remaining defects are **model properties, not pipeline bugs**, which means no
parameter fixes them:

- **Colour drift.** TRELLIS.2 is documented as producing textures that deviate from the
  input image. The Snag's golden-amber source coming out grey-green is exactly this. No
  setting corrects it.
- **Style mismatch.** The base model is trained on Objaverse-XL — general 3D objects. It
  has no prior for "this creature is *deliberately* low-poly and faceted", so it fights the
  Forest Variant's art direction rather than reproducing it.
- **Family coherence.** The dungeon-crawl and worklings casts need to look like one world.
  Per-asset correction cannot deliver that; a learned prior can.

## What is trainable

Five independently trainable components:

| Component | Controls |
|-----------|----------|
| Shape SC-VAE | how geometry is encoded/decoded to O-Voxel |
| Texture SC-VAE | how PBR material is encoded |
| Sparse-structure flow | stage 1 — where the object exists in space |
| Shape flow | stage 2 — the surface |
| **Texture flow** | stage 3 — colour and material ← *most relevant to colour drift* |

Upstream ships `*_ft_512.json` and `*_ft1024.json` configs that **fine-tune from existing
checkpoints**, so this need not be training from scratch. No LoRA support is mentioned —
though absence from a README has been a weak signal in this project.

## Requirements

**Hardware:** NVIDIA GPU with **≥24 GB VRAM**, verified on A100/H100, CUDA 12.4, Python 3.8+.

This is a genuine capacity constraint — memory footprint plus CUDA sparse-3D kernels — not
a caution to be tested away. Our Macs will not train a 4B flow model. **That is a "where",
not a "whether":** an A100 rents for roughly $1–2/hour, and a resolution fine-tune from an
existing checkpoint is tens of hours, not weeks.

**Data — the real cost.** Assets must be converted to the **O-Voxel representation**: mesh
conversion, compact structured latent generation, and metadata preparation. The dataset
layout expects mesh dumps, dual-grid representations, PBR voxel data, precomputed
sparse-structure and shape/texture latents, and conditioning renders.

Realistically **hundreds to low thousands** of examples, not a dozen.

## The command shape

```sh
python train.py \
  --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json \
  --output_dir results/shape_vae_next_dc_f16c32_fp16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \
               \"mesh_dump\": \"datasets/ObjaverseXL_sketchfab/mesh_dumps\", \
               \"dual_grid\": \"datasets/ObjaverseXL_sketchfab/dual_grid_256\", \
               \"asset_stats\": \"datasets/ObjaverseXL_sketchfab/asset_stats\"}}"
```

Equivalent commands exist for the texture VAE and each of the three flow models.

## Sequencing — why not now

Today established that our problems were **pipeline bugs, not model limitations**. A better
model will not fix a remesh kernel that drops quads, and training now would confound the
two: we would not know which improvement came from where.

1. **Finish the pipeline** — remesh, the Forest Variant's blade holes, the unexplained
   residual gap to the official demo on the old Flicker.
2. **Then measure what remains.** If the leftover gap is colour, palette and style, that is
   precisely what fine-tuning addresses and it justifies the spend.
3. **Start hoarding data now** — see below. This is the one part worth doing immediately.

## Start collecting the dataset today

Converting to O-Voxel later is far easier than re-sourcing art later. For **every** asset
generated and approved, keep:

- the **source artwork** at full resolution
- the **manifest** used (settings, seed) — already emitted per run
- the **approved output GLB**
- a note on **what made it good or bad**, which is the labelling that turns a pile of files
  into a dataset

The repo already emits a `.provenance.json` sidecar per run, so most of this is captured;
what is missing is the approval judgement. Worth adding a field.
