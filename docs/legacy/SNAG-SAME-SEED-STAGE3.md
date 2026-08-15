# Snag same-seed result and Stage-3 experiment boundary

**Date:** 2026-08-14  
**Start here for the next Snag material session.**

The exact hosted-demo seed/settings run succeeded locally. It establishes that the corrected
Mac path produces a solid, authentic Snag with comparable geometry. The remaining difference
is the sampled material field: local is greener and more saturated than the hosted control.
This is no longer the old hollow/lattice failure.

The complete artifact-level record is
`output/snag_same_seed_hf/RUN.md`; machine-verifiable paths, byte sizes, and hashes are in
`output/snag_same_seed_hf/MANIFEST.json`. Those files live under the intentionally ignored
`output/` directory and must stay on disk or be copied to artifact storage before cleaning the
workspace.

## Accepted baseline

- source: `assets_to_test/3-4th-snag-roots-alpha.png`
- hosted control: `assets_to_test/trellis-snag-huggingface.glb`
- local baseline: `output/snag_same_seed_hf/snag_seed614089393.glb`
- shape/material latents: `output/snag_same_seed_hf/snag_seed614089393_latents.pt`
- decoded mesh/material field: `output/snag_same_seed_hf/snag_seed614089393_decode.pt`
- seed: `614089393`
- pipeline: `1024_cascade`, 12 steps, 2048 texture, requested 3M faces
- raw decode: 20,210,054 triangles
- final local GLB: 2,793,927 triangles, 151.5 MB
- hosted GLB: 284,529 triangles, 13.7 MB
- wall time: about 52m15s

The requested 3M target is not what the hosted demo ultimately shipped. Its measured 284.5K
faces explain why its GLB is roughly one tenth the size. Match that measured output after
choosing a material candidate; do not spend another diffusion run investigating file size.

## What the comparison proves

With backface culling enabled, local and hosted assets are both visually solid. Shape is
comparable rather than pixel-identical, which is the product goal. The local result has an
intact amber eye and convincing organic surface detail.

Base-colour medians:

| | luminance | saturation | RGB |
|---|---:|---:|---|
| local | 0.196 | 0.708 | 45, 55, 15 |
| hosted | 0.195 | 0.569 | 53, 52, 23 |
| source | 0.299 | 0.667 | 96, 75, 31 |

Local and hosted lightness already match. The visible issue is excessive global green and
weak spatial separation between brown vine and moss—not exposure, Blender lighting, missing
textures, or deleted PBR channels.

## Do not confuse finishing with parity

Three fast derived files exist under `output/snag_same_seed_hf/material_tests/`:

- `chroma_025.glb`
- `chroma_050.glb`
- `living_v2c.glb`

The first two show that a global hue correction cannot create semantic separation. The third
uses a manually located eye mask and a deterministic living-organic recipe. It is useful as a
product-finishing demonstration, but it is **not** evidence of hosted-demo parity. Upstream
Stage 3 generates its base-colour and metallic-roughness fields without a human mask.

Do not continue tuning those derived files until the Stage-3-only lane has been evaluated.

## Correct next experiment

Build a runner that reconstructs the saved `SparseTensor` shape latent, reloads the original
image conditioning, and calls only `sample_tex_slat` plus the material decoder. Keep geometry
fixed and give Stage 3 its own explicit `texture_seed`.

First candidate uses the official material defaults unchanged:

```text
steps              12
guidance_strength   1.0
guidance_rescale    0.0
guidance_interval   0.6, 0.9
rescale_t           3.0
```

No mask, colour grade, lightness lift, or material override is allowed in this comparison.
Record the independent texture seed, latent hash, decoded attribute statistics, timings, and
output hashes.

The original full run called `torch.manual_seed` once before all three stages. Because the
saved bundle does not include the RNG state immediately before Stage 3, passing the original
shape seed to a Stage-3-only runner does not recreate the original texture noise. Treat
`texture_seed` as an explicit new reproducibility dimension.

After one default candidate validates the runner, batch two additional texture seeds with one
pipeline load. If seed selection does not produce adequate brown/moss separation, sweep one
parameter at a time—guidance strength first—on the best seed. Bake only the accepted candidate
at 2048 and approximately 285K faces.
