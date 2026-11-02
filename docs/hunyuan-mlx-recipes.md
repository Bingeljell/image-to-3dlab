# Hunyuan3D-MLX shape models — quick reference

Repo: `hunyuan_mlx/shape` + `hunyuan_mlx/paint` (ZimengXiong's port, MIT, tracked in this
repo; the weights download separately).

Setup from a fresh clone:
```
uv sync --project hunyuan_mlx/shape
uv sync --project hunyuan_mlx/paint
hunyuan_mlx/shape/.venv/bin/python hunyuan_mlx/download_weights.py
```
`download_weights.py` fetches the default route: shape model `2.0` plus paint, about
13.3 GB. It names that route and asks before it fetches. These weights are not licensed
for use in the EU, the UK or South Korea. `--all` is the extra fetch (shape `2.1` and
`2.0-turbo` as well). Paint comes from Xiong's MLX-ready repo
(`zimengxiong/hunyuan3d-mlx-paint-large`, pinned), which carries the UNet, VAE, DINOv2 and
RealESRGAN super-res weights together. Tencent's own paint folder ships the UNet as a
torch `.bin` and has no DINOv2, so it cannot be used directly.
`hunyuan_mlx/paint/scripts/convert_realesrgan.py` still exists for converting RealESRGAN
from the original release yourself; the paint code only falls back to its output.

CLI: `python -m hy3dmlx.pipeline --weights <dir> [flags] --out out.glb`, or the full
end-to-end wrapper: `hunyuan_mlx/shape/.venv/bin/python scripts/hunyuan_mlx_xiong_generate.py
input.png output.glb --model 2.0 [flags]`

| Model | `--model` | Verdict |
|---|---|---|
| 2.1 | `2.1` | not recommended by Xiong; weaker (DINOv2-large) conditioner |
| 2.0 | `2.0` | **best so far — cleanest shape, default** |
| 2.0-turbo | `2.0-turbo` | fast, but distillation noise (dents/tears); more `--steps` (up to `pcm_timesteps`, 100) helps but doesn't fully clear it |
| 2mini | not downloaded | untested |

**Current best recipe:** 2.0, `--octree-decode --quantize 8 --steps 30`, octree=512.
~9 min shape+paint end to end.

**Paint stage** (`hunyuan_mlx/paint/scripts/run_paint_pbr.py`) is shared by every shape
model above, including dgrauet's, which stays cloned separately in `vendor/hunyuan-mlx`
because its code is Tencent-licensed. Keep `decimation_target` at or under 500,000: the
paint stage's UV unwrap stalls above that.
