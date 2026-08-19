# Hunyuan3D-MLX shape models — quick reference

Repo: `hunyuan_mlx/shape` + `hunyuan_mlx/paint` (ZimengXiong's own port, MIT, tracked
in-repo since 2026-08-19 — a clone of this repo alone has the code; weights are
downloaded separately).

Setup from a fresh clone:
```
uv sync --project hunyuan_mlx/shape
uv sync --project hunyuan_mlx/paint
hunyuan_mlx/shape/.venv/bin/python hunyuan_mlx/download_weights.py
```
`download_weights.py` pulls shape weights (2.1, 2.0, 2.0-turbo) and paint weights from
Hugging Face. RealESRGAN super-res weights
(`hunyuan_mlx/paint/weights/realesrgan/rrdbnet.npz`) aren't part of the official Tencent
HF repos, so they're separate: `hunyuan_mlx/paint/scripts/convert_realesrgan.py` downloads
the official `xinntao/Real-ESRGAN` release and converts it (needs a torch venv, dev-time
only — matches the paint module's other oracle/convert scripts; not a runtime dependency).

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
model above, including dgrauet's (still vendored at `vendor/hunyuan-mlx` — Tencent-licensed
code, kept because its shape output is still the cleanest we've tested, see
`docs/info_and_credits.md`). The texture-tear-on-concave-geometry bug (inner
thigh/armpit/ear folds) that affected all of them is fixed as of 2026-08-19 — occluded
texels now fill from their nearest neighbor in 3D surface space, not 2D atlas-pixel space.
