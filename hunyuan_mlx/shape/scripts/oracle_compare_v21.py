"""v2-1 fast parity: dump DINOv2-large output + a single HunYuanDiTPlain forward.
(The VAE is the same ShapeVAE already proven on 2mini — only num_latents differs.)
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/oracle_compare_v21.py
"""
import sys

import numpy as np
import yaml

WD = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"
IMG = "reference/Hunyuan3D-2.1/assets/demo.png"


def split(state, prefix):
    return {key[len(prefix):]: value.float() for key, value in state.items()
            if key.startswith(prefix)}


def main():
    sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
    import torch
    from hy3dshape.models.conditioner import SingleImageEncoder
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from hy3dshape.preprocessors import ImageProcessorV2
    from safetensors.torch import load_file

    with open(f"{WD}/config.yaml") as config_file:
        cfg = yaml.safe_load(config_file)
    sd = load_file(f"{WD}/model.fp16.safetensors", device="cpu")

    print(">> building hy3dshape v2-1 dit + cond (cpu fp32)...", flush=True)
    dit = HunYuanDiTPlain(**cfg["model"]["params"]).eval()
    cond = SingleImageEncoder(**cfg["conditioner"]["params"]).eval()
    print("  dit missing:", len(dit.load_state_dict(split(sd, "model."), strict=False).missing_keys),
          "| cond missing:", len(cond.load_state_dict(split(sd, "conditioner."), strict=False).missing_keys), flush=True)
    dit.float()
    cond.float()
    proc = ImageProcessorV2(**cfg["image_processor"]["params"])

    image = proc(IMG)["image"]
    dino_pixels = cond.main_image_encoder.transform((image + 1) / 2.0).float()
    np.save("/tmp/v21_pixels.npy", dino_pixels.numpy())
    with torch.no_grad():
        dino_t = cond.main_image_encoder.model(dino_pixels).last_hidden_state
    np.save("/tmp/v21_dino.npy", dino_t.numpy())
    print(f"[oracle] DINO-large {tuple(dino_t.shape)} std {dino_t.std():.4f}", flush=True)

    # single DiT forward (CFG batch of 2) at t=0.5
    n = cfg["vae"]["params"]["num_latents"]
    noise = np.random.RandomState(0).randn(1, n, 64).astype(np.float32)
    np.save("/tmp/v21_noise.npy", noise)
    with torch.no_grad():
        ce = cond(image=image)["main"]
        ue = cond.unconditional_embedding(1)["main"]
        cc = {"main": torch.cat([ce, ue], 0)}
        inp = torch.from_numpy(np.concatenate([noise, noise], 0)).float()
        t = torch.tensor([0.5, 0.5]).float()
        v = dit(inp, t, cc)
    np.save("/tmp/v21_cond.npy", cc["main"].numpy())
    np.save("/tmp/v21_dit_v.npy", v.numpy())
    print(f"[oracle] DiT single-forward v {tuple(v.shape)} std {v.std():.4f} mean {v.mean():.4f}", flush=True)
    print("ORACLE DONE", flush=True)


if __name__ == "__main__":
    main()
