"""Per-stage parity: torch reference (oracle) vs MLX port, with IDENTICAL inputs.

Builds the hy3dshape (Hunyuan3D-2.1) modules directly (the 2mini config targets the
`hy3dgen.*` namespace, not present here — same architecture, different module path),
loads the bundled checkpoint by prefix, dumps reference tensors to /tmp.

Run:  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/oracle_compare.py
"""
import sys

import numpy as np
import yaml

WD = "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini"
CKPT, CFG = f"{WD}/model.fp16.safetensors", f"{WD}/config.yaml"
IMG = "reference/Hunyuan3D-2.1/assets/demo.png"
STEPS, GS = 30, 5.0


def split(state, prefix):
    return {key[len(prefix):]: value.float() for key, value in state.items()
            if key.startswith(prefix)}


def main():
    sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
    import torch
    from hy3dshape.models.autoencoders.model import ShapeVAE
    from hy3dshape.models.conditioner import SingleImageEncoder
    from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT
    from hy3dshape.preprocessors import ImageProcessorV2
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file

    with open(CFG) as config_file:
        cfg = yaml.safe_load(config_file)
    sd = load_file(CKPT, device="cpu")

    print(">> building hy3dshape modules (cpu, fp32)...")
    dit = Hunyuan3DDiT(**cfg["model"]["params"]).eval()
    vae = ShapeVAE(**cfg["vae"]["params"]).eval()
    cond = SingleImageEncoder(**cfg["conditioner"]["params"]).eval()
    print("  dit:", dit.load_state_dict(split(sd, "model."), strict=False).__class__.__name__,
          "| vae:", len(vae.load_state_dict(split(sd, "vae."), strict=False).missing_keys), "missing",
          "| cond:", len(cond.load_state_dict(split(sd, "conditioner."), strict=False).missing_keys), "missing")
    dit.float()
    vae.float()
    cond.float()
    sched = FlowMatchEulerDiscreteScheduler(**cfg["scheduler"]["params"])
    proc = ImageProcessorV2(**cfg["image_processor"]["params"])

    # ---- shared inputs ----
    out = proc(IMG)
    image = out["image"]                              # [1,3,512,512] in [-1,1]
    img01 = (image - (-1)) / 2.0
    dino_pixels = cond.main_image_encoder.transform(img01).float()  # [1,3,518,518]
    np.save("/tmp/dino_pixels.npy", dino_pixels.numpy())

    with torch.no_grad():
        dino_out_t = cond.main_image_encoder.model(dino_pixels).last_hidden_state
    np.save("/tmp/dino_out_torch.npy", dino_out_t.numpy())
    print(f"[oracle] DINO out {tuple(dino_out_t.shape)} mean {dino_out_t.mean():.4f} std {dino_out_t.std():.4f}")

    noise = np.random.RandomState(0).randn(1, 512, 64).astype(np.float32)
    np.save("/tmp/noise.npy", noise)
    with torch.no_grad():
        cond_emb = cond(image=image)["main"]
        uncond_emb = cond.unconditional_embedding(1)["main"]
        cond_cat = {"main": torch.cat([cond_emb, uncond_emb], 0)}
    np.save("/tmp/cond_torch.npy", cond_cat["main"].numpy())

    # ---- torch denoise loop ----
    sigmas = np.linspace(0, 1, STEPS)
    sched.set_timesteps(sigmas=sigmas, device="cpu")
    timesteps = sched.timesteps
    lat = torch.from_numpy(noise).float()
    with torch.no_grad():
        for t in timesteps:
            inp = torch.cat([lat] * 2)
            ts = t.expand(inp.shape[0]).float() / sched.config.num_train_timesteps
            v = dit(inp, ts, cond_cat, guidance=None)
            vc, vu = v.chunk(2)
            v = vu + GS * (vc - vu)
            lat = sched.step(v, t, lat).prev_sample
    np.save("/tmp/latents_torch.npy", lat.numpy())
    print(f"[oracle] final latents mean {lat.mean():.4f} std {lat.std():.4f}")

    # ---- torch VAE grid (vanilla, low res) ----
    OCT = 96
    with torch.no_grad():
        dec = vae((1.0 / vae.scale_factor) * lat)
        grid_t = vae.volume_decoder(dec, vae.geo_decoder, bounds=1.01, num_chunks=8000,
                                    octree_resolution=OCT, enable_pbar=False)
    grid_t = grid_t[0].cpu().numpy()
    np.save("/tmp/grid_torch.npy", grid_t)
    print(f"[oracle] grid {grid_t.shape} range {grid_t.min():.4f}..{grid_t.max():.4f} frac>0 {(grid_t > 0).mean():.4f}")
    print("ORACLE DONE")


if __name__ == "__main__":
    main()
