"""Measure PyTorch-on-MPS peak RAM for a Hunyuan3D shape model: load to mps (fp16)
+ one CFG DiT forward + VAE decode. Reports torch MPS allocator + process RSS.
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/measure_mps.py <model_dir>
"""
import sys

import yaml


def split(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def main():
    import resource

    sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
    import torch
    from hy3dshape.models.autoencoders.model import ShapeVAE
    from hy3dshape.models.conditioner import SingleImageEncoder
    from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from safetensors.torch import load_file

    md = sys.argv[1]
    dev = torch.device("mps")
    with open(f"{md}/config.yaml") as config_file:
        cfg = yaml.safe_load(config_file)
    sd = load_file(f"{md}/model.fp16.safetensors", device="cpu")

    dit_type = HunYuanDiTPlain if cfg["model"]["target"].endswith("HunYuanDiTPlain") else Hunyuan3DDiT
    dit = dit_type(**cfg["model"]["params"]).eval()
    dit.load_state_dict(split(sd, "model."), strict=False)
    vae = ShapeVAE(**cfg["vae"]["params"]).eval()
    vae.load_state_dict(split(sd, "vae."), strict=False)
    cond = SingleImageEncoder(**cfg["conditioner"]["params"]).eval()
    cond.load_state_dict(split(sd, "conditioner."), strict=False)

    for m in (dit, vae, cond):
        m.to(dev, dtype=torch.float16)
    del sd
    torch.mps.synchronize()
    weights_alloc = torch.mps.current_allocated_memory() / 1e9

    nl = cfg["vae"]["params"]["num_latents"]
    ctx = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]["config"]["hidden_size"]
    with torch.no_grad():
        lat = torch.randn(1, nl, 64, device=dev, dtype=torch.float16)
        contexts = {"main": torch.zeros(2, 1370, ctx, device=dev, dtype=torch.float16)}
        t = torch.tensor([0.5, 0.5], device=dev, dtype=torch.float16)
        # Keep both outputs alive until after sampling allocator memory.
        _denoised = dit(torch.cat([lat, lat]), t, contexts)
        torch.mps.synchronize()
        _decoded = vae((1.0 / vae.scale_factor) * lat)
        torch.mps.synchronize()

    current = torch.mps.current_allocated_memory() / 1e9
    driver = torch.mps.driver_allocated_memory() / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"MPS  {md.split('/')[-1]:24s} fp16: "
          f"weights {weights_alloc:5.2f} GB | mps-current {current:5.2f} GB | "
          f"mps-driver {driver:5.2f} GB | process-RSS {rss:5.2f} GB")


if __name__ == "__main__":
    main()
