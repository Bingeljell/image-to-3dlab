"""Real-ESRGAN x4plus in PyTorch: a 4x upscale for the hero route's head crop.

One 67 MB weights file (BSD-3-Clause, xinntao/Real-ESRGAN), run from the project's own
environment on a Mac, NVIDIA or AMD. Hero used to borrow the upscaler from the Hunyuan3D-MLX
paint install, which made a multi-GB painting engine a requirement for one small network.
The weights arrive only through Setup or `scripts/bootstrap_upscaler.py`, after a yes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_BYTES = 67_040_989
WEIGHTS_FILE = REPO / "vendor" / "realesrgan" / "RealESRGAN_x4plus.pth"
LICENSE = "BSD-3-Clause (xinntao/Real-ESRGAN)"
TILE = 256   # pixels per tile edge: bounds memory on big pictures; seams are hidden by overlap


def installed(path: Path = WEIGHTS_FILE) -> bool:
    return path.is_file() and path.stat().st_size == WEIGHTS_BYTES


def network():
    """The RRDBNet that Real-ESRGAN x4plus uses: 23 residual-in-residual dense blocks."""
    import torch
    from torch import nn

    def lrelu(x):
        return nn.functional.leaky_relu(x, 0.2)

    class Dense(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

        def forward(self, x):
            x1 = lrelu(self.conv1(x))
            x2 = lrelu(self.conv2(torch.cat([x, x1], 1)))
            x3 = lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
            x4 = lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
            return x + self.conv5(torch.cat([x, x1, x2, x3, x4], 1)) * 0.2

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = Dense(), Dense(), Dense()

        def forward(self, x):
            return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))

    class RRDBNet(nn.Module):
        def __init__(self, blocks=23, nf=64):
            super().__init__()
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(blocks)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

        def forward(self, x):                  # x: [N,3,H,W] in 0..1 -> [N,3,4H,4W]
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            up = nn.functional.interpolate
            feat = lrelu(self.conv_up1(up(feat, scale_factor=2, mode="nearest")))
            feat = lrelu(self.conv_up2(up(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(lrelu(self.conv_hr(feat)))

    return RRDBNet()


def state_dict(path: Path) -> dict:
    """Weights from the release .pth (its `params_ema`), or a same-named .npz copy."""
    import torch

    if path.suffix == ".npz":
        return {k: torch.from_numpy(v) for k, v in np.load(path).items()}
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    return loaded.get("params_ema", loaded.get("params", loaded))


def device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(path: Path = WEIGHTS_FILE):
    if not path.is_file():
        raise FileNotFoundError(
            "Real-ESRGAN is not installed: get it (67 MB) from Setup & Status or "
            "scripts/bootstrap_upscaler.py. Nothing was downloaded.")
    model = network()
    model.load_state_dict(state_dict(path))
    return model.eval().to(device())


def upscale(model, rgb: np.ndarray, tile: int = TILE, overlap: int = 10) -> np.ndarray:
    """[H,W,3] floats in 0..1 -> [4H,4W,3], run tile by tile with an overlap."""
    import torch

    h, w = rgb.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    dev = next(model.parameters()).device
    with torch.no_grad():
        for top in range(0, h, tile):
            for left in range(0, w, tile):
                bottom, right = min(top + tile, h), min(left + tile, w)
                t0, l0 = max(top - overlap, 0), max(left - overlap, 0)
                b0, r0 = min(bottom + overlap, h), min(right + overlap, w)
                piece = torch.from_numpy(np.ascontiguousarray(rgb[t0:b0, l0:r0].transpose(2, 0, 1)))[None]
                done = model(piece.float().to(dev))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                out[top * 4:bottom * 4, left * 4:right * 4] = done[
                    (top - t0) * 4:(top - t0 + bottom - top) * 4, (left - l0) * 4:(left - l0 + right - left) * 4]
    return out
