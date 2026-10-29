"""Is this image already cut out? Judged by its contents, never by its mode.

Qwen-Image through `stable-diffusion.cpp` writes RGBA whose alpha is opaque noise with
nothing transparent in it. Backends that read "has an alpha channel" as "already matted"
skip background removal, and the backdrop comes back as geometry: sheets beside a fox's
head in Pixal3D (2026-09-22), a grey slab around every SF3D model (2026-09-24).
"""

from __future__ import annotations

import os

# A real cutout leaves a lot of the frame empty -- a centred subject is typically 30-60%
# transparent. This floor only has to separate that from an alpha channel that cuts nothing.
MATTE_MIN_TRANSPARENT = 0.02
MATTE_TRANSPARENT_BELOW = 16


def is_matted(image) -> bool:
    """True only when a meaningful share of the PIL image is actually transparent."""
    if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    transparent = sum(alpha.histogram()[:MATTE_TRANSPARENT_BELOW])
    return transparent / (alpha.width * alpha.height) >= MATTE_MIN_TRANSPARENT


# --- Cutting the subject out ----------------------------------------------------------------
#
# BiRefNet-lite, when installed: on 2026-09-28 u2net ate a white robot's upper arms on a
# light backdrop, an axe handle and a sword + shield, and BiRefNet (full or lite) kept all
# of them; lite matched full at full size and ran ~25x faster on this Mac. u2net stays as
# the fallback so a machine without lite still works. Never BRIA RMBG, which this repo's
# generation pipeline must not load.
#
# Nothing here downloads lite: AGENTS.md forbids fetching weights without an explicit
# choice, so it arrives only through scripts/bootstrap_matte.py / Setup & Status.

LITE_MODEL = "birefnet-general-lite"
FALLBACK_MODEL = "u2net"
LITE_URL = ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx")
LITE_BYTES = 224_005_088
LITE_MD5 = "4fab47adc4ff364be1713e97b7e66334"  # rembg's own known hash for this file

# rembg's u2net.onnx, rounded to the nearest MB for error messages.
FALLBACK_BYTES = 176_099_093


def model_home():
    """Where rembg keeps its models (its own U2NET_HOME convention)."""
    from pathlib import Path

    return Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))


def model_file(name: str):
    return model_home() / f"{name}.onnx"


def matte_model() -> str:
    """The best remover this machine has: lite once installed, else u2net."""
    return LITE_MODEL if model_file(LITE_MODEL).is_file() else FALLBACK_MODEL


def _model_size(model: str) -> str:
    # Match bootstrap_matte.py's MB convention (1e6) so the same file reads the same size
    # everywhere it is mentioned.
    if model == LITE_MODEL:
        return f"{LITE_BYTES / 1e6:.0f} MB"
    if model == FALLBACK_MODEL:
        return f"{FALLBACK_BYTES / 1e6:.0f} MB"
    return "unknown size"


def _ensure_model(model: str, allow_download: bool) -> None:
    """Raise a helpful RuntimeError if the model file is missing and downloads are disabled."""
    if allow_download or model_file(model).is_file():
        return
    size = _model_size(model)
    raise RuntimeError(
        f"{model} weights are not installed ({size}). "
        "Fetch them with scripts/bootstrap_matte.py or via Setup & Status, "
        "or pass allow_download=True to download automatically."
    )


def fallback_note(model: str) -> str | None:
    """What to tell the user when a run falls back to u2net, or None."""
    if model != FALLBACK_MODEL:
        return None
    return ("using u2net, which can eat thin or light-coloured parts; install "
            "BiRefNet-lite (224 MB) from Setup & Status or scripts/bootstrap_matte.py")


def clean_edges(rgba, solid: int = 250, reach: int = 4):
    """Give half-transparent edge pixels the colour of the subject, not the old backdrop.

    A soft matte keeps each edge pixel's original colour, which is part backdrop: a thin
    grey or dark outline that a 3D generator then paints onto the model's rim. Colours are
    pulled in from the nearest solid pixels, `reach` pixels at a time; alpha is untouched.
    Takes and returns an H x W x 4 uint8 array.
    """
    import numpy as np

    rgb = rgba[..., :3].astype(np.float64)
    known = rgba[..., 3] >= solid
    height, width = known.shape
    for _ in range(reach):
        padded_rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)))
        padded_known = np.pad(known, 1)
        total = np.zeros_like(rgb)
        count = np.zeros(known.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                k = padded_known[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
                total += padded_rgb[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] * k[..., None]
                count += k
        grow = ~known & (count > 0)
        if not grow.any():
            break
        rgb[grow] = total[grow] / count[grow, None]
        known = known | grow
    out = rgba.copy()
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def new_session(model: str | None = None, *, allow_download: bool = False):
    # Newer rembg keeps models in ~/.rembg/models unless told otherwise, so a backend's own
    # venv missed BiRefNet-lite here and fetched u2net. One folder for every rembg.
    os.environ.setdefault("U2NET_HOME", str(model_home()))
    import rembg

    model = model or matte_model()
    _ensure_model(model, allow_download)
    return rembg.new_session(model)


def cut_out(image, session=None, *, allow_download: bool = False):
    """Remove the background from a PIL image; returns (RGBA image, model name used)."""
    import numpy as np
    import rembg
    from PIL import Image

    model = matte_model() if session is None else getattr(session, "model_name", "custom")
    session = session or new_session(model, allow_download=allow_download)
    cut = rembg.remove(image.convert("RGB"), session=session).convert("RGBA")
    return Image.fromarray(clean_edges(np.asarray(cut))), model
