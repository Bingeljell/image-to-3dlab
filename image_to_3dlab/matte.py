"""Is this image already cut out? Judged by its contents, never by its mode.

Qwen-Image through `stable-diffusion.cpp` writes RGBA whose alpha is opaque noise with
nothing transparent in it. Backends that read "has an alpha channel" as "already matted"
skip background removal, and the backdrop comes back as geometry: sheets beside a fox's
head in Pixal3D (2026-09-22), a grey slab around every SF3D model (2026-09-24).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

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
#
# LEAN is the same BiRefNet-lite weights, exported with ONNX's own DeformConv layer. rembg's
# file spells every deformable convolution out as thousands of gather/scatter steps, and
# those steps are what made one 1024 px cut-out peak at 9.5 GB on a Mac. The lean file
# peaks at 4.0 GB, runs ~2.5x faster and gives the same mask (IoU >= 0.998 on our test
# pictures). It has its own name because rembg checks its file's hash on every load and
# silently re-downloads its own copy on a mismatch; we run the lean file ourselves.

LEAN_MODEL = "birefnet-lite-lean"
LEAN_URL = ("https://huggingface.co/AssetFurnace/birefnet-lite-onnx/resolve/main/"
            "birefnet-lite-1024.onnx")
LEAN_BYTES = 180_666_557
LEAN_SHA256 = "a97176abb6fe55525b11113bf5bead5a9b5866b2c0d4ac058186cabb86cadfff"
LEAN_SIZE = 1024  # the square input the file was exported at

LITE_MODEL = "birefnet-general-lite"
FALLBACK_MODEL = "u2net"
FALLBACK_MD5 = "60024c5c889badc19c04ad937298a77b"
LITE_URL = ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx")
LITE_BYTES = 224_005_088
LITE_MD5 = "4fab47adc4ff364be1713e97b7e66334"  # rembg's own known hash for this file

def model_home():
    """Where rembg keeps its models (its own U2NET_HOME convention)."""
    from pathlib import Path

    return Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))


def model_file(name: str):
    return model_home() / f"{name}.onnx"


def matte_model() -> str:
    """The best remover this machine has: the lean file, then rembg's lite, then u2net."""
    for name in (LEAN_MODEL, LITE_MODEL):
        if model_file(name).is_file():
            return name
    return FALLBACK_MODEL


def _ensure_model(model: str, allow_download: bool) -> None:
    """Raise a helpful RuntimeError if the model file is missing and downloads are disabled."""
    if (allow_download and model != LEAN_MODEL) or model_file(model).is_file():
        return
    # Whichever model was missing, the fix is the one Setup installs.
    raise RuntimeError(
        "The background remover is not installed. Install \"Background remover "
        f"(BiRefNet-lite)\" on Setup & Status ({LEAN_BYTES / 1e6:.0f} MB), "
        "or run scripts/bootstrap_matte.py."
    )


def fallback_note(model: str) -> str | None:
    """What to tell the user when a run used something other than the lean file, or None."""
    if model == LITE_MODEL:
        return (f"using the older background remover file, which needs about twice the memory; "
                f"install the lighter BiRefNet-lite ({LEAN_BYTES / 1e6:.0f} MB) from Setup & Status "
                f"or scripts/bootstrap_matte.py, and the old file ({LITE_BYTES / 1e6:.0f} MB) is removed")
    if model != FALLBACK_MODEL:
        return None
    return ("using u2net, which can eat thin or light-coloured parts; install "
            f"BiRefNet-lite ({LEAN_BYTES / 1e6:.0f} MB) from Setup & Status or "
            "scripts/bootstrap_matte.py")


def retire_older_copy(loads=None):
    """Delete rembg's older BiRefNet-lite file once the lean one is here and loads.

    Returns the removed path, or None. The older file is the fallback when the lean one
    will not load (an onnxruntime too old for DeformConv), so it goes only after the lean
    file has opened in this environment.
    """
    older, lean = model_file(LITE_MODEL), model_file(LEAN_MODEL)
    if not (older.is_file() and lean.is_file()):
        return None
    try:
        (loads or LeanSession)(lean)
    except Exception as error:
        print(f"[matte] kept the older remover file: the lighter one would not load ({error})",
              flush=True)
        return None
    older.unlink()
    return older


class LeanSession:
    """The lean BiRefNet-lite file, run with onnxruntime directly (rembg would re-download).

    Same steps as rembg's BiRefNet session: resize to the export size, ImageNet-normalise,
    sigmoid the raw scores, stretch to 0-1, resize the mask back to the picture.
    """

    model_name = LEAN_MODEL
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, path, inner=None):
        if inner is None:
            import onnxruntime as ort

            inner = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inner = inner

    def mask(self, image):
        """A PIL 'L' mask the size of `image` (a PIL RGB picture)."""
        import numpy as np
        from PIL import Image

        small = image.convert("RGB").resize((LEAN_SIZE, LEAN_SIZE), Image.Resampling.LANCZOS)
        x = (np.asarray(small, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        scores = self.inner.run(None, {"input_image": x})[0][0, 0]
        pred = 1.0 / (1.0 + np.exp(-scores))
        span = float(pred.max() - pred.min()) or 1.0
        pred = (pred - pred.min()) / span
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        return mask.resize(image.size, Image.Resampling.LANCZOS)


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


def _verified_model_file(model: str) -> Path:
    digests = {LEAN_MODEL: ("sha256", LEAN_SHA256), LITE_MODEL: ("md5", LITE_MD5),
               FALLBACK_MODEL: ("md5", FALLBACK_MD5)}
    if model not in digests:
        raise RuntimeError(f"No verified offline background remover for {model!r}. Use Setup & Status.")
    algorithm, expected = digests[model]
    path = model_file(model)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Background remover {model!r} is damaged. Reinstall it from Setup & Status.")
    return path


def _offline_rembg_session(model: str):
    """Use rembg's prediction class with an exclusively local model resolver."""
    import onnxruntime as ort
    from rembg.sessions import sessions_class

    path = _verified_model_file(model)
    session_class = next((cls for cls in sessions_class if cls.name() == model), None)
    if session_class is None:
        raise RuntimeError(f"This rembg version cannot load {model!r}. Update it from Setup & Status.")

    class LocalSession(session_class):
        @classmethod
        def download_models(cls, *args, **kwargs):
            return path

    options = ort.SessionOptions()
    if "OMP_NUM_THREADS" in os.environ:
        options.inter_op_num_threads = options.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
    return LocalSession(model, options)


def new_session(model: str | None = None, *, allow_download: bool = False):
    # Newer rembg keeps models in ~/.rembg/models unless told otherwise, so a backend's own
    # venv missed BiRefNet-lite here and fetched u2net. One folder for every rembg.
    os.environ.setdefault("U2NET_HOME", str(model_home()))
    import rembg

    model = model or matte_model()
    _ensure_model(model, allow_download)
    if model == LEAN_MODEL:
        try:
            return LeanSession(_verified_model_file(LEAN_MODEL))
        except Exception as error:  # an onnxruntime too old for DeformConv, say
            fallback = LITE_MODEL if model_file(LITE_MODEL).is_file() else FALLBACK_MODEL
            print(f"[matte] lean remover would not load ({error}); using {fallback}", flush=True)
            model = fallback
            _ensure_model(model, allow_download)
    return rembg.new_session(model) if allow_download else _offline_rembg_session(model)


def cut_out(image, session=None, *, allow_download: bool = False):
    """Remove the background from a PIL image; returns (RGBA image, model name used)."""
    import numpy as np
    from PIL import Image

    session = session or new_session(allow_download=allow_download)
    model = getattr(session, "model_name", "custom")
    rgb = image.convert("RGB")
    if isinstance(session, LeanSession):
        # rembg's naive cut-out: picture where the mask is, fully transparent elsewhere
        cut = Image.composite(rgb.convert("RGBA"), Image.new("RGBA", rgb.size, 0), session.mask(rgb))
    else:
        import rembg

        cut = rembg.remove(rgb, session=session).convert("RGBA")
    return Image.fromarray(clean_edges(np.asarray(cut))), model
