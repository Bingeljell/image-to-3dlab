"""An alpha channel is not a matte: only real transparency counts as a cutout."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab.matte import is_matted


def _rgba(alpha: np.ndarray) -> Image.Image:
    rgb = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def test_qwen_noise_alpha_is_not_a_matte():
    # Qwen via stable-diffusion.cpp: alpha is noise in 101-255, nothing transparent.
    noise = np.random.default_rng(0).integers(101, 256, (64, 64))
    assert is_matted(_rgba(noise)) is False


def test_real_cutout_is_a_matte():
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[16:48, 16:48] = 255
    assert is_matted(_rgba(alpha)) is True


def test_rgb_is_not_a_matte():
    assert is_matted(Image.new("RGB", (8, 8))) is False


def test_stray_transparent_pixel_is_not_a_matte():
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[0, 0] = 0
    assert is_matted(_rgba(alpha)) is False


# --- choosing and running the remover ------------------------------------------------------


from image_to_3dlab import matte as mt


def test_lite_is_used_once_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    assert mt.matte_model() == "u2net"
    (tmp_path / "birefnet-general-lite.onnx").write_bytes(b"x")
    assert mt.matte_model() == "birefnet-general-lite"


def test_falling_back_says_how_to_get_lite():
    assert "Setup & Status" in mt.fallback_note("u2net")
    assert mt.fallback_note("birefnet-general-lite") is None


def test_edge_pixels_take_the_subjects_colour_not_the_backdrops():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[:, :3] = (200, 30, 30, 255)            # solid red subject, left
    rgba[:, 3] = (40, 40, 40, 128)              # soft edge still carrying grey backdrop
    rgba[:, 4] = (40, 40, 40, 0)                # backdrop
    out = mt.clean_edges(rgba)
    assert tuple(out[2, 3]) == (200, 30, 30, 128)   # red now, same transparency
    assert (out[..., 3] == rgba[..., 3]).all()      # alpha never changes
    assert (out[:, :3] == rgba[:, :3]).all()        # solid pixels untouched


def test_edge_cleaning_reaches_only_so_far():
    rgba = np.zeros((1, 10, 4), dtype=np.uint8)
    rgba[0, 0] = (255, 255, 255, 255)
    rgba[0, 1:] = (0, 0, 0, 100)
    out = mt.clean_edges(rgba, reach=3)
    assert tuple(out[0, 3, :3]) == (255, 255, 255)
    assert tuple(out[0, 4, :3]) == (0, 0, 0)


def test_the_lite_download_is_pinned_to_a_known_file():
    assert mt.LITE_URL.endswith("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx")
    assert mt.LITE_BYTES == 224_005_088 and len(mt.LITE_MD5) == 32


def test_every_rembg_looks_in_the_same_model_folder(tmp_path, monkeypatch):
    # Seen on an NVIDIA pod: TRELLIS's newer rembg kept models in ~/.rembg/models, so it
    # could not see BiRefNet-lite in ~/.u2net and quietly fetched u2net mid-generation.
    import sys
    import types

    seen = {}
    fake = types.SimpleNamespace(new_session=lambda name: seen.setdefault("home", __import__("os").environ.get("U2NET_HOME")))
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.setattr(mt, "model_home", lambda: tmp_path)
    (tmp_path / "u2net.onnx").write_bytes(b"x")
    mt.new_session("u2net", allow_download=False)
    assert seen["home"] == str(tmp_path)


def _fake_rembg(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np
    from PIL import Image

    def fake_remove(image, session=None):
        out = image.convert("RGBA")
        a = np.zeros((image.height, image.width), dtype=np.uint8)
        a[8:24, 8:24] = 255
        out.putalpha(Image.fromarray(a))
        return out

    fake_session = types.SimpleNamespace(model_name="birefnet-general-lite")
    fake = types.SimpleNamespace(
        new_session=lambda name: fake_session,
        remove=fake_remove,
    )
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.setattr(mt, "model_home", lambda: tmp_path)


def test_cut_out_refuses_to_download_without_consent(tmp_path, monkeypatch):
    _fake_rembg(monkeypatch, tmp_path)
    image = Image.new("RGB", (32, 32), (200, 120, 60))

    with pytest.raises(RuntimeError) as exc:
        mt.cut_out(image)

    message = str(exc.value)
    # Names what Setup installs, not the fallback: there is no "u2net" to find on Setup.
    assert "BiRefNet-lite" in message and "Setup & Status" in message and "MB" in message


def test_new_session_refuses_to_download_without_consent(tmp_path, monkeypatch):
    _fake_rembg(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError) as exc:
        mt.new_session(allow_download=False)

    assert "Setup & Status" in str(exc.value)


def test_cut_out_downloads_when_allowed(tmp_path, monkeypatch):
    _fake_rembg(monkeypatch, tmp_path)
    (tmp_path / "birefnet-general-lite.onnx").write_bytes(b"x")
    image = Image.new("RGB", (32, 32), (200, 120, 60))

    cut, model = mt.cut_out(image, allow_download=True)

    assert cut.mode == "RGBA"
    assert model == "birefnet-general-lite"


def test_new_session_downloads_when_allowed(tmp_path, monkeypatch):
    _fake_rembg(monkeypatch, tmp_path)

    session = mt.new_session(allow_download=True)

    assert session.model_name == "birefnet-general-lite"


def test_cut_out_with_existing_weight_needs_no_permission(tmp_path, monkeypatch):
    _fake_rembg(monkeypatch, tmp_path)
    (tmp_path / "birefnet-general-lite.onnx").write_bytes(b"x")
    image = Image.new("RGB", (32, 32), (200, 120, 60))

    cut, model = mt.cut_out(image)

    assert model == "birefnet-general-lite"
    assert cut.mode == "RGBA"


# --- the lean BiRefNet-lite file -----------------------------------------------------------


def test_the_lean_file_is_preferred_over_rembgs_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    (tmp_path / "birefnet-general-lite.onnx").write_bytes(b"x")
    assert mt.matte_model() == "birefnet-general-lite"
    (tmp_path / "birefnet-lite-lean.onnx").write_bytes(b"x")
    assert mt.matte_model() == "birefnet-lite-lean"


def test_the_lean_download_is_pinned_to_a_known_file():
    assert mt.LEAN_URL.startswith("https://huggingface.co/AssetFurnace/birefnet-lite-onnx/")
    assert mt.LEAN_BYTES == 180_666_557 and len(mt.LEAN_SHA256) == 64
    assert mt.LEAN_SIZE == 1024


def test_the_lean_file_is_never_fetched_behind_the_users_back(tmp_path, monkeypatch):
    # rembg would download its own copy on allow_download; the lean file only comes from Setup.
    monkeypatch.setattr(mt, "model_home", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Setup & Status"):
        mt._ensure_model(mt.LEAN_MODEL, allow_download=True)


class _Scores:
    """Stands in for onnxruntime: returns raw scores, high inside a square."""

    def __init__(self):
        self.fed = None

    def run(self, outputs, feeds):
        self.fed = feeds["input_image"]
        scores = np.full((1, 1, mt.LEAN_SIZE, mt.LEAN_SIZE), -8.0, dtype=np.float32)
        scores[..., 256:768, 256:768] = 8.0
        return [scores]


def test_the_lean_session_feeds_a_normalised_1024_square_and_returns_a_full_size_mask():
    inner = _Scores()
    session = mt.LeanSession(None, inner=inner)
    picture = Image.new("RGB", (200, 100), (124, 116, 104))   # ImageNet's mean colour
    mask = session.mask(picture)

    assert inner.fed.shape == (1, 3, 1024, 1024) and inner.fed.dtype == np.float32
    assert abs(float(inner.fed.mean())) < 0.02                # mean colour normalises to ~0
    assert mask.mode == "L" and mask.size == (200, 100)
    m = np.asarray(mask)
    assert m[50, 100] > 250 and m[2, 2] < 5                   # inside the square / outside


def test_cut_out_with_the_lean_session_keeps_the_subject_and_clears_the_rest(monkeypatch):
    session = mt.LeanSession(None, inner=_Scores())
    picture = Image.new("RGB", (64, 64), (200, 30, 30))
    cut, model = mt.cut_out(picture, session=session)

    assert model == "birefnet-lite-lean" and cut.mode == "RGBA"
    a = np.asarray(cut)
    assert a[32, 32, 3] == 255 and tuple(a[32, 32, :3]) == (200, 30, 30)
    assert a[1, 1, 3] == 0


def test_a_lean_file_that_will_not_load_falls_back_instead_of_failing(tmp_path, monkeypatch, capsys):
    _fake_rembg(monkeypatch, tmp_path)
    (tmp_path / "birefnet-lite-lean.onnx").write_bytes(b"x")
    (tmp_path / "birefnet-general-lite.onnx").write_bytes(b"x")

    def too_old(*a, **k):
        raise RuntimeError("Could not find an implementation for DeformConv")

    monkeypatch.setattr(mt, "LeanSession", too_old)
    session = mt.new_session()
    assert session.model_name == "birefnet-general-lite"
    assert "lean remover would not load" in capsys.readouterr().out
