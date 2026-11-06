"""SF3D picks the fastest device the machine has."""

from __future__ import annotations

import pytest

from image_to_3dlab import sf3d_backend as sf3d
from image_to_3dlab.sf3d_backend import pick_device


@pytest.mark.parametrize("forced_cpu,cuda,mps,expected", [
    (False, True, False, "cuda"),
    (False, False, True, "mps"),
    (False, False, False, "cpu"),
    (True, True, False, "cpu"),
    (True, False, True, "cpu"),
])
def test_pick_device(forced_cpu, cuda, mps, expected):
    assert pick_device(forced_cpu, cuda, mps) == expected


# Upstream `remove_background` skips rembg on any RGBA with alpha < 255, so a Qwen image
# with noise alpha went in un-cut and every SF3D model came out inside a grey slab
# (3090 Ti pod run 4, 2026-09-24). We decide by contents and force rembg when needed.

def _noisy_rgba():
    import numpy as np
    from PIL import Image

    noise = np.random.default_rng(0).integers(101, 256, (32, 32))
    rgb = np.full((32, 32, 3), 128)
    return Image.fromarray(np.dstack([rgb, noise]).astype(np.uint8), "RGBA")


@pytest.mark.parametrize("noisy", [False, True])
def test_default_lean_session_cuts_out_opaque_input(noisy):
    import numpy as np
    from PIL import Image

    from image_to_3dlab.matte import LeanSession

    class Inference:
        def run(self, names, inputs):
            scores = np.full((1, 1, 32, 32), -10.0)
            scores[:, :, 8:24, 8:24] = 10.0
            return [scores]

    session = LeanSession("unused.onnx", inner=Inference())
    image = _noisy_rgba() if noisy else Image.new("RGB", (32, 32), "white")
    cut = sf3d.cut_out(image, session=session)
    assert cut.mode == "RGBA"
    assert cut.getpixel((0, 0))[3] == 0
    assert cut.getpixel((16, 16))[3] == 255


def test_real_matte_is_left_alone(monkeypatch):
    import numpy as np
    from PIL import Image

    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    image = Image.fromarray(np.dstack([np.zeros((32, 32, 3)), alpha]).astype(np.uint8))
    monkeypatch.setattr(sf3d.matte, "new_session", lambda **kw: pytest.fail("loaded a model"))
    assert sf3d.cut_out(image) is image


def test_a_hub_id_with_no_snapshot_is_refused_before_sf3d_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr(sf3d, "_load_sf3d", lambda repo: pytest.fail("loaded the code"))
    options = sf3d.SF3DOptions(repo=tmp_path, model="stabilityai/stable-fast-3d")
    with pytest.raises(SystemExit, match="about 4.9 GB") as raised:
        sf3d._run(tmp_path / "in.png", tmp_path / "out.glb", options, "cpu")
    message = str(raised.value)
    assert "Stable Fast 3D" in message
    assert "Mac" in message or "AMD" in message


def test_the_setup_hint_names_this_machines_installer():
    """On AMD SF3D lives in a venv of its own; the Mac script refuses that machine."""
    from image_to_3dlab import host
    from image_to_3dlab.sf3d_backend import setup_command

    assert setup_command(host.AMD) == "scripts/bootstrap_sf3d_rocm.py"
    assert setup_command(host.APPLE) == "scripts/bootstrap_sf3d.py"
