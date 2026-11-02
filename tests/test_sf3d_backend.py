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


def test_noisy_alpha_forces_background_removal():
    from image_to_3dlab.sf3d_backend import cut_out

    calls = []

    def remove_background(image, session, force=False):
        calls.append(force)
        return image

    cut_out(_noisy_rgba(), remove_background, session=None)
    assert calls == [True]


def test_real_matte_is_left_alone():
    import numpy as np
    from PIL import Image

    from image_to_3dlab.sf3d_backend import cut_out

    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    image = Image.fromarray(np.dstack([np.zeros((32, 32, 3)), alpha]).astype(np.uint8), "RGBA")
    calls = []
    cut_out(image, lambda *a, **k: calls.append(1), session=None)
    assert calls == []


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
