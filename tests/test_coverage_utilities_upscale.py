"""Exercise the upscale command with local images and a model boundary."""

import runpy
import sys

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import upscale
from scripts import upscale_esrgan


def test_upscale_cli_reports_missing_weights_without_opening_source(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(upscale, "installed", lambda: False)

    def unexpected_load():
        pytest.fail("attempted to load missing weights")

    monkeypatch.setattr(upscale, "load", unexpected_load)
    dest = tmp_path / "output.png"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("sys.argv", ["upscale_esrgan.py", str(tmp_path / "missing.png"), str(dest)])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(upscale_esrgan.__file__, run_name="__main__")
    assert status.value.code == 2
    assert "67 MB" in capsys.readouterr().err
    assert not dest.exists()


@pytest.mark.parametrize("alpha", [255, 120])
def test_upscale_cli_rounds_and_clips_rgb_and_carries_alpha(monkeypatch, tmp_path, capsys, alpha):
    src = tmp_path / "input.png"
    dest = tmp_path / "new" / "output.png"
    Image.new("RGBA", (2, 3), (64, 128, 255, alpha)).save(src)
    model = object()
    monkeypatch.setattr(upscale, "installed", lambda: True)
    monkeypatch.setattr(upscale, "load", lambda: model)

    def model_upscale(instance, pixels):
        assert instance is model
        assert pixels.shape == (3, 2, 3)
        assert pixels.dtype == np.float32
        np.testing.assert_allclose(pixels[0, 0], [64 / 255, 128 / 255, 1])
        return np.broadcast_to([-0.2, 0.5, 1.2], (12, 8, 3))

    monkeypatch.setattr(upscale, "upscale", model_upscale)
    monkeypatch.setattr("sys.argv", ["upscale_esrgan.py", str(src), str(dest), "--size", "4"])
    assert upscale_esrgan.main() == 0
    assert "input.png (2, 3) -> output.png (4, 4)" in capsys.readouterr().out
    with Image.open(dest) as result:
        assert result.size == (4, 4)
        expected = (0, 128, 255) if alpha == 255 else (0, 128, 255, alpha)
        assert result.getpixel((1, 2)) == expected
