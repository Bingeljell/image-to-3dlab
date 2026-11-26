"""Check the saved normal map through the bake command's postprocessing."""

import sys

import blender_bake_normals as bake
import pytest
from PIL import Image


@pytest.mark.parametrize("mode", ["correct", "keep", "missing"])
def test_bake_resolves_sign_in_the_saved_map(mode, tmp_path, monkeypatch, capsys):
    output = tmp_path / "bake" / "normal.png"
    low, high = tmp_path / "low.glb", tmp_path / "high.ply"
    argv = [
        "bake",
        str(low),
        str(high),
        str(output),
        "--device",
        "CPU",
        "--flat-source",
        "--permutation",
        "gltf_to_blender",
    ]
    if mode == "keep":
        argv.append("--keep-sign")
    monkeypatch.setattr(sys, "argv", argv)
    good, reversed_normal = (128, 128, 255), (175, 23, 45)

    def send(code, host, port):
        assert (host, port) == ("127.0.0.1", 9876)
        assert output.parent.is_dir()
        compile(code, "<normal bake request>", "exec")
        if mode != "missing":
            image = Image.new("RGBA", (2, 1))
            image.putdata([(*good, 255), (*reversed_normal, 255)])
            image.save(output)
        return "bake response"

    monkeypatch.setattr(bake, "send", send)

    assert bake.main() == 0
    log = capsys.readouterr().out
    assert "bake response" in log
    if mode == "missing":
        assert not output.exists()
    else:
        with Image.open(output) as result:
            if mode == "keep":
                assert result.mode == "RGBA"
                assert result.getpixel((1, 0)) == (*reversed_normal, 255)
            else:
                assert result.mode == "RGB"
                assert result.getpixel((0, 0)) == good
                assert result.getpixel((1, 0)) == (80, 232, 210)
    assert ("negated 50.0% of texels" in log) == (mode == "correct")
