"""Exercise comparison files and rendering commands without launching Blender."""

import json
import sys
from subprocess import CompletedProcess

import numpy as np
import pytest
from PIL import Image

from scripts import compare_to_source, render_glb_comparison


@pytest.mark.parametrize("sweep", [False, True])
def test_comparison_command_builds_requested_panels(tmp_path, monkeypatch, sweep):
    source, asset, output = tmp_path / "art.png", tmp_path / "asset.glb", tmp_path / "preview"
    art = Image.new("RGBA", (28, 24), (0, 0, 0, 0))
    art.paste((50, 80, 100, 255), (4, 4, 24, 20))
    art.save(source)
    calls = []

    def run(command, *, capture_output, text, check):
        calls.append(command)
        request = json.loads(command[-1])
        assert request["asset_path"] == str(asset)
        assert request["output_dir"] == str(output)
        if sweep:
            assert request["angles"] == [0, 45]
            assert not request["grey_mode"]
        else:
            assert request["angles"] == [27]
            assert request["grey_mode"] == (len(calls) == 2)
        mode = "grey" if request["grey_mode"] else "tex"
        for angle in request["angles"]:
            render = Image.new("RGBA", (28, 24), (0, 0, 0, 0))
            render.paste((150, 150, 150, 255), (8, 4, 20, 20))
            render.save(output / f"cmp_{mode}_az{round(angle)}.png")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(compare_to_source.subprocess, "run", run)
    argv = ["compare_to_source", str(source), str(asset), str(output), "--height", "16"]
    if sweep:
        argv.extend(["--sweep", "0,45"])
    monkeypatch.setattr(sys, "argv", argv)
    assert compare_to_source.main() == 0
    sheet_path = output / ("cmp_sweep.png" if sweep else "cmp_vs_source.png")
    with Image.open(sheet_path) as sheet:
        assert sheet.height == 82
        colors = {tuple(pixel) for pixel in np.asarray(sheet).reshape(-1, 3)}
    assert (50, 80, 100) in colors
    assert (150, 150, 150) in colors
    assert (compare_to_source.SOURCE_ONLY in colors) is not sweep
    assert len(calls) == (1 if sweep else 2)


@pytest.mark.parametrize("result_has_frame", [False, True])
def test_render_comparison_uses_frame_presence_as_success(tmp_path, monkeypatch, result_has_frame):
    blender, asset = tmp_path / "blender", tmp_path / "asset.glb"
    blender.touch()
    asset.touch()
    output = tmp_path / "out" / "comparison.jpg"

    def run(command, *, capture_output, text, check):
        assert command[0] == str(blender)
        assert command[3] == str(render_glb_comparison.BLENDER_WORKER)
        assert command[5] == str(asset)
        assert command[-1] == "1"
        if result_has_frame:
            frame = Image.new("RGB", (80, 80), "black")
            frame.paste((255, 255, 255), (25, 25, 55, 55))
            frame.save(command[6])
        return CompletedProcess(command, 1 if result_has_frame else 0, "stdout", "stderr")

    monkeypatch.setattr(render_glb_comparison.subprocess, "run", run)
    args = [str(output), f"{asset}:Fixture", "--blender", str(blender), "--clay", "--width", "180"]
    if result_has_frame:
        assert render_glb_comparison.main(args) == 0
        with Image.open(output) as sheet:
            assert sheet.width == 180
            assert sheet.getpixel((90, sheet.height * 3 // 4))[0] > 240
    else:
        with pytest.raises(SystemExit, match="render failed"):
            render_glb_comparison.main(args)
        assert not output.exists()
