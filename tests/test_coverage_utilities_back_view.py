"""Check back-view command persistence without starting an image model."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts import make_back_view


@pytest.mark.parametrize(("returncode", "creates_picture"), [(0, True), (3, False), (0, False)])
def test_back_view_records_provenance_only_when_process_creates_picture(
    monkeypatch, tmp_path, capsys, returncode, creates_picture,
):
    front = tmp_path / "front.png"
    back = tmp_path / "subdir" / "back.png"
    Image.new("RGB", (4, 4), (0, 200, 0)).save(front)
    weights = {key: tmp_path / f"{key}.test" for key in ("diffusion_model", "llm", "vae")}
    vision = tmp_path / "vision.test"
    monkeypatch.setattr(make_back_view.image_api, "resolve_weights", lambda: weights)
    monkeypatch.setattr(make_back_view.image_api, "resolve_vision", lambda: vision)
    matte_calls = []

    def cut_out(pixels):
        matte_calls.append(pixels.mode)
        matte = pixels.convert("RGBA")
        matte.putpixel((0, 0), (0, 0, 0, 0))
        return matte, {}

    def run(command, **kwargs):
        assert command[command.index("--seed") + 1] == "5"
        assert command[command.index("--llm_vision") + 1] == str(vision)
        assert kwargs["cwd"] == str(make_back_view.REPO)
        reference = Path(command[command.index("-r") + 1])
        assert reference == back.with_name("back__reference.png")
        with Image.open(reference) as generated_reference:
            assert generated_reference.getpixel((0, 0)) == (make_back_view.back_view.GREY,) * 3
            assert generated_reference.getpixel((2, 2)) == (0, 200, 0)
        if creates_picture:
            Image.new("RGB", (4, 4), (200, 0, 0)).save(command[command.index("-o") + 1])
        return subprocess.CompletedProcess(command, returncode, "diagnostic ", "tail")

    monkeypatch.setattr(make_back_view.matte, "cut_out", cut_out)
    monkeypatch.setattr(subprocess, "run", run)
    assert make_back_view.main([str(front), str(back), "--seed", "5"]) == (0 if creates_picture else 1)
    assert matte_calls == ["RGB"]
    sidecar = back.with_suffix(".provenance.json")
    assert sidecar.exists() == creates_picture
    if creates_picture:
        record = json.loads(sidecar.read_text())
        assert record["reference"] == {"front": str(front), "mode": "edit (back view)"}
        assert record["settings"]["seed"] == 5
        assert record["output"] == "back.png"
        assert record["license"]["classification"]
    else:
        assert "diagnostic tail" in capsys.readouterr().out


def test_missing_front_fails_before_resolving_model(monkeypatch, tmp_path):
    monkeypatch.setattr(make_back_view.image_api, "resolve_vision", lambda: pytest.fail("selected a model"))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("sys.argv", ["make_back_view.py", str(tmp_path / "missing.png"),
                                      str(tmp_path / "output.png")])
    with pytest.raises(SystemExit, match="not found:"):
        runpy.run_path(make_back_view.__file__, run_name="__main__")
