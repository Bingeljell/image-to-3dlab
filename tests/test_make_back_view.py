"""Tests for scripts/make_back_view.py (no GPU: the command and the guards only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import make_back_view as mbv

WEIGHTS = {"diffusion_model": Path("/w/dit.gguf"), "llm": Path("/w/llm.gguf"),
           "vae": Path("/w/vae.safetensors")}


def test_the_back_is_drawn_at_768_in_8_steps_from_the_reference():
    command = mbv.command_for(Path("/r/ref.png"), Path("/o/back.png"), 3, WEIGHTS,
                              Path("/w/mmproj.gguf"), binary=Path("/bin/sd-cli"))
    assert command[command.index("-W") + 1] == "768" and command[command.index("-H") + 1] == "768"
    assert command[command.index("--steps") + 1] == "8"
    assert command[command.index("-r") + 1] == "/r/ref.png"
    assert command[command.index("-p") + 1] == mbv.back_view.BACK_PROMPT


def test_a_missing_vision_file_stops_and_says_how_to_get_it(monkeypatch, tmp_path, capsys):
    front = tmp_path / "front.png"
    Image.new("RGB", (8, 8)).save(front)
    monkeypatch.setattr(mbv.image_api, "resolve_vision", lambda *a: None)
    monkeypatch.setattr(mbv.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("ran Qwen")))
    assert mbv.main([str(front), str(tmp_path / "back.png")]) == 1
    out = capsys.readouterr().out
    assert "mmproj" in out and "0.75 GB" in out and "--vision" in out


def test_a_real_cut_out_is_used_as_it_is(tmp_path, monkeypatch):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = (10, 200, 10, 255)  # subject; the rest is clear
    front = tmp_path / "front.png"
    Image.fromarray(rgba, "RGBA").save(front)
    monkeypatch.setattr(mbv.matte, "cut_out", lambda *a: (_ for _ in ()).throw(
        AssertionError("cut out again")))
    ref = mbv.reference_for(front, tmp_path / "ref.png")
    pixels = np.asarray(Image.open(ref))
    assert tuple(pixels[16, 16]) == (10, 200, 10)
    assert tuple(pixels[0, 0]) == (mbv.back_view.GREY,) * 3
