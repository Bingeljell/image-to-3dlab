"""Tests for scripts/make_side_views.py (no GPU: the prompt, the mirror, the command and the guards)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import make_side_views as msv

WEIGHTS = {"diffusion_model": Path("/w/dit.gguf"), "llm": Path("/w/llm.gguf"),
           "vae": Path("/w/vae.safetensors")}


def test_the_side_prompt_names_no_parts_and_says_which_way_the_subject_faces():
    prompt = msv.back_view.SIDE_PROMPT
    assert "right side" in prompt and "profile" in prompt and "right edge" in prompt
    assert "turnaround sheet" in prompt   # the wording that gave true profiles (back_view.py)
    # naming a part summons it (back_view's rule): no legs, tails, arms or heads in the words
    named = re.findall(r"\b(legs?|tails?|arms?|heads?|wheels?|ears?|eyes?)\b", prompt.lower())
    assert named == []


def test_the_left_side_is_the_right_side_mirrored():
    right = np.zeros((4, 6, 3), dtype=np.uint8)
    right[:, 0] = (200, 10, 10)      # the subject's back at the left edge ...
    right[:, 5] = (10, 10, 200)      # ... its front at the right edge
    left = np.asarray(msv.back_view.mirror_side(Image.fromarray(right)))
    assert tuple(left[0, 0]) == (10, 10, 200) and tuple(left[0, 5]) == (200, 10, 10)


def test_the_right_side_is_drawn_from_the_reference_with_the_side_prompt():
    engine = msv.image_api.FLUX
    command = msv.command_for(Path("/r/ref.png"), Path("/o/right.png"), 3, WEIGHTS, None, engine,
                              binary=Path("/bin/sd-cli"))
    assert command[command.index("-r") + 1] == str(Path("/r/ref.png"))  # a Windows path on Windows
    assert command[command.index("-p") + 1] == msv.back_view.SIDE_PROMPT
    assert command[command.index("--seed") + 1] == "3"


def test_an_engine_that_needs_a_vision_file_stops_without_it(monkeypatch, tmp_path, capsys):
    front = tmp_path / "front.png"
    Image.new("RGB", (8, 8)).save(front)
    monkeypatch.setattr(msv.image_api, "pick_engine", lambda *a, **k: msv.image_api.QWEN)
    monkeypatch.setattr(msv.image_api, "resolve_vision", lambda *a: None)
    monkeypatch.setattr(msv.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("drew a picture")))
    assert msv.main([str(front), str(tmp_path / "r.png"), str(tmp_path / "l.png")]) == 1
    assert "vision" in capsys.readouterr().out


def test_a_missing_front_is_refused(tmp_path):
    try:
        msv.main([str(tmp_path / "nope.png"), str(tmp_path / "r.png"), str(tmp_path / "l.png")])
    except SystemExit as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("a missing front was accepted")


def test_the_left_is_written_as_the_mirror_and_both_get_a_record(monkeypatch, tmp_path):
    front = tmp_path / "front.png"
    Image.new("RGB", (16, 16), (200, 200, 200)).save(front)
    monkeypatch.setattr(msv.image_api, "pick_engine", lambda *a, **k: msv.image_api.FLUX)
    monkeypatch.setattr(msv.image_api, "resolve_weights", lambda *a: WEIGHTS)
    monkeypatch.setattr(msv, "reference_for", lambda f, d: d)

    def fake_run(command, **kw):
        out = Path(command[command.index("-o") + 1])
        half = np.zeros((8, 8, 3), dtype=np.uint8)
        half[:, 7] = (10, 10, 200)       # the subject's front at the right edge
        Image.fromarray(half).save(out)

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    monkeypatch.setattr(msv.subprocess, "run", fake_run)
    right, left = tmp_path / "right.png", tmp_path / "left.png"
    assert msv.main([str(front), str(right), str(left), "--seed", "5"]) == 0
    assert tuple(np.asarray(Image.open(left))[0, 0]) == (10, 10, 200)
    assert right.with_suffix(".provenance.json").is_file() and left.with_suffix(".provenance.json").is_file()
    assert "mirror" in left.with_suffix(".provenance.json").read_text()


def _side_png(path: Path, legs_apart: bool) -> None:
    """A cut-out side view (alpha), the subject facing right: four legs apart, or one block to the floor."""
    rgba = np.zeros((120, 200, 4), dtype=np.uint8)
    rgba[..., :3] = 90
    rgba[10:70, 20:180, 3] = 255
    if legs_apart:
        rgba[70:100, 20:50, 3] = 255
        rgba[70:100, 150:180, 3] = 255
    else:
        rgba[70:100, 20:180, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _fake_drawer(monkeypatch, apart_from_seed: int, seeds: list):
    monkeypatch.setattr(msv.image_api, "pick_engine", lambda *a, **k: msv.image_api.FLUX)
    monkeypatch.setattr(msv.image_api, "resolve_weights", lambda *a: WEIGHTS)
    monkeypatch.setattr(msv, "reference_for", lambda f, d: d)

    def fake_run(command, **kw):
        seed = int(command[command.index("--seed") + 1])
        seeds.append(seed)
        _side_png(Path(command[command.index("-o") + 1]), legs_apart=seed >= apart_from_seed)

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    monkeypatch.setattr(msv.subprocess, "run", fake_run)


def test_a_four_legged_side_whose_legs_fuse_is_drawn_again_with_the_next_seed(monkeypatch, tmp_path, capsys):
    front = tmp_path / "front.png"
    Image.new("RGB", (16, 16), (200, 200, 200)).save(front)
    seeds: list = []
    _fake_drawer(monkeypatch, apart_from_seed=5, seeds=seeds)
    right, left = tmp_path / "right.png", tmp_path / "left.png"
    assert msv.main([str(front), str(right), str(left), "--seed", "3", "--four-legged"]) == 0
    assert seeds == [3, 4, 5]                     # drawn until the legs stood apart
    _, found = msv.leg_check.check_picture(right, "side")
    assert found == []                            # the kept picture is the one that passed
    record = right.with_suffix(".provenance.json").read_text()
    assert '"seed": 5' in record and "open_under_body" in record
    assert "legs apart" in capsys.readouterr().out
    assert not list(tmp_path.glob("*__try*"))     # the rejected tries are cleaned up


def test_when_no_try_passes_the_best_is_kept_and_named(monkeypatch, tmp_path, capsys):
    front = tmp_path / "front.png"
    Image.new("RGB", (16, 16), (200, 200, 200)).save(front)
    seeds: list = []
    _fake_drawer(monkeypatch, apart_from_seed=99, seeds=seeds)
    right, left = tmp_path / "right.png", tmp_path / "left.png"
    assert msv.main([str(front), str(right), str(left), "--four-legged", "--tries", "2"]) == 0
    assert len(seeds) == 2 and right.is_file() and left.is_file()
    out = capsys.readouterr().out
    assert "legs still touch" in out and "open under the body" in out


def test_without_four_legged_one_picture_is_drawn_and_not_checked(monkeypatch, tmp_path):
    front = tmp_path / "front.png"
    Image.new("RGB", (16, 16), (200, 200, 200)).save(front)
    seeds: list = []
    _fake_drawer(monkeypatch, apart_from_seed=99, seeds=seeds)
    assert msv.main([str(front), str(tmp_path / "r.png"), str(tmp_path / "l.png")]) == 0
    assert seeds == [3]


def test_draw_left_draws_the_left_side_from_the_mirrored_front(monkeypatch, tmp_path):
    # a subject whose sides differ (a tail curled to one side): the mirror of the right side would put the tail on
    # both sides, so the left side is drawn too: from the front mirrored, then mirrored back
    front = tmp_path / "front.png"
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :4] = (220, 30, 30)                       # something on the picture's left only
    Image.fromarray(rgb).save(front)
    monkeypatch.setattr(msv.image_api, "pick_engine", lambda *a, **k: msv.image_api.FLUX)
    monkeypatch.setattr(msv.image_api, "resolve_weights", lambda *a: WEIGHTS)
    refs = []

    def fake_reference(f, d):
        refs.append(np.asarray(Image.open(f).convert("RGB"))[0, 0].tolist())
        return d
    monkeypatch.setattr(msv, "reference_for", fake_reference)

    def fake_run(command, **kw):
        out = Path(command[command.index("-o") + 1])
        half = np.zeros((8, 8, 3), dtype=np.uint8)
        half[:, 7] = (10, 10, 200)                   # whatever is drawn faces right
        Image.fromarray(half).save(out)

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()
    monkeypatch.setattr(msv.subprocess, "run", fake_run)
    right, left = tmp_path / "right.png", tmp_path / "left.png"
    assert msv.main([str(front), str(right), str(left), "--draw-left"]) == 0
    assert refs == [[220, 30, 30], [0, 0, 0]]        # the front, then the front mirrored
    # the left picture is the drawing mirrored back: it faces left
    assert tuple(np.asarray(Image.open(left))[0, 0]) == (10, 10, 200)
    assert "drawn from the mirrored front" in left.with_suffix(".provenance.json").read_text()
