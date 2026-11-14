"""Tests for `scripts/restyle_views.py`: the parts of the in-app restyle that need no GPU.

The shape guide's five views are repainted by the studio's own picture editor, one edit per
view with the view as the reference picture; the prompts and the file names are plain logic.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent))
import restyle_views as rv

PROMPTS = {
    "subject": "a modern 9 m European low-floor midibus",
    "shared": "Repaint this detailed model into a photorealistic photo of {subject}. Keep every part where it is.",
    "ending": "Plain light grey background.",
    "sides": {"front": "Seen straight from the front.", "top": "Seen from directly above."},
}


def test_a_view_prompt_is_the_shared_part_its_side_and_the_ending():
    prompt = rv.view_prompt(PROMPTS, "front")
    assert prompt == ("Repaint this detailed model into a photorealistic photo of a modern 9 m European "
                      "low-floor midibus. Keep every part where it is. Seen straight from the front. "
                      "Plain light grey background.")


def test_a_side_without_its_own_words_gets_the_shared_part_and_the_ending():
    assert rv.view_prompt(PROMPTS, "left").endswith("Keep every part where it is. Plain light grey background.")


def test_the_pictures_come_with_their_side_and_their_rig_view_from_the_crops_file(tmp_path):
    crops = {"pictures": [{"file": "midibus_1_front.png", "side": "front", "rig_view": "view_000.png"},
                          {"file": "midibus_5_top.png", "side": "top", "rig_view": "view_top.png"}]}
    (tmp_path / "crops.json").write_text(json.dumps(crops), encoding="utf-8")
    for name in ("midibus_1_front.png", "midibus_5_top.png"):
        (tmp_path / name).write_bytes(b"png")
    views = rv.views_of(tmp_path)
    assert views == [(tmp_path / "midibus_1_front.png", "front", "view_000"),
                     (tmp_path / "midibus_5_top.png", "top", "view_top")]


def test_the_edit_reads_the_view_as_its_reference_at_the_asked_size():
    weights = {"diffusion_model": Path("m.gguf"), "llm": Path("l.gguf"), "vae": Path("v.safetensors")}
    command = rv.edit_command("paint it", Path("front.png"), Path("out.png"), weights, size=1024, steps=8,
                              seed=42, binary=Path("sd-cli"))
    assert command[0] == "sd-cli"
    assert command[command.index("-r") + 1] == "front.png"
    assert command[command.index("-W") + 1] == "1024" and command[command.index("-H") + 1] == "1024"
    assert command[command.index("--steps") + 1] == "8" and command[command.index("--seed") + 1] == "42"
    assert command[command.index("-o") + 1] == "out.png"


def test_with_a_strength_the_edit_starts_from_the_view_so_its_outline_stays():
    # FLUX.2 klein, given the view only as a reference, redrew a side as a three-quarter view
    # (2026-11-14); starting from the view itself (img2img) keeps every edge where it was
    weights = {"diffusion_model": Path("m.gguf"), "llm": Path("l.gguf"), "vae": Path("v.safetensors")}
    command = rv.edit_command("paint it", Path("left.png"), Path("out.png"), weights, size=1024, steps=8,
                              seed=42, binary=Path("sd-cli"), strength=0.7)
    assert command[command.index("-i") + 1] == "left.png"
    assert command[command.index("--strength") + 1] == "0.7"
    assert command[command.index("-r") + 1] == "left.png"


def test_without_a_strength_the_edit_reads_the_view_only_as_a_reference():
    weights = {"diffusion_model": Path("m.gguf"), "llm": Path("l.gguf"), "vae": Path("v.safetensors")}
    command = rv.edit_command("paint it", Path("left.png"), Path("out.png"), weights, size=1024, steps=8,
                              seed=42, binary=Path("sd-cli"), strength=None)
    assert "-i" not in command and "--strength" not in command
