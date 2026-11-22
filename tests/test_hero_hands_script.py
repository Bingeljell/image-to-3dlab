"""The Hero hands command's own logic: what it checks before starting, and the licence record it keeps.

The heavy steps (the picture editor, Pixal3D, its clean-up) run real models and are exercised by hand; these pin
the cheap parts that decide whether a run starts at all and what its result's licence record says.
"""
from __future__ import annotations

import json

from scripts import hero_hands


def test_hero_hands_stops_before_any_work_when_a_part_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hero_hands, "needed_parts", lambda: {"Pixal3D (3D models)": tmp_path / "gone"})
    monkeypatch.setattr(hero_hands, "picture_engine",
                        lambda: (None, "the picture editor (FLUX.2 klein, edit mode) is not set up"))
    code = hero_hands.main([str(tmp_path / "rigged.glb"), str(tmp_path / "front.png"), "--out", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert code == 2
    assert "Pixal3D (3D models)" in err and "picture editor" in err and "nothing was downloaded" in err
    assert not (tmp_path / "run").exists()                   # nothing started, nothing written


def test_a_builds_drawing_lends_its_licence_with_the_classification_on_top(tmp_path):
    drawing = {"kind": "generated-image", "license": {"name": "Apache-2.0", "classification": "commercial-clear"}}
    (tmp_path / "right_draw_s3.provenance.json").write_text(json.dumps(drawing))
    record = hero_hands._picture_record(tmp_path / "right_draw_s3_b42.glb")
    assert record["classification"] == "commercial-clear" and record["license"]["name"] == "Apache-2.0"
    assert hero_hands._picture_record(tmp_path / "left_draw_s1_b7.glb") is None


def test_the_defaults_are_round_22s_tested_ones():
    # 2 of 8 drawings had five digits, and both seeds built them right (round 22 of the AssetFurnace Support notes)
    assert (hero_hands.TRIES, hero_hands.KEEP, hero_hands.SEEDS) == (8, 2, (42, 7))
