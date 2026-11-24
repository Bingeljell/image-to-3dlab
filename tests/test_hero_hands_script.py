"""The Hero hands command's own logic: what it checks before starting, and the licence record it keeps.

The heavy steps (the picture editor, Pixal3D, its clean-up) run real models and are exercised by hand; these pin
the cheap parts that decide whether a run starts at all and what its result's licence record says.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def test_a_builds_drawing_joins_the_licence_ancestry(tmp_path):
    from image_to_3dlab.provenance import write_derived_sidecar

    # shaped as viewer/image_api.provenance writes it: "output" is a bare file name, and it once stopped the run
    drawing = {"schema_version": 1, "kind": "generated-image", "model": {"id": "flux2-klein-4b"},
               "license": {"name": "Apache-2.0", "classification": "commercial-clear",
                           "inherited_by_derivatives": "yours"}, "output": "right_draw_s3.png"}
    (tmp_path / "right_draw_s3.provenance.json").write_text(json.dumps(drawing))
    record = hero_hands._picture_record(tmp_path / "right_draw_s3_b42.glb")
    assert record["classification"] == "commercial-clear" and record["license"]["name"] == "Apache-2.0"
    assert hero_hands._picture_record(tmp_path / "left_draw_s1_b7.glb") is None
    model, result = tmp_path / "rigged.glb", tmp_path / "rigged_hands.glb"
    model.write_bytes(b"a")
    result.write_bytes(b"b")
    source = {"license": {"name": "MIT"}, "output": {"classification": "commercial-clear"}}
    sidecar = write_derived_sidecar(model, result, operation="hero_hands", source_record=source,
                                    component_records=[record])
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["output"]["classification"] == "commercial-clear"
    assert "Apache-2.0" in written["license"]["name"] and written["components"][0]["license"] == "Apache-2.0"


def test_the_defaults_are_round_22s_tested_ones():
    # 2 of 8 drawings had five digits, and both seeds built them right (round 22 of the AssetFurnace Support notes)
    assert (hero_hands.TRIES, hero_hands.KEEP, hero_hands.SEEDS) == (8, 2, (42, 7))


def test_a_build_with_a_hole_for_a_palm_is_not_kept(tmp_path):
    # Pixal3D built the Waechter's first gauntlet as a plate round an empty inside: five digits and a thumb, and a
    # hole where the palm should be, which showed on the figure
    raw = tmp_path / "right_draw_s3_b42.glb"
    counted = raw.with_suffix(".count.json")
    five = {"digits": 5, "thumbs": 1, "ok": True, "per_view": [5, 3, 2]}
    counted.write_text(json.dumps({**five, "hollow": 0.184}))
    assert not hero_hands.build_count(raw)["kept"]
    counted.write_text(json.dumps({**five, "hollow": 0.0}))
    assert hero_hands.build_count(raw)["kept"]
    counted.write_text(json.dumps({**five, "ok": False, "digits": 4, "hollow": 0.0}))
    assert not hero_hands.build_count(raw)["kept"]


def test_each_model_takes_the_hand_at_the_size_its_density_asks(monkeypatch, tmp_path):
    from tests.test_hero_hands import tiny_glb

    raw, reference = tmp_path / "right_draw_s3_b42.glb", tmp_path / "right_draw_s3_b42_clean20k.glb"
    raw.write_bytes(tiny_glb(skinned=False))                    # one triangle
    asked = []
    monkeypatch.setattr(hero_hands, "clean_hand", lambda run, path, faces: asked.append(faces) or tmp_path / "c.glb")
    hand = {"raw": raw, "reference": reference}
    assert hero_hands.glb_faces(raw) == 1
    assert hero_hands.hand_size(None, hand, 3000, rigged=True) == raw           # not finer than the raw build
    hand["raw_faces"] = 960058                                                  # a real raw build's size
    assert hero_hands.hand_size(None, hand, 3000, rigged=False) == tmp_path / "c.glb" and asked == [3000]
    # --hand-detail 0: as before the option, the reference clean-up for the rigged model, the raw for the others
    assert hero_hands.hand_size(None, hand, None, rigged=True) == reference
    assert hero_hands.hand_size(None, hand, None, rigged=False) == raw


def test_a_build_whose_wrist_will_not_close_is_not_kept(tmp_path):
    # the Waechter's gauntlet build (drawing 3, seed 7) counted five digits and a thumb with a solid palm, but at the
    # figure's cut its wrist was open plates: the stitch closed a seventh of it; the next build is tried instead
    import argparse

    folder = tmp_path / "hands"
    folder.mkdir()
    five = {"digits": 5, "thumbs": 1, "ok": True, "per_view": [5, 3, 2], "hollow": 0.0}
    four = {**five, "digits": 4, "ok": False}
    for name, count in (("right_draw_s3_b7", five), ("right_draw_s3_b42", four), ("right_draw_s6_b7", four),
                        ("right_draw_s6_b42", five)):
        (folder / f"{name}.glb").write_bytes(b"built")                    # built already: nothing is run
        (folder / f"{name}.count.json").write_text(json.dumps(count))
    run = hero_hands.Run(tmp_path)
    args = argparse.Namespace(seeds=[7, 42])
    drawings = [folder / "right_draw_s3.png", folder / "right_draw_s6.png"]
    asked = []

    def fits(raw):
        asked.append(raw.name)
        return 0.13 if "s3" in raw.name else 0.99

    assert hero_hands.build_hand(run, "right", drawings, folder, args, fits) == folder / "right_draw_s6_b42.glb"
    assert asked == ["right_draw_s3_b7.glb", "right_draw_s6_b42.glb"]
    assert "closes only 13%" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_both_hands_of_one_build_share_one_clean_up(tmp_path):
    # a left hand mirrored from the right asked for 33k faces where the right asked for 32k: two clean-ups of one build
    one, other = tmp_path / "right_draw_s6_b42.glb", tmp_path / "left_draw_s2_b7.glb"
    hands = {"right": {"raw": one, "area": 0.08}, "left": {"raw": one, "area": 0.085}}
    sizes = hero_hands.clean_sizes(35000, hands, 1.0)
    assert sizes["right"] == sizes["left"] == 3000
    hands["left"]["raw"] = other                                              # --left own: a build of its own
    assert hero_hands.clean_sizes(35000, hands, 1.0) == {"right": 3000, "left": 3000}
    hands["left"]["area"] = 0.2
    assert hero_hands.clean_sizes(35000, hands, 1.0) == {"right": 3000, "left": 7000}
    assert hero_hands.clean_sizes(35000, hands, 0.0) == {"right": None, "left": None}


def test_a_step_that_always_runs_forgets_its_old_notes(tmp_path):
    run = hero_hands.Run(tmp_path)
    run.current = "check"
    run.flag("the grafted right hand counts 5 digits and 2 thumb(s)")
    run.current = "draw"
    run.flag("the picture editor failed on seed 3")
    again = hero_hands.Run(tmp_path)                          # a rerun reads the notes back from disk ...
    hero_hands.fresh(again, "check")                          # ... and the check runs again
    assert again.unsure == ["the picture editor failed on seed 3"]
    assert "check" not in json.loads((tmp_path / "unsure.json").read_text(encoding="utf-8"))


def drawn(tmp_path, colour, name="right_draw_s2"):
    """A drawing on grey and the cut-out Pixal3D keeps beside it: COLOUR where the subject is."""
    from PIL import Image

    picture = Image.new("RGB", (8, 8), (128, 128, 128))
    picture.paste(colour, (2, 2, 6, 6))
    picture.save(tmp_path / f"{name}.png")
    cut = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    cut.paste((*colour, 255), (2, 2, 6, 6))
    cut.save(tmp_path / f"{name}__matted.png")
    return tmp_path / f"{name}.png"


def test_the_new_hands_take_their_drawings_tone(monkeypatch, tmp_path):
    # the summoner (round 28 of the AssetFurnace Support notes): Pixal3D's clean-up painted the new skin 59/17/24
    # (L*a*b*) where the drawing and the figure's own hands were 69/7/13
    from image_to_3dlab import hero_hands as hh
    from tests.test_hero_hands import painted_glb

    _, _, built = hh.read_part(painted_glb((184, 130, 100)))
    _, _, both = hh.read_part(painted_glb((184, 130, 100)))
    monkeypatch.setattr(hh, "TONE_LEAST_FACES", 1)
    run = hero_hands.Run(tmp_path)
    figure = np.array([[190.0, 163, 145]])
    toned = hero_hands.match_paint(run, tmp_path / "figure_hands.glb", built, drawn(tmp_path, (190, 163, 145)), both,
                                   figure)
    medians = toned["paint_lab_median"]
    assert toned["paint"] == "matched"
    assert np.abs(np.subtract(medians["matched"], medians["drawing"])).max() < 2
    assert np.abs(np.subtract(medians["built"], medians["drawing"])).max() > 8
    assert medians["figure"] == np.round(hh.to_lab(figure)[0], 1).tolist()      # reported beside it
    assert "the drawing's" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_hands_without_paint_to_match_are_kept(tmp_path):
    from image_to_3dlab import hero_hands as hh
    from tests.test_hero_hands import tiny_glb

    _, _, plain = hh.read_part(tiny_glb(skinned=False, with_image=False))
    run = hero_hands.Run(tmp_path)
    kept = hero_hands.match_paint(run, tmp_path / "f.glb", plain, drawn(tmp_path, (190, 163, 145)), dict(plain))
    assert kept["paint"].startswith("kept")


def test_a_build_names_its_drawing():
    assert hero_hands.drawing_of(Path("hands/right_draw_s2_b42.glb")) == Path("hands/right_draw_s2.png")
    assert hero_hands.drawing_of(Path("hands/left_draw_s10_b7_clean6500.glb")).name.startswith("left_draw_s10")
