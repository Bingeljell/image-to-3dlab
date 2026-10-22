"""The hero command's own logic: what it checks before starting, and what the studio reads.

The heavy steps run real models and are exercised by hand; these pin the cheap parts that
decide whether a 25-minute run starts at all and whether the studio can follow it.
"""
from __future__ import annotations

import io

from scripts import hero_route


def test_missing_parts_names_only_what_is_absent(tmp_path):
    present = tmp_path / "here"
    present.write_text("x")
    parts = {"Pixal3D": present, "Auto-rig": tmp_path / "gone"}
    assert hero_route.missing_parts(parts) == ["Auto-rig"]
    assert hero_route.missing_parts({"Pixal3D": present}) == []


def test_stage_prints_the_marker_the_studio_reads(capsys, tmp_path):
    class Run:
        log = io.StringIO()

    hero_route.stage(Run, "head", "Building the head")
    assert capsys.readouterr().out == "I2L_STAGE::head::Building the head\n"
    assert "I2L_STAGE::head::Building the head" in Run.log.getvalue()


def test_stages_are_in_running_order():
    assert hero_route.STAGES == ("body", "measure", "crop", "head", "graft", "join", "check")


def test_hero_needs_the_upscaler_not_the_paint_install():
    parts = hero_route.needed_parts()
    assert any("upscaler" in name.lower() for name in parts)
    assert not any("hunyuan" in name.lower() or "hunyuan" in str(path).lower() for name, path in parts.items())
