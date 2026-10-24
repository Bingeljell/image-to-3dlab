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


def test_hero_runs_its_steps_with_the_python_running_it():
    """Not a hard-coded .venv/bin/python: that path does not exist on Windows."""
    import sys
    from pathlib import Path

    assert hero_route.PYTHON == Path(sys.executable)
    assert hero_route.PYTHON.exists()


def test_hero_checks_for_pixal3ds_own_clean_up_before_starting():
    from image_to_3dlab import pixal3d_native

    assert pixal3d_native.POST_REPLAY in hero_route.needed_parts().values()


def _body(tmp_path, views=True, dump=True):
    body = tmp_path / "model.glb"
    body.write_bytes(b"glb")
    folder = tmp_path / "model.svviews"
    if views:
        folder.mkdir()
        (folder / "transforms.json").write_text("{}")
    post = tmp_path / "model.post.bin"
    if dump:
        post.write_bytes(b"dump")
    return body, folder, post


def test_a_body_with_cameras_and_a_dump_is_accepted(tmp_path):
    assert hero_route.body_problem(*_body(tmp_path)) is None


def test_a_body_without_cameras_is_refused(tmp_path):
    assert "cameras" in hero_route.body_problem(*_body(tmp_path, views=False))


def test_a_body_without_its_clean_up_dump_is_refused_in_plain_words(tmp_path):
    """Without the dump the body kept its raw million triangles, silently."""
    problem = hero_route.body_problem(*_body(tmp_path, dump=False))
    assert problem and "again" in problem


def test_no_body_given_is_no_problem():
    assert hero_route.body_problem(None, None, None) is None


def test_a_failed_step_prints_the_end_of_its_own_output(capsys, tmp_path):
    import sys

    import pytest

    run = hero_route.Run(tmp_path)
    script = "import sys\nfor i in range(30): print(f'line {i}')\nsys.exit('the real error')"
    with pytest.raises(SystemExit, match="step failed"):
        run.run([sys.executable, "-c", script])
    out = capsys.readouterr().out
    assert "the real error" in out and "line 29" in out
    assert "line 0\n" not in out                      # only the end, so the report keeps room
    assert "line 0" in (tmp_path / "run.log").read_text()


def test_a_step_that_works_prints_nothing_of_its_own(capsys, tmp_path):
    import sys

    hero_route.Run(tmp_path).run([sys.executable, "-c", "print('quiet')"])
    assert "quiet" not in capsys.readouterr().out.splitlines()   # the $ line names it; output does not
