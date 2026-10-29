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
    with pytest.raises(SystemExit, match=r"step failed \(1\): the real error; see "):
        run.run([sys.executable, "-u", "-c", script])   # -u, as every real step runs
    out = capsys.readouterr().out
    assert "the real error" in out and "line 29" in out
    assert "line 0\n" not in out                      # only the end, so the report keeps room
    assert "line 0" in (tmp_path / "run.log").read_text()


def test_a_step_that_works_prints_nothing_of_its_own(capsys, tmp_path):
    import sys

    hero_route.Run(tmp_path).run([sys.executable, "-c", "print('quiet')"])
    assert "quiet" not in capsys.readouterr().out.splitlines()   # the $ line names it; output does not


def test_a_step_that_works_on_its_second_try_prints_nothing_of_its_own(capsys, tmp_path):
    import sys

    flag = tmp_path / "tried"
    script = (f"import pathlib, sys; f = pathlib.Path({str(flag)!r})\n"
              "if not f.exists(): f.write_text('x'); print('first go broke'); sys.exit(1)")
    assert hero_route.Run(tmp_path).run([sys.executable, "-c", script], tries=2)
    assert "first go broke" not in capsys.readouterr().out.splitlines()


def test_the_log_takes_any_folder_name_and_any_byte(tmp_path):
    import sys

    run = hero_route.Run(tmp_path / "Ünïcødé 名前")
    run.say("hero route: 名前.png")
    run.run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ok \\x81\\xff\\n')"])
    run.flag("名前 looked odd")
    text = (tmp_path / "Ünïcødé 名前" / "run.log").read_text(encoding="utf-8")
    assert "名前.png" in text and "ok " in text
    assert "名前 looked odd" in hero_route.Run(tmp_path / "Ünïcødé 名前").unsure


def test_hero_stops_before_any_work_on_a_build_that_cannot_save_its_clean_up(monkeypatch, tmp_path, capsys):
    import sys

    import pixal3d_generate

    monkeypatch.setattr(hero_route, "needed_parts", dict)
    monkeypatch.setattr(pixal3d_generate, "dump_supported", lambda cli=None: False)
    monkeypatch.setattr(sys, "argv", ["hero_route.py", str(tmp_path / "x.png"), "--out", str(tmp_path / "run")])
    assert hero_route.main() == 2
    assert "Rebuild for Pixal3D" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()                # nothing started


def _fake_finish(calls):
    def finish(run, model, picture, views, out, faces, dump, classic=False):
        calls.append((model.name, out.name, classic))
        out.write_text("finished")
    return finish


def test_a_head_clean_up_that_crashes_takes_the_classic_finish(monkeypatch, tmp_path):
    run = hero_route.Run(tmp_path)
    monkeypatch.setattr(run, "run", lambda command, **_: False)   # post-replay exits 1, as on Windows
    calls = []
    monkeypatch.setattr(hero_route, "finish", _fake_finish(calls))
    raw, clean, head = tmp_path / "head_raw.glb", tmp_path / "head_clean.glb", tmp_path / "head.glb"
    hero_route.finish_head(run, raw, tmp_path / "crop.png", clean, head, 16000)
    assert calls == [("head_raw.glb", "head.glb", True)]           # once, classic, straight from the raw head
    assert head.read_text() == "finished" and clean.exists()       # a rerun skips both steps
    assert any("classic" in note for note in run.unsure)


def test_a_head_clean_up_that_works_is_pixel_matched_as_it_is(monkeypatch, tmp_path):
    run = hero_route.Run(tmp_path)
    monkeypatch.setattr(hero_route, "head_cleanup", lambda run, raw, out, faces: out.write_text("clean") or True)
    calls = []
    monkeypatch.setattr(hero_route, "finish", _fake_finish(calls))
    hero_route.finish_head(run, tmp_path / "head_raw.glb", tmp_path / "crop.png",
                           tmp_path / "head_clean.glb", tmp_path / "head.glb", 16000)
    assert calls == [("head_clean.glb", "head.glb", False)]


def test_finish_asks_for_the_route_it_was_given(monkeypatch, tmp_path):
    import json

    run = hero_route.Run(tmp_path)
    commands = []
    monkeypatch.setattr(run, "run", lambda command, **_: commands.append([str(c) for c in command]) or True)
    out = tmp_path / "head.glb"
    out.with_suffix(".retopo-repaint.json").write_text(json.dumps({}))
    for classic in (True, False):
        hero_route.finish(run, tmp_path / "raw.glb", tmp_path / "p.png", tmp_path / "v", out, 16000, None, classic)
    classic_cmd, kept_cmd = commands
    assert "--keep-mesh" not in classic_cmd and "--native-dump" not in classic_cmd
    assert "--keep-mesh" in kept_cmd


def test_the_measuring_rig_gets_free_ports_like_the_rig_tab(monkeypatch, tmp_path):
    # Hero rigs the body to find the head; on a Windows machine that reserves SkinTokens'
    # fixed ports (#193) that rig failed the whole run.
    import sys

    sys.path.insert(0, str(hero_route.REPO / "viewer"))
    import autorig_api

    seen = tmp_path / "ports.txt"
    script = (f"import os; open({str(seen)!r}, 'w').write("
              "os.environ['SKINTOKENS_PORT'] + ' ' + os.environ['SKINTOKENS_BPY_PORT'])")
    monkeypatch.setattr(autorig_api, "rig_commands",
                        lambda body, rig: [([sys.executable, "-c", script], tmp_path)])
    monkeypatch.setattr(autorig_api, "ensure_port_patch", lambda root=None: None)
    hero_route.measure_rig(hero_route.Run(tmp_path), tmp_path / "body.glb", tmp_path / "rig.glb")
    port, bpy_port = seen.read_text().split()
    assert port != bpy_port
