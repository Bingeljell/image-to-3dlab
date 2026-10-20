"""POST /api/fix-spot: a fixed copy saved as a new Finish run, the original untouched."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("fixspot_api", REPO / "viewer" / "fixspot_api.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)

from test_spot_fix import _sphere_with_pocket


def _finished(tmp_path):
    run = tmp_path / "out" / "finish" / "ball__finish__20261020-120000"
    (run / "input").mkdir(parents=True)
    glb, _pos, _front = _sphere_with_pocket(tmp_path)
    (run / "ball_50k.glb").write_bytes(glb.read_bytes())
    (run / "input" / "source.glb").write_bytes(b"raw")
    return tmp_path / "out", "finish/ball__finish__20261020-120000/ball_50k.glb"


def test_a_fix_is_a_new_version_and_the_original_is_kept(tmp_path, monkeypatch):
    monkeypatch.setitem(fx.spot_fix.RADII, "large", 0.15)   # the test sphere's triangles are coarse
    output, rel = _finished(tmp_path)
    before = (output / rel).read_bytes()
    status, body = fx.fix_spot_response(output, {"file": rel, "point": [0, 0, 0.9], "size": "large"})
    assert status == 200, body
    new = output / body["file"]
    assert new.is_file() and new.parent != (output / rel).parent
    assert fx.FINISH_RUN.fullmatch(new.parent.name)
    assert (new.parent / "input" / "source.glb").read_bytes() == b"raw"
    assert (output / rel).read_bytes() == before
    assert body["cut"] > 0


def test_only_finished_models_can_be_fixed(tmp_path):
    output, _rel = _finished(tmp_path)
    for bad in ["../../etc/passwd.glb", "finish/ball__finish__20261020-120000/../x.glb", "images/a.glb", 3, None]:
        assert fx.fix_spot_response(output, {"file": bad, "point": [0, 0, 1]})[0] == 400


def test_the_point_and_size_are_checked(tmp_path):
    output, rel = _finished(tmp_path)
    assert fx.fix_spot_response(output, {"file": rel, "point": [0, 0]})[0] == 400
    assert fx.fix_spot_response(output, {"file": rel, "point": [0, 0, float("nan")]})[0] == 400
    assert fx.fix_spot_response(output, {"file": rel, "point": [0, 0, 1], "size": "huge"})[0] == 400


def test_a_click_off_the_model_saves_nothing(tmp_path):
    output, rel = _finished(tmp_path)
    status, _ = fx.fix_spot_response(output, {"file": rel, "point": [9, 9, 9]})
    assert status == 422
    assert len(list((output / "finish").iterdir())) == 1


def test_two_fixes_in_the_same_second_get_their_own_folders(tmp_path):
    run = tmp_path / "x__finish__20261020-120000"
    run.mkdir()
    first = fx.new_run_dir(run, now=0)
    first.mkdir()
    assert fx.new_run_dir(run, now=0) != first


def test_the_studio_slider_sends_a_share_of_the_height(tmp_path):
    output, rel = _finished(tmp_path)
    assert fx.fix_spot_response(output, {"file": rel, "point": [0, 0, 0.9], "radius": 0.5})[0] == 400
    assert fx.fix_spot_response(output, {"file": rel, "point": [0, 0, 0.9], "radius": "big"})[0] == 400
    status, body = fx.fix_spot_response(output, {"file": rel, "point": [0, 0, 0.9], "radius": 0.08})
    assert status in (200, 422), body
