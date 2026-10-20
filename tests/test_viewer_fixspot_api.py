"""The Fix a spot session: mark, Patch to a preview, Keep one version or Discard."""

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

RUN = "ball__finish__20261020-120000"
SPOT = {"point": [0, 0, 0.95], "radius": 0.08, "mode": "fill"}   # 0.08 x height 2 = the sphere's coarse cells


def _finished(tmp_path, n=48):
    run = tmp_path / "out" / "finish" / RUN
    (run / "input").mkdir(parents=True)
    glb, _pos, _front = _sphere_with_pocket(tmp_path, n=n)
    (run / "ball_50k.glb").write_bytes(glb.read_bytes())
    (run / "input" / "source.glb").write_bytes(b"raw")
    return tmp_path / "out", f"finish/{RUN}/ball_50k.glb"


def test_patch_makes_a_preview_not_a_version(tmp_path):
    output, rel = _finished(tmp_path)
    status, body = fx.preview_response(output, {"from": rel, "spots": [SPOT]})
    assert status == 200, body
    assert body["preview"].startswith(".fix-previews/")
    assert len(list((output / "finish").iterdir())) == 1, "no new version until Keep"


def test_keep_makes_one_version_from_many_spots_and_keeps_the_original(tmp_path):
    output, rel = _finished(tmp_path)
    before = (output / rel).read_bytes()
    second = {**SPOT, "point": [0.3, 0.2, 0.9]}
    _s, first = fx.preview_response(output, {"from": rel, "spots": [SPOT]})
    _s, again = fx.preview_response(output, {"from": rel, "base": first["preview"], "spots": [second]})
    assert not (output / first["preview"]).exists(), "the superseded preview is cleared"
    status, kept = fx.keep_response(output, {"from": rel, "preview": again["preview"]})
    assert status == 200, kept
    runs = sorted((output / "finish").iterdir())
    assert len(runs) == 2 and fx.FINISH_RUN.fullmatch(runs[1].name)
    assert (runs[1] / "input" / "source.glb").read_bytes() == b"raw"
    assert (output / rel).read_bytes() == before
    assert not (output / again["preview"]).exists()


def test_discard_removes_the_preview(tmp_path):
    output, rel = _finished(tmp_path)
    _s, body = fx.preview_response(output, {"from": rel, "spots": [SPOT]})
    assert fx.discard_response(output, {"preview": body["preview"]}) == (200, {"discarded": True})
    assert not (output / body["preview"]).exists()


def test_only_finished_models_and_our_own_previews_are_touched(tmp_path):
    output, rel = _finished(tmp_path)
    for bad in ["../../etc/passwd.glb", f"finish/{RUN}/../x.glb", "images/a.glb", 3, None]:
        assert fx.preview_response(output, {"from": bad, "spots": [SPOT]})[0] == 400
    assert fx.preview_response(output, {"from": rel, "base": rel, "spots": [SPOT]})[0] == 400
    assert fx.keep_response(output, {"from": rel, "preview": rel})[0] == 400
    assert fx.discard_response(output, {"preview": rel}) == (200, {"discarded": False})
    assert (output / rel).exists()


def test_spots_are_checked(tmp_path):
    output, rel = _finished(tmp_path)
    for bad in [[], [{"point": [0, 0], "radius": 0.02}], [{"point": [0, 0, 1], "radius": 0.5}],
                [{"point": [0, 0, 1], "radius": 0.02, "mode": "melt"}], [{"point": [0, 0, float("nan")], "radius": 0.02}],
                [SPOT] * (fx.MAX_SPOTS + 1)]:
        assert fx.preview_response(output, {"from": rel, "spots": bad})[0] == 400


def test_repaint_needs_the_picture(tmp_path):
    output, rel = _finished(tmp_path)
    status, body = fx.preview_response(output, {"from": rel, "spots": [{**SPOT, "mode": "repaint"}]})
    assert status == 422 and "picture" in body["error"]


def test_a_click_off_the_model_saves_nothing(tmp_path):
    output, rel = _finished(tmp_path)
    status, _ = fx.preview_response(output, {"from": rel, "spots": [{**SPOT, "point": [9, 9, 9]}]})
    assert status == 422
    assert not list((output / fx.PREVIEWS).glob("*.glb"))


def test_two_keeps_in_the_same_second_get_their_own_folders(tmp_path):
    run = tmp_path / "x__finish__20261020-120000"
    run.mkdir()
    first = fx.new_run_dir(run, now=0)
    first.mkdir()
    assert fx.new_run_dir(run, now=0) != first
