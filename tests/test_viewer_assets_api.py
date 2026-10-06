"""The studio Library: every run on disk grouped into assets, oldest picture to newest clip."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api as aa  # noqa: E402

IMG = "a-chunky-knight__20261005-205524"
GEN = f"{IMG}-cutout__pixal3d__20261005-210000"
FIN = "a-chunky-knight__20261005-205524-cutout__pixal3d__202610__finish__20261005-210615"
RIG = "a-chunky-knight_40k__rig__20261005-210643"


def _write(path: Path, data: bytes | str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    # one full character: picture -> 3D -> finish -> rig -> two clips
    _write(out / "images" / IMG / f"{IMG}.png", b"png", 1000)
    _write(out / "commercial_conditional" / GEN / f"{GEN}.glb", b"GENERATED", 2000)
    _write(out / "finish" / FIN / "a-chunky-knight_40k.glb", b"FINISHED", 3000)
    _write(out / "finish" / FIN / "input" / "source.glb", b"GENERATED", 3000)
    _write(out / "animate" / RIG / "a-chunky-knight_40k_rigged.glb", b"RIGGED", 4000)
    _write(out / "animate" / RIG / "a-chunky-knight_40k_rigged.provenance.json",
           json.dumps({"input": {"name": "a-chunky-knight_40k.glb",
                                 "sha256": hashlib.sha256(b"FINISHED").hexdigest()}}), 4000)
    _write(out / "animate" / RIG / "a-chunky-knight_40k_dance.glb", b"DANCE", 4100)
    _write(out / "animate" / RIG / "a-chunky-knight_40k_run.glb", b"RUN", 4200)
    _write(out / "animate" / RIG / "a-chunky-knight_40k_run_b.glb", b"RUN B", 4200)
    # a picture nobody turned into 3D
    _write(out / "images" / "a-lone-fox__20261006-080100" / "a-lone-fox__20261006-080100.png", b"png", 5000)
    # a 3D model with no picture on disk (uploaded or from before images were kept)
    _write(out / "research_only" / "orc__trellis__20260903-090915" / "orc__trellis__20260903-090915.glb", b"ORC", 500)
    # a prop sheet: its source.glb is a copy of a generated sheet model
    sheet = "a-3x3-grid-of-props__20261003-140407"
    sheet_gen = f"{sheet}__pixal3d__20261003-141000"
    _write(out / "commercial_conditional" / sheet_gen / f"{sheet_gen}.glb", b"SHEET", 600)
    props = out / "props" / f"{sheet}__pixal3d__20261003-141__props__20261003-141650"
    _write(props / "source.glb", b"SHEET", 700)
    for name in ("barrel", "chest"):
        _write(props / "finished" / name / f"{name}_LOD0.glb", name, 700)
    return out


def _by_name(assets):
    return {a["name"]: a for a in assets}


def test_a_full_chain_becomes_one_animated_asset(output):
    knight = _by_name(aa.list_assets(output))["a chunky knight"]
    assert knight["kind"] == "character"
    assert knight["stage"] == "animated"
    assert knight["picture"].endswith(f"{IMG}.png")
    assert knight["model"].endswith(f"{GEN}.glb")
    assert knight["finished"].endswith("a-chunky-knight_40k.glb")
    assert knight["rigged"].endswith("a-chunky-knight_40k_rigged.glb")
    # second takes ("_b") are not separate moves
    assert [c["name"] for c in knight["clips"]] == ["dance", "run"]


def test_loose_runs_still_show_up_at_the_step_they_reached(output):
    assets = _by_name(aa.list_assets(output))
    assert assets["a lone fox"]["stage"] == "picture"
    assert assets["a lone fox"]["model"] is None
    orc = assets["orc"]
    assert orc["stage"] == "model" and orc["picture"] is None


def test_a_prop_sheet_is_one_asset_holding_its_props(output):
    sheet = _by_name(aa.list_assets(output))["a 3x3 grid of props"]
    assert sheet["kind"] == "prop set"
    assert sheet["stage"] == "finished"
    assert sorted(p["name"] for p in sheet["props"]) == ["barrel", "chest"]
    # the generated sheet model belongs to the set, not to a second asset of its own
    holders = [a["name"] for a in aa.list_assets(output) if a["model"] and a["model"].endswith("__pixal3d__20261003-141000.glb")]
    assert holders == ["a 3x3 grid of props"]


def test_a_finish_links_to_its_picture_even_without_a_3d_run(output):
    # a model made outside the Generate tab: no 3D run on disk, but the finish kept the picture
    _write(output / "images" / "a-mech__20261005-200000" / "a-mech__20261005-200000.png", b"mech png", 1500)
    run = output / "finish" / "mech__finish__20261005-203000"
    _write(run / "mech_40k.glb", b"MECH FINISHED", 1600)
    _write(run / "input" / "source.glb", b"MECH RAW", 1600)
    _write(run / "input" / "source.png", b"mech png", 1600)
    # the name someone gave at the finish step ("mech") wins over the picture's prompt
    mech = _by_name(aa.list_assets(output))["mech"]
    assert mech["picture"].endswith("a-mech__20261005-200000.png")
    assert mech["finished"].endswith("mech_40k.glb")
    assert mech["stage"] == "finished"


def test_newest_activity_first(output):
    names = [a["name"] for a in aa.list_assets(output)]
    assert names[0] == "a lone fox"           # touched at 5000
    assert names[1] == "a chunky knight"      # last clip at 4200
    assert names[-1] == "orc"                 # 500


def test_every_path_stays_inside_output(output):
    for asset in aa.list_assets(output):
        for key in ("picture", "model", "finished", "rigged"):
            if asset[key]:
                assert not Path(asset[key]).is_absolute() and ".." not in asset[key]


def test_file_hashes_are_cached_and_refreshed_when_a_file_changes(output):
    aa.list_assets(output)
    index = json.loads((output / ".assets-index.json").read_text())
    finished = f"finish/{FIN}/a-chunky-knight_40k.glb"
    assert index[finished]["sha256"] == hashlib.sha256(b"FINISHED").hexdigest()
    _write(output / finished, b"FINISHED AGAIN", 9000)
    aa.list_assets(output)
    index = json.loads((output / ".assets-index.json").read_text())
    assert index[finished]["sha256"] == hashlib.sha256(b"FINISHED AGAIN").hexdigest()


def test_an_empty_or_missing_output_folder_is_an_empty_library(tmp_path):
    assert aa.list_assets(tmp_path / "nope") == []
    (tmp_path / "output").mkdir()
    assert aa.list_assets(tmp_path / "output") == []


def test_the_library_endpoint_serves_the_list(output, monkeypatch):
    import http.server
    import threading
    import urllib.request

    import generate_api as api

    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/assets", timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["base"] == "/output/"
        assert "a chunky knight" in [a["name"] for a in payload["assets"]]
    finally:
        server.shutdown()
        server.server_close()


def test_hiding_an_asset_keeps_its_files_and_can_be_undone(output):
    knight = _by_name(aa.list_assets(output))["a chunky knight"]
    assert knight["hidden"] is False
    aa.set_hidden(output, knight["id"], True)
    assert _by_name(aa.list_assets(output))["a chunky knight"]["hidden"] is True
    assert (output / "finish" / FIN / "a-chunky-knight_40k.glb").is_file()  # nothing deleted
    aa.set_hidden(output, knight["id"], False)
    assert _by_name(aa.list_assets(output))["a chunky knight"]["hidden"] is False


def test_only_real_asset_ids_can_be_hidden(output):
    with pytest.raises(ValueError):
        aa.set_hidden(output, "../../etc", True)
    with pytest.raises(ValueError):
        aa.set_hidden(output, "0123456789ab", True)  # well formed, but no such asset


def test_the_recipes_endpoint_serves_the_shared_file():
    import http.server
    import threading
    import urllib.request

    import generate_api as api

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/recipes", timeout=30) as response:
            recipes = json.loads(response.read())
        assert "T-pose" in recipes["goals"]["character"]["wording"]
    finally:
        server.shutdown()
        server.server_close()


def test_each_rig_says_whether_it_fits_the_moves_and_the_answer_is_cached(output, monkeypatch):
    import rig_check
    calls = []
    monkeypatch.setattr(rig_check, "check", lambda f: calls.append(f) or {"humanoid": False})
    # (a humanoid would also carry "forward", the way its feet point)
    knight = _by_name(aa.list_assets(output))["a chunky knight"]
    assert knight["fits_moves"] is False
    assert _by_name(aa.list_assets(output))["a chunky knight"]["fits_moves"] is False
    assert len(calls) == 1  # the second listing read it from the cache
    assert _by_name(aa.list_assets(output))["a lone fox"]["fits_moves"] is None  # no rig, no answer


def test_an_unreadable_rig_leaves_the_moves_alone(output):
    # the fixture's rig is not a real GLB: the check cannot tell, so nothing is hidden
    assert _by_name(aa.list_assets(output))["a chunky knight"]["fits_moves"] is None


def test_thumbnails_are_small_made_once_and_never_reach_outside_output(tmp_path):
    from PIL import Image
    Image.new("RGBA", (1024, 1024), (200, 80, 20, 255)).save(tmp_path / "big.png")
    thumb = aa.thumbnail(tmp_path, "big.png")
    with Image.open(thumb) as small:
        assert max(small.size) == 128
    made = thumb.stat().st_mtime_ns
    assert aa.thumbnail(tmp_path, "big.png") == thumb and thumb.stat().st_mtime_ns == made  # kept, not remade
    with pytest.raises(ValueError):
        aa.thumbnail(tmp_path / "sub", "../big.png")
    with pytest.raises(ValueError):
        aa.thumbnail(tmp_path, "missing.png")
