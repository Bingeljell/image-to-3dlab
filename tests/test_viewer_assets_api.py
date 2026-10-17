"""The studio Library: every run on disk grouped into assets, oldest picture to newest clip."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api as aa

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
        assert "A-pose" in recipes["goals"]["character"]["wording"]
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


def test_a_rename_changes_only_the_label_and_an_empty_name_restores_the_old_one(output):
    knight = _by_name(aa.list_assets(output))["a chunky knight"]
    files_before = sorted(p for p in output.rglob("*") if p.is_file() and not p.name.startswith("."))
    assert aa.set_name(output, knight["id"], "  Sir   Clank  ") == "Sir Clank"
    renamed = {a["id"]: a for a in aa.list_assets(output)}[knight["id"]]
    assert renamed["name"] == "Sir Clank" and renamed["rigged"] == knight["rigged"]
    assert sorted(p for p in output.rglob("*") if p.is_file() and not p.name.startswith(".")) == files_before
    assert aa.set_name(output, knight["id"], "") == "a chunky knight"
    assert len(aa.set_name(output, knight["id"], "x" * 500)) == aa.NAME_LIMIT
    with pytest.raises(ValueError):
        aa.set_name(output, "../../etc", "nope")


# --- single props made through Create's prop route --------------------------------------------


def _single_prop(output, stem="a-wooden-barrel__20261012-150000", prompt="A wooden barrel"):
    _write(output / "images" / stem / f"{stem}.png", b"barrel-png", 6000)
    _write(output / "images" / stem / f"{stem}.provenance.json", json.dumps({"prompt": prompt}), 6000)
    gen = f"{stem}__pixal3d__20261012-150500"
    _write(output / "commercial_conditional" / gen / f"{gen}.glb", b"BARREL", 6100)
    return stem


def test_a_prop_the_chain_asked_for_is_filed_as_a_prop(output):
    from viewer import activity_api
    stem = _single_prop(output)
    activity_api.append_history(output, {"chain": "c1", "step": "picture", "result": "done", "want": "prop",
                                         "made": f"images/{stem}/{stem}.png"})
    assert _by_name(aa.list_assets(output))["a wooden barrel"]["kind"] == "prop"


def test_an_older_prop_is_known_by_the_prop_wording_in_its_prompt(output):
    wording = json.loads((Path(aa.__file__).resolve().parents[1] / "image_to_3dlab" / "prompt_recipes.json")
                         .read_text())["goals"]["prop"]["wording"]
    _single_prop(output, prompt=f"A wooden barrel, {wording}")
    assert _by_name(aa.list_assets(output))["a wooden barrel"]["kind"] == "prop"


def test_a_model_nobody_asked_to_be_a_prop_stays_a_model(output):
    _single_prop(output)
    assert _by_name(aa.list_assets(output))["a wooden barrel"]["kind"] == "model"


def test_an_asset_carries_its_back_picture_when_there_is_one(output):
    stem = _single_prop(output)
    _write(output / "images" / stem / f"{stem}__back.png", b"back-png", 6050)
    assets = _by_name(aa.list_assets(output))
    assert assets["a wooden barrel"]["back"] == f"images/{stem}/{stem}__back.png"
    assert assets["a lone fox"]["back"] is None
    assert "a wooden barrel back" not in assets   # the back is not an asset of its own


def test_an_asset_carries_each_steps_time_from_the_history(output):
    from viewer import activity_api
    for step, made, seconds in (("picture", f"images/{IMG}/{IMG}.png", 251.4),
                                ("model", f"commercial_conditional/{GEN}/{GEN}.glb", 318.0),
                                ("finished", f"finish/{FIN}/a-chunky-knight_40k.glb", 95.2)):
        activity_api.append_history(output, {"step": step, "result": "done", "made": made, "seconds": seconds})
    activity_api.append_history(output, {"step": "rigged", "result": "error", "made": None, "seconds": 3.0})
    knight = _by_name(aa.list_assets(output))["a chunky knight"]
    assert knight["timings"] == {"picture": 251.4, "model": 318.0, "finished": 95.2}
    assert _by_name(aa.list_assets(output))["a lone fox"]["timings"] == {}


def _rig_again(out: Path, of: bytes = b"FINISHED", stamp: str = "20261006-090000", mtime: float = 6000) -> Path:
    run = out / "animate" / f"a-chunky-knight_40k__rig__{stamp}"
    _write(run / "a-chunky-knight_40k_rigged.glb", b"RIGGED " + stamp.encode(), mtime)
    _write(run / "a-chunky-knight_40k_rigged.provenance.json",
           json.dumps({"input": {"sha256": hashlib.sha256(of).hexdigest()}}), mtime)
    return run


def _knights(out: Path) -> list[dict]:
    return sorted((a for a in aa.list_assets(out) if a["name"].startswith("a chunky knight")), key=lambda a: a["version"])


def test_rigging_again_adds_a_version_and_keeps_the_first(output):
    before = _by_name(aa.list_assets(output))["a chunky knight"]
    _rig_again(output)
    first, second = _knights(output)
    assert first["id"] == before["id"] and first["version"] == 1 and first["name"] == "a chunky knight"
    assert [c["name"] for c in first["clips"]] == ["dance", "run"]      # untouched
    assert second["version"] == 2 and second["name"] == "a chunky knight · v2"
    assert second["id"] != first["id"] and second["clips"] == []
    # same picture, model and finish; only the rig differs
    for key in ("picture", "model", "finished"):
        assert second[key] == first[key]
    assert second["rigged"] != first["rigged"]
    assert first["family"] == second["family"] and first["versions"] == second["versions"] == 2


def test_a_second_3d_model_from_the_same_picture_is_a_new_version(output):
    again = f"{IMG}-cutout__pixal3d__20261006-120000"
    _write(output / "commercial_conditional" / again / f"{again}.glb", b"GENERATED TWO", 7000)
    first, second = _knights(output)
    assert first["stage"] == "animated" and first["model"].endswith(f"{GEN}.glb")
    assert second["stage"] == "model" and second["model"].endswith(f"{again}.glb")
    assert second["picture"] == first["picture"]


def test_rigging_the_raw_model_skips_finish_as_its_own_version(output):
    _rig_again(output, of=b"GENERATED")
    first, second = _knights(output)
    assert first["finished"] and second["finished"] is None
    assert second["model"] == first["model"] and second["stage"] == "rigged"


def test_a_new_move_on_a_rig_is_not_a_new_version(output):
    _write(output / "animate" / RIG / "a-chunky-knight_40k_walk.glb", b"WALK", 8000)
    (knight,) = _knights(output)
    assert knight["versions"] == 1
    assert [c["name"] for c in knight["clips"]] == ["dance", "run", "walk"]


def test_a_versions_id_survives_its_own_next_step_and_a_later_version(output):
    _rig_again(output)
    second_id = _knights(output)[1]["id"]
    _write(output / "animate" / "a-chunky-knight_40k__rig__20261006-090000" / "a-chunky-knight_40k_walk.glb", b"W", 8000)
    _rig_again(output, stamp="20261007-090000", mtime=9000)
    knights = _knights(output)
    assert [k["version"] for k in knights] == [1, 2, 3]
    assert knights[1]["id"] == second_id and len({k["id"] for k in knights}) == 3
    assert [c["name"] for c in knights[1]["clips"]] == ["walk"]


def test_each_family_counts_its_own_versions(output):
    _rig_again(output)
    fox = _by_name(aa.list_assets(output))["a lone fox"]
    assert fox["versions"] == 1 and fox["version"] == 1


def test_a_picture_a_finish_was_made_from_is_not_a_lone_item(output):
    # the same knight finished from another copy of its picture (say, the uncut one)
    other = "a-chunky-knight__20261005-205524-uncut"
    _write(output / "images" / other / f"{other}.png", b"png uncut", 1100)
    run = output / "finish" / "a-chunky-knight__finish__20261005-230000"
    _write(run / "a-chunky-knight_40k.glb", b"FINISHED UNCUT", 6500)
    _write(run / "input" / "source.glb", b"GENERATED", 6500)
    _write(run / "input" / "source.png", b"png uncut", 6500)
    names = [a["picture"] for a in aa.list_assets(output) if a["stage"] == "picture"]
    assert not any(n and n.endswith(f"{other}.png") for n in names)


def test_one_version_per_family_can_be_the_main_one(output):
    _rig_again(output)
    first, second = _knights(output)
    assert first["main"] is False and second["main"] is False
    aa.set_main(output, second["id"], True)
    first, second = _knights(output)
    assert second["main"] is True and first["main"] is False
    aa.set_main(output, first["id"], True)          # starring another one moves the star
    first, second = _knights(output)
    assert first["main"] is True and second["main"] is False
    aa.set_main(output, first["id"], False)
    assert not any(a["main"] for a in _knights(output))
    assert (output / "finish" / FIN / "a-chunky-knight_40k.glb").is_file()   # nothing deleted
    with pytest.raises(ValueError):
        aa.set_main(output, "0123456789ab", True)


def test_an_asset_carries_the_prompt_its_picture_was_drawn_from(output):
    _write(output / "images" / IMG / f"{IMG}.provenance.json", json.dumps({"prompt": "A chunky knight"}), 1000)
    assert _by_name(aa.list_assets(output))["a chunky knight"]["prompt"] == "A chunky knight"
    assert _by_name(aa.list_assets(output))["a lone fox"]["prompt"] is None       # no record kept


# --- deleting a version --------------------------------------------------------------------------


def test_deleting_a_version_removes_only_what_it_alone_owns(output):
    run = _rig_again(output)
    second = _knights(output)[1]
    plan = aa.plan_delete(output, second["id"])
    assert [f["path"] for f in plan["folders"]] == [f"animate/{run.name}"] and plan["bytes"] > 0
    aa.set_name(output, second["id"], "Spare rig")
    aa.set_hidden(output, second["id"], True)
    aa.delete_version(output, second["id"])
    assert not run.exists()
    first, = _knights(output)
    assert first["versions"] == 1 and first["rigged"] and [c["name"] for c in first["clips"]] == ["dance", "run"]
    for shared in (output / "finish" / FIN, output / "images" / IMG):
        assert shared.is_dir()                                  # picture, model and finish stay
    meta = json.loads((output / aa.META_NAME).read_text())
    assert second["id"] not in meta.get("hidden", []) and second["id"] not in meta.get("names", {})


def test_the_original_stays_while_later_versions_were_built_from_it(output):
    _rig_again(output)
    first = _knights(output)[0]
    plan = aa.plan_delete(output, first["id"])
    assert "original" in plan["refused"]
    with pytest.raises(ValueError, match="original"):
        aa.delete_version(output, first["id"])
    assert (output / "animate" / RIG).is_dir()


def test_a_lone_version_takes_its_picture_and_model_with_it(output):
    fox = _by_name(aa.list_assets(output))["a lone fox"]
    plan = aa.plan_delete(output, fox["id"])
    assert plan["refused"] is None and [f["path"] for f in plan["folders"]] == ["images/a-lone-fox__20261006-080100"]
    aa.delete_version(output, fox["id"])
    assert "a lone fox" not in _by_name(aa.list_assets(output))


def test_only_real_asset_ids_can_be_deleted(output):
    for bad in ("../../etc", "0123456789ab"):
        with pytest.raises(ValueError):
            aa.delete_version(output, bad)
    assert (output / "images").is_dir()


def test_deleting_never_reaches_outside_output(output, tmp_path):
    import shutil

    run = output / "images" / "a-lone-fox__20261006-080100"
    outside = tmp_path / "keep-me"
    _write(outside / "a-lone-fox__20261006-080100.png", b"png")
    shutil.rmtree(run)
    run.symlink_to(outside, target_is_directory=True)        # a run folder that is really a link elsewhere
    fox = _by_name(aa.list_assets(output))["a lone fox"]
    with pytest.raises(ValueError, match="refusing"):
        aa.delete_version(output, fox["id"])
    assert (outside / "a-lone-fox__20261006-080100.png").is_file()


def test_the_delete_endpoints_preview_first_and_need_a_confirmation(output, monkeypatch):
    import http.server
    import threading
    import urllib.error
    import urllib.request

    import generate_api as api

    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    run = _rig_again(output)
    second = _knights(output)[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/api/assets/{second['id']}"

    def call(url, body=None):
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())
    try:
        status, plan = call(f"{base}/delete-plan")
        assert status == 200 and plan["folders"][0]["path"] == f"animate/{run.name}" and run.is_dir()
        assert call(f"{base}/delete", {})[0] == 422 and run.is_dir()                  # no confirmation, nothing happens
        status, done = call(f"{base}/delete", {"confirm": True})
        assert status == 200 and done["deleted"] == [f"animate/{run.name}"] and not run.exists()
        assert call(f"{base}/delete", {"confirm": True})[0] == 422                    # already gone
    finally:
        server.shutdown()
        server.server_close()
