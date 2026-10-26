"""Skins: new paint on an existing Pixal3D model, from approved front and back pictures.

What matters: only a finished Pixal3D model that kept its shape can take a skin, and the
person is told why not otherwise; the front is approved before the back, a new front
invalidates the back, and nothing builds until every needed side is approved; the build
repaints the saved shape with the model's own seed, steps and Finish settings (same face
count, so the same mesh); the repaint's scratch is deleted; the Library lists a model's
finished skins and never mistakes a skin's files for a new model.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api as aa
import skins_api as sk

PIC = "a-paladin__20261019-133432"
GEN = f"{PIC}__pixal3d__20261019-145"
FIN = f"{GEN}__finish__20261019-145515"
FINISH = {"faces": 20000, "texture_size": 2048, "skip_paint": True, "skip_photo": False}


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


@pytest.fixture
def output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    _write(out / "images" / PIC / f"{PIC}.png", b"front")
    _write(out / "images" / PIC / f"{PIC}__back.png", b"back")
    _write(out / GEN / f"{GEN}.glb", b"MODEL")
    _write(out / GEN / f"{GEN}.shape.bin", b"SHAPE")
    _write(out / GEN / f"{GEN}.json", json.dumps({
        "backend": "pixal3d", "parameters": {"seed": 15629060, "steps": 8, "views": "front + back"}}))
    _write(out / "finish" / FIN / "paladin_20k.glb", b"FINISHED")
    _write(out / "finish" / FIN / "input" / "source.glb", b"MODEL")
    _write(out / "finish" / FIN / "input" / "settings.json", json.dumps(FINISH))
    return out


def _asset(**changes):
    asset = {"id": "abc123def456", "name": "a paladin", "picture": f"images/{PIC}/{PIC}.png",
             "back": f"images/{PIC}/{PIC}__back.png", "model": f"{GEN}/{GEN}.glb",
             "finished": f"finish/{FIN}/paladin_20k.glb"}
    asset.update(changes)
    return asset


# --- what a skin starts from ---------------------------------------------------------------------


def test_the_base_carries_the_shape_pictures_seed_and_finish_settings(output):
    base = sk.base_from_asset(output, _asset())
    assert base["shape"] == f"{GEN}/{GEN}.shape.bin"
    assert base["front"].endswith(f"{PIC}.png") and base["back"].endswith("__back.png")
    assert (base["seed"], base["steps"]) == (15629060, 8)
    assert base["finish"]["faces"] == 20000


def test_a_version_changed_by_fix_a_spot_cannot_take_a_skin(output):
    """A skin repaints the shape the model was made with, so the fixed spots would come back."""
    _write(output / "finish" / FIN / "fix-spot.json", "{}")
    with pytest.raises(sk.SkinError, match="before Fix a spot"):
        sk.base_from_asset(output, _asset())


@pytest.mark.parametrize("change, words", [
    ({"finished": None}, "Finish this one first"),
    ({"picture": None}, "no picture"),
    ({"back": None}, "back picture is missing"),
])
def test_a_model_that_cannot_take_a_skin_says_why(output, change, words):
    with pytest.raises(sk.SkinError, match=words):
        sk.base_from_asset(output, _asset(**change))


def test_a_model_from_before_skins_is_told_to_make_it_again(output):
    (output / GEN / f"{GEN}.shape.bin").unlink()
    with pytest.raises(sk.SkinError, match="made before skins"):
        sk.base_from_asset(output, _asset())


def test_only_pixal3d_models_take_skins(output):
    _write(output / GEN / f"{GEN}.json", json.dumps({"backend": "trellis", "parameters": {}}))
    with pytest.raises(sk.SkinError, match="Only Pixal3D"):
        sk.base_from_asset(output, _asset())


def test_without_its_finish_settings_a_skin_could_not_match_the_mesh(output):
    (output / "finish" / FIN / "input" / "settings.json").unlink()
    with pytest.raises(sk.SkinError, match="Finish settings"):
        sk.base_from_asset(output, _asset())


def test_a_front_only_model_needs_no_back(output):
    _write(output / GEN / f"{GEN}.json", json.dumps({"backend": "pixal3d", "parameters": {"seed": 1}}))
    skin = sk.SkinStore(output).create(_asset(back=None), "gold skirt")
    assert skin.sides() == ("front",) and skin.base["back"] is None


# --- what Qwen is asked ----------------------------------------------------------------------------


def test_the_prompt_asks_for_the_change_and_keeps_everything_else():
    prompt = sk.edit_prompt("make the skirt gold.", "front")
    assert prompt.startswith("Change only this: make the skirt gold.")
    for kept in ("same pose", "same face", "position and size in the frame", "Do not add or remove"):
        assert kept in prompt


def test_the_back_prompt_says_it_is_the_back():
    assert "back view of the same character" in sk.edit_prompt("gold skirt", "back")


# --- tries and approvals ---------------------------------------------------------------------------


def test_each_try_draws_from_the_models_own_picture_with_a_new_seed(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    ref1, out1, _, seed1 = store.picture_try(skin, "front")
    _, out2, prompt2, seed2 = store.picture_try(skin, "front", "gold skirt, darker")
    assert ref1 == (output / "images" / PIC / f"{PIC}.png").resolve()
    assert (out1.name, out2.name) == ("front-1.png", "front-2.png")
    assert seed1 != seed2 and "darker" in prompt2
    assert out1.parent == output / "skins" / skin.id


def test_the_back_waits_for_an_approved_front(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    with pytest.raises(sk.SkinError, match="front picture first"):
        store.picture_try(skin, "back")


def test_a_try_still_drawing_cannot_be_approved(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    store.picture_try(skin, "front")
    with pytest.raises(sk.SkinError, match="not ready"):
        store.approve(skin, "front")


def _drawn(output: Path, skin, side: str) -> None:
    _, target, _, _ = sk.SkinStore(output).picture_try(skin, side)
    _write(target, b"picture")


def test_front_then_back_makes_it_ready(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    _drawn(output, skin, "front")
    store.approve(skin, "front")
    assert skin.next_side() == "back" and not skin.ready()
    _drawn(output, skin, "back")
    store.approve(skin, "back")
    assert skin.ready() and skin.next_side() is None
    assert store.get(skin.id).ready()  # saved, not only in memory


def test_a_new_front_try_unapproves_the_back(output):
    """The back must match the front; a changed front can no longer vouch for it."""
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    _drawn(output, skin, "front")
    store.approve(skin, "front")
    _drawn(output, skin, "back")
    store.approve(skin, "back")
    store.picture_try(skin, "front", "gold skirt, matte")
    assert skin.approved == {"front": None, "back": None}


def test_an_empty_or_huge_change_is_refused(output):
    store = sk.SkinStore(output)
    with pytest.raises(sk.SkinError, match="Say what to change"):
        store.create(_asset(), "   ")
    with pytest.raises(sk.SkinError, match="under"):
        store.create(_asset(), "x" * (sk.MAX_INSTRUCTION + 1))


def test_ids_from_urls_are_checked(output):
    with pytest.raises(sk.SkinError):
        sk.SkinStore(output).get("../../etc")


# --- the build -------------------------------------------------------------------------------------


def _ready(output: Path):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    for side in ("front", "back"):
        _drawn(output, skin, side)
        store.approve(skin, side)
    return store, skin


def test_the_repaint_uses_the_saved_shape_and_the_models_seed_and_steps(output):
    _, skin = _ready(output)
    command = sk.repaint_command(output, skin, python="py")
    assert command[command.index("--reuse-shape") + 1].endswith(f"{GEN}.shape.bin")
    assert command[command.index("--seed") + 1] == "15629060"
    assert command[command.index("--steps") + 1] == "8"
    assert command[command.index("--back") + 1].endswith("back-1.png")
    assert command[3].endswith("front-1.png")


def test_finish_uses_the_models_own_face_count(output):
    """A different face count gives a different mesh, and the rig would no longer fit."""
    _, skin = _ready(output)
    command = sk.finish_command(output, skin)
    assert command[command.index("--faces") + 1] == "20000"
    assert command[5].endswith("gold-skirt_20k.glb")


def _same_box(path):
    return (0.2, 0.04, 0.8, 0.95)


class _FakeProcess:
    def __init__(self, lines, code=0, on_wait=None):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.code, self.on_wait = code, on_wait

    def wait(self):
        if self.on_wait:
            self.on_wait()
        return self.code

    def poll(self):
        return self.code


def test_a_build_repaints_finishes_and_tidies_up(output):
    store, skin = _ready(output)
    folder = output / "skins" / skin.id
    seen = []

    def runner(command, **kwargs):
        seen.append(command)
        if len(seen) == 1:   # the repaint leaves its scratch behind, and the camera Pixel Match needs
            return _FakeProcess(["[4/6] decode", "[6/6] write"], on_wait=lambda: [
                _write(folder / "repaint" / "repaint.post.bin", b"x"),
                _write(folder / "repaint" / "repaint.ply", b"x"),
                _write(folder / "repaint" / "repaint_base.png", b"x"),
                _write(folder / "repaint" / "repaint.svviews" / "transforms.json", "{}"),
                _write(folder / "repaint" / "repaint.glb", b"glb")])
        return _FakeProcess(["I2L_STAGE::photo::Pixel Match"],
                            on_wait=lambda: _write(folder / "gold-skirt_20k.glb", b"skin"))

    _write(output / "finish" / FIN / "input" / "source.post.bin", b"dump")   # the model took Pixal3D's own route
    skin.base["native"] = True
    job = sk.SkinBuilds(store).start(skin, background=False, runner=runner, box=_same_box)
    assert job.status == "done"
    # Finish was built after the repaint wrote its files: same route as the model, with Pixel Match
    assert "--native-dump" in seen[1] and "--views" in seen[1]
    saved = store.get(skin.id)
    assert saved.status == "done" and saved.result == f"skins/{skin.id}/gold-skirt_20k.glb"
    assert not (folder / "repaint" / "repaint.post.bin").exists()
    assert not (folder / "repaint" / "repaint.ply").exists()
    assert not (folder / "repaint" / "repaint_base.png").exists()
    assert (folder / "repaint" / "repaint.glb").is_file()
    assert [e["overall_pct"] for e in job.events if "overall_pct" in e][-1] == 100


def test_a_failed_step_marks_the_skin_and_can_be_built_again(output):
    store, skin = _ready(output)
    builds = sk.SkinBuilds(store)
    job = builds.start(skin, background=False, runner=lambda c, **k: _FakeProcess(["boom"], code=1), box=_same_box)
    assert job.status == "error" and "repaint" in job.error
    assert store.get(skin.id).status == "error"
    again = builds.start(store.get(skin.id), background=False,
                         runner=lambda c, **k: _FakeProcess(["boom"], code=1), box=_same_box)
    assert again.status == "error"  # allowed to try again; it simply failed again here


def test_a_build_starts_its_steps_in_their_own_process_group(output):
    """So Cancel can stop the engine and Blender the step started, not just the step itself."""
    store, skin = _ready(output)
    seen = []
    sk.SkinBuilds(store).start(skin, background=False, box=_same_box,
                               runner=lambda c, **k: seen.append(k) or _FakeProcess(["boom"], code=1))
    assert seen and all(k.get("start_new_session") or k.get("creationflags") for k in seen)


def test_cancel_stops_the_whole_group_not_just_the_step(output):
    store, skin = _ready(output)
    stopped = []
    builds = sk.SkinBuilds(store, stop=stopped.append)

    class Running(_FakeProcess):
        pid = 4242

        def poll(self):
            return None

    def runner(command, **kwargs):
        process = Running(["[1/6] load"], on_wait=lambda: builds.cancel(builds.active))
        return process

    job = builds.start(skin, background=False, runner=runner, box=_same_box)
    assert stopped == [4242]
    assert job.status == "cancelled"


def test_a_cancel_while_a_step_is_starting_still_stops_it(output):
    """Cancel lands after the last check but before the process exists: it had nothing to stop."""
    store, skin = _ready(output)
    stopped = []
    builds = sk.SkinBuilds(store, stop=stopped.append)

    class Started(_FakeProcess):
        pid = 5151

    def runner(command, **kwargs):
        builds.cancel(builds.active)                   # job.process is not set yet
        return Started(["[1/6] load"])

    job = builds.start(skin, background=False, runner=runner, box=_same_box)
    assert stopped == [5151]
    assert job.status == "cancelled"


def test_a_cancel_during_the_picture_check_starts_no_step(output):
    store, skin = _ready(output)
    builds = sk.SkinBuilds(store, stop=lambda pid: None)
    launched = []

    def box(path):
        builds.cancel(builds.active)
        return _same_box(path)

    job = builds.start(skin, background=False, box=box,
                       runner=lambda c, **k: launched.append(c) or _FakeProcess([]))
    assert launched == []
    assert job.status == "cancelled" and store.get(skin.id).status == "pictures"


def test_nothing_builds_before_every_side_is_approved(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    _drawn(output, skin, "front")
    store.approve(skin, "front")
    with pytest.raises(sk.SkinError, match="back picture"):
        sk.SkinBuilds(store).start(skin, background=False)


@pytest.mark.parametrize("stage, line, expected", [
    ("repaint", "[3/6] shape", 40.0), ("repaint", "[6/6] write", 80.0), ("repaint", "noise", None),
    ("finish", "I2L_STAGE::photo::Pixel Match", 88.0), ("finish", "plain line", None),
])
def test_progress_reads_both_steps(stage, line, expected):
    assert sk.progress(stage, line) == expected


# --- the Library -------------------------------------------------------------------------------------


def test_a_model_lists_only_its_own_finished_skins(output):
    store, skin = _ready(output)
    skin.status, skin.result = "done", f"skins/{skin.id}/gold-skirt_20k.glb"
    store.save(skin)
    other = store.create(_asset(), "red boots")          # still at its pictures
    listed = store.skins_for(f"finish/{FIN}/paladin_20k.glb")
    assert [s["id"] for s in listed] == [skin.id] and other.id not in [s["id"] for s in listed]
    assert store.skins_for("finish/elsewhere/model.glb") == []


def test_the_library_never_counts_a_skins_repaint_as_a_new_model(output):
    store, skin = _ready(output)
    _write(output / "skins" / skin.id / "repaint" / "repaint.glb", b"REPAINT")
    skin.status, skin.result = "done", f"skins/{skin.id}/gold-skirt_20k.glb"
    store.save(skin)
    assets = aa.list_assets(output)
    assert all("skins/" not in (a.get("model") or "") for a in assets)
    finished = [a for a in assets if a.get("finished") == f"finish/{FIN}/paladin_20k.glb"]
    assert finished and [s["id"] for s in finished[0]["skins"]] == [skin.id]


# --- the studio endpoints (the MCP calls the same ones) ------------------------------------------


def test_the_endpoints_walk_a_skin_from_front_to_back(output, monkeypatch, tmp_path):
    """Picture tries are faked: a test must never start Qwen on the real install."""
    import http.server
    import threading
    import types
    import urllib.error
    import urllib.request

    import generate_api as api

    store = sk.SkinStore(output)
    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(sk, "STORE", store)
    monkeypatch.setattr(sk, "BUILDS", sk.SkinBuilds(store))
    engine = _write(tmp_path / "sd-cli", b"")
    monkeypatch.setattr(api.image_api, "BINARY", engine)
    monkeypatch.setattr(api.image_api, "resolve_weights", dict)
    monkeypatch.setattr(api.image_api, "resolve_vision", lambda: engine)
    started = []

    def fake_edit(reference, target, prompt, seed, mode="edit (skin)"):
        started.append((reference, target, prompt, seed))
        return types.SimpleNamespace(id=f"{len(started):032x}")
    monkeypatch.setattr(api.image_api, "start_edit", fake_edit)

    asset = next(a for a in aa.list_assets(output) if a.get("finished"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/api/skins"

    def call(url, body=None):
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json",
                                                  "X-AssetFurnace-Request": "studio"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())
    try:
        assert call(base, {"asset": "000000000000", "instruction": "gold skirt"})[0] == 422
        status, made = call(base, {"asset": asset["id"], "instruction": "make the skirt gold"})
        assert status == 202 and len(started) == 1 and "make the skirt gold" in started[0][2]
        skin_id = made["skin"]["id"]
        assert call(f"{base}/{skin_id}/approve", {"side": "front"})[0] == 422   # still drawing
        _write(output / "skins" / skin_id / "front-1.png", b"drawn")
        status, after = call(f"{base}/{skin_id}/approve", {"side": "front"})
        assert status == 200 and after["skin"]["next_side"] == "back"
        assert len(started) == 2 and started[1][1].name == "back-1.png"           # the back followed at once
        assert call(f"{base}/{skin_id}/build", {})[0] == 422                       # back not approved yet
        status, state = call(f"{base}/{skin_id}")
        assert status == 200 and state["approved"]["front"].endswith("front-1.png")
        assert call(f"{base}/nothing-here")[0] == 404
        busy = types.SimpleNamespace(status="running")
        monkeypatch.setattr(api.image_api.MANAGER, "active", "busy-job")
        monkeypatch.setitem(api.image_api.MANAGER.jobs, "busy-job", busy)
        before = call(f"{base}/{skin_id}")[1]["tries"]["back"]
        assert call(f"{base}/{skin_id}/try", {"side": "back"})[0] == 409
        assert call(f"{base}/{skin_id}")[1]["tries"]["back"] == before   # no dead try left behind
    finally:
        server.shutdown()
        server.server_close()


def test_the_state_says_which_tries_have_their_picture(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "gold skirt")
    _drawn(output, skin, "front")
    store.picture_try(skin, "front", "again")
    tries = store.state(skin, build_running=False)["tries"]["front"]
    assert [t["ready"] for t in tries] == [True, False]


def test_a_build_the_studio_lost_reads_as_one_to_start_again(output):
    store, skin = _ready(output)
    skin.status = "building"
    store.save(skin)
    assert store.state(skin, build_running=True)["status"] == "building"
    lost = store.state(skin, build_running=False)
    assert lost["status"] == "error" and "interrupted" in lost["error"]
    assert sk.SkinBuilds(store).start(store.get(skin.id), background=False,
                                      runner=lambda c, **k: _FakeProcess([], code=1), box=_same_box).status == "error"


def test_deleting_a_finished_model_deletes_its_skins_too(output):
    store, skin = _ready(output)
    skin.status, skin.result = "done", f"skins/{skin.id}/gold-skirt_20k.glb"
    _write(output / "skins" / skin.id / "gold-skirt_20k.glb", b"skin")
    store.save(skin)
    stranger = sk.Skin(name="red boots", instruction="red boots",
                       base={**skin.base, "finished": "finish/elsewhere/other_20k.glb"})
    store.save(stranger)
    model = next(a for a in aa.list_assets(output) if a.get("finished") == f"finish/{FIN}/paladin_20k.glb")
    paths = [f["path"] for f in aa.plan_delete(output, model["id"])["folders"]]
    assert f"finish/{FIN}" in paths and f"skins/{skin.id}" in paths
    assert f"skins/{stranger.id}" not in paths
    aa.delete_version(output, model["id"])
    assert not (output / "skins" / skin.id).exists() and (output / "skins" / stranger.id).is_dir()


# --- the route a skin is finished by, installs that can't, and discarding ------------------------


def test_a_model_finished_by_pixal3ds_own_clean_up_gets_its_skins_finished_the_same_way(output):
    _write(output / "finish" / FIN / "input" / "source.post.bin", b"dump")
    _, skin = _ready(output)
    _write(output / "skins" / skin.id / "repaint" / "repaint.post.bin", b"dump")
    assert skin.base["native"] is True
    assert "--native-dump" in sk.finish_command(output, skin)


def test_a_model_finished_the_classic_way_gets_no_dump_for_its_skins(output):
    """Same route as the model, or the mesh differs and the model's rig no longer fits."""
    _, skin = _ready(output)
    _write(output / "skins" / skin.id / "repaint" / "repaint.post.bin", b"dump")
    assert skin.base["native"] is False
    assert "--native-dump" not in sk.finish_command(output, skin)


def test_an_install_that_cannot_keep_shapes_says_so_instead_of_remake_it(output, monkeypatch):
    (output / GEN / f"{GEN}.shape.bin").unlink()
    monkeypatch.setattr(sk, "install_keeps_shapes", lambda: False)
    with pytest.raises(sk.SkinError, match="install can't keep"):
        sk.base_from_asset(output, _asset())
    monkeypatch.setattr(sk, "install_keeps_shapes", lambda: True)
    with pytest.raises(sk.SkinError, match="made before skins"):
        sk.base_from_asset(output, _asset())


def test_skin_records_are_utf8_on_every_os(output):
    store = sk.SkinStore(output)
    skin = store.create(_asset(), "botas de piel blanca, café, 白い毛皮のブーツ")
    raw = (output / "skins" / skin.id / "skin.json").read_bytes()
    raw.decode("utf-8")
    assert store.get(skin.id).instruction.endswith("ブーツ")


def test_discard_removes_the_skin_and_everything_it_made(output):
    store, skin = _ready(output)
    store.discard(skin)
    assert not (output / "skins" / skin.id).exists()
    with pytest.raises(sk.SkinError):
        store.get(skin.id)


def test_a_preview_can_show_the_model_wearing_a_skin(output, tmp_path):
    import preview_api

    asset = {"name": "paladin", "finished": f"finish/{FIN}/paladin_20k.glb",
             "skins": [{"id": "k1", "name": "gold skirt", "file": "skins/k1/gold_20k.glb"}]}
    _write(output / "skins" / "k1" / "gold_20k.glb", b"glb")
    with pytest.raises(ValueError, match="no finished skin"):
        preview_api.preview(output, asset, skin="nope")
    source = (output / "skins" / "k1" / "gold_20k.glb").resolve()
    cached = output / preview_api.PREVIEWS_DIR / f"{preview_api.cache_key(source, 0.0, 'paint')}.png"
    _write(cached, b"png")
    path, shown = preview_api.preview(output, asset, skin="k1")
    assert path == cached and shown == "skin: gold skirt"


# --- the framing check (white boots on light grey) --------------------------------------------------


def test_matching_cut_outs_pass(output):
    _, skin = _ready(output)
    assert sk.framing_problem(output, skin, box=_same_box) is None


def test_a_part_lost_to_the_background_stops_the_build_before_painting(output):
    """2026-10-19, the user's worry: white fur boots on a light background. Lose the boots and
    the cut-out ends higher, Pixal3D frames the picture shorter, and all the paint slides."""
    store, skin = _ready(output)

    def boots_lost(path):
        return (0.2, 0.04, 0.8, 0.80) if "skins" in str(path) else (0.2, 0.04, 0.8, 0.95)
    ran = []
    job = sk.SkinBuilds(store).start(skin, background=False, box=boots_lost,
                                     runner=lambda c, **k: ran.append(c))
    assert job.status == "error" and "framed lower" in job.error and "front" in job.error
    assert ran == []                                         # nothing was painted


def test_an_empty_cut_out_is_caught(output):
    _, skin = _ready(output)
    assert "came out empty" in sk.framing_problem(output, skin, box=lambda p: None if "skins" in str(p) else (0, 0, 1, 1))


# --- wearing a skin: rigged models, moves, one file with every skin ----------------------------


def _glb(paint: bytes) -> bytes:
    from tests.test_glb_skins import _glb as make
    return make(paint, paint + b"-SHINE")


def _finished_skin(output, name="gold skirt", paint=b"GOLD"):
    store, skin = _ready(output)
    skin.name, skin.status, skin.result = name, "done", f"skins/{skin.id}/{sk.slug(name)}_20k.glb"
    _write(output / skin.result, _glb(paint))
    store.save(skin)
    return store, skin


def test_a_rigged_model_is_dressed_once_and_kept(output):
    from image_to_3dlab.glb_skins import paint_images
    from image_to_3dlab.glb_turn import parse_glb

    _, skin = _finished_skin(output)
    _write(output / "animate" / "r" / "p_rigged.glb", _glb(b"MODEL-PAINT"))
    made = sk.dressed(output, skin, "animate/r/p_rigged.glb")
    assert made.startswith(f"skins/{skin.id}/dressed/") and b"GOLD" in (output / made).read_bytes()
    document, _ = parse_glb((output / made).read_bytes())
    assert set(paint_images(document)) == {"baseColorTexture", "metallicRoughnessTexture"}
    first = (output / made).stat().st_mtime
    assert sk.dressed(output, skin, "animate/r/p_rigged.glb") == made and (output / made).stat().st_mtime == first


def test_the_finished_model_in_a_skin_is_the_skin_itself(output):
    _, skin = _finished_skin(output)
    assert sk.dressed(output, skin, skin.base["finished"]) == skin.result


def test_an_unfinished_skin_or_a_stray_path_is_refused(output):
    _, skin = _ready(output)
    with pytest.raises(sk.SkinError, match="not finished"):
        sk.dressed(output, skin, skin.base["finished"])
    _, done = _finished_skin(output)
    with pytest.raises(sk.SkinError):
        sk.dressed(output, done, "../../etc/passwd")


def test_one_file_carries_every_finished_skin(output):
    from image_to_3dlab.glb_skins import variant_names

    _, gold = _finished_skin(output, "gold skirt", b"GOLD")
    _, purple = _finished_skin(output, "purple", b"PURPLE")
    _write(output / "animate" / "r" / "p_rigged.glb", _glb(b"MODEL-PAINT"))
    made = sk.all_skins(output, "animate/r/p_rigged.glb", [gold, purple])
    assert variant_names((output / made).read_bytes()) == ["Original paint", "gold skirt", "purple"]
    with pytest.raises(sk.SkinError, match="no finished skins"):
        sk.all_skins(output, "animate/r/p_rigged.glb", [])


def test_every_build_lands_in_the_activity_history(output):
    """Activity lists it, and Report a bug sends a failed one's log like any other step's."""
    from activity_api import read_history

    store, skin = _ready(output)
    sk.SkinBuilds(store).start(skin, background=False, box=_same_box,
                               runner=lambda c, **k: _FakeProcess(["[3/6] shape", "boom"], code=1))
    record = read_history(output)[0]
    assert record["step"] == "skin" and record["result"] == "error" and "repaint" in record["note"]
    assert "boom" in record["log"] and record["prompt"] == "gold skirt"


def test_two_rigs_of_one_model_get_their_own_dressed_files(output):
    """A re-rig the same day shares the file name and all but the time in its folder name."""
    _, skin = _finished_skin(output)
    first = "animate/p_20k__rig__20261019-203017/p_rigged.glb"
    second = "animate/p_20k__rig__20261019-213544/p_rigged.glb"
    _write(output / first, _glb(b"RIG-ONE"))
    _write(output / second, _glb(b"RIG-TWO"))
    assert sk.dressed(output, skin, first) != sk.dressed(output, skin, second)


@pytest.mark.parametrize("instruction, name, expected", [
    ("Make the boots white fur boots", None, "Boots white fur boots"),
    ("make her skirt gold.", None, "Skirt gold"),
    ("silver armour, red cloak", None, "Silver armour, red cloak"),
    ("Make the boots white fur boots", "  Fur   boots ", "Fur boots"),
])
def test_a_skin_gets_a_short_name_for_lists_and_game_engines(instruction, name, expected):
    assert sk.skin_name(instruction, name) == expected


def test_the_module_imports_on_its_own():
    """It used to import only because another studio module had put the repo on sys.path first."""
    import subprocess
    import sys

    viewer = Path(__file__).resolve().parents[1] / "viewer"
    code = f"import sys; sys.path.insert(0, {str(viewer)!r}); import skins_api"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, cwd="/")
