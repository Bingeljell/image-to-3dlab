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
        if len(seen) == 1:   # the repaint leaves its scratch behind
            return _FakeProcess(["[4/6] decode", "[6/6] write"], on_wait=lambda: [
                _write(folder / "repaint" / "repaint.post.bin", b"x"),
                _write(folder / "repaint" / "repaint.ply", b"x"),
                _write(folder / "repaint" / "repaint.glb", b"glb")])
        return _FakeProcess(["I2L_STAGE::photo::Pixel Match"],
                            on_wait=lambda: _write(folder / "gold-skirt_20k.glb", b"skin"))

    job = sk.SkinBuilds(store).start(skin, background=False, runner=runner)
    assert job.status == "done"
    saved = store.get(skin.id)
    assert saved.status == "done" and saved.result == f"skins/{skin.id}/gold-skirt_20k.glb"
    assert not (folder / "repaint" / "repaint.post.bin").exists()
    assert not (folder / "repaint" / "repaint.ply").exists()
    assert (folder / "repaint" / "repaint.glb").is_file()
    assert [e["overall_pct"] for e in job.events if "overall_pct" in e][-1] == 100


def test_a_failed_step_marks_the_skin_and_can_be_built_again(output):
    store, skin = _ready(output)
    builds = sk.SkinBuilds(store)
    job = builds.start(skin, background=False, runner=lambda c, **k: _FakeProcess(["boom"], code=1))
    assert job.status == "error" and "repaint" in job.error
    assert store.get(skin.id).status == "error"
    again = builds.start(store.get(skin.id), background=False,
                         runner=lambda c, **k: _FakeProcess(["boom"], code=1))
    assert again.status == "error"  # allowed to try again; it simply failed again here


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
                                         headers={"Content-Type": "application/json"})
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
    finally:
        server.shutdown()
        server.server_close()
