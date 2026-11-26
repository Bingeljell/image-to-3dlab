"""Exercise registered tools through the MCP server and its HTTP boundary."""

import asyncio
import json
import subprocess
import sys
from collections import deque
from types import SimpleNamespace

import pytest
import requests

from image_to_3dlab import mcp_server as ms

BASE = "http://localhost:8777"
ASSET = {"id": "a", "name": "Fox", "kind": "character", "stage": "finished",
         "picture": "fox.png", "back": "fox-back.png", "finished": "fox.glb",
         "skins": [{"id": "s", "name": "Rust", "file": "rust.glb"}]}
SKIN = {"id": "s", "name": "Rust", "instruction": "red coat", "status": "draft",
        "base": {"finished": "fox.glb", "front": "fox.png"}, "needs_back": True,
        "tries": {"front": [{"number": 1, "instruction": "red coat", "picture": "front.png"}]},
        "approved": {"front": "front.png", "back": None}, "ready": True}


class Http:
    """Only declared requests can be answered; no sockets are opened."""

    def __init__(self, gets=(), posts=()):
        self.gets = {k: deque(v) for k, v in dict(gets).items()}
        self.posts = dict(posts)
        self.calls = []

    def get(self, url, **kwargs):
        path = url.removeprefix(BASE + "/")
        self.calls.append(("GET", path, kwargs))
        replies = self.gets[path]
        reply = replies.popleft() if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return self.response(reply)

    def post(self, url, **kwargs):
        path = url.removeprefix(BASE + "/")
        self.calls.append(("POST", path, kwargs))
        return self.response(self.posts[path])

    @staticmethod
    def response(reply):
        if isinstance(reply, dict):
            return SimpleNamespace(status_code=200, json=lambda: reply)
        return reply


@pytest.fixture
def studio_factory():
    def make(gets=(), posts=()):
        defaults = {
            "api/catalog": [{}], "api/assets": [{"assets": [ASSET]}],
            "api/animate/models": [{"installed": False}], "api/chains": [{}],
            "api/usage": [{"noticed": True}], "api/notices": [{}],
            "api/models-folder": [{}], "api/skins/s": [SKIN],
        }
        http = Http(defaults | dict(gets), posts)
        return ms.Studio(BASE, http=http, start=lambda: pytest.fail("unexpected server launch"))
    return make


def invoke(server, name, arguments):
    response = asyncio.run(server.call_tool(name, arguments))
    assert not response.is_error
    return response.content


@pytest.mark.parametrize(("tool", "arguments", "path", "params", "field", "value"), [
    ("status", {}, "api/models-folder", None, "ready_engines", []),
    ("list_engines", {}, "api/catalog", None, None, []),
    ("list_moves", {}, "api/animate/models", None, "installed", False),
    ("run_status", {"wait_seconds": 0}, "api/chains", None, "status", "idle"),
    ("list_assets", {"search": "fox", "kind": "character"}, "api/assets", None, "total", 1),
    ("list_versions", {"asset_id": "a"}, "api/assets", None, "family", "a"),
    ("back_versions", {"asset_id": "a"}, "api/image/versions", {"back": "fox-back.png"}, "versions", []),
    ("picture_status", {"job_id": "a" * 32}, "api/image/" + "a" * 32 + "/status", None, "status", "done"),
    ("skin_status", {"skin_id": "s"}, "api/skins/s", None, "skin_id", "s"),
    ("export_skins", {"asset_id": "a"}, "api/skins/all", {"asset": "a", "file": "fox.glb"}, "file", "output/all.glb"),
    ("model_info", {"asset_id": "a", "stage": "finished"}, "api/model-info", {"asset": "a", "stage": "finished"}, "file", "fox.glb"),
    ("usage_counts", {}, "api/usage", None, "sharing", False),
    ("list_addons", {}, "api/addons", None, "addons", []),
])
def test_tool_queries_studio(studio_factory, tool, arguments, path, params, field, value):
    studio = studio_factory({
        "api/image/versions": [{"versions": []}],
        "api/image/" + "a" * 32 + "/status": [{"status": "done"}],
        "api/skins/all": [{"file": "all.glb"}],
        "api/model-info": [{"file": "fox.glb"}],
        "api/addons": [{"addons": []}],
    })
    content = invoke(ms.build_server(studio), tool, arguments)
    if field is None:
        assert content == value
    else:
        assert json.loads(content[0].text)[field] == value
    assert any(method == "GET" and route == path and request["params"] == params
               for method, route, request in studio.http.calls if "params" in request)


@pytest.mark.parametrize(("tool", "arguments", "path", "body", "reply", "field", "value"), [
    ("cancel_run", {"run_id": "r"}, "api/chains/r/cancel", None, {"cancelled": False}, "cancelled", False),
    ("back_versions", {"asset_id": "a", "use": "old.png"}, "api/image/versions/use",
     {"back": "fox-back.png", "version": "old.png"}, {"back": "old.png"}, "back", "old.png"),
    ("make_skin", {"asset_id": "a", "change": "red coat", "name": "Rust"}, "api/skins",
     {"asset": "a", "instruction": "red coat", "name": "Rust"}, {"skin": SKIN}, "skin_id", "s"),
    ("approve_skin_picture", {"skin_id": "s", "side": "front", "number": 1}, "api/skins/s/approve",
     {"side": "front", "number": 1}, {"skin": SKIN}, "approved", {"front": "output/front.png"}),
    ("redo_skin_picture", {"skin_id": "s", "side": "back", "change": "darker"}, "api/skins/s/try",
     {"side": "back", "instruction": "darker"}, {"skin": SKIN}, "skin_id", "s"),
    ("build_skin", {"skin_id": "s"}, "api/skins/s/build", {}, {}, "skin_id", "s"),
    ("cancel_skin", {"skin_id": "s"}, "api/skins/build/b/cancel", {}, {"cancelled": True}, "cancelled", True),
    ("show_in_folder", {"asset_id": "a"}, "api/reveal", {"file": "fox.glb"}, {"shown": "fox.glb"}, "shown", "fox.glb"),
    ("export_roblox", {"asset_id": "a", "kind": "character", "height_studs": 5.5}, "api/roblox-export",
     {"file": "fox.glb", "kind": "character", "height_studs": 5.5}, {"passed": True}, "passed", True),
    ("report_bug", {"message": "broken"}, "api/bug-report", {"message": "broken", "details": "version"},
     {"text": "draft", "site": "site", "github": "github", "discord": "discord"}, "report", "draft"),
    ("install_addon", {"addon_id": "test"}, "api/addons/test/install", {"confirm": False},
     {"licence": "MIT"}, "licence", "MIT"),
    ("run_addon", {"addon_id": "test", "tool": "view"}, "api/addons/test/run/view", {"inputs": {}},
     {"job_id": "j", "status": "queued"}, "run_id", "addon-j"),
])
def test_tool_sends_action_to_studio(studio_factory, tool, arguments, path, body, reply, field, value):
    studio = studio_factory({
        "api/skins/s": [SKIN | {"build_job": {"job_id": "b"}}],
        "api/model-info": [{"file": "fox.glb"}], "api/bug-report": [{"details": "version"}],
    }, {path: reply})

    content = invoke(ms.build_server(studio), tool, arguments)

    assert json.loads(content[0].text)[field] == value
    posts = [(route, request.get("json", request.get("files")))
             for method, route, request in studio.http.calls if method == "POST"]
    assert posts == [(path, body)]


@pytest.mark.parametrize(("tool", "arguments", "error"), [
    ("create", {"want": "character"}, "Describe what you want"),
    ("continue_asset", {"asset_id": "a", "up_to": "bad"}, "up_to must be one of"),
    ("brush_back", {"asset_id": "a", "mask_path": "not-used", "change": "", "which": "side"}, "which is"),
    ("list_versions", {"asset_id": "unknown"}, "No asset with id"),
    ("install_backend", {"backend_id": "unknown"}, "No backend"),
    ("skin_picture", {"skin_id": "s", "side": "back"}, "no back picture"),
    ("preview", {"asset_id": "unknown"}, "No such asset"),
])
def test_tool_returns_validation_error(studio_factory, tool, arguments, error):
    preview = SimpleNamespace(status_code=404, json=lambda: {"error": "No such asset"})
    studio = studio_factory({"api/preview": [preview]})
    content = invoke(ms.build_server(studio), tool, arguments)
    assert error in json.loads(content[0].text)["error"]
    assert all(method != "POST" for method, *_ in studio.http.calls)


@pytest.mark.parametrize("tool", ["preview", "skin_picture"])
def test_tool_returns_picture_and_metadata(studio_factory, tool, tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(ms, "REPO", tmp_path)
    picture = tmp_path / "output" / "front.png"
    picture.parent.mkdir()
    Image.new("RGB", (2, 2), "red").save(picture)
    response = SimpleNamespace(status_code=200, content=picture.read_bytes(),
                               headers={"X-Preview-Stage": "finished", "X-Preview-Path": "front.png"})
    studio = studio_factory({"api/preview": [response]})
    arguments = {"asset_id": "a", "move": "walk", "skin": "s"} if tool == "preview" else {"skin_id": "s", "number": 1}
    image, description = invoke(ms.build_server(studio), tool, arguments)
    assert image.type == "image"
    assert image.mime_type == "image/png"
    assert json.loads(description.text)["file"] == str(picture)


def test_tool_options_and_helper_catalog(studio_factory):
    server = ms.build_server(studio_factory())
    options = json.loads(invoke(server, "options", {})[0].text)
    helpers = json.loads(invoke(server, "blender_helpers", {})[0].text)
    assert "advanced" in options
    assert helpers["helpers"]


def test_skin_status_polls_until_build_finishes(studio_factory):
    building = SKIN | {"status": "building", "build_job": {"last_event": {"message": "Painting"}}}
    complete = SKIN | {"status": "done", "result": "rust.glb"}
    studio = studio_factory({"api/skins/s": [building, complete]})
    sleeps = []
    result = ms.skin_status(studio, skin_id="s", wait_seconds=1, sleep=sleeps.append)
    assert sleeps == [2]
    assert result["result"] == "output/rust.glb"
    assert "finished" in result["next"]
    assert ms.skin_summary(building)["building"] == "Painting"
    assert "Building" in ms.skin_summary(building)["next"]


@pytest.mark.parametrize("skin", [None, "missing", "s"])
def test_export_skin_includes_only_existing_rig_and_moves(studio_factory, skin):
    asset = ASSET | {"clips": [{"name": "walk", "file": "fox-walk.glb"}]}
    studio = studio_factory({"api/assets": [{"assets": [asset]}],
                             "api/skins/s/dressed": [{"file": "rust-walk.glb"}],
                             "api/skins/all": [{"file": "all.glb"}]})
    if skin == "missing":
        with pytest.raises(ms.StudioError, match="no finished skin"):
            ms.export_skins(studio, asset_id="a", skin_id=skin)
    elif skin is None:
        assert ms.export_skins(studio, asset_id="a")["skins"] == ["Rust"]
    else:
        assert ms.export_skins(studio, asset_id="a", skin_id=skin)["files"] == {
            "finished": "output/rust.glb", "walk": "output/rust-walk.glb",
        }


def test_skin_picture_rejects_path_outside_output(studio_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "REPO", tmp_path)
    (tmp_path / "elsewhere.png").write_bytes(b"private")
    skin = SKIN | {"tries": {"front": [{"number": 1, "picture": "../elsewhere.png"}]}}
    studio = studio_factory({"api/skins/s": [skin]})
    with pytest.raises(ms.StudioError, match="not a file in output"):
        ms.skin_picture(studio, skin_id="s")


@pytest.mark.parametrize("eventually_ready", [False, True])
def test_studio_start_waits_or_reports_timeout(monkeypatch, eventually_ready):
    http = Http({"api/catalog": [OSError("offline"), OSError("starting"),
                                {} if eventually_ready else OSError("offline")]})
    started, sleeps = [], []
    times = iter([0, 0, ms.START_TIMEOUT + 1])
    monkeypatch.setattr(ms, "time", SimpleNamespace(time=lambda: next(times), sleep=sleeps.append))
    studio = ms.Studio(BASE, http=http, start=lambda: started.append(True))
    if eventually_ready:
        assert studio.get("api/catalog") == {}
    else:
        with pytest.raises(ms.StudioError, match="did not start"):
            studio.get("api/catalog")
    assert started == [True]
    assert sleeps == [.5]


def test_studio_http_error_without_json(studio_factory):
    failure = requests.Response()
    failure.status_code = 503
    failure._content = b"offline"
    studio = studio_factory({"api/models-folder": [failure]})
    with pytest.raises(ms.StudioError, match="answered 503"):
        studio.get("api/models-folder")


def test_studio_launch_uses_its_port_and_a_local_log(monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "REPO", tmp_path)
    calls = []

    def launch(argv, **kwargs):
        kwargs["stdout"].write(b"ready")
        calls.append((argv, kwargs))

    monkeypatch.setattr(ms.subprocess, "Popen", launch)
    ms.Studio(BASE, http=Http())._start_server()
    argv, kwargs = calls[0]
    assert argv == [sys.executable, str(tmp_path / "viewer/serve.py"), "--auto", "--no-browser", "--port", "8777"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["stderr"] == subprocess.STDOUT
    assert (tmp_path / "output/.studio-mcp.log").read_bytes() == b"ready"
    with pytest.raises(ms.StudioError, match="not this machine"):
        ms.Studio("http://example.invalid", http=Http())._start_server()


@pytest.mark.parametrize(("args", "expected"), [
    ([], ("stdio", {})),
    (["--http"], ("streamable-http", {"host": "127.0.0.1", "port": 8779})),
    (["--http", "8788"], ("streamable-http", {"host": "127.0.0.1", "port": 8788})),
])
def test_mcp_entry_point_selects_transport(args, expected, monkeypatch):
    from mcp.server.mcpserver import MCPServer

    calls = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kw: calls.append((transport, kw)))
    assert ms.main(args) == 0
    assert calls == [expected]


READY = {"backends": [{"id": "pixal3d", "kind": "3d", "state": "ready"}],
         "front_back": {"ready": True, "shape_ready": True}}


@pytest.mark.parametrize(("catalog", "options", "message"), [
    (READY, {"model_path": "model.glb"}, "not both"),
    ({"backends": READY["backends"]}, {"back_image_path": "back.png"}, "Front and back isn't installed"),
    ({"backends": [{"id": "trellis", "kind": "3d", "state": "ready"}],
      "front_back": {"shape_ready": True}}, {"back_image_path": "back.png"}, "needs Pixal3D"),
    ({}, {"front_and_back": True}, "Front and back isn't installed"),
    ({}, {}, "No 3D engine"),
])
def test_create_refuses_incompatible_inputs_before_upload(studio_factory, catalog, options, message):
    studio = studio_factory({"api/catalog": [catalog]})
    with pytest.raises(ms.StudioError, match=message):
        ms.create(studio, want="character", image_path="not-uploaded.png", **options)
    assert all(method != "POST" for method, *_ in studio.http.calls)


def test_new_character_can_rig_raw_model_and_stop_when_no_moves(studio_factory, tmp_path):
    picture = tmp_path / "in.png"
    picture.write_bytes(b"picture")
    studio = studio_factory({"api/catalog": [READY]}, {"api/chains": {"id": "r"}})
    result = ms.create(studio, want="character", image_path=str(picture),
                       advanced={"rig_from": "model", "in_place": False})
    assert result["steps"] == ["back", "model", "rigged"]
    fields = studio.http.calls[-1][2]["files"]
    assert "finish_settings" not in fields
    assert "move_settings" not in fields
    assert fields["file"] == ("in.png", b"picture", "image/png")


def test_continue_raw_rig_uses_model_instead_of_finished(studio_factory):
    asset = ASSET | {"model": "raw.glb"}
    studio = studio_factory({"api/assets": [{"assets": [asset]}]}, {"api/chains": {"id": "r"}})
    result = ms.continue_asset(studio, asset_id="a", up_to="rigged", advanced={"rig_from": "model"})
    assert result["steps"] == ["rigged"]
    assert result["starts_from"] == {"asset_id": "a", "version": 1, "file": "output/raw.glb"}
    data = studio.http.calls[-1][2]["files"]
    assert "finished" not in json.loads(data["made"][1])


@pytest.mark.parametrize(("asset", "opts", "message"), [
    (ASSET, {"up_to": "rigged", "advanced": {"rig_from": "model"}}, "no raw 3D model"),
    (ASSET, {"up_to": "model"}, "No 3D engine"),
    (ASSET | {"rigged": "rigged.glb", "model": "raw.glb"}, {"up_to": "animated", "move": "walk"}, "No move"),
    (ASSET | {"rigged": "rigged.glb", "model": "raw.glb"}, {"up_to": "animated"}, "Animation is not set up"),
])
def test_continue_refuses_missing_inputs(studio_factory, asset, opts, message):
    studio = studio_factory({"api/assets": [{"assets": [asset]}]}, {"api/chains": {"id": "r"}})
    with pytest.raises(ms.StudioError, match=message):
        ms.continue_asset(studio, asset_id="a", **opts)
    assert all(method != "POST" for method, *_ in studio.http.calls)


def test_continue_model_tracks_motion_settings_and_replaced_move(studio_factory):
    asset = ASSET | {"model": "raw.glb", "rigged": "rigged.glb", "clips": [{"name": "walk", "file": "old.glb"}]}
    moves = {"installed": True, "presets": [{"id": "walk", "label": "Walk", "in_place": True}]}
    studio = studio_factory({"api/assets": [{"assets": [asset]}], "api/animate/models": [moves]},
                             {"api/chains": {"id": "r"}})
    result = ms.continue_asset(studio, asset_id="a", up_to="animated", move="walk",
                              advanced={"arms_away": 20, "in_place": False})
    assert result["steps"] == ["animated"]
    assert "replaces the old" in result["replaces"]
    fields = studio.http.calls[-1][2]["files"]
    assert json.loads(fields["move_settings"][1]) == {"arm_spread": 20.0, "in_place": False}
    assert fields["first_move"] == (None, "walk")
    assert ms.list_moves(studio)["default"] == "walk"


def test_repeat_model_preserves_option_to_keep_everything(studio_factory):
    studio = studio_factory({"api/catalog": [READY]}, {"api/chains": {"id": "r"}})
    result = ms.continue_asset(studio, asset_id="a", up_to="model", again=True,
                              advanced={"keep_everything": True})
    assert result["steps"] == ["model"]
    form = studio.http.calls[-1][2]["files"]
    assert form["keep_everything"] == (None, "1")
    assert form["engine"] == (None, "pixal3d")


@pytest.mark.parametrize(("which", "asset", "change", "message"), [
    ("front", ASSET | {"picture": None}, "darker", "has no picture"),
    ("back", ASSET | {"back": None}, "darker", "has no back picture"),
    ("back", ASSET, "  ", "Say what goes"),
    ("back", ASSET, "darker", "No file"),
])
def test_brush_rejects_missing_picture_prompt_or_mask(studio_factory, tmp_path, which, asset, change, message):
    studio = studio_factory({"api/assets": [{"assets": [asset]}]})
    with pytest.raises(ms.StudioError, match=message):
        ms.brush_back(studio, asset_id="a", mask_path=str(tmp_path / "missing.png"), change=change, which=which)
    assert all(method != "POST" for method, *_ in studio.http.calls)


@pytest.mark.parametrize(("addon", "finished"), [(False, False), (True, False), (True, True)])
def test_run_waits_for_progress_or_completion(studio_factory, addon, finished):
    if addon:
        path = "api/addons/jobs/r/status"
        run_id = "addon-r"
        in_flight = {"status": "running", "addon_id": "addon", "tool_id": "tool"}
        result = in_flight | ({"status": "done", "result": "mesh.glb"} if finished
                              else {"status": "error", "error": "stopped", "log_tail": "worker failed"})
    else:
        path = "api/chains/r"
        run_id = "r"
        in_flight = {"id": "r", "status": "running", "done": []}
        result = in_flight | {"done": ["picture"]}
    studio = studio_factory({path: [in_flight, result]})
    sleeps = []
    out = ms.run_status(studio, run_id=run_id, wait_seconds=1, sleep=sleeps.append)
    assert sleeps == [2]
    assert out["status"] == result["status"]
    if finished:
        assert out["made"] == {"model": "mesh.glb"}
    elif addon:
        assert out["error"] == "stopped" and out["log"] == "worker failed"
    else:
        assert out["done"] == ["picture"]


def test_cancel_addon_reports_when_job_has_ended(studio_factory):
    error = SimpleNamespace(status_code=409, json=lambda: {"error": "already ended"})
    studio = studio_factory(posts={"api/addons/jobs/r/cancel": error})
    result = ms.cancel_run(studio, run_id="addon-r")
    assert result["cancelled"] is False
    assert "not running" in result["note"]


def test_picture_status_waits_or_returns_running_snapshot(studio_factory):
    job_id = "b" * 32
    path = f"api/image/{job_id}/status"
    studio = studio_factory({path: [{"status": "running"}, {"status": "running"},
                                    {"status": "done", "picture": "back.png"}]})
    sleeps = []
    assert ms.picture_status(studio, job_id=job_id, wait_seconds=0, sleep=sleeps.append)["status"] == "running"
    assert sleeps == []
    assert ms.picture_status(studio, job_id=job_id, wait_seconds=1, sleep=sleeps.append)["picture"] == "back.png"
    assert sleeps == [5]


def test_open_studio_front_page_uses_short_link(studio_factory):
    studio = studio_factory()
    opened = []
    result = ms.open_in_studio(studio, opener=lambda url: opened.append(url) or True)
    assert result == {"opened": True, "link": BASE + "/viewer/studio.html"}
    assert opened == [result["link"]]


def test_installing_ready_backend_never_requests_download(studio_factory):
    studio = studio_factory({"api/catalog": [READY]})
    assert ms.install_backend(studio, backend_id="pixal3d", confirm=True) == {"already_installed": "pixal3d"}
    assert all(method != "POST" for method, *_ in studio.http.calls)


@pytest.mark.parametrize("ready", [False, True])
def test_skin_picture_waits_for_actual_file(studio_factory, ready, tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "REPO", tmp_path)
    studio = studio_factory()
    picture = tmp_path / "output/front.png"
    if ready:
        picture.parent.mkdir()
        picture.write_bytes(b"front")
        assert ms.skin_picture(studio, skin_id="s")[0] == b"front"
    else:
        with pytest.raises(ms.StudioError, match="still being drawn"):
            ms.skin_picture(studio, skin_id="s")


def test_invalid_advanced_options_do_not_reach_http(studio_factory):
    studio = studio_factory()
    with pytest.raises(ms.StudioError, match="named choices"):
        ms.create(studio, want="character", advanced=["bad"])
    with pytest.raises(ms.StudioError, match="arms_away is a number"):
        ms.create(studio, want="character", advanced={"arms_away": "bad"})
    assert studio.http.calls == []


def test_finished_run_reports_log_when_worker_fails(studio_factory):
    studio = studio_factory({"api/chains": [{"running": {"id": "r"}}],
                             "api/chains/r": [{"id": "r", "status": "error", "full_log": "\x1b[31mcrash\x1b[0m"}]})
    assert ms.run_status(studio)["log"] == "crash"


def test_raw_brush_masks_must_be_png(studio_factory, tmp_path):
    path = tmp_path / "mask.txt"
    path.write_text("mask")
    studio = studio_factory()
    with pytest.raises(ms.StudioError, match="is not one of .png"):
        ms.brush_back(studio, asset_id="a", mask_path=str(path), change="darker")
    assert all(method != "POST" for method, *_ in studio.http.calls)


def test_full_detail_cannot_be_rigged_and_prop_finish_keeps_user_options(studio_factory):
    animal = ASSET | {"model": "raw.glb", "finished": "fox_full.glb"}
    studio = studio_factory({"api/assets": [{"assets": [animal]}]}, {"api/chains": {"id": "r"}})
    with pytest.raises(ms.StudioError, match="Full detail"):
        ms.continue_asset(studio, asset_id="a", up_to="rigged")
    assert all(method != "POST" for method, *_ in studio.http.calls)
    assert ms.continue_asset(studio, asset_id="a", up_to="finished", want="prop", again=True,
                             advanced={"upright": False, "height_m": 1.2})["steps"] == ["finished"]
    form = studio.http.calls[-1][2]["files"]
    assert json.loads(form["finish_settings"][1]) == {"height_m": 1.2, "upright": False}


def test_repeat_rig_sends_bone_names(studio_factory):
    asset = ASSET | {"model": "raw.glb"}
    studio = studio_factory({"api/assets": [{"assets": [asset]}]}, {"api/chains": {"id": "r"}})
    choices = ms.ADVANCED["bone_names"]["choices"]
    preset = next(value for value in choices if value != ms.ADVANCED["bone_names"]["default"])
    assert ms.continue_asset(studio, asset_id="a", up_to="rigged", again=True,
                             advanced={"bone_names": preset})["steps"] == ["rigged"]
    assert json.loads(studio.http.calls[-1][2]["files"]["rig_settings"][1]) == {"bone_names": preset}


def test_continue_drops_back_for_engine_that_cannot_use_it(studio_factory):
    asset = {"id": "a", "name": "Fox", "kind": "character", "stage": "picture", "picture": "fox.png"}
    catalog = READY | {"backends": [{"id": "trellis", "kind": "3d", "state": "ready"}]}
    studio = studio_factory({"api/assets": [{"assets": [asset]}], "api/catalog": [catalog]}, {"api/chains": {"id": "r"}})
    assert ms.continue_asset(studio, asset_id="a", up_to="model")["steps"] == ["model"]
    assert studio.http.calls[-1][2]["files"]["engine"] == (None, "trellis")


def test_brush_back_sends_selected_picture_engine(studio_factory, tmp_path):
    path = tmp_path / "mask.png"
    path.write_bytes(b"mask")
    studio = studio_factory(posts={"api/image/brush": {"job_id": "job"}})
    result = ms.brush_back(studio, asset_id="a", mask_path=str(path), change=" darker  ", picture_engine="qwen")
    form = studio.http.calls[-1][2]["files"]
    assert form["engine"] == (None, "qwen")
    assert form["instruction"] == (None, "darker")
    assert result["job_id"] == "job"


def test_pending_skin_without_approved_picture_has_no_build_instruction(studio_factory):
    state = SKIN | {"ready": False, "approved": {}}
    studio = studio_factory({"api/skins/s": [state]})
    assert "next" not in ms.skin_status(studio, skin_id="s")


def test_mcp_module_can_start_as_main(monkeypatch):
    import runpy

    from mcp.server.mcpserver import MCPServer

    calls = []
    monkeypatch.setattr(sys, "argv", ["mcp_server.py"])
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kwargs: calls.append(transport))
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(ms.__file__, run_name="__main__")
    assert exit.value.code == 0 and calls == ["stdio"]
