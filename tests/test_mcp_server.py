"""The MCP server: what agents can reach, what they send, and the download consent rule."""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

from image_to_3dlab import mcp_server as ms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))


# --------------------------------------------------------------------------- a fake studio
class Response:
    def __init__(self, status: int, body=None, content: bytes = b"", headers=None):
        self.status_code, self._body, self.content, self.headers = status, body, content, headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


CATALOG = {"host": {"label": "Apple Silicon Mac"}, "front_back": {"ready": True}, "backends": [
    {"id": "trellis", "kind": "3d", "rank": 3, "state": "ready", "license": {"name": "MIT"}},
    {"id": "pixal3d", "kind": "3d", "rank": 1, "state": "ready", "license": {"name": "MIT"}},
    {"id": "qwen", "kind": "image", "rank": 1, "state": "ready"},
    {"id": "hunyuan", "kind": "3d", "rank": 2, "state": "missing", "supported_here": True, "label": "Hunyuan3D",
     "license": {"name": "Tencent Hunyuan Community", "url": "https://example.test/licence"},
     "caveat": "Not licensed in the EU, UK or South Korea.", "human_expected": "12 GB", "best_for": "Detail",
     "weights": [{"label": "shape", "source": "tencent/x", "human_expected": "12 GB", "present": False}]},
]}
PRESETS = {"installed": True, "presets": [{"id": "idle", "label": "Idle", "take": 0}, {"id": "idle_b", "label": "Idle B", "take": 1},
                                          {"id": "wave", "label": "Wave", "take": 0}]}
KNIGHT = {"id": "aaaaaaaaaaaa", "name": "knight", "kind": "character", "stage": "finished", "updated": 0,
          "picture": "images/k/k.png", "model": "k/k.glb", "finished": "finish/k/k_20k.glb", "clips": [], "props": [],
          "prompt": "a knight"}
DANCER = {**KNIGHT, "id": "bbbbbbbbbbbb", "name": "dancer", "stage": "animated", "rigged": "animate/d/d_rigged.glb",
          "clips": [{"name": "idle", "file": "animate/d/d_idle.glb"}]}
PICTURE = {"id": "cccccccccccc", "name": "fox", "kind": "picture", "stage": "picture", "updated": 0,
           "picture": "images/f/f.png", "clips": [], "props": []}


class FakeHttp:
    def __init__(self):
        self.posts: list[tuple[str, dict]] = []
        self.alive = True

    def get(self, url, params=None, timeout=None):
        path = url.split("8777/", 1)[1]
        self.last_params = params
        if not self.alive:
            raise ConnectionError("refused")
        body = {
            "api/catalog": CATALOG, "api/animate/models": PRESETS,
            "api/recipes": {"goals": {"character": {"wording": "A-pose, plain background"}}},
            "api/assets": {"assets": [KNIGHT, DANCER, PICTURE]},
            "api/chains": {"running": None, "recent": [{"id": "r1", "status": "done", "done": ["picture"], "made": {"picture": "images/x.png"}}]},
            "api/chains/r1": {"id": "r1", "status": "done", "done": ["picture"], "made": {"picture": "images/x.png"}},
        }.get(path)
        if path == "api/bug-report":
            return Response(200, {"details": "AssetFurnace 0.4.5 · macOS", "failure": True, "has_prompt": True})
        if path == "api/model-info":
            return Response(200, {"file": "finish/k/k_20k.glb", "path": "/out/finish/k/k_20k.glb",
                                  "folder": "/out/finish/k", "stats": {"triangles": 19981}})
        if path == "api/preview":
            return Response(200, content=b"PNG", headers={"X-Preview-Stage": "finished", "X-Preview-Path": ".previews/x.png"})
        return Response(200, body) if body is not None else Response(404, {"error": f"no {path}"})

    def post(self, url, files=None, json=None, timeout=None):
        path = url.split("8777/", 1)[1]
        self.posts.append((path, files or json or {}))
        if path == "api/bug-report":
            return Response(200, {"site": "https://assetfurnace.com/report#report=x", "github": "https://github.com/i",
                                  "text": json["message"] + "\n\n" + json["details"], "discord": "https://discord.gg/x"})
        if path == "api/reveal":
            return Response(200, {"shown": "/out/" + json["file"]})
        if path == "api/chains":
            return Response(202, {"id": "new-run"})
        if path.endswith("/cancel"):
            return Response(200, {"cancelled": True})
        return Response(202, {"status_url": "/api/setup/hunyuan/status"})


@pytest.fixture
def studio():
    return ms.Studio("http://127.0.0.1:8777", http=FakeHttp(), start=lambda: None)


def sent(studio) -> dict:
    """The last plan posted, as plain fields."""
    path, files = studio.http.posts[-1]
    assert path == "api/chains"
    return {k: (v[1] if v[0] is None else v) for k, v in files.items()}


# --------------------------------------------------------------------------- every studio action is covered
def post_routes() -> set[str]:
    """Every POST route the studio server answers, read from its handler."""
    import generate_api

    source = inspect.getsource(generate_api.Handler.do_POST)
    routes = set()
    for cond in re.findall(r"if (.*?):\n", source):
        if "parts" not in cond or "api" not in cond:
            continue
        fixed = re.search(r"parts == \[(.*?)\]", cond)
        if fixed:
            routes.add("/".join(json.loads(f"[{fixed[1]}]")))
            continue
        head = re.search(r"parts\[:(\d)\] == \[(.*?)\]", cond)
        length = re.search(r"len\(parts\) == (\d)", cond)
        if not (head and length):
            continue
        parts = json.loads(f"[{head[2]}]") + ["*"] * (int(length[1]) - int(head[1]))
        last = int(length[1]) - 1
        tail = re.search(rf"parts\[{last}\] == \"(\w[\w-]*)\"", cond)
        options = re.search(rf"parts\[{last}\] in \{{(.*?)\}}", cond)
        for word in ([tail[1]] if tail else json.loads(f"[{options[1]}]") if options else ["*"]):
            routes.add("/".join(parts[:-1] + [word]))
    return routes


def test_every_studio_action_is_a_tool_or_marked_studio_only():
    routes = post_routes()
    assert {"api/chains", "api/setup/*/download", "api/assets/*/delete"} <= routes  # the reader still works
    unaccounted = routes - set(ms.STUDIO_ONLY) - ms.USED
    assert not unaccounted, (f"New studio actions agents can't reach: {sorted(unaccounted)}. "
                             "Add a tool in image_to_3dlab/mcp_server.py, or list them in STUDIO_ONLY with a reason.")
    stale = (set(ms.STUDIO_ONLY) | ms.USED) - routes
    assert not stale, f"Listed but gone from the server: {sorted(stale)}"


def test_the_server_registers_every_tool():
    pytest.importorskip("mcp")
    import asyncio

    server = ms.build_server(ms.Studio("http://127.0.0.1:8777", http=FakeHttp(), start=lambda: None))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"status", "list_engines", "list_moves", "create", "continue_asset", "run_status", "cancel_run",
                     "list_assets", "list_versions", "options", "preview", "open_in_studio", "install_backend",
                     "model_info", "show_in_folder", "report_bug"}


# --------------------------------------------------------------------------- create
def test_an_idea_for_a_character_runs_the_studios_full_plan_with_its_wording(studio):
    out = ms.create(studio, want="character", description="a knight.")
    form = sent(studio)
    assert json.loads(form["steps"]) == ["picture", "back", "model", "finished", "rigged", "animated"]
    assert form["prompt"] == "a knight, A-pose, plain background"
    assert form["engine"] == "pixal3d"             # best ready engine, not the first listed
    assert form["first_move"] == "idle"            # the first move that is not a second take
    assert out["run_id"] == "new-run"


def test_wording_can_be_left_off_and_the_plan_cut_short(studio):
    ms.create(studio, want="character", description="a knight", add_wording=False, stop_after_step="model")
    form = sent(studio)
    assert form["prompt"] == "a knight"
    assert json.loads(form["steps"]) == ["picture", "back", "model"]
    assert "first_move" not in form


def test_back_picture_only_with_an_engine_that_uses_it(studio):
    ms.create(studio, want="prop", description="a barrel", engine="trellis")
    assert json.loads(sent(studio)["steps"]) == ["picture", "model", "finished"]


def test_a_picture_on_disk_is_uploaded_and_skips_drawing(studio, tmp_path):
    image = tmp_path / "hero.png"
    image.write_bytes(b"\x89PNG fake")
    ms.create(studio, want="prop", image_path=str(image), front_and_back=False)
    form = sent(studio)
    assert form["file"][0] == "hero.png" and form["file"][1] == b"\x89PNG fake"
    assert json.loads(form["steps"]) == ["model", "finished"]
    assert "prompt" not in form


@pytest.mark.parametrize("kwargs, words", [
    ({"want": "dragon", "description": "x"}, "want must be"),
    ({"want": "character"}, "Describe what you want"),
    ({"want": "character", "description": "x", "engine": "hunyuan"}, "isn't ready"),
    ({"want": "character", "description": "x", "move": "moonwalk"}, "No move"),
    ({"want": "prop", "image_path": "/nope/missing.png"}, "No file"),
])
def test_bad_requests_say_why_and_send_nothing(studio, kwargs, words):
    with pytest.raises(ms.StudioError, match=words):
        ms.create(studio, **kwargs)
    assert studio.http.posts == []


# --------------------------------------------------------------------------- continue
def test_continuing_a_finished_character_rigs_and_animates_from_its_files(studio):
    ms.continue_asset(studio, asset_id=KNIGHT["id"])
    form = sent(studio)
    assert json.loads(form["steps"]) == ["rigged", "animated"]
    assert json.loads(form["made"]) == {"picture": "images/k/k.png", "model": "k/k.glb", "finished": "finish/k/k_20k.glb"}


def test_an_animated_character_can_be_given_another_move(studio):
    ms.continue_asset(studio, asset_id=DANCER["id"], up_to="animated", move="wave")
    form = sent(studio)
    assert json.loads(form["steps"]) == ["animated"] and form["first_move"] == "wave"


def test_remaking_a_move_it_already_has_says_so(studio):
    assert "replaces" not in ms.continue_asset(studio, asset_id=DANCER["id"], up_to="animated", move="wave")
    out = ms.continue_asset(studio, asset_id=DANCER["id"], up_to="animated", move="idle")
    assert "already had 'idle'" in out["replaces"]


def test_a_picture_goes_on_to_3d_with_a_back_view_when_installed(studio):
    ms.continue_asset(studio, asset_id=PICTURE["id"], up_to="model")
    assert json.loads(sent(studio)["steps"]) == ["back", "model"]


def test_nothing_to_do_sends_nothing(studio):
    out = ms.continue_asset(studio, asset_id=KNIGHT["id"], up_to="finished")
    assert "nothing_to_do" in out and studio.http.posts == []


# --------------------------------------------------------------------------- the rest
def test_run_status_without_an_id_reports_the_latest_run(studio):
    out = ms.run_status(studio, sleep=lambda s: None)
    assert out["status"] == "done" and out["made"] == {"picture": "output/images/x.png"}


def test_install_asks_first_and_downloads_nothing_without_confirm(studio):
    plan = ms.install_backend(studio, backend_id="hunyuan")
    assert studio.http.posts == []
    assert plan["total_size"] == "12 GB" and "EU" in plan["caveat"] and plan["licence"].startswith("Tencent")
    assert "confirm=true" in plan["to_start"]
    ms.install_backend(studio, backend_id="hunyuan", confirm=True)
    assert studio.http.posts[-1][0] == "api/setup/hunyuan/download"


def test_preview_returns_the_picture_and_a_studio_link(studio):
    data, info = ms.preview(studio, asset_id=KNIGHT["id"])
    assert data == b"PNG" and info["shown"] == "finished"
    assert info["studio"].endswith("/viewer/studio.html#asset=aaaaaaaaaaaa")


def test_open_in_studio_opens_the_assets_link(studio):
    opened = []
    out = ms.open_in_studio(studio, asset_id=KNIGHT["id"], opener=lambda url: opened.append(url) or True)
    assert opened == [out["link"]] and out["link"].endswith("#asset=aaaaaaaaaaaa")
    with pytest.raises(ms.StudioError):
        ms.open_in_studio(studio, asset_id="ffffffffffff", opener=lambda url: True)


def test_model_info_gives_the_stats_and_where_the_file_is(studio):
    out = ms.model_info(studio, asset_id=KNIGHT["id"])
    assert out["stats"]["triangles"] == 19981 and out["path"] == "/out/finish/k/k_20k.glb"
    assert studio.http.last_params == {"asset": KNIGHT["id"], "stage": "latest"}
    assert "triangles" in out["tip"]


def test_show_in_folder_opens_the_steps_file_on_this_machine(studio):
    out = ms.show_in_folder(studio, asset_id=KNIGHT["id"], stage="finished")
    assert studio.http.last_params == {"asset": KNIGHT["id"], "stage": "finished"}
    assert studio.http.posts[-1] == ("api/reveal", {"file": "finish/k/k_20k.glb"})
    assert out["shown"] == "/out/finish/k/k_20k.glb"


def test_report_bug_drafts_the_report_and_never_sends_it(studio):
    out = ms.report_bug(studio, message="Finish crashed", include_prompt=True)
    assert studio.http.last_params == {"prompt": "1"}
    assert studio.http.posts[-1] == ("api/bug-report", {"message": "Finish crashed", "details": "AssetFurnace 0.4.5 · macOS"})
    assert out["send_here"].startswith("https://assetfurnace.com/report")
    assert "user" in out["tip"].lower() and "send" in out["tip"].lower()


def test_a_studio_that_is_not_running_is_started_once():
    http, started = FakeHttp(), []
    http.alive = False

    def start():
        started.append(1)
        http.alive = True

    studio = ms.Studio("http://127.0.0.1:8777", http=http, start=start)
    ms.status(studio)
    ms.status(studio)
    assert started == [1]


def test_a_remote_studio_is_never_started_from_here():
    http = FakeHttp()
    http.alive = False
    with pytest.raises(ms.StudioError, match="not this machine"):
        ms.Studio("http://10.0.0.5:8777", http=http)._ensure()


def test_preview_can_show_a_move(studio):
    ms.preview(studio, asset_id=DANCER["id"], move="idle")
    assert studio.http.last_params["move"] == "idle"
    ms.preview(studio, asset_id=DANCER["id"])
    assert "move" not in studio.http.last_params


def test_a_finished_run_names_its_asset_and_logs_read_as_words(studio):
    out = ms.run_summary({"id": "r", "status": "done", "made": {"picture": "images/k/k.png", "model": "k/k.glb",
                                                                 "finished": "finish/k/k_20k.glb"}}, studio)
    assert out["asset_id"] == KNIGHT["id"] and out["studio"].endswith("#asset=" + KNIGHT["id"])
    running = ms.run_summary({"id": "r", "status": "running", "log": "|===>  | 2/8\x1b[K\n\x1b[1;32mdone\x1b[0m"}, studio)
    assert running["log"] == "|===>  | 2/8\ndone"


def test_a_new_move_is_traced_to_its_character():
    assets = [KNIGHT, DANCER]
    assert ms.asset_for_run(assets, {"rigged": "animate/d/d_rigged.glb", "animated": "animate/d/d_idle.glb"}) == DANCER["id"]
    assert ms.asset_for_run(assets, {"picture": "images/nowhere.png"}) is None


# --------------------------------------------------------------------------- Finish choices
def test_every_finish_detail_in_the_studio_is_one_agents_can_ask_for():
    source = (REPO / "viewer" / "studio" / "steps.js").read_text()
    menu = source[source.index("function finishDetail()"):]
    menu = menu[: menu.index("</select>")]
    studio_values = set(re.findall(r'<option value="([^"]+)"', menu)) - {"custom"}
    ours = {str(v.get("faces")) if "faces" in v else "full" for v in ms.FINISH_DETAILS.values()}
    assert studio_values == ours, "The studio's Finish Detail menu and FINISH_DETAILS in mcp_server.py disagree."


@pytest.mark.parametrize("detail, pm, expected", [
    (None, True, {}), ("full", True, {"keep_mesh": True}), ("game", False, {"faces": 20000, "skip_photo": True}),
    ("30000", True, {"faces": 30000}), (12000, True, {"faces": 12000}),
])
def test_finish_choices(detail, pm, expected):
    assert ms.finish_choices(detail, pm) == expected


def test_edge_fade_is_on_unless_asked_and_means_nothing_without_pixel_match():
    assert ms.finish_choices("full", True, edge_fade=False) == {"keep_mesh": True, "skip_edge_fade": True}
    assert ms.finish_choices("game", False, edge_fade=False) == {"faces": 20000, "skip_photo": True}


def test_nudge_is_on_unless_asked_and_means_nothing_without_pixel_match():
    assert ms.finish_choices("game", True, nudge=False) == {"faces": 20000, "skip_nudge": True}
    assert ms.finish_choices(None, True, edge_fade=False, nudge=False) == {"skip_edge_fade": True, "skip_nudge": True}
    assert ms.finish_choices("game", False, nudge=False) == {"faces": 20000, "skip_photo": True}


def test_nudge_travels_with_the_plan(studio):
    ms.create(studio, want="prop", description="a barrel", advanced={"nudge": False})
    assert json.loads(sent(studio)["finish_settings"]) == {"skip_nudge": True}


def test_edge_fade_travels_with_the_plan(studio):
    ms.create(studio, want="prop", description="a barrel", finish_detail="full", advanced={"edge_fade": False})
    assert json.loads(sent(studio)["finish_settings"]) == {"keep_mesh": True, "skip_edge_fade": True}


@pytest.mark.parametrize("detail", ["huge", "500", "999999"])
def test_unknown_finish_details_say_what_is_allowed(detail):
    with pytest.raises(ms.StudioError, match="finish_detail is one of"):
        ms.finish_choices(detail)


def test_a_full_detail_finish_travels_with_the_plan(studio):
    ms.create(studio, want="prop", description="a barrel", finish_detail="full")
    assert json.loads(sent(studio)["finish_settings"]) == {"keep_mesh": True}


def test_no_finish_choice_leaves_finish_on_its_defaults(studio):
    ms.create(studio, want="prop", description="a barrel")
    assert "finish_settings" not in sent(studio)


def test_full_detail_is_never_rigged(studio):
    with pytest.raises(ms.StudioError, match="not rigging"):
        ms.create(studio, want="character", description="a knight", finish_detail="full")
    full = {**KNIGHT, "finished": "finish/k/k_full.glb"}
    assert "not rigging" in ms.full_detail_refusal(["rigged", "animated"], {}, full)   # an existing full version
    assert ms.full_detail_refusal(["finished"], {"keep_mesh": True}) is None              # finishing alone is fine
    assert ms.full_detail_refusal(["rigged"], {}, KNIGHT) is None


def test_finish_again_makes_a_new_version_at_another_detail(studio):
    ms.continue_asset(studio, asset_id=KNIGHT["id"], up_to="finished", again=True, finish_detail="full")
    form = sent(studio)
    assert json.loads(form["steps"]) == ["finished"]
    assert json.loads(form["finish_settings"]) == {"keep_mesh": True}
    assert json.loads(form["made"])["model"] == "k/k.glb"


def test_again_needs_a_step_that_can_be_remade(studio):
    with pytest.raises(ms.StudioError, match="again=True works with"):
        ms.continue_asset(studio, asset_id=KNIGHT["id"], up_to="picture", again=True)
    with pytest.raises(ms.StudioError, match="has no model"):
        ms.continue_asset(studio, asset_id=PICTURE["id"], up_to="finished", again=True)


# --------------------------------------------------------------------------- versions
def test_list_versions_gathers_one_family_oldest_first_from_any_member():
    v1 = {**KNIGHT, "family": "fam", "version": 1, "main": True}
    v2 = {**KNIGHT, "id": "dddddddddddd", "family": "fam", "version": 2, "finished": "finish/k2/k_full.glb", "hidden": True}
    other = {**PICTURE, "family": "other"}
    studio = ms.Studio("http://127.0.0.1:8777", http=FakeHttp(), start=lambda: None)
    studio.get = lambda path, **_: {"assets": [v2, other, v1]}
    out = ms.list_versions(studio, asset_id="dddddddddddd")
    assert out["family"] == "fam" and out["prompt"] == "a knight"
    assert [(v["id"], v["version"], v["main"], v["hidden"]) for v in out["versions"]] == [
        ("aaaaaaaaaaaa", 1, True, False), ("dddddddddddd", 2, False, True)]
    assert out["versions"][1]["files"]["model"] == "output/k/k.glb"       # the raw model, before Finish
    assert out["versions"][1]["files"]["finished"] == "output/finish/k2/k_full.glb"


def test_an_asset_without_a_family_is_its_own_family(studio):
    out = ms.list_versions(studio, asset_id="cccccccccccc")
    assert out["family"] == "cccccccccccc" and [v["id"] for v in out["versions"]] == ["cccccccccccc"]


def test_list_versions_names_an_unknown_id(studio):
    with pytest.raises(ms.StudioError, match="No asset with id"):
        ms.list_versions(studio, asset_id="nope")


def test_list_assets_rows_say_their_family_and_version(studio):
    row = ms.list_assets(studio)["assets"][0]
    assert row["family"] == row["id"] and row["version"] == 1


# --------------------------------------------------------------------------- advanced knobs
def test_options_lists_the_main_and_every_advanced_knob():
    out = ms.options()
    assert set(out["advanced"]) == {"edge_fade", "nudge", "texture_size", "rig_from", "speed", "arms_away",
                                    "keep_everything"}
    assert all("what" in spec and "default" in spec and "step" in spec for spec in out["advanced"].values())
    assert {"finish_detail", "pixel_match", "move"} <= set(out["main"])


def test_every_studio_control_has_an_mcp_twin():
    """The studio's Create and Steps controls, each matched to its main or advanced MCP knob."""
    source = "".join((REPO / "viewer" / "studio" / f).read_text() for f in ("steps.js", "create.js"))
    controls = set(re.findall(r'data-k="(\w+)"', source)) - {"licence"}   # licence is a label, not a knob
    twin = {"faces": "finish_detail", "customFaces": "finish_detail", "photo": "pixel_match", "engine": "engine",
            "frontback": "front_and_back", "recipe": "add_wording", "move": "move", "text": "description",
            "edgefade": "edge_fade", "nudge": "nudge", "texture": "texture_size", "rigsource": "rig_from",
            "speed": "speed", "spread": "arms_away", "keep": "keep_everything"}
    missing = sorted(controls - set(twin))
    assert not missing, f"Studio controls with no MCP twin: {missing}. Add them to ADVANCED (or main) and here."
    known = set(ms.ADVANCED) | set(ms.options()["main"]) | {"description"}
    assert {twin[c] for c in controls} <= known


@pytest.mark.parametrize("advanced, error", [
    ({"colour": "red"}, "Unknown advanced choice 'colour'"), ({"nudge": "no"}, "nudge is true or false"),
    ({"texture_size": 3000}, "texture_size is one of"), ({"speed": 5}, "speed is from 0.5 to 2"),
    ({"rig_from": "picture"}, "rig_from is one of"),
])
def test_bad_advanced_choices_say_what_is_allowed(advanced, error):
    with pytest.raises(ms.StudioError, match=error):
        ms.read_advanced(advanced)


def test_advanced_texture_speed_and_arms_travel_with_the_plan(studio):
    ms.create(studio, want="character", description="a knight",
              advanced={"texture_size": 4096, "speed": 1.5, "arms_away": 10, "keep_everything": True})
    form = sent(studio)
    assert json.loads(form["finish_settings"]) == {"texture_size": 4096}
    assert json.loads(form["move_settings"]) == {"speed": 1.5, "arm_spread": 10}
    assert form["keep_everything"] == "1"


def test_defaults_add_nothing_to_the_plan(studio):
    ms.create(studio, want="character", description="a knight", advanced={})
    form = sent(studio)
    assert "finish_settings" not in form and "move_settings" not in form and "keep_everything" not in form


def test_rig_from_the_raw_model_skips_finish(studio):
    ms.create(studio, want="character", description="a knight", advanced={"rig_from": "model"})
    assert "finished" not in json.loads(sent(studio)["steps"])
    ms.continue_asset(studio, asset_id="aaaaaaaaaaaa", up_to="rigged", advanced={"rig_from": "model"})
    form = sent(studio)
    assert json.loads(form["steps"]) == ["rigged"]
    assert "finished" not in json.loads(form["made"]) and json.loads(form["made"])["model"] == "k/k.glb"
