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


CATALOG = {"host": {"label": "Apple Silicon Mac"}, "front_back": {"ready": True, "shape_ready": True},
           "picture_engines": [{"id": "flux", "installed": True}, {"id": "qwen", "installed": False}], "backends": [
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
        self.assets = [KNIGHT, DANCER, PICTURE]

    def get(self, url, params=None, timeout=None):
        path = url.split("8777/", 1)[1]
        self.last_params = params
        if not self.alive:
            raise ConnectionError("refused")
        body = {
            "api/catalog": CATALOG, "api/animate/models": PRESETS,
            "api/recipes": {"goals": {"character": {"wording": "A-pose, plain background"}}},
            "api/assets": {"assets": self.assets},
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

    source = inspect.getsource(generate_api.Handler._route_post)
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
                     "list_assets", "list_versions", "options", "preview", "open_in_studio", "install_backend", "blender_helpers",
                     "model_info", "show_in_folder", "report_bug", "make_skin", "skin_status", "skin_picture",
                     "approve_skin_picture", "redo_skin_picture", "build_skin", "cancel_skin", "export_skins",
                     "usage_counts", "brush_back", "picture_status"}


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


# --------------------------------------------------------------------------- engines
def test_an_engine_with_an_update_says_so_and_where_to_get_it(studio, monkeypatch):
    """Rebuild is a Setup-page button; an agent can still tell the user it is there."""
    pixal = next(b for b in CATALOG["backends"] if b["id"] == "pixal3d")
    monkeypatch.setitem(pixal, "rebuild_reason", "An older download: the new one adds the new Finish.")
    rows = {e["id"]: e for e in ms.list_engines(studio)}
    assert "new Finish" in rows["pixal3d"]["update"] and "Setup" in rows["pixal3d"]["update"]
    assert "update" not in rows["trellis"]


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


# The 2026-10-18 dwarf: "3D model again" made version 2, then Finish was asked for on the family id
# (version 1), which quietly finished version 1's old model.
OLD_GNOME = {**PICTURE, "id": "dddddddddddd", "name": "gnome", "stage": "model", "family": "dddddddddddd",
             "version": 1, "model": "g/g_v1.glb"}
NEW_GNOME = {**OLD_GNOME, "id": "eeeeeeeeeeee", "version": 2, "model": "g/g_v2.glb"}


def test_continuing_says_which_version_and_file_it_starts_from(studio):
    studio.http.assets = [OLD_GNOME, NEW_GNOME]
    out = ms.continue_asset(studio, asset_id=NEW_GNOME["id"], up_to="finished")
    assert out["starts_from"] == {"asset_id": "eeeeeeeeeeee", "version": 2, "file": "output/g/g_v2.glb"}
    assert "newer_version" not in out


def test_continuing_an_older_version_points_at_the_newer_one(studio):
    studio.http.assets = [OLD_GNOME, NEW_GNOME]
    out = ms.continue_asset(studio, asset_id=OLD_GNOME["id"], up_to="finished")
    assert out["starts_from"]["file"] == "output/g/g_v1.glb"
    assert "eeeeeeeeeeee" in out["newer_version"] and "Version 2" in out["newer_version"]


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


def test_install_refusal_gives_the_catalogues_reason(studio):
    CATALOG["backends"].append({"id": "old-mac", "label": "TRELLIS.2", "supported_here": False, "requires": "an Apple Silicon Mac",
                                "platform_note": "TRELLIS.2 on a Mac needs macOS 26 or newer; this Mac has macOS 15."})
    try:
        with pytest.raises(ms.StudioError, match="needs macOS 26"):
            ms.install_backend(studio, backend_id="old-mac")
    finally:
        CATALOG["backends"].pop()


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
    menu = source[source.index("function finishDetail("):]
    menu = menu[: menu.index("</select>")]
    studio_values = set(re.findall(r'<option value="([^"]+)"', menu)) - {"custom"}
    ours = {"high" if v.get("high_detail") else "hero" if v.get("hero") else str(v.get("faces")) if "faces" in v else "full"
            for v in ms.FINISH_DETAILS.values()}
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
            "speed": "speed", "spread": "arms_away", "keep": "keep_everything",
            "picengine": "picture_engine", "change": "back_change"}
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


# --------------------------------------------------------------------------- skins
SKIN = {"id": "5k1n5k1n5k1n", "name": "gold skirt", "instruction": "make the skirt gold", "status": "pictures",
        "base": {"finished": "finish/k/k_20k.glb", "front": "images/k/k.png", "back": "images/k/k__back.png"},
        "tries": {"front": [{"number": 1, "instruction": "make the skirt gold", "seed": 42,
                             "picture": "skins/5k1n5k1n5k1n/front-1.png"}], "back": []},
        "approved": {"front": None, "back": None}, "needs_back": True, "ready": False, "next_side": "front",
        "result": None, "error": None, "picture_job": {"status": "running"}, "build_job": None}


class SkinHttp(FakeHttp):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def get(self, url, params=None, timeout=None):
        if "api/skins/" in url:
            return Response(200, self.state)
        return super().get(url, params, timeout)

    def post(self, url, files=None, json=None, timeout=None):
        path = url.split("8777/", 1)[1]
        self.posts.append((path, json or files or {}))
        if path.startswith("api/skins/build/"):
            return Response(202, {"cancelled": True})
        return Response(202 if path == "api/skins" else 200, {"skin": self.state, "job_id": "b1"})


def test_make_skin_sends_the_change_and_says_qwen_is_drawing():
    studio = ms.Studio("http://127.0.0.1:8777", http=SkinHttp(SKIN), start=lambda: None)
    out = ms.make_skin(studio, asset_id="aaaaaaaaaaaa", change="make the skirt gold")
    assert studio.http.posts[-1] == ("api/skins", {"asset": "aaaaaaaaaaaa", "instruction": "make the skirt gold"})
    assert out["skin_id"] == SKIN["id"] and out["drawing"] is True and "skin_status" in out["next"]


def test_a_drawn_front_asks_for_the_users_verdict():
    state = {**SKIN, "picture_job": {"status": "done"}}
    out = ms.skin_summary(state)
    assert "skin_picture" in out["next"] and "approve_skin_picture" in out["next"]
    assert out["tries"]["front"][0]["file"] == "output/skins/5k1n5k1n5k1n/front-1.png"


def test_both_approved_points_at_build():
    state = {**SKIN, "picture_job": {"status": "done"}, "ready": True, "next_side": None,
             "approved": {"front": "skins/x/front-1.png", "back": "skins/x/back-1.png"}}
    assert "build_skin" in ms.skin_summary(state)["next"]


def test_redo_and_approve_send_side_change_and_number():
    studio = ms.Studio("http://127.0.0.1:8777", http=SkinHttp(SKIN), start=lambda: None)
    ms.redo_skin_picture(studio, skin_id=SKIN["id"], side="back", change="keep the belt blue")
    assert studio.http.posts[-1] == (f"api/skins/{SKIN['id']}/try", {"side": "back", "instruction": "keep the belt blue"})
    ms.approve_skin_picture(studio, skin_id=SKIN["id"], side="front", number=2)
    assert studio.http.posts[-1] == (f"api/skins/{SKIN['id']}/approve", {"side": "front", "number": 2})


def test_a_picture_still_drawing_is_not_shown():
    studio = ms.Studio("http://127.0.0.1:8777", http=SkinHttp(SKIN), start=lambda: None)
    with pytest.raises(ms.StudioError, match="still being drawn"):
        ms.skin_picture(studio, skin_id=SKIN["id"])


def test_cancel_reaches_the_running_build_only():
    idle = ms.Studio("http://127.0.0.1:8777", http=SkinHttp(SKIN), start=lambda: None)
    assert ms.cancel_skin(idle, skin_id=SKIN["id"])["cancelled"] is False
    building = ms.Studio("http://127.0.0.1:8777", http=SkinHttp({**SKIN, "build_job": {"job_id": "b1"}}),
                         start=lambda: None)
    assert ms.cancel_skin(building, skin_id=SKIN["id"])["cancelled"] is True
    assert building.http.posts[-1][0] == "api/skins/build/b1/cancel"


def test_list_assets_shows_a_models_skins():
    asset = {**KNIGHT, "skins": [{"id": "s1", "name": "gold skirt", "file": "skins/s1/gold-skirt_20k.glb"}]}
    assert ms.asset_summary(asset)["skins"] == [{"id": "s1", "name": "gold skirt",
                                                 "file": "output/skins/s1/gold-skirt_20k.glb"}]


def test_preview_can_show_a_skin(studio):
    ms.preview(studio, asset_id="aaaaaaaaaaaa", skin="k1")
    assert studio.http.last_params["skin"] == "k1"


class ExportHttp(FakeHttp):
    def get(self, url, params=None, timeout=None):
        path = url.split("8777/", 1)[1]
        self.last_params = params
        if path == "api/assets":
            dancer = {**DANCER, "skins": [{"id": "k1", "name": "gold", "file": "skins/k1/gold_20k.glb"}]}
            return Response(200, {"assets": [dancer]})
        if path == "api/skins/all":
            return Response(200, {"file": "skins/all/d__skins-abc.glb"})
        if path == "api/skins/k1/dressed":
            return Response(200, {"file": "skins/k1/dressed/" + params["file"].split("/")[-1]})
        return super().get(url, params, timeout)


def test_export_dresses_the_rig_and_every_move_in_the_skin():
    studio = ms.Studio("http://127.0.0.1:8777", http=ExportHttp(), start=lambda: None)
    out = ms.export_skins(studio, asset_id=DANCER["id"], skin_id="k1")
    assert out["files"] == {"finished": "output/skins/k1/gold_20k.glb",
                            "rigged": "output/skins/k1/dressed/d_rigged.glb",
                            "idle": "output/skins/k1/dressed/d_idle.glb"}


def test_export_without_a_skin_is_one_file_with_every_skin():
    studio = ms.Studio("http://127.0.0.1:8777", http=ExportHttp(), start=lambda: None)
    out = ms.export_skins(studio, asset_id=DANCER["id"])
    assert out["file"] == "output/skins/all/d__skins-abc.glb" and studio.http.last_params["file"] == DANCER["rigged"]


def test_export_says_when_there_is_nothing_to_export(studio):
    with pytest.raises(ms.StudioError, match="no finished skins"):
        ms.export_skins(studio, asset_id=KNIGHT["id"])


def test_agents_can_ask_for_high_detail():
    assert ms.finish_choices("high") == {"faces": 50000, "high_detail": True}


def test_agents_are_told_where_the_guide_is_and_that_fix_a_spot_is_hands_on():
    from image_to_3dlab import mcp_server
    text = mcp_server.INSTRUCTIONS
    assert "https://assetfurnace.com/blog/how-to-use-assetfurnace" in text
    assert "Fix a spot" in text and "open_in_studio" in text
    assert 'finish_detail="hero"' in text


# --------------------------------------------------------------------------- usage counts
class UsageHttp(FakeHttp):
    def __init__(self, noticed):
        super().__init__()
        self.usage = {"enabled": True, "noticed": noticed, "sends": {"id": "a" * 32},
                      "policy": "https://assetfurnace.com/privacy#usage-counts"}

    def get(self, url, params=None, timeout=None):
        if url.endswith("api/usage"):
            return Response(200, self.usage)
        return super().get(url, params, timeout)

    def post(self, url, files=None, json=None, timeout=None):
        if url.endswith("api/usage"):
            self.posts.append(("api/usage", json))
            return Response(200, {**self.usage, "noticed": True})
        return super().post(url, files, json, timeout)


def test_status_hands_the_agent_the_question_until_the_user_answered():
    asked = ms.status(ms.Studio("http://127.0.0.1:8777", http=UsageHttp(noticed=False), start=lambda: None))
    assert asked["usage_counts"]["ask_user"] == ms.USAGE_ASK
    assert "user_said_yes" in asked["usage_counts"]["then"]
    answered = ms.status(ms.Studio("http://127.0.0.1:8777", http=UsageHttp(noticed=True), start=lambda: None))
    assert answered["usage_counts"] is None


def test_an_older_studio_without_usage_counts_asks_nothing(studio):
    assert ms.status(studio)["usage_counts"] is None


def test_an_agent_passes_on_a_yes_and_can_never_switch_sharing():
    http = UsageHttp(noticed=False)
    studio = ms.Studio("http://127.0.0.1:8777", http=http, start=lambda: None)
    looked = ms.usage_counts(studio)
    assert http.posts == [] and "About page" in looked["to_change"]
    ms.usage_counts(studio, user_said_yes=True)
    assert http.posts == [("api/usage", {"noticed": True})]     # never {"share": ...}


def test_status_says_where_models_go_and_survives_an_older_studio():
    class Answers:
        def __init__(self, answer):
            self.answer = answer

        def get(self, path):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    assert ms.models_folder(Answers({"in_use": "/Volumes/Big/models"})) == "/Volumes/Big/models"
    assert ms.models_folder(Answers({"in_use": None})) is None
    assert ms.models_folder(Answers(RuntimeError("404: no such route"))) is None


def test_blender_helpers_lists_every_blender_script_with_how_to_run_it():
    got = ms.blender_helpers()
    on_disk = {p.name for p in (ms.REPO / "scripts").glob("*.py") if ms.is_blender_helper(p.name)}
    assert {h["script"] for h in got["helpers"]} == on_disk
    for h in got["helpers"]:
        assert h["mode"] in ms.BLENDER_HOW, f"{h['script']}: add a 'Run (live|headless|internal): ...' line to its docstring"
        assert h["run"] and h["does"] and h["group"]
        if h["mode"] != "internal":
            assert h["script"] in h["run"], f"{h['script']}: its Run line should name the script"
    assert set(got["how"]) >= set(ms.BLENDER_HOW)


def test_every_run_line_gives_the_options_its_script_requires():
    # An agent runs the line as written: blender_agent.py's left out --out-dir, so it stopped on an argparse error.
    import ast

    for h in ms.blender_helpers()["helpers"]:
        if h["mode"] == "internal" or h["run"].endswith("--help"):
            continue   # a --help line sends the agent to the full list first, which is fine
        tree = ast.parse((ms.REPO / "scripts" / h["script"]).read_text(encoding="utf-8"))
        if any(getattr(n.func, "attr", "") == "add_subparsers" for n in ast.walk(tree) if isinstance(n, ast.Call)):
            continue   # sub-commands each need their own options; the line shows one and points to the rest
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "add_argument"):
                continue
            required = any(k.arg == "required" and getattr(k.value, "value", False) is True for k in call.keywords)
            flags = [a.value for a in call.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
            if required and flags:
                assert any(f in h["run"] for f in flags), f"{h['script']}: its Run line leaves out {flags[0]}"


def test_live_helpers_warn_that_they_clear_the_open_scene():
    # Most live helpers delete everything in the scene they run in; "changes the scene" undersold that.
    assert "CLEAR" in ms.BLENDER_HOW["live"] and "empty scene" in ms.BLENDER_HOW["live"]


def test_run_line_reads_the_mode_and_command():
    assert ms.run_line("Does a thing.\n\nRun (headless): blender -b -P x.py -- a b\n\nWhy.") == ("headless", "blender -b -P x.py -- a b")
    assert ms.run_line("Does a thing.") is None
    assert not ms.is_blender_helper("bootstrap_blender.py")     # an installer, not a helper


def test_create_passes_an_installed_picture_engine_and_refuses_a_missing_one(studio):
    ms.create(studio, want="picture", description="a fox", picture_engine="flux")
    assert sent(studio)["picture_engine"] == "flux"
    with pytest.raises(ms.StudioError, match="'qwen' isn't installed. Installed: flux"):
        ms.create(studio, want="picture", description="a fox", picture_engine="qwen")
    ms.create(studio, want="picture", description="a fox")
    assert "picture_engine" not in sent(studio)   # the studio's default: Flux when installed


def test_create_sends_your_own_back_and_draws_none(studio, tmp_path):
    front, back = tmp_path / "orc.png", tmp_path / "orc-back.png"
    front.write_bytes(b"front"), back.write_bytes(b"back")
    out = ms.create(studio, want="prop", image_path=str(front), back_image_path=str(back))
    _, files = studio.http.posts[-1]
    assert files["back_file"][0] == "orc-back.png" and files["back_file"][1] == b"back"
    assert "back" not in out["steps"] and "model" in out["steps"] and out["engine"] == "pixal3d"


@pytest.mark.parametrize("kwargs, error", [
    ({"back_image_path": "x.png"}, "goes with image_path"),
    ({"image_path": "FRONT", "back_image_path": "BACK", "engine": "trellis"}, "Pixal3D only"),
])
def test_a_back_picture_that_cannot_be_used_is_refused_up_front(studio, tmp_path, kwargs, error):
    for name in ("FRONT", "BACK"):
        (tmp_path / f"{name}.png").write_bytes(b"x")
    kwargs = {k: str(tmp_path / f"{v}.png") if v in ("FRONT", "BACK") else v for k, v in kwargs.items()}
    with pytest.raises(ms.StudioError, match=error):
        ms.create(studio, want="prop", **kwargs)


def test_status_counts_an_installed_picture_engine_even_if_its_card_is_partial(studio):
    """Qwen draws without its projector; a partial card must not tell agents there are no pictures."""
    catalog = {**CATALOG, "backends": [b for b in CATALOG["backends"] if b["kind"] != "image"]}
    original = studio.http.get
    studio.http.get = lambda url, **k: Response(200, catalog) if url.endswith("api/catalog") else original(url, **k)
    assert ms.status(studio)["pictures"] is True


def test_drawing_the_back_again_sends_a_fresh_seed_and_what_to_change(studio):
    ms.continue_asset(studio, asset_id=KNIGHT["id"], up_to="back", again=True, back_change="  a red cloak  ")
    form = sent(studio)
    assert json.loads(form["steps"]) == ["back"]
    settings = json.loads(form["back_settings"])
    assert settings["instruction"] == "a red cloak" and isinstance(settings["seed"], int)
    with pytest.raises(ms.StudioError, match="goes with up_to='back'"):
        ms.continue_asset(studio, asset_id=KNIGHT["id"], back_change="a red cloak")


def test_brush_back_sends_the_back_the_mask_and_what_goes_there(studio, tmp_path):
    knight = {**KNIGHT, "back": "images/k/k__back.png"}
    studio.http.assets = [knight]
    original = studio.http.post
    studio.http.post = lambda url, files=None, **k: (studio.http.posts.append((url.split("8777/", 1)[1], files))
                                                     or Response(202, {"job_id": "j1"}))
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"PNG")
    out = ms.brush_back(studio, asset_id=knight["id"], mask_path=str(mask), change=" a red satchel ")
    path, files = studio.http.posts[-1]
    assert path == "api/image/brush" and out["job_id"] == "j1"
    assert files["picture"] == (None, "images/k/k__back.png") and files["instruction"] == (None, "a red satchel")
    assert files["mask"][0] == "mask.png"
    studio.http.post = original
    studio.http.assets = [KNIGHT]
    with pytest.raises(ms.StudioError, match="has no back picture"):
        ms.brush_back(studio, asset_id=KNIGHT["id"], mask_path=str(mask), change="x")
