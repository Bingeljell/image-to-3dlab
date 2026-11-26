"""Exercise the studio's job routes through real HTTP request parsing and responses."""

import io
import json
from email.parser import BytesParser
from types import SimpleNamespace

import generate_api as api
import pytest
from finish_api import FinishJob
from rig_api import RigJob


class RequestSocket:
    """The two streams a request handler gets from a TCP connection."""

    def __init__(self, request):
        self.incoming = io.BytesIO(request)
        self.outgoing = bytearray()

    def makefile(self, mode, *args):
        assert mode == "rb"
        return self.incoming

    def sendall(self, data):
        self.outgoing.extend(data)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(api, "JOBS", api.JobManager())
    monkeypatch.setattr(api, "FINISH_JOBS", type(api.FINISH_JOBS)(output / "finish", output))
    monkeypatch.setattr(api, "RIG_JOBS", type(api.RIG_JOBS)(output / "rig"))
    monkeypatch.setattr(api, "ANIMATE_JOBS", type(api.ANIMATE_JOBS)(output / "animate"))
    monkeypatch.setattr(api, "PROPS_JOBS", type(api.PROPS_JOBS)(output / "props"))
    monkeypatch.setattr(api.image_api, "MANAGER", api.image_api.ImageJobManager(output / "images"))
    monkeypatch.setattr(api.addons_api, "ADDON_JOBS", api.addons_api.AddonJobManager())
    monkeypatch.setattr(api, "SETUP_RUNS", {})
    store = api.skins_api.SkinStore(output)
    monkeypatch.setattr(api.skins_api, "STORE", store)
    monkeypatch.setattr(api.skins_api, "BUILDS", api.skins_api.SkinBuilds(store))
    monkeypatch.setattr(api, "_STARTING", {})
    monkeypatch.setattr(api, "SETUP_ACTIVE", None)
    monkeypatch.setattr(api, "download_active", lambda: None)

    class Handler(api.Handler):
        studio_token = None
        _chain_runner = None

    def request(path, body=None, *, content_type="application/json", method=None,
                headers=None, client="127.0.0.1"):
        if method is None:
            method = "POST" if body is not None else "GET"
        payload = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else b""
        fields = {
            "Host": "localhost",
            "X-AssetFurnace-Request": "studio",
            "Content-Type": content_type,
            "Content-Length": str(len(payload)),
            **(headers or {}),
        }
        wire = f"{method} {path} HTTP/1.0\r\n"
        wire += "".join(f"{name}: {value}\r\n" for name, value in fields.items())
        sock = RequestSocket(wire.encode() + b"\r\n" + payload)
        Handler(sock, (client, 10100), SimpleNamespace(server_address=("127.0.0.1", 8765)),
                directory=tmp_path)
        head, result = bytes(sock.outgoing).split(b"\r\n\r\n", 1)
        status, raw_headers = head.split(b"\r\n", 1)
        parsed = BytesParser().parsebytes(raw_headers, headersonly=True)
        if result and parsed.get_content_type() == "application/json":
            result = json.loads(result)
        assert api._STARTING == {}
        return int(status.split()[1]), parsed, result

    return request


JOB_ID = "a" * 32


@pytest.fixture(params=["generate", "finish", "rig/rebind"])
def model_job(request, studio, tmp_path):
    route = request.param
    directory = api.OUTPUT_ROOT / "sample"
    directory.mkdir()
    if route == "generate":
        manager, job = api.JOBS, api.Job(JOB_ID, directory, tmp_path / "source.png", directory / "model.glb", {}, "sf3d")
        artifacts = {"result.glb": job.output_path, "manifest.json": job.manifest_path}
    elif route == "finish":
        manager, job = api.FINISH_JOBS, FinishJob(JOB_ID, directory)
        artifacts = {action: getattr(job, attr) for action, (attr, _) in api.FINISH_ARTIFACTS.items()}
    else:
        manager, job = api.RIG_JOBS, RigJob(JOB_ID, directory)
        artifacts = {action: getattr(job, attr) for action, (attr, _) in api.RIG_ARTIFACTS.items()}
    manager.jobs[JOB_ID] = job
    return f"/api/{route}/{JOB_ID}", job, artifacts


def test_model_jobs_report_status_and_only_serve_completed_artifacts(studio, model_job):
    route, job, artifacts = model_job
    assert studio(route + "/status")[2]["status"] == "queued"
    job.status = "running"

    for action, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"{}" if action.endswith(".json") else b"fixture bytes"
        path.write_bytes(data)
        assert studio(route + "/" + action)[0] == 404

    job.status = "done"
    for action, path in artifacts.items():
        data = path.read_bytes()
        status, headers, body = studio(route + "/" + action)
        assert status == 200
        assert headers["Content-Length"] == str(len(data))
        disposition = "inline" if action == "result.glb" or "/generate/" in route else "attachment"
        assert headers["Content-Disposition"] == f'{disposition}; filename="{path.name}"'
        assert body == ({} if headers.get_content_type() == "application/json" else data)
    job.emit({"phase": "done", "message": "ready"})
    assert studio(route + "/status")[2]["last_event"]["message"] == "ready"
    artifacts["result.glb"].unlink()
    assert studio(route + "/result.glb")[0] == 404

    missing = route.replace(JOB_ID, "f" * 32)
    for action in ("status", "events", "result.glb"):
        assert studio(missing + "/" + action)[0] == 404
    assert studio(route + "/unknown")[0] == 404


def test_model_job_events_and_cancellation(studio, model_job, monkeypatch):
    route, job, _ = model_job
    monkeypatch.setattr(job.condition, "wait", lambda timeout: None)
    job.emit({"phase": "starting"})
    assert studio(route.replace(JOB_ID, "b" * 32) + "/cancel", {})[0] == 404
    status, _, body = studio(route + "/cancel", {})
    assert (status, body) == (202, {"job_id": JOB_ID, "status": "cancelling"})
    assert job.cancel_requested is True

    job.status = "cancelled"
    assert studio(route + "/cancel", {})[0] == 409
    status, headers, body = studio(route + "/events")
    assert status == 200
    assert headers["X-Accel-Buffering"] == "no"
    assert b'"phase":"starting"' in body
    assert body.endswith(b": keep-alive\n\n")


def test_images_report_progress_and_hide_results_until_ready(studio, monkeypatch):
    manager = api.image_api.MANAGER
    path = api.OUTPUT_ROOT / "result.png"
    job = api.image_api.ImageJob(JOB_ID, path.parent, path, "stone texture", {})
    manager.jobs[job.id] = job
    route = f"/api/image/{job.id}"
    assert studio("/api/image/invalid/status")[0] == 404
    job.set_status("running")
    path.write_bytes(b"png bytes")
    assert studio(route + "/status")[2]["result_url"] is None
    assert studio(route + "/result.png")[0] == 404
    assert studio(route + "/unknown")[0] == 404
    assert studio("/api/image/missing/cancel", {})[0] == 404
    status, _, body = studio(route + "/cancel", {})
    assert (status, body["status"]) == (202, "cancelling")
    assert job.cancel_requested

    job.set_status("done")
    assert studio(route + "/cancel", {})[0] == 409
    assert studio(route + "/status")[2]["result_url"] == route + "/result.png"
    status, headers, body = studio(route + "/result.png")
    assert (status, headers.get_content_type(), body) == (200, "image/png", b"png bytes")
    path.unlink()
    assert studio(route + "/result.png")[0] == 404
    monkeypatch.setattr(job.condition, "wait", lambda timeout: None)
    status, headers, body = studio(route + "/events")
    assert (status, headers.get_content_type()) == (200, "text/event-stream")
    assert b'"status":"done"' in body


def test_addon_status_events_and_cancellation(studio, monkeypatch):
    addons = api.addons_api
    job = addons.AddonJob(JOB_ID, "shape-tools", "check", {}, "test-run")
    addons.ADDON_JOBS.jobs[JOB_ID] = job
    job.append_log("checked geometry")
    route = f"/api/addons/jobs/{job.id}"
    assert studio("/api/addons/jobs/missing/status")[0] == 404
    assert studio("/api/addons/jobs/missing/events")[0] == 404
    assert studio("/api/addons/jobs/missing/cancel", {})[0] == 422
    assert studio(route + "/status")[2]["log_tail"] == "checked geometry"
    assert studio(route + "/events")[2]["events"] == job.events

    def finish(timeout):
        assert timeout == 25
        job.status = "done"
        job.emit({"phase": "done"})

    assert studio(route + "/cancel", {})[2] == {"cancelled": True}
    assert job.cancel_event.is_set()
    monkeypatch.setattr(job.condition, "wait", finish)
    assert studio(route + "/events?last=1")[2] == {"events": [], "status": "done"}
    assert studio(route + "/events?last=0")[2]["events"] == job.events[1:]
    assert studio(route + "/cancel", {})[0] == 422


@pytest.mark.parametrize("kind", ["props", "animate"])
def test_props_and_animation_job_routes(studio, monkeypatch, kind):
    from autorig_api import AnimateJob
    from props_api import PropsJob

    directory = api.OUTPUT_ROOT / kind / "sample"
    if kind == "animate":
        book = api.ANIMATE_JOBS
        job = AnimateJob(JOB_ID, "rig", directory, directory / "rigged.glb", [], 2, {})
    else:
        book = api.PROPS_JOBS
        job = PropsJob(JOB_ID, directory)

    book.jobs[JOB_ID] = job
    route = f"/api/{kind}/{JOB_ID}"
    assert studio(route + "/status")[2]["status"] == "queued"
    assert studio(route + "/unknown")[0] == 404
    assert studio(f"/api/{kind}/missing/status")[0] == 404
    assert studio(f"/api/{kind}/missing/cancel", {})[0] == 404
    assert studio(route + "/cancel", {})[2] == {"job_id": JOB_ID, "status": "cancelling"}
    assert job.cancel_requested

    job.emit({"phase": "error", "message": "stopped"})
    job.status = "cancelled"
    assert studio(route + "/cancel", {})[0] == 409
    monkeypatch.setattr(job.condition, "wait", lambda timeout: None)
    assert b'"message":"stopped"' in studio(route + "/events")[2]


def test_setup_and_download_status_without_launching_installers(studio, monkeypatch):
    from backend_catalog import resolve
    from download_api import DownloadRun

    setup = api.SetupRun(JOB_ID)
    api.SETUP_RUNS[JOB_ID] = setup
    run = DownloadRun(resolve("sf3d"))
    monkeypatch.setattr(api, "DOWNLOADS", {"sf3d": run})
    for job in (setup, run):
        job.status = "error"
        job.emit({"phase": "setup_done", "message": "stopped"})
        monkeypatch.setattr(job.condition, "wait", lambda timeout: None)
    assert studio("/api/setup/sf3d/status")[2]["last_event"]["message"] == "stopped"
    assert b'"message":"stopped"' in studio("/api/setup/sf3d/events")[2]
    assert b'"message":"stopped"' in studio(f"/api/setup/run/{JOB_ID}/events")[2]
    assert studio("/api/setup/unknown/status")[0] == 404
    assert studio("/api/setup/run/missing/events")[0] == 404


def test_library_renames_stars_hides_and_deletes_a_run(studio):
    from PIL import Image

    directory = api.OUTPUT_ROOT / "images" / "stone-wall__20260101-080000"
    directory.mkdir(parents=True)
    path = directory / f"{directory.name}.png"
    Image.new("RGB", (16, 12), (100, 20, 10)).save(path)
    asset = studio("/api/assets")[2]["assets"][0]
    route = f'/api/assets/{asset["id"]}'

    status, headers, data = studio(f'/api/thumb?path={asset["picture"]}')
    assert (status, headers.get_content_type()) == (200, "image/png")
    assert Image.open(io.BytesIO(data)).size == (16, 12)
    assert studio("/api/thumb?path=missing.png")[0] == 404
    assert studio(route + "/name", {"name": "  Stone   bricks  "})[2]["name"] == "Stone bricks"
    assert studio(route + "/hidden", {})[2]["hidden"] is True
    assert studio(route + "/main", {})[2]["main"] is True
    assert studio(route + "/delete", {"confirm": False})[0] == 422
    assert path.is_file()
    assert studio(route + "/delete-plan")[2]["bytes"] == path.stat().st_size
    assert studio(route + "/delete", {"confirm": True})[2]["deleted"] == [
        directory.relative_to(api.OUTPUT_ROOT).as_posix(),
    ]
    assert not directory.exists()
    assert studio("/api/assets")[2]["assets"] == []
    assert studio(route + "/delete-plan")[0] == 422


@pytest.mark.parametrize("action", ["name", "hidden", "main", "delete"])
def test_library_refuses_invalid_mutations(studio, action):
    route = "/api/assets/bad-id/" + action
    assert studio(route, {"confirm": True})[0] == 422
    assert studio(route, [])[0] == 422
    assert list(api.OUTPUT_ROOT.iterdir()) == []


def test_models_folder_saves_for_next_start(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_SETTING_REPO", tmp_path)
    monkeypatch.setattr(api, "MODELS_STARTED_WITH", None)
    route = "/api/models-folder"
    selected = tmp_path / "models"
    assert studio(route)[2]["restart_needed"] is False
    assert studio(route, {"path": "relative"})[0] == 422
    assert studio(route, [])[0] == 422
    assert studio(route, {"path": str(selected)})[2]["restart_needed"] is True
    assert selected.is_dir()
    assert studio(route)[2]["saved"] == str(selected)
    assert studio(route, method="POST")[2]["saved"] is None


def test_plan_routes_list_and_cancel_a_running_plan(studio, monkeypatch):
    chain = api.activity_api.Chain("A stone wall", ["picture"])
    runner = api.ChainRunner(api.OUTPUT_ROOT)
    monkeypatch.setattr(api, "ChainRunner", lambda output, request: runner)
    assert studio("/api/chains")[2] == {"running": None, "recent": []}
    runner.chains[chain.id] = chain
    route = f"/api/chains/{chain.id}"
    assert studio("/api/chains")[2]["running"]["title"] == chain.title
    assert studio(route)[2]["steps"] == ["picture"]
    assert studio("/api/chains/missing")[0] == 404
    assert studio("/api/chains/missing/cancel", {})[0] == 404
    assert studio(route + "/cancel", {})[2] == {"cancelled": True}
    assert chain.cancel_requested


def test_notices_persist_the_read_state_even_when_update_lookup_fails(studio, monkeypatch):
    def update():
        raise OSError("offline")

    monkeypatch.setattr(api, "update_check", update)
    monkeypatch.delenv("I3D_NO_NOTICES", raising=False)
    monkeypatch.setattr(api.notices_api, "CACHE", api.OUTPUT_ROOT / "notice-cache.json")
    monkeypatch.setattr(api.notices_api, "READ", api.OUTPUT_ROOT / "notice-read.json")
    monkeypatch.setattr(api.notices_api, "fetch_notices", lambda: {"notices": [{"id": "news", "title": "New tools"}]})
    assert studio("/api/notices")[2]["unread"] == 1
    assert studio("/api/notices/read", b"{bad json")[2]["unread"] == 1
    assert studio("/api/notices/read", [])[2]["unread"] == 1
    assert studio("/api/notices/read", {"ids": "news"})[2]["unread"] == 1
    assert studio("/api/notices/read", {"ids": ["news"]})[2]["unread"] == 0
    assert studio("/api/notices")[2]["notices"][0]["read"] is True


def test_recipes_report_invalid_or_missing_local_data(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPO", tmp_path)
    route = "/api/recipes"
    assert studio(route)[0] == 500
    recipes = tmp_path / "image_to_3dlab" / "prompt_recipes.json"
    recipes.parent.mkdir()
    recipes.write_text("{broken", encoding="utf-8")
    assert studio(route)[2]["error"].startswith("prompt recipes unreadable:")
    recipes.write_text('{"model": ["flat lighting"]}', encoding="utf-8")
    assert studio(route)[2] == {"model": ["flat lighting"]}


def test_head_and_unknown_routes(studio, tmp_path):
    (tmp_path / "local.txt").write_text("studio file", encoding="utf-8")
    status, headers, body = studio("/local.txt", method="HEAD")
    assert (status, headers["Content-Length"], body) == (200, "11", b"")
    assert studio("/output/")[0] == 404
    assert studio("/no-such-api", {})[0] == 404


@pytest.fixture
def picture_asset(studio):
    from PIL import Image

    directory = api.OUTPUT_ROOT / "images" / "stone-wall__20260101-080000"
    directory.mkdir(parents=True)
    path = directory / f"{directory.name}.png"
    Image.new("RGB", (16, 12), (100, 20, 10)).save(path)
    return studio("/api/assets")[2]["assets"][0]


def test_picture_preview_and_file_information(studio, picture_asset, monkeypatch):
    monkeypatch.setattr(api, "find_blender", lambda: None)
    asset = picture_asset["id"]
    picture = picture_asset["picture"]
    assert studio("/api/preview?asset=missing")[0] == 404
    assert studio(f"/api/preview?asset={asset}&look=wrong")[0] == 422
    status, headers, body = studio(f"/api/preview?asset={asset}")
    assert (status, headers["X-Preview-Stage"], headers.get_content_type()) == (200, "picture", "image/png")
    assert body == (api.OUTPUT_ROOT / headers["X-Preview-Path"]).read_bytes()

    assert studio(f"/api/model-info?file={picture}")[2]["file"] == picture
    assert studio(f"/api/model-info?asset={asset}")[2]["stats"] is None
    assert studio(f"/api/model-info?asset={asset}&stage=rigged")[0] == 422
    assert studio("/api/model-info?asset=missing")[0] == 422

    generated = api.OUTPUT_ROOT / "research_only" / "wall__trellis__20260101-080000"
    generated.mkdir(parents=True)
    generated.joinpath(f"{generated.name}.glb").write_bytes(b"model")
    model = next(item for item in studio("/api/assets")[2]["assets"] if item["model"])
    status, _, error = studio(f'/api/preview?asset={model["id"]}')
    assert status == 503
    assert "need Blender" in error["error"]


def test_status_pages_use_local_state(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "find_blender", lambda: None)
    from functools import partial

    monkeypatch.setattr(api, "catalog_status", lambda: {"backends": []})
    monkeypatch.setattr(api, "finish_capabilities", partial(api.finish_capabilities, find=lambda: None))
    monkeypatch.setattr(api, "update_check", lambda: {"newer": False})
    monkeypatch.setattr(api.addons_api, "ADDONS_DIR", tmp_path / "addons")
    monkeypatch.setattr(api.addons_api.ai, "load_registry", lambda index: [])
    status, headers, _ = studio("/")
    assert (status, headers["Location"]) == (302, "/viewer/studio.html")
    assert studio("/api/catalog")[2]["backends"] == []
    assert studio("/api/update-check")[2] == {"newer": False}
    assert studio("/api/addons")[2]["addons"] == []
    assert studio("/api/activity")[2]["history"] == []
    assert studio("/api/animate/models")[2]["models"] == []
    assert studio("/api/props/runs")[2]["runs"] == []
    assert studio("/api/props/tools")[0] == 200
    assert studio("/api/finish/runs")[2]["runs"] == []
    assert studio("/api/finish/capabilities")[0] == 200
    assert studio("/api/backends")[2]["backends"]
    result = studio("/api/image/defaults")[2]
    assert result["engine"] in {item["id"] for item in result["engines"]}
    assert studio("/api/setup?backend=missing")[0] == 422


@pytest.mark.parametrize(("route", "function"), [
    ("/api/assets", "assets_payload"),
    ("/api/props/runs", "list_props_runs"),
    ("/api/animate/models", "animate_models"),
])
def test_broken_run_folder_returns_json_error(studio, monkeypatch, route, function):
    def broken(_):
        raise OSError("bad run folder")

    monkeypatch.setattr(api, function, broken)
    status, _, body = studio(route)
    assert (status, body) == (500, {"error": "bad run folder"})


def test_bug_report_routes_supply_a_draft_without_sending_it(studio):
    info = studio("/api/bug-report?chain=unknown&step=model&prompt=1")[2]
    assert info["failure"] is False
    assert info["has_prompt"] is False
    assert studio("/api/bug-report", {})[0] == 422
    assert studio("/api/bug-report", b"{malformed")[0] == 422
    status, _, result = studio("/api/bug-report", {"message": "  Missing texture ", "details": "fixture"})
    assert status == 200
    assert result["text"] == "Missing texture\n\nfixture"


def test_trellis_advice_validates_upload_and_cleans_up_temporary_picture(studio, monkeypatch):
    route = "/api/trellis/input-advice"
    assert studio(route, method="POST")[0] == 400
    assert studio(route, method="POST", headers={"Content-Length": str(51 * 1024 * 1024)})[0] == 400
    wrong, wrong_type = api.activity_api.multipart({}, {"image": ("img.txt", b"not image")})
    assert studio(route, wrong, content_type=wrong_type)[0] == 422
    no_image, no_image_type = api.activity_api.multipart({"setting": "value"}, {})
    assert studio(route, no_image, content_type=no_image_type)[0] == 422
    body, content_type = api.activity_api.multipart({}, {"image": ("image.JPG", b"fixture picture")})

    files = []

    def advisory(path):
        assert path.suffix == ".jpg"
        assert path.read_bytes() == b"fixture picture"
        files.append(path)
        return {"verdict": "uncertain", "flat_risk": 0.5}

    monkeypatch.setattr(api, "run_trellis_input_advisor", advisory)
    assert studio(route, body, content_type=content_type)[2]["verdict"] == "uncertain"
    assert not files[0].exists()

    def missing(_):
        raise RuntimeError("not installed")

    monkeypatch.setattr(api, "run_trellis_input_advisor", missing)
    status, _, result = studio(route, body, content_type=content_type)
    assert (status, result) == (503, {"error": "not installed", "advisory_only": True})
    def broken(_):
        raise OSError("input checker unreadable")

    monkeypatch.setattr(api, "run_trellis_input_advisor", broken)
    assert studio(route, body, content_type=content_type)[0] == 500


def test_hf_status_and_sign_in_handle_missing_token(studio, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert studio("/api/hf/status")[2]["signed_in"] is False
    assert studio("/api/hf/sign-in", b"{bad json")[0] == 400
    assert studio("/api/hf/sign-in", method="POST")[0] == 400
    assert studio("/api/hf/sign-in", {"token": ""})[0] == 422


@pytest.mark.parametrize(("action", "expected"), [("preview", 400), ("keep", 400), ("discard", 200)])
def test_fix_spot_validates_json_before_editing(studio, action, expected):
    route = "/api/fix-spot/" + action
    assert studio(route, method="POST")[0] == expected
    assert studio(route, b"{bad json")[0] == expected
    assert studio(route, {})[0] == expected
    assert list(api.OUTPUT_ROOT.iterdir()) == []


def test_roblox_export_checks_format_and_constraints(studio):
    route = "/api/roblox-export"
    assert studio(route, b"", content_type="text/plain")[0] == 415
    assert studio(route, [])[0] == 422
    assert studio(route, {"file": "missing.glb", "height_studs": "bad"})[0] == 422
    assert studio(route, {"file": "missing.glb", "height_studs": "1.5"})[2]["error"] == "no such file: missing.glb"


def test_back_versions_can_be_swapped_then_deleted(studio):
    folder = api.OUTPUT_ROOT / "images"
    folder.mkdir()
    back, archived = (folder / "stone__back.png", folder / "stone__back_prev_20260101-080000.png")
    back.write_bytes(b"current")
    archived.write_bytes(b"older")
    relative = back.relative_to(api.OUTPUT_ROOT).as_posix()
    route = "/api/image/versions"
    assert studio(route)[0] == 422
    variants = studio(f"{route}?back={relative}")[2]
    assert variants["current"]["file"] == relative
    assert len(variants["versions"]) == 1
    assert studio(f"{route}/use", b"{bad json")[0] == 422
    data = {"back": relative, "version": variants["versions"][0]["file"]}
    changed = studio(f"{route}/use", data)[2]
    assert back.read_bytes() == b"older"
    assert not archived.is_file()
    data["version"] = changed["versions"][0]["file"]
    assert studio(f"{route}/delete", data)[2]["versions"] == []
    assert back.read_bytes() == b"older"
    with pytest.raises(ValueError, match="unknown action"):
        api.change_back_version("edit", {"back": relative, "version": relative})


def test_skin_listing_status_and_cancel_use_saved_state(studio):
    skins = api.skins_api
    skin = skins.Skin("Dark stone", "make the stone dark", {"finished": "sample.glb"})
    skins.STORE.save(skin)
    assert studio("/api/skins")[2]["skins"][0]["id"] == skin.id
    assert len(studio("/api/skins?finished=missing.glb")[2]["skins"]) == 0

    route = f"/api/skins/{skin.id}"
    assert studio(route)[2]["name"] == "Dark stone"
    assert studio("/api/skins/missing")[0] == 404
    assert studio(f"{route}/dressed?file=sample.glb")[0] in {404, 422}
    assert studio(f"{route}/other")[0] == 404
    assert studio("/api/skins/all?file=sample.glb")[0] == 404

    job = skins.SkinBuild(JOB_ID, skin.id)
    skins.BUILDS.jobs[JOB_ID] = job
    skins.BUILDS.active = JOB_ID
    active = studio(route)[2]["build_job"]
    assert active["job_id"] == job.id
    assert active["status"] == "queued"
    assert studio(route + "/discard", {})[0] == 409
    assert studio(route + "/approve", {})[0] == 409
    assert studio("/api/skins/build/missing/status")[0] == 404
    assert studio("/api/skins/build/missing/cancel", {})[0] == 409
    build = f"/api/skins/build/{job.id}"
    assert studio(f"{build}/status")[2]["skin_id"] == skin.id
    assert studio(f"{build}/cancel", {})[2] == {"cancelled": True}
    assert job.cancel_requested
    job.status = "cancelled"

    assert studio(route + "/build", {})[0] == 422
    assert studio(route + "/discard", {})[2] == {"discarded": skin.id}
    assert studio(route)[0] == 404
    assert studio("/api/skins", [])[0] == 400
    assert studio("/api/skins", b" " * 9000)[0] == 400


def test_addon_routes_validate_input_and_report_install_state(studio, tmp_path, monkeypatch):
    addons = api.addons_api
    monkeypatch.setattr(addons, "ADDONS_DIR", tmp_path / "addons")

    def install(**kwargs):
        if kwargs["addon_id"] == "missing":
            raise ValueError("unknown add-on")
        if kwargs["addon_id"] == "broken":
            raise RuntimeError("invalid manifest")
        assert kwargs["addon_id"] == "shape-tools"
        if kwargs["confirm"]:
            return {"id": "shape-tools"}
        plan = {
            "author": "Test", "licence": "MIT", "version": "1.0", "models": [],
            "setup": [], "runs": [], "warning": "", "tier": "listed",
        }
        return {"plan": plan, "listing": SimpleNamespace(id="shape-tools"),
                "manifest": SimpleNamespace(name="Shape tools")}

    monkeypatch.setattr(addons.ai, "install_addon", install)
    route = "/api/addons/shape-tools/install"
    assert studio(route, b"plain", content_type="text/plain")[0] == 415
    assert studio(route, [])[0] == 422
    assert studio(route, {})[2]["licence"] == "MIT"
    assert studio(route, {"confirm": True})[0] == 200
    assert studio("/api/addons/missing/install", {})[0] == 422
    assert studio("/api/addons/broken/install", {})[0] == 500
    assert studio("/api/addons/missing/remove", {})[0] == 422
    installed = addons.ADDONS_DIR / "shape-tools"
    installed.mkdir(parents=True)
    assert studio("/api/addons/shape-tools/remove", {})[2]["removed"]
    assert not installed.exists()
    assert studio("/api/addons/shape-tools/run/check", {"inputs": ["wrong"]})[0] == 422
    assert studio("/api/addons/shape-tools/run/check", [])[0] == 422
    assert studio("/api/addons/shape-tools/run/check", {})[2]["error"].endswith("is not installed")


def test_generation_with_user_supplied_back_requires_weights_and_right_file_type(studio, tmp_path, monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock

    monkeypatch.setattr(api, "REPO", tmp_path)
    for engine in ("trellis", "pixal3d"):
        monkeypatch.setitem(api.BACKENDS, engine, replace(api.BACKENDS[engine], readiness=lambda: {"ready": True},
                                                        gpu_memory_wanted=1))
    monkeypatch.setattr(api, "gpu_busy_note", lambda wanted: "Only a little memory is free")
    pending = Mock()
    monkeypatch.setattr(api.threading, "Thread", pending)

    def form(backend, back_name):
        settings = {"settings": json.dumps({"backend": backend})}
        pictures = {"image": ("front.png", b"front"), "back": (back_name, b"back")}
        body, content_type = api.activity_api.multipart(settings, pictures)
        return studio("/api/generate", body, content_type=content_type)

    assert form("trellis", "back.png")[0] == 422
    monkeypatch.setattr(api, "pixal3d_front_back_ready", lambda: (False, "Download first"))
    assert form("pixal3d", "back.png")[2] == {"error": "Download first", "needs_setup": True}
    monkeypatch.setattr(api, "pixal3d_front_back_ready", lambda: (True, None))
    assert form("pixal3d", "back.bmp")[0] == 422
    pending.assert_not_called()
    assert list(api.OUTPUT_ROOT.iterdir()) == []

    status, _, answer = form("pixal3d", "BACK.JPG")
    assert status == 202
    job = api.JOBS.get(answer["job_id"])
    pending.assert_called_once_with(target=api._run_job, args=(job,), daemon=True, name=None)
    assert job.image_path.read_bytes() == b"front"
    assert (job.directory / "back.jpg").read_bytes() == b"back"
    assert job.settings["back_name"] == "back.jpg"
    assert answer["warning"] == "Heads up: Only a little memory is free This run may fail."
    assert job.log_lines[-1] == answer["warning"]


def test_generation_cleans_up_if_image_check_cannot_run(studio, tmp_path, monkeypatch):
    import sys
    from dataclasses import replace

    monkeypatch.setattr(api, "REPO", tmp_path)
    monkeypatch.setitem(api.BACKENDS, "trellis", replace(api.BACKENDS["trellis"], readiness=lambda: {"ready": True}))
    monkeypatch.setitem(sys.modules, "PIL", None)
    upload, content_type = api.activity_api.multipart({}, {"image": ("front.png", b"front")})
    status, _, answer = studio("/api/generate", upload, content_type=content_type)
    assert status == 500
    assert "Pillow is not installed" in answer["error"]
    assert list(api.OUTPUT_ROOT.iterdir()) == []
    assert not api.JOBS.jobs


def test_catalog_setup_preserves_weight_rows_when_generator_does_not_report_them(studio, monkeypatch):
    from dataclasses import replace

    def readiness(backend):
        return {"name": backend, "weights": ["shape"], "missing_weights": ["shape"], "ready": False}

    monkeypatch.setattr(api, "catalog_readiness", readiness)
    monkeypatch.setitem(api.BACKENDS, "sf3d", replace(
        api.BACKENDS["sf3d"], readiness=lambda: {"weights": {}, "missing_weights": [], "ready": True},
    ))
    live = studio("/api/setup?backend=sf3d")[2]
    assert live["backend"] == "sf3d" and live["ready"]
    assert live["weights"] == live["missing_weights"] == ["shape"]
    assert studio("/api/setup?backend=image-only")[2] == readiness("image-only")
    monkeypatch.setitem(api.BACKENDS, "sf3d", replace(
        api.BACKENDS["sf3d"], readiness=lambda: {"weights": {"installed": {}}, "missing_weights": ["texture"]},
    ))
    assert studio("/api/setup?backend=sf3d")[2]["missing_weights"] == ["texture"]


def test_plan_refuses_empty_oversized_and_busy_requests_before_launch(studio, monkeypatch):
    from unittest.mock import Mock

    route = "/api/chains"
    assert studio(route, method="POST")[0] == 400
    assert studio(route, method="POST", headers={"Content-Length": str(api.CHAIN_UPLOAD_LIMIT + 1)})[0] == 413
    runner = api.ChainRunner(api.OUTPUT_ROOT)
    running = api.activity_api.Chain("Fixture", ["model"])
    runner.chains[running.id] = running
    monkeypatch.setattr(api, "ChainRunner", lambda output, request: runner)
    monkeypatch.setattr(api.image_api, "is_installed", lambda engine: True)
    thread = Mock()
    monkeypatch.setattr(api.threading, "Thread", thread)
    body, content_type = api.activity_api.multipart({"title": "Another run", "steps": '["picture"]'}, {})
    status, _, answer = studio(route, body, content_type=content_type)
    assert status == 409
    assert "already being made" in answer["error"]
    thread.assert_not_called()


def test_prop_turn_validates_request_and_reserves_job_without_editing_run(studio, monkeypatch):
    from unittest.mock import Mock

    manager = api.PROPS_JOBS
    started = manager.create("test sheet.glb", b"model", {})
    (started.directory / "split").mkdir()
    record = started.directory / "split" / "props.json"
    record.write_text('{"props":[{"name":"barrel"}]}', encoding="utf-8")
    route = f"/api/props/runs/{started.directory.name}/turn"
    assert studio(route, {"prop": "barrel"})[0] == 409
    manager.finish(started)
    assert studio(route, method="POST")[0] == 400
    assert studio(route, b"{malformed")[0] == 400
    assert studio(route, [])[0] == 400
    assert studio(route, {"prop": "barrel", "degrees": 181})[0] == 422
    thread = Mock()
    monkeypatch.setattr(api.threading, "Thread", thread)
    status, _, answer = studio(route, {"prop": "barrel"})
    assert status == 202
    job = manager.get(answer["job_id"])
    assert job.only == "barrel"
    assert answer["settings"]["turns"] == {"barrel": 90.0}
    assert json.loads(job.settings_path.read_text(encoding="utf-8"))["turns"] == {}
    assert job.staging not in record.parents
    thread.assert_called_once_with(target=api.run_props_job, args=(job,), daemon=True,
                                   name=f"props-{job.id[:8]}")


def test_reveal_uses_local_file_manager_and_reports_launch_failure(studio, picture_asset, monkeypatch):
    from unittest.mock import Mock

    runner = Mock()
    monkeypatch.setattr(api.model_info_api.subprocess, "Popen", runner)
    picture = picture_asset["picture"]
    route = "/api/reveal"
    assert studio(route, {"file": picture}, client="192.0.2.43")[0] == 403
    assert studio(route, {"file": "missing.png"})[0] == 422
    runner.assert_not_called()
    status, _, answer = studio(route, {"file": picture})
    path = (api.OUTPUT_ROOT / picture).resolve()
    assert (status, answer) == (200, {"shown": str(path)})
    command = runner.call_args.args[0]
    assert command == api.model_info_api.reveal_command(path, api.model_info_api.os_family())

    runner.side_effect = OSError("desktop not available")
    status, _, answer = studio(route, {"file": picture})
    assert status == 503
    assert "desktop not available" in answer["error"]


def test_cancel_generate_stops_the_running_group(studio, monkeypatch):
    from unittest.mock import Mock

    manager = api.JOBS
    job = manager.create(api.OUTPUT_ROOT / "front.png", {}, "sf3d", "front", None, api.OUTPUT_ROOT)
    job.process = SimpleNamespace(pid=3210, poll=lambda: None)
    terminate = Mock()
    monkeypatch.setattr(api, "_killpg_if_alive", terminate)
    status, _, answer = studio(f"/api/generate/{job.id}/cancel", {})
    assert (status, answer["status"]) == (202, "cancelling")
    terminate.assert_called_once_with(job.process.pid)
    assert job.cancel_requested
    assert job.events[-1]["message"] == "Cancellation requested"


def test_back_version_mutations_wait_while_a_picture_is_running(studio):
    book = api.image_api.MANAGER
    book.create("new picture", {})
    status, _, result = studio("/api/image/versions/delete", {"back": "fixture.png"})
    assert status == 409
    assert "wait for it to finish" in result["error"]


def test_skin_downloads_use_real_variant_and_dressed_file_logic(studio):
    from tests.test_glb_skins import _glb

    skins = api.skins_api
    directory = api.OUTPUT_ROOT / "finish" / "statue__finish__20260101-080000"
    directory.mkdir(parents=True)
    model = directory / "statue_20k.glb"
    model.write_bytes(_glb(b"BASE", b"SHINE"))
    asset = studio("/api/assets")[2]["assets"][0]
    skin = skins.Skin("Copper", "paint copper", {"finished": asset["finished"]}, status="done")
    folder = skins.STORE.folder(skin.id)
    folder.mkdir(parents=True)
    paint = folder / "paint.glb"
    paint.write_bytes(_glb(b"COPPER", b"SHINE"))
    skin.result = paint.relative_to(api.OUTPUT_ROOT).as_posix()
    skins.STORE.save(skin)
    dressed = studio(f'/api/skins/{skin.id}/dressed?file={asset["finished"]}')
    assert dressed[0] == 200
    assert dressed[2]["file"] == skin.result
    route = f'/api/skins/all?asset={asset["id"]}&file={asset["finished"]}'
    assert studio(route.replace(asset["finished"], "wrong.glb"))[0] == 404
    status, _, variants = studio(route)
    assert status == 200
    result = (api.OUTPUT_ROOT / variants["file"]).read_bytes()
    assert b"BASE" in result and b"COPPER" in result
    assert b"Original paint" in result


def test_setup_refuses_cancel_for_unknown_route(studio, monkeypatch):
    import download_api

    monkeypatch.setattr(download_api, "DOWNLOADS", {})
    assert studio("/api/setup/unknown/cancel", {})[0] == 409
    assert studio("/api/setup/unknown/remove", {})[0] == 404
