"""The studio's start routes, asked the way the page asks them.

Every big job (a generation, a Finish run, a prop sheet, setup and engine downloads, and then
pictures, skins, rig rebinds and animations) is started here over HTTP, against the real
request handler, and so is a whole plan. Only the job runners are replaced: a run costs
minutes and needs a model.
Before this file nothing ran these handlers, so a wrong status code, a size limit that stopped
working or a missing wait would have shipped.
"""

from __future__ import annotations

import dataclasses
import http.client
import http.server
import json
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import activity_api
import generate_api as api
import pytest

GLB = b"glTF\x02\x00\x00\x00" + b"\x00" * 16       # never parsed here: the runners are stubs
PICTURE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16      # nor is this, unless a test asks for a cut-out
MIB = 1024 * 1024


def _soon(check, seconds=5.0):
    """Whether `check()` comes true within a few seconds. A runner is handed its job on a
    thread of its own, a moment after the answer is sent."""
    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """A real studio server on a free port. It writes into a temp folder and starts no job:
    `started` collects what each runner was handed instead."""
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(api, "REPO", tmp_path)
    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    # Every book the waiting rule reads is this test's own and empty: a job another test left
    # in the real ones, or a place one failed test never gave back, would make this one wait.
    monkeypatch.setattr(api, "_STARTING", {})
    monkeypatch.setattr(api, "SETUP_ACTIVE", None)
    monkeypatch.setattr(api, "download_active", lambda: None)
    monkeypatch.setattr(api, "JOBS", api.JobManager())
    monkeypatch.setattr(api, "FINISH_JOBS", api.FINISH_JOBS.__class__(output / "finish", output))
    monkeypatch.setattr(api, "PROPS_JOBS", api.PROPS_JOBS.__class__(output / "props"))
    monkeypatch.setattr(api, "RIG_JOBS", api.RIG_JOBS.__class__(output / "rig"))
    monkeypatch.setattr(api, "ANIMATE_JOBS", api.ANIMATE_JOBS.__class__(output / "animate"))
    monkeypatch.setattr(api.image_api, "MANAGER", api.image_api.ImageJobManager(output / "images"))
    monkeypatch.setattr(api.skins_api, "STORE", api.skins_api.SkinStore(output))
    monkeypatch.setattr(api.skins_api, "BUILDS", api.skins_api.SkinBuilds(api.skins_api.STORE))
    monkeypatch.setattr(api.addons_api, "ADDON_JOBS", api.addons_api.AddonJobManager())
    started = []
    for runner in ("_run_job", "run_finish_job", "run_props_job"):
        monkeypatch.setattr(api, runner, lambda job, runner=runner: started.append((runner, job)))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    # a short poll: shutdown waits one poll, and the default half second adds up over this file
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        yield types.SimpleNamespace(port=server.server_port, output=output, started=started)
    finally:
        server.shutdown()
        server.server_close()
    # Every request hands back the slot it claimed, whatever it answered: one that did not
    # would make every other kind of job wait until the studio is restarted. It is handed back
    # a moment after the answer is sent, on the request's own thread, so allow it that moment.
    assert _soon(lambda: api._STARTING == {}, seconds=2.0), f"a place was never given back: {api._STARTING}"


@pytest.fixture
def engine(monkeypatch):
    """TRELLIS.2 with its real settings check, counted as installed, taking a picture as it comes."""
    spec = dataclasses.replace(api.BACKENDS["trellis"], readiness=lambda: {"ready": True},
                               requires_alpha=False, gpu_memory_wanted=None)
    monkeypatch.setitem(api.BACKENDS, "trellis", spec)
    return spec


# Straight to the studio: a machine with http_proxy set would send these requests to the proxy.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _ask(studio, path, body, content_type):
    request = urllib.request.Request(
        f"http://127.0.0.1:{studio.port}{path}", data=body,
        headers={"Content-Type": content_type, "X-AssetFurnace-Request": "studio"})
    try:
        with _DIRECT.open(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _post(studio, path, fields=None, files=None):
    """POST a form the way the page's FormData does; returns (status, the JSON answer)."""
    return _ask(studio, path, *activity_api.multipart(fields or {}, files or {}))


def _send(studio, path, body):
    """POST JSON the way the page's fetch does; returns (status, the JSON answer)."""
    return _ask(studio, path, body if isinstance(body, bytes) else json.dumps(body).encode(), "application/json")


def _announce(studio, path, length):
    """Send only the headers of an upload that says it is `length` bytes long.

    A size the route refuses must be refused before the body is read, so none is sent: a
    route that started reading would wait for bytes that never come, and this would time out.
    """
    connection = http.client.HTTPConnection("127.0.0.1", studio.port, timeout=20)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "multipart/form-data; boundary=x")
        connection.putheader("Content-Length", str(length))
        connection.putheader("X-AssetFurnace-Request", "studio")
        connection.endheaders()
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _folders(studio):
    """What is in the output folder, by name: an upload that was refused must leave nothing."""
    return sorted(path.name for path in studio.output.iterdir())


def _a_picture_is_being_made(monkeypatch, tmp_path):
    pictures = api.image_api.ImageJobManager(tmp_path / "pictures")
    pictures.jobs["x"], pictures.active = types.SimpleNamespace(status="running"), "x"
    monkeypatch.setattr(api.image_api, "MANAGER", pictures)


# --- Generate: POST /api/generate -----------------------------------------------------------

def test_generate_saves_the_picture_and_starts_the_run_it_describes(studio, engine):
    status, answer = _post(studio, "/api/generate",
                           {"settings": json.dumps({"backend": "trellis", "seed": "7", "output_name": "Rune Knight"})},
                           {"image": ("Knight Art.PNG", PICTURE)})

    assert status == 202
    [(runner, job)] = studio.started
    assert runner == "_run_job"
    assert answer == {"job_id": job.id, "events_url": f"/api/generate/{job.id}/events",
                      "output_dir": str(Path("output") / "Rune-Knight"),      # backslashes on Windows
                      "warning": None}
    assert job.directory == studio.output / "Rune-Knight"
    assert job.output_path == job.directory / "Rune-Knight.glb"
    assert job.image_path == job.directory / "input.png" and job.image_path.read_bytes() == PICTURE
    assert job.backend_id == "trellis" and job.settings["seed"] == 7      # checked, not passed through
    assert _folders(studio) == ["Rune-Knight"]                            # the upload's own folder is gone


def test_generate_needs_a_picture(studio, engine):
    assert _post(studio, "/api/generate", {"settings": "{}"}) == (
        400, {"error": "multipart field 'image' is required"})
    assert studio.started == [] and _folders(studio) == []


@pytest.mark.parametrize("length", [0, 50 * MIB + 1])
def test_generate_refuses_a_missing_or_oversized_upload_before_reading_it(studio, engine, length):
    assert _announce(studio, "/api/generate", length) == (
        400, {"error": "image upload is missing or larger than 50 MiB"})


@pytest.mark.parametrize("settings, status, error", [
    ("{not json", 422, "invalid settings JSON"),
    ("[1, 2]", 422, "settings must be a JSON object"),
    ('{"backend": "no-such-engine"}', 422, "unknown backend 'no-such-engine'"),
    ('{"resolution": "9999"}', 422, "resolution must be one of 512, 1024, or 1536"),
    ('{"output_dir": "../elsewhere"}', 422, "output_dir must be inside"),
])
def test_generate_refuses_settings_it_cannot_use_and_keeps_nothing(studio, engine, settings, status, error):
    got, answer = _post(studio, "/api/generate", {"settings": settings}, {"image": ("knight.png", PICTURE)})

    assert got == status and error in answer["error"]
    assert studio.started == [] and _folders(studio) == []


def test_generate_refuses_a_file_that_is_not_a_picture(studio, engine):
    assert _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.gif", PICTURE)}) == (
        422, {"error": "unsupported image type; use PNG, JPG, WebP, or BMP"})
    assert studio.started == [] and _folders(studio) == []


def test_generate_says_when_the_engine_is_not_installed(studio, engine, monkeypatch):
    monkeypatch.setitem(api.BACKENDS, "trellis", dataclasses.replace(engine, readiness=lambda: {"ready": False}))

    status, answer = _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)})

    assert status == 503 and answer == {"error": f"{engine.label} is not installed/ready"}
    assert studio.started == [] and _folders(studio) == []


def test_generate_waits_for_another_big_job(studio, engine, monkeypatch, tmp_path):
    _a_picture_is_being_made(monkeypatch, tmp_path)

    assert _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)}) == (
        409, {"error": "a picture is being made; wait for it to finish"})
    assert studio.started == [] and _folders(studio) == []


def test_generate_refuses_a_second_generation_and_keeps_only_the_first(studio, engine):
    first, _ = _post(studio, "/api/generate", {"settings": json.dumps({"output_name": "one"})},
                     {"image": ("knight.png", PICTURE)})
    second, answer = _post(studio, "/api/generate", {"settings": json.dumps({"output_name": "two"})},
                           {"image": ("knight.png", PICTURE)})

    assert (first, second) == (202, 409) and answer == {"error": "a generation is already running"}
    assert len(studio.started) == 1 and _folders(studio) == ["one"]


def test_generate_reports_a_bug_in_the_settings_check_in_words(studio, engine, monkeypatch, capsys):
    # A KeyError on a setting name reached the browser as just 'max_num_view', with nothing in
    # the server's terminal: the run looked as if it had simply stopped.
    def broken(_raw):
        raise KeyError("max_num_view")
    monkeypatch.setitem(api.BACKENDS, "trellis", dataclasses.replace(engine, validate_settings=broken))

    status, answer = _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)})

    assert status == 500
    assert answer["error"].startswith(f"{engine.label} could not read the settings for this run (KeyError: 'max_num_view')")
    assert "not in your picture" in answer["error"]
    assert "[generate] trellis validate_settings raised KeyError: 'max_num_view'" in capsys.readouterr().out
    assert studio.started == [] and _folders(studio) == []


def test_generate_cuts_out_a_picture_the_engine_cannot_take_as_it_is(studio, engine, monkeypatch):
    import io

    from PIL import Image

    picture = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(picture, "PNG")       # no transparency: it needs the cut
    monkeypatch.setitem(api.BACKENDS, "trellis", dataclasses.replace(engine, requires_alpha=True))

    def cut(path):       # the remover is a model; what the route does with its answer is ours
        cut_path = path.with_name("input__matted.png")
        cut_path.write_bytes(b"cut out")
        return cut_path, "birefnet-general-lite"
    monkeypatch.setattr(api, "matte_upload", cut)

    status, _ = _post(studio, "/api/generate", {"settings": json.dumps({"output_name": "knight"})},
                      {"image": ("knight.png", picture.getvalue())})

    assert status == 202
    [(_, job)] = studio.started
    assert job.settings["matted_with"] == "birefnet-general-lite"
    assert job.image_path == job.directory / "input__matted.png" and job.image_path.read_bytes() == b"cut out"
    assert (job.directory / "input.png").read_bytes() == picture.getvalue()      # the original is kept


def test_generate_says_when_the_cut_out_fails_and_keeps_nothing(studio, engine, monkeypatch):
    import io

    from PIL import Image

    picture = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(picture, "PNG")
    monkeypatch.setitem(api.BACKENDS, "trellis", dataclasses.replace(engine, requires_alpha=True))

    def fail(_path):
        raise RuntimeError("the remover is not installed")
    monkeypatch.setattr(api, "matte_upload", fail)

    status, answer = _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", picture.getvalue())})

    assert status == 422
    assert answer == {"error": "This picture needs its background removed and the remover failed "
                               "(the remover is not installed). Upload a PNG with a transparent background."}
    assert studio.started == [] and _folders(studio) == []


# --- Finish: POST /api/finish ---------------------------------------------------------------

@pytest.fixture
def blender(monkeypatch):
    monkeypatch.setattr(api, "find_blender", lambda: "/Applications/Blender.app/Contents/MacOS/Blender")


def test_finish_saves_the_model_and_its_picture_and_starts_the_run(studio, blender):
    status, answer = _post(studio, "/api/finish", {"settings": json.dumps({"faces": 12000, "no_such_setting": 1})},
                           {"asset": ("Rune Knight.GLB", GLB), "image": ("knight.png", PICTURE)})

    assert status == 202
    [(runner, job)] = studio.started
    assert runner == "run_finish_job"
    assert answer == {"job_id": job.id, "settings": job.settings,
                      "events_url": f"/api/finish/{job.id}/events", "status_url": f"/api/finish/{job.id}/status"}
    assert job.asset_path.read_bytes() == GLB and job.image_path.read_bytes() == PICTURE
    assert job.settings["faces"] == 12000 and "no_such_setting" not in job.settings
    assert job.directory.parent == studio.output / "finish" and job.directory.name.startswith("Rune-Knight__finish__")


def test_finish_without_a_picture_cleans_the_model_up_and_paints_nothing(studio, blender):
    status, answer = _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)})

    assert status == 202
    assert answer["settings"]["skip_photo"] is True and answer["settings"]["skip_paint"] is True
    [(_, job)] = studio.started
    assert not job.image_path.exists()


@pytest.mark.parametrize("files, error", [
    ({"asset": ("knight.obj", GLB)}, "multipart field 'asset' must be a .glb file"),
    ({"image": ("knight.png", PICTURE)}, "multipart field 'asset' must be a .glb file"),
    ({"asset": ("knight.glb", GLB), "image": ("knight.gif", PICTURE)},
     "multipart field 'image' must be the source art (PNG/JPG/WebP)"),
])
def test_finish_refuses_the_wrong_kind_of_file(studio, blender, files, error):
    assert _post(studio, "/api/finish", {"settings": "{}"}, files) == (422, {"error": error})
    assert studio.started == [] and _folders(studio) == []


@pytest.mark.parametrize("settings, error", [
    ("{not json", "invalid settings JSON"),
    ('"faces"', "settings must be a JSON object"),
    ('{"faces": "many"}', "faces must be a number, got 'many'"),
])
def test_finish_refuses_settings_it_cannot_use_and_keeps_nothing(studio, blender, settings, error):
    status, answer = _post(studio, "/api/finish", {"settings": settings}, {"asset": ("knight.glb", GLB)})

    assert status == 422 and error in answer["error"]
    assert studio.started == [] and not any((studio.output / "finish").glob("*"))


@pytest.mark.parametrize("length", [0, 768 * MIB + 1])
def test_finish_refuses_a_missing_or_oversized_upload_before_reading_it(studio, blender, length):
    assert _announce(studio, "/api/finish", length) == (
        400, {"error": "finish bundle is missing or larger than 768 MiB"})


def test_finish_says_how_to_get_blender_when_it_is_missing(studio, monkeypatch):
    # Checked here rather than a minute into the run, where it surfaced as a bare exit code.
    monkeypatch.setattr(api, "find_blender", lambda: None)

    status, answer = _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)})

    assert status == 409 and answer == {"error": api.blender_missing_help()}
    assert "Blender" in answer["error"] and studio.started == []


def test_finish_waits_for_a_running_generation(studio, blender, engine):
    _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)})

    assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == (
        409, {"error": "a generation is running; wait for it to finish"})
    assert [runner for runner, _ in studio.started] == ["_run_job"]


def _an_add_on_tool_is_running(monkeypatch, _tmp_path):
    tools = api.addons_api.AddonJobManager()
    job = api.addons_api.AddonJob("d" * 32, "vehicle-kit", "build", {}, "run")
    job.status = "running"
    tools.jobs[job.id], tools.active = job, job.id
    monkeypatch.setattr(api.addons_api, "ADDON_JOBS", tools)


@pytest.mark.parametrize("something_else_runs, words", [
    (_a_picture_is_being_made, "a picture is being made"),
    (_an_add_on_tool_is_running, "an add-on is running"),
])
def test_finish_waits_for_every_other_big_job_not_only_the_ones_it_listed(
        studio, blender, monkeypatch, tmp_path, something_else_runs, words):
    # Finish kept its own list of what to wait for (a generation, setup, a prop bake, a rig run),
    # written before pictures, rebinds, skins and add-on tools joined the shared rule. It started
    # beside any of those; resuming a Finish run already asked the shared rule.
    something_else_runs(monkeypatch, tmp_path)

    assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == (
        409, {"error": f"{words}; wait for it to finish"})
    assert studio.started == [] and not any((studio.output / "finish").glob("*"))


def test_finish_refuses_a_second_finish(studio, blender):
    first, _ = _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)})

    assert first == 202
    assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == (
        409, {"error": "a finishing job is already running"})
    assert len(studio.started) == 1


# --- Prop sheets: POST /api/props -----------------------------------------------------------

def test_props_saves_an_uploaded_sheet_and_starts_the_run(studio):
    status, answer = _post(studio, "/api/props", {"settings": json.dumps({"compress": False, "no_such_setting": 1})},
                           {"asset": ("Tavern Sheet.GLB", GLB)})

    assert status == 202
    [(runner, job)] = studio.started
    assert runner == "run_props_job"
    assert answer == {"job_id": job.id, "directory": job.directory.name, "settings": job.settings,
                      "events_url": f"/api/props/{job.id}/events", "status_url": f"/api/props/{job.id}/status"}
    assert job.source_glb.read_bytes() == GLB
    assert job.settings["compress"] is False and "no_such_setting" not in job.settings
    assert job.directory.parent == studio.output / "props" and job.directory.name.startswith("Tavern-Sheet__props__")


def test_props_takes_a_model_the_studio_made_by_the_name_it_was_listed_under(studio):
    made = studio.output / "tavern" / "tavern.glb"
    made.parent.mkdir()
    made.write_bytes(b"glTF made here")

    status, _ = _post(studio, "/api/props", {"generated": "tavern/tavern.glb", "settings": "{}"})

    assert status == 202
    [(_, job)] = studio.started
    assert job.source_glb.read_bytes() == b"glTF made here"


@pytest.mark.parametrize("generated", ["../secret.glb", "tavern/other.glb", "/etc/hosts"])
def test_props_takes_no_other_file_from_disk(studio, tmp_path, generated):
    (tmp_path / "secret.glb").write_bytes(b"not for the studio")
    made = studio.output / "tavern"
    made.mkdir()
    (made / "tavern.glb").write_bytes(GLB)
    (made / "other.glb").write_bytes(GLB)        # beside a made model, but not one itself

    assert _post(studio, "/api/props", {"generated": generated, "settings": "{}"}) == (
        422, {"error": f"not a generated model: {generated!r}"})
    assert studio.started == []


def test_props_needs_a_sheet(studio):
    assert _post(studio, "/api/props", {"settings": "{}"}, {"asset": ("sheet.obj", GLB)}) == (
        422, {"error": "send a .glb as 'asset', or a generated model as 'generated'"})
    assert studio.started == []


@pytest.mark.parametrize("settings, error", [
    ("{not json", "invalid settings JSON"),
    ("7", "settings must be a JSON object"),
    ('{"atlas": 999}', "atlas must be one of"),
])
def test_props_refuses_settings_it_cannot_use_and_keeps_nothing(studio, settings, error):
    status, answer = _post(studio, "/api/props", {"settings": settings}, {"asset": ("sheet.glb", GLB)})

    assert status == 422 and error in answer["error"]
    assert studio.started == [] and not any((studio.output / "props").glob("*"))


@pytest.mark.parametrize("length", [0, 768 * MIB + 1])
def test_props_refuses_a_missing_or_oversized_upload_before_reading_it(studio, length):
    assert _announce(studio, "/api/props", length) == (
        400, {"error": "prop sheet is missing or larger than 768 MiB"})


def test_props_waits_for_another_big_job(studio, monkeypatch, tmp_path):
    _a_picture_is_being_made(monkeypatch, tmp_path)

    assert _post(studio, "/api/props", {"settings": "{}"}, {"asset": ("sheet.glb", GLB)}) == (
        409, {"error": "a picture is being made; wait for it to finish"})
    assert studio.started == [] and _folders(studio) == []


# --- Setup starts: POST /api/setup/run, /api/blender/install, /api/gltfpack/install ---------

SETUP_ROUTES = ["/api/setup/run", "/api/blender/install", "/api/gltfpack/install"]


@pytest.fixture
def installers(monkeypatch):
    """Every setup may start as far as the machine goes, and none really runs: the list
    collects the id of each run that was started."""
    started = []

    def start(run_id, command=None):
        started.append(run_id)
        return api.SetupRun(run_id)

    monkeypatch.setattr(api, "_start_setup_run", start)
    monkeypatch.setattr(api, "SETUP_RUNS", {})
    monkeypatch.setattr(api, "setup_available", lambda: (True, None))
    monkeypatch.setattr(api, "clean_port_build_present", lambda: False)
    monkeypatch.setattr(api, "finish_capabilities", lambda: {"blender_installable": True})
    monkeypatch.setattr(api, "props_tools_payload", lambda: {"gltfpack_installable": True})
    monkeypatch.setattr(api, "download_active", lambda: None)
    return started


@pytest.mark.parametrize("route", SETUP_ROUTES)
def test_a_setup_starts_when_nothing_else_is_running(studio, installers, route):
    status, answer = _post(studio, route)

    assert status == 202
    [run_id] = installers
    assert answer == {"setup_run_id": run_id, "events_url": f"/api/setup/run/{run_id}/events"}
    assert api.SETUP_ACTIVE == run_id       # what makes every other big job wait for it


@pytest.mark.parametrize("route", SETUP_ROUTES)
@pytest.mark.parametrize("something_else_runs, words", [
    (_a_picture_is_being_made, "a picture is being made"),
    (_an_add_on_tool_is_running, "an add-on is running"),
])
def test_a_setup_waits_for_every_other_big_job_not_only_a_generation(
        studio, installers, monkeypatch, tmp_path, route, something_else_runs, words):
    # Each setup start asked only whether a generation was running. An install or a build
    # started beside a picture, Finish, a prop bake, a rig run, a skin or an add-on tool,
    # though all of those wait for a setup.
    something_else_runs(monkeypatch, tmp_path)

    assert _post(studio, route) == (409, {"error": f"{words}; wait for it to finish"})
    assert installers == [] and api.SETUP_ACTIVE is None


@pytest.mark.parametrize("route", SETUP_ROUTES)
def test_a_setup_waits_for_a_running_generation(studio, installers, engine, route):
    _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)})

    assert _post(studio, route) == (409, {"error": "a generation is running; wait for it to finish"})
    assert installers == [] and api.SETUP_ACTIVE is None


@pytest.mark.parametrize("route, words", [
    ("/api/setup/run", "setup is already running"),
    ("/api/blender/install", "another setup is running; wait for it to finish"),
    ("/api/gltfpack/install", "another setup is running; wait for it to finish"),
])
def test_a_second_setup_is_refused(studio, installers, route, words):
    first, _ = _post(studio, route)

    assert first == 202
    assert _post(studio, route) == (409, {"error": words})
    assert len(installers) == 1


# --- Engines from Setup & Status: POST /api/setup/<engine>/download|rebuild|remove|cancel ----

ENGINE_ACTIONS = ["download", "rebuild", "remove"]


@pytest.fixture
def engines(monkeypatch):
    """Setup & Status's actions, recorded instead of run: nothing is downloaded, built or
    deleted here. The list collects (action, engine) for each one the route let through."""
    asked = []
    monkeypatch.setattr(api, "start_download", lambda backend_id, rebuild=False: asked.append(
        ("rebuild" if rebuild else "download", backend_id)))
    monkeypatch.setattr(api, "remove_weights", lambda backend_id: asked.append(("remove", backend_id)) or {"freed": 0})
    monkeypatch.setattr(api, "cancel_download", lambda backend_id: asked.append(("cancel", backend_id)))
    monkeypatch.setattr(api, "download_active", lambda: None)
    return asked


@pytest.mark.parametrize("action", ENGINE_ACTIONS)
def test_an_engine_is_set_up_rebuilt_or_removed_when_nothing_else_is_running(studio, engines, action):
    status, _ = _post(studio, f"/api/setup/pixal3d/{action}")

    assert status == (200 if action == "remove" else 202)
    assert engines == [(action, "pixal3d")]


@pytest.mark.parametrize("action", ENGINE_ACTIONS)
@pytest.mark.parametrize("something_else_runs, words", [
    (_a_picture_is_being_made, "a picture is being made"),
    (_an_add_on_tool_is_running, "an add-on is running"),
])
def test_an_engine_is_not_set_up_rebuilt_or_removed_while_something_is_being_made(
        studio, engines, monkeypatch, tmp_path, action, something_else_runs, words):
    # Only a second download was refused. An engine could be rebuilt, or its weights deleted,
    # under the job that was using it.
    something_else_runs(monkeypatch, tmp_path)

    assert _post(studio, f"/api/setup/pixal3d/{action}") == (409, {"error": f"{words}; wait for it to finish"})
    assert engines == []


@pytest.mark.parametrize("action", ENGINE_ACTIONS)
def test_an_engine_is_not_set_up_rebuilt_or_removed_during_a_generation(studio, engines, engine, action):
    _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)})

    assert _post(studio, f"/api/setup/pixal3d/{action}") == (
        409, {"error": "a generation is running; wait for it to finish"})
    assert engines == []


def test_nothing_else_starts_while_an_engine_is_being_set_up(studio, engines, engine, blender, monkeypatch):
    # A download or a build was not a setup as far as the shared rule knew, so a generation
    # could start on an engine that was being rebuilt under it.
    monkeypatch.setattr(api, "download_active", lambda: object())
    wait = (409, {"error": "setup is running; wait for it to finish"})

    assert _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)}) == wait
    assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == wait
    assert _post(studio, "/api/props", {"settings": "{}"}, {"asset": ("sheet.glb", GLB)}) == wait
    assert studio.started == []


@pytest.mark.parametrize("action", ENGINE_ACTIONS)
def test_an_engine_is_not_set_up_rebuilt_or_removed_during_another_setup(studio, engines, monkeypatch, action):
    monkeypatch.setattr(api, "SETUP_ACTIVE", "a" * 32)       # Blender is being installed, say

    assert _post(studio, f"/api/setup/pixal3d/{action}") == (
        409, {"error": "another setup is running; wait for it to finish"})
    assert engines == []


def _being_set_up(engine_id):
    """What the download list answers while that engine's weights are coming down."""
    return lambda: types.SimpleNamespace(backend=types.SimpleNamespace(id=engine_id))


def test_an_engine_is_not_removed_while_another_engine_is_being_set_up(studio, engines, monkeypatch):
    # Engines share some weights (the two Hunyuan routes' paint model): removing one could delete
    # what the other's download is writing. Remove only refused a download of the same engine.
    monkeypatch.setattr(api, "download_active", _being_set_up("hunyuan_xiong"))

    assert _post(studio, "/api/setup/pixal3d/remove") == (
        409, {"error": "another engine is being set up; wait for it to finish"})
    assert engines == []


def test_removing_the_engine_that_is_itself_downloading_still_says_to_cancel_it_first(studio, engines, monkeypatch):
    def downloading(_backend_id):       # what the real remover answers for its own engine
        raise RuntimeError("that backend is downloading right now; cancel it first")
    monkeypatch.setattr(api, "download_active", _being_set_up("pixal3d"))
    monkeypatch.setattr(api, "remove_weights", downloading)

    assert _post(studio, "/api/setup/pixal3d/remove") == (
        409, {"error": "that backend is downloading right now; cancel it first"})


def test_the_mac_setup_waits_for_an_engine_being_set_up(studio, installers, monkeypatch):
    monkeypatch.setattr(api, "download_active", lambda: object())

    assert _post(studio, "/api/setup/run") == (409, {"error": "setup is already running"})
    assert installers == []


def test_a_download_can_be_cancelled_whatever_else_is_running(studio, engines, monkeypatch, tmp_path):
    _a_picture_is_being_made(monkeypatch, tmp_path)

    assert _post(studio, "/api/setup/pixal3d/cancel") == (202, {"backend": "pixal3d", "status": "cancelling"})
    assert engines == [("cancel", "pixal3d")]


# --- A start holds the slot from its waiting check until its job is on the books ------------

def _plain_picture() -> bytes:
    """A real picture with no transparency: the kind an engine needs cut out first."""
    import io

    from PIL import Image

    picture = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(picture, "PNG")
    return picture.getvalue()


def _hold(monkeypatch, name, answer=None, on=api):
    """Make `on.<name>` stop where it is called until `let_go` is set, then answer (as it
    did before, unless `answer` says otherwise).
    Returns (reached, let_go): `reached` is set once a request is waiting inside it."""
    reached, let_go = threading.Event(), threading.Event()
    answer = answer or getattr(on, name)

    def held(*args, **more):
        reached.set()
        assert let_go.wait(20), "the test never let this start go on"
        return answer(*args, **more)

    monkeypatch.setattr(on, name, held)
    return reached, let_go


def _in_the_background(call):
    answers = []
    thread = threading.Thread(target=lambda: answers.append(call()), daemon=True)
    thread.start()
    return thread, answers


def _answered(thread, answers):
    """The answer of a start made `_in_the_background`, once the test has let it go on."""
    thread.join(20)
    assert not thread.is_alive(), "the start that was held never answered"
    return answers[0]


def test_a_generation_that_is_still_starting_already_makes_finish_wait(studio, engine, blender, monkeypatch):
    # The waiting check and the job's entry on the books were two steps, seconds apart while
    # the picture is cut out. Finish asked in between, saw nothing running, and started too.
    monkeypatch.setitem(api.BACKENDS, "trellis", dataclasses.replace(engine, requires_alpha=True))
    reached, let_go = _hold(monkeypatch, "matte_upload", lambda path: (path, "birefnet-general-lite"))
    generating, answers = _in_the_background(lambda: _post(
        studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", _plain_picture())}))
    assert reached.wait(20)
    try:
        assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == (
            409, {"error": "a generation is running; wait for it to finish"})
    finally:
        let_go.set()

    assert _answered(generating, answers)[0] == 202
    assert [runner for runner, _ in studio.started] == ["_run_job"]


def test_a_finish_that_is_still_starting_already_makes_a_generation_wait(studio, engine, monkeypatch):
    reached, let_go = _hold(monkeypatch, "find_blender", lambda: "/usr/bin/blender")
    finishing, answers = _in_the_background(lambda: _post(
        studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}))
    assert reached.wait(20)
    try:
        assert _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)}) == (
            409, {"error": "a model is being finished; wait for it to finish"})
    finally:
        let_go.set()

    assert _answered(finishing, answers)[0] == 202
    assert [runner for runner, _ in studio.started] == ["run_finish_job"]


def test_a_start_that_is_refused_gives_the_slot_back(studio, engine, blender):
    refused, _ = _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.gif", PICTURE)})

    assert refused == 422       # after its waiting check: it had the slot
    assert _finish_starts(studio)


def test_a_slot_stays_held_until_every_start_of_that_kind_has_let_go(studio):
    # Two generations can ask together; their own manager settles which one runs. The one it
    # refuses lets go first, and the other is still on its way to the books.
    assert api.claim_start("generate") is None and api.claim_start("generate") is None
    try:
        api.release_start("generate")
        assert api.busy_reason("finish") == "a generation is running; wait for it to finish"
    finally:
        api.release_start("generate")
    assert api.busy_reason("finish") is None


# --- Pictures, skins, rig rebinds and animations --------------------------------------------
#
# The other four kinds of big job. What would draw, repaint, rebind or rig is replaced, and
# `studio.started` collects what each was handed, as it does for the three kinds above.

BLEND = b"BLENDER-v405" + b"\x00" * 16             # never opened here: the rebind runner is a stub
KNIGHT = "images/knight/knight.png"                # a cut-out picture with no back yet
PALADIN = "a-paladin__20261019-133432"             # a picture run,
MODEL = f"{PALADIN}__pixal3d__20261019-145"        # the Pixal3D model made from it,
FINISHED = f"{MODEL}__finish__20261019-145515"     # and that model finished: what a skin starts from
RIGGED = "animate/knight__rig__20261005-120000/knight_rigged.glb"
NO_ENGINE = "no picture engine 'dall-e'; choose "      # ...and then the engines there are


def _picture(mode="RGB", fill="white", middle=None) -> bytes:
    """A real 64-pixel picture; `middle` paints a square in its centre."""
    import io

    from PIL import Image, ImageDraw

    picture = Image.new(mode, (64, 64), fill)
    if middle is not None:
        ImageDraw.Draw(picture).rectangle((16, 16, 47, 47), fill=middle)
    out = io.BytesIO()
    picture.save(out, "PNG")
    return out.getvalue()


def _finish_starts(studio):
    """Whether another kind of big job is let through: nothing else holds a place. A place is
    handed back a moment after its answer is sent, so it is given that moment first."""
    _soon(lambda: not api._STARTING)
    return _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)})[0] == 202


@pytest.fixture
def workshop(studio, monkeypatch, tmp_path):
    """Pictures, skins, rebinds and the auto-rig, each counted as installed and running nothing.

    On disk: a cut-out knight with no back yet; a paladin with a front, a back, a Pixal3D
    model and its finished version (what a skin starts from), with one skin still at its
    pictures and one with both approved; and a rigged knight.
    """
    out, started = studio.output, studio.started

    def keep(relative, data):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return relative

    keep(KNIGHT, _picture("RGBA", (0, 0, 0, 0), (200, 40, 40, 255)))
    front = keep(f"images/{PALADIN}/{PALADIN}.png", _picture("RGBA", (0, 0, 0, 0), (40, 40, 200, 255)))
    back = keep(f"images/{PALADIN}/{PALADIN}__back.png", _picture("RGB", "grey", "navy"))
    keep(f"{MODEL}/{MODEL}.glb", GLB)
    keep(f"{MODEL}/{MODEL}.shape.bin", b"SHAPE")
    keep(f"{MODEL}/{MODEL}.json", {"backend": "pixal3d",
                                    "parameters": {"seed": 15629060, "steps": 8, "views": "front + back"}})
    finished = keep(f"finish/{FINISHED}/paladin_20k.glb", GLB + b"finished")
    keep(f"finish/{FINISHED}/input/source.glb", GLB)
    keep(f"finish/{FINISHED}/input/settings.json",
         {"faces": 20000, "texture_size": 2048, "skip_paint": True, "skip_photo": False})
    keep(RIGGED, GLB + b"rigged")

    # the picture engine: never this machine's own build or its downloaded files
    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"")
    pictures = api.image_api.ImageJobManager(out / "images")
    monkeypatch.setattr(api.image_api, "MANAGER", pictures)
    monkeypatch.setattr(api.image_api, "BINARY", binary)
    monkeypatch.setattr(api.image_api, "resolve_weights", lambda engine=None: {})
    monkeypatch.setattr(api.image_api, "resolve_vision", lambda engine=None: binary)
    monkeypatch.setattr(api.image_api, "run_job", lambda job, manager: started.append(("picture", job)))

    def no_cut_out(_picture):       # the remover is a model: a front here is already cut out
        raise AssertionError("a route test must never load the background remover")
    monkeypatch.setattr(api.image_api.matte, "cut_out", no_cut_out)

    rigs = api.RIG_JOBS.__class__(out / "rig")
    monkeypatch.setattr(api, "RIG_JOBS", rigs)
    monkeypatch.setattr(api, "run_rig_job", lambda job: started.append(("rebind", job)))

    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "walk.npz").write_bytes(b"x")
    (motions / "presets.json").write_text(json.dumps({"presets": [{"id": "walk", "label": "Walk", "seconds": 4}]}))
    animations = api.ANIMATE_JOBS.__class__(out / "animate", motions)
    monkeypatch.setattr(api, "ANIMATE_JOBS", animations)
    monkeypatch.setattr(api, "run_animate_job", lambda job: started.append(("animate", job)))
    monkeypatch.setattr(api, "autorig_installed", lambda: True)

    skins = api.skins_api.SkinStore(out)
    builds = api.skins_api.SkinBuilds(skins)
    monkeypatch.setattr(builds, "run", lambda job, skin, runner, box: started.append(("skin", job)))
    monkeypatch.setattr(api.skins_api, "STORE", skins)
    monkeypatch.setattr(api.skins_api, "BUILDS", builds)
    paladin = next(asset for asset in api.list_assets(out) if asset.get("finished"))
    draft = skins.create(paladin, "red boots")
    approved = skins.create(paladin, "gold skirt")
    for side in ("front", "back"):
        _, drawn, _, _ = skins.picture_try(approved, side)
        drawn.write_bytes(PICTURE)
        skins.approve(approved, side)

    return types.SimpleNamespace(
        pictures=pictures, rigs=rigs, animations=animations, skins=skins, builds=builds,
        paladin=paladin, front=front, back=back, finished=finished, draft=draft, approved=approved)


def _on_the_books(shop):
    return [*shop.pictures.jobs.values(), *shop.builds.jobs.values(),
            *shop.rigs.jobs.values(), *shop.animations.jobs.values()]


def _stroke(painted=True) -> bytes:
    """What the brush sends: black, white where the person painted."""
    return _picture("L", "black", "white" if painted else None)


def _rig_sidecar(asset: bytes, scene: bytes) -> bytes:
    """The smallest rig record a rebind takes: one joint, moved, bound to one Blender bone."""
    import hashlib

    return json.dumps({
        "schemaVersion": 1, "rigProfile": "test",
        "assetFingerprint": "sha256:" + hashlib.sha256(asset).hexdigest(),
        "coordinateSpace": "armature-local", "mirror": {"axis": "X", "origin": 0},
        "joints": {"joint": {"label": "Joint", "position": [0, 0, 0], "sourceBone": "DEF-joint",
                             "parent": None, "mirrorOf": None}},
        "corrections": {"joint": {"sourcePosition": [0, 0, 0], "targetPosition": [0, 1, 0],
                                  "delta": [0, 1, 0], "mirrored": False}},
        "binding": {"adapter": "test", "sceneFingerprint": "sha256:" + hashlib.sha256(scene).hexdigest(),
                    "metarigObjectId": "object",
                    "joints": {"joint": {"targets": [{"boneId": "bone", "boneName": "bone", "endpoint": "head"}]}}},
    }).encode()


def _rebind_bundle(**changes):
    return {"asset": ("knight.glb", GLB), "scene": ("knight.blend", BLEND),
            "sidecar": ("knight.rig.json", _rig_sidecar(GLB, BLEND)), **changes}


# Every start of these four kinds, asked for in a way that is let through.

def _draw(studio, shop):
    return _send(studio, "/api/image", {"prompt": "a rune knight", "engine": "qwen"})


def _draw_a_back(studio, shop):
    return _send(studio, "/api/image/back", {"picture": KNIGHT, "engine": "flux"})


def _brush(studio, shop, **changes):
    fields = {"picture": shop.back, "instruction": "a red cape", "engine": "flux", **changes}
    return _post(studio, "/api/image/brush", fields, {"mask": ("mask.png", _stroke())})


def _new_skin(studio, shop):
    return _send(studio, "/api/skins", {"asset": shop.paladin["id"], "instruction": "make the skirt gold"})


def _try_a_skin_again(studio, shop):
    return _send(studio, f"/api/skins/{shop.draft.id}/try", {"side": "front", "instruction": "matte red boots"})


def _build_a_skin(studio, shop):
    return _send(studio, f"/api/skins/{shop.approved.id}/build", {})


def _rebind(studio, shop):
    return _post(studio, "/api/rig/rebind", files=_rebind_bundle())


def _rig(studio, shop):
    return _post(studio, "/api/animate/rig", files={"asset": ("knight.glb", GLB)})


def _play(studio, shop):
    return _send(studio, "/api/animate/play", {"model": RIGGED, "preset": "walk"})


# name: (how it is asked for, a step between its waiting check and its job's entry on the
# books, what it makes the other kinds answer)
STARTS = {
    "a picture": (_draw, lambda shop: (api.image_api, "resolve_weights"), "a picture is being made"),
    "a back": (_draw_a_back, lambda shop: (api.image_api, "check_ready"), "a picture is being made"),
    "a brush": (_brush, lambda shop: (api.image_api, "check_ready"), "a picture is being made"),
    "a new skin": (_new_skin, lambda shop: (api.image_api, "resolve_weights"), "a picture is being made"),
    "a skin try": (_try_a_skin_again, lambda shop: (api.image_api, "resolve_weights"), "a picture is being made"),
    "a skin build": (_build_a_skin, lambda shop: (shop.builds, "start"), "a skin is being made"),
    "a rebind": (_rebind, lambda shop: (shop.rigs, "create"), "a rig rebind is running"),
    "an auto-rig": (_rig, lambda shop: (api, "autorig_installed"), "an auto-rig or animation is running"),
    "a move": (_play, lambda shop: (api, "autorig_installed"), "an auto-rig or animation is running"),
}
each_start = pytest.mark.parametrize("ask, held_at, words", STARTS.values(), ids=STARTS)


@each_start
def test_each_of_these_starts_is_let_through_when_nothing_else_runs(studio, workshop, ask, held_at, words):
    status, _ = ask(studio, workshop)

    assert status == 202
    [job] = _on_the_books(workshop)
    assert _soon(lambda: [handed for _, handed in studio.started] == [job])
    assert _soon(lambda: not api._STARTING)     # the place is back: only the job on the books says so now
    assert api.busy_reason("finish") == f"{words}; wait for it to finish"


@each_start
def test_each_of_these_starts_waits_for_another_big_job(studio, workshop, monkeypatch, tmp_path, ask, held_at, words):
    _an_add_on_tool_is_running(monkeypatch, tmp_path)

    status, answer = ask(studio, workshop)

    assert (status, answer["error"]) == (409, "an add-on is running; wait for it to finish")
    assert _on_the_books(workshop) == [] and studio.started == []


@each_start
def test_each_of_these_starts_holds_its_place_before_its_job_is_on_the_books(
        studio, workshop, monkeypatch, ask, held_at, words):
    # The same gap generate and Finish had: told it may go, a start loads or checks something
    # before its manager knows of it, and another kind asking in between saw nothing running.
    on, step = held_at(workshop)
    reached, let_go = _hold(monkeypatch, step, on=on)
    starting, answers = _in_the_background(lambda: ask(studio, workshop))
    assert reached.wait(20)
    try:
        assert _on_the_books(workshop) == []      # so only the place it holds can say so
        assert _post(studio, "/api/finish", {"settings": "{}"}, {"asset": ("knight.glb", GLB)}) == (
            409, {"error": f"{words}; wait for it to finish"})
    finally:
        let_go.set()

    assert _answered(starting, answers)[0] == 202 and len(_on_the_books(workshop)) == 1


# --- Pictures: POST /api/image --------------------------------------------------------------

def test_a_picture_starts_with_its_prompt_trimmed_and_its_settings_made_safe(studio, workshop):
    status, answer = _send(studio, "/api/image", {
        "prompt": "  a rune knight  ", "engine": "qwen", "settings": {"width": 700, "seed": "7", "steps": 9999}})

    assert status == 202
    [job] = workshop.pictures.jobs.values()
    assert answer == {"job_id": job.id, "settings": job.settings,
                      "events_url": f"/api/image/{job.id}/events", "status_url": f"/api/image/{job.id}/status"}
    assert (job.prompt, job.engine.id) == ("a rune knight", "qwen")
    assert (job.settings["width"], job.settings["seed"], job.settings["steps"]) == (704, 7, 50)
    assert job.directory.parent == studio.output / "images"
    assert _soon(lambda: studio.started == [("picture", job)])


def test_a_picture_that_is_the_end_product_gets_its_engines_extra_steps(studio, workshop):
    flux = api.image_api.FLUX
    assert flux.picture_only_steps != flux.defaults["steps"]

    _send(studio, "/api/image", {"prompt": "a rune knight", "engine": "flux", "picture_only": True})

    [job] = workshop.pictures.jobs.values()
    assert job.settings["steps"] == flux.picture_only_steps


@pytest.mark.parametrize("request_body, error", [
    ({"prompt": "   "}, "a prompt is required"),
    ({"prompt": "x" * 2001}, "prompt is longer than 2000 characters"),
    ({"prompt": "a rune knight", "engine": "dall-e"}, NO_ENGINE),
])
def test_a_picture_refuses_a_request_it_cannot_draw(studio, workshop, request_body, error):
    status, answer = _send(studio, "/api/image", request_body)

    assert status == 422 and answer["error"].startswith(error)
    assert workshop.pictures.jobs == {}


@pytest.mark.parametrize("length", [0, 64 * 1024 + 1])
def test_a_picture_refuses_a_missing_or_oversized_request_before_reading_it(studio, workshop, length):
    assert _announce(studio, "/api/image", length) == (400, {"error": "expected a small JSON body"})


def test_a_picture_refuses_a_request_that_is_not_json(studio, workshop):
    status, answer = _send(studio, "/api/image", b"{not json")

    assert status == 400 and answer["error"].startswith("could not read the request: ")
    assert workshop.pictures.jobs == {}


NOT_OBJECTS = [[1, 2], "a rune knight", 7, None]       # each is valid JSON, and none has fields


@pytest.mark.parametrize("request_body", NOT_OBJECTS, ids=repr)
def test_a_picture_request_that_is_not_a_json_object_is_answered_not_dropped(studio, workshop, request_body):
    # Reading a field from a list raised, and the connection closed with no answer at all: an
    # agent or a script saw "remote end closed connection" and nothing about why.
    assert _send(studio, "/api/image", request_body) == (
        400, {"error": "could not read the request: expected a JSON object"})
    assert workshop.pictures.jobs == {}


def test_a_picture_says_when_the_picture_engine_is_not_installed(studio, workshop, monkeypatch, tmp_path):
    monkeypatch.setattr(api.image_api, "BINARY", tmp_path / "nothing-here")

    assert _send(studio, "/api/image", {"prompt": "a rune knight"}) == (
        503, {"error": "stable-diffusion.cpp is not installed (vendor/sdcpp/sd-cli).", "needs_setup": True})
    assert workshop.pictures.jobs == {}


def _nothing_downloaded(monkeypatch):
    def missing(engine=None):
        raise api.image_api.MissingWeights(["weights.gguf"], engine)
    monkeypatch.setattr(api.image_api, "resolve_weights", missing)


def test_a_picture_names_the_files_its_engine_still_needs_and_gives_its_place_back(
        studio, workshop, blender, monkeypatch):
    _nothing_downloaded(monkeypatch)

    status, answer = _send(studio, "/api/image", {"prompt": "a rune knight", "engine": "qwen"})

    assert status == 503 and answer["needs_setup"] is True and answer["missing"] == ["weights.gguf"]
    assert answer["error"].startswith("Qwen-Image 2.1 is not set up yet. Missing: weights.gguf.")
    assert workshop.pictures.jobs == {}
    assert _finish_starts(studio)       # refused after its waiting check: it had the place


def test_a_second_picture_is_refused_while_the_first_is_being_made(studio, workshop):
    first, _ = _draw(studio, workshop)
    second, answer = _draw(studio, workshop)

    assert (first, second) == (202, 409) and answer == {"error": "an image is already being generated"}
    assert len(workshop.pictures.jobs) == 1


# --- A back: POST /api/image/back -----------------------------------------------------------

def test_a_back_is_drawn_from_the_front_laid_on_grey(studio, workshop):
    status, answer = _send(studio, "/api/image/back", {
        "picture": KNIGHT, "engine": "flux", "seed": 11, "instruction": "  a red   cape "})

    assert status == 202
    [job] = workshop.pictures.jobs.values()
    front = (studio.output / KNIGHT).resolve()
    assert answer == {"job_id": job.id, "settings": job.settings,
                      "events_url": f"/api/image/{job.id}/events", "status_url": f"/api/image/{job.id}/status"}
    assert job.output_path == front.with_name("knight__back.png") and job.replaces is None
    assert job.reference == front.with_name("knight__back_reference.png") and job.reference.is_file()
    assert (job.engine.id, job.settings["seed"], job.instruction) == ("flux", 11, "a red cape")
    assert _soon(lambda: studio.started == [("picture", job)])


def test_a_new_back_is_drawn_beside_the_one_already_there(studio, workshop):
    back = (studio.output / workshop.back).resolve()
    as_it_was = back.read_bytes()

    status, _ = _send(studio, "/api/image/back", {"picture": workshop.front, "engine": "flux"})

    [job] = workshop.pictures.jobs.values()
    assert status == 202
    assert job.output_path == back.with_name(f"{PALADIN}__back_new.png") and job.replaces == back
    assert back.read_bytes() == as_it_was       # swapped in only once the new one is done


@pytest.mark.parametrize("request_body, error", [
    ({"picture": "../secret.png"}, "not a picture in output/: ../secret.png"),
    ({"picture": "images/knight/nothing.png"}, "not a picture in output/: images/knight/nothing.png"),
    ({}, "not a picture in output/: "),
    ({"picture": KNIGHT, "engine": "dall-e"}, NO_ENGINE),
])
def test_a_back_refuses_a_request_it_cannot_draw(studio, workshop, request_body, error):
    status, answer = _send(studio, "/api/image/back", request_body)

    assert status == 422 and answer["error"].startswith(error)
    assert workshop.pictures.jobs == {}


@pytest.mark.parametrize("request_body", NOT_OBJECTS, ids=repr)
def test_a_back_request_that_is_not_a_json_object_is_answered_not_dropped(studio, workshop, request_body):
    assert _send(studio, "/api/image/back", request_body) == (422, {"error": "expected a JSON object"})
    assert workshop.pictures.jobs == {}


def test_a_back_says_when_the_picture_engine_is_not_installed(studio, workshop, monkeypatch, tmp_path):
    monkeypatch.setattr(api.image_api, "BINARY", tmp_path / "nothing-here")

    assert _draw_a_back(studio, workshop) == (
        503, {"error": "stable-diffusion.cpp is not installed.", "needs_setup": True})


def test_a_back_names_the_files_its_engine_still_needs_and_gives_its_place_back(
        studio, workshop, blender, monkeypatch):
    _nothing_downloaded(monkeypatch)

    status, answer = _draw_a_back(studio, workshop)

    assert status == 503 and answer["needs_setup"] is True and answer["missing"] == ["weights.gguf"]
    assert workshop.pictures.jobs == {}
    assert _finish_starts(studio)


def test_a_front_that_cannot_be_read_is_refused_and_leaves_pictures_free(studio, workshop):
    (studio.output / "images/knight/torn.png").write_bytes(b"not a picture at all")

    status, _ = _send(studio, "/api/image/back", {"picture": "images/knight/torn.png", "engine": "flux"})

    assert status == 422
    assert not (studio.output / "images/knight/torn__back_reference.png").exists()
    assert _draw(studio, workshop)[0] == 202        # the refused back does not count as running


# --- The brush: POST /api/image/brush -------------------------------------------------------

def test_a_brush_repaints_the_painted_part_of_a_back_and_keeps_the_old_one(studio, workshop):
    status, answer = _brush(studio, workshop, seed="5")

    assert status == 202
    [job] = workshop.pictures.jobs.values()
    back = (studio.output / workshop.back).resolve()
    assert answer["job_id"] == job.id and answer["status_url"] == f"/api/image/{job.id}/status"
    assert job.replaces == back and job.output_path == back.with_name(f"{PALADIN}__back_new.png")
    assert job.mask.is_file() and job.paste_mask.is_file() and job.keep_old is True
    assert (job.engine.id, job.settings["seed"], job.instruction) == ("flux", 5, "a red cape")
    assert _soon(lambda: studio.started == [("picture", job)])


def test_a_front_no_model_was_made_from_can_be_brushed_and_is_simply_replaced(studio, workshop):
    status, _ = _brush(studio, workshop, picture=KNIGHT)

    [job] = workshop.pictures.jobs.values()
    assert status == 202 and job.replaces == (studio.output / KNIGHT).resolve() and job.keep_old is False


@pytest.mark.parametrize("changes, error", [
    ({"instruction": "  "}, "say what goes in the painted area"),
    ({"picture": "../secret.png"}, "not a picture in output/: ../secret.png"),
    ({"picture": f"images/{PALADIN}/{PALADIN}.png"},
     "a 3D model was already made from this picture; fix its back, or draw another picture"),
])
def test_a_brush_refuses_a_request_it_cannot_paint(studio, workshop, changes, error):
    assert _brush(studio, workshop, **changes) == (422, {"error": error})
    assert workshop.pictures.jobs == {}


@pytest.mark.parametrize("mask, error", [
    (None, "paint over the part to change first"),
    (b"not a picture at all", "the painted area could not be read ("),
])
def test_a_brush_needs_a_painted_area_it_can_read(studio, workshop, mask, error):
    fields = {"picture": workshop.back, "instruction": "a red cape", "engine": "flux"}

    status, answer = _post(studio, "/api/image/brush", fields, {"mask": ("mask.png", mask)} if mask else {})

    assert status == 422 and answer["error"].startswith(error)
    assert workshop.pictures.jobs == {}


def test_a_brush_with_nothing_painted_is_refused_and_gives_its_place_back(studio, workshop, blender):
    fields = {"picture": workshop.back, "instruction": "a red cape", "engine": "flux"}

    assert _post(studio, "/api/image/brush", fields, {"mask": ("mask.png", _stroke(painted=False))}) == (
        422, {"error": "paint over the part to change first"})
    assert workshop.pictures.jobs == {}
    assert _finish_starts(studio)


@pytest.mark.parametrize("length", [0, 8 * MIB + 1])
def test_a_brush_refuses_a_missing_or_oversized_request_before_reading_it(studio, workshop, length):
    assert _announce(studio, "/api/image/brush", length) == (
        422, {"error": "expected a picture path, a painted mask and what to change"})


def test_a_brush_says_when_the_picture_engine_is_not_installed(studio, workshop, monkeypatch, tmp_path):
    monkeypatch.setattr(api.image_api, "BINARY", tmp_path / "nothing-here")

    assert _brush(studio, workshop) == (
        503, {"error": "stable-diffusion.cpp is not installed.", "needs_setup": True})


def test_a_brush_names_the_files_its_engine_still_needs(studio, workshop, monkeypatch):
    _nothing_downloaded(monkeypatch)

    status, answer = _brush(studio, workshop)

    assert status == 503 and answer["needs_setup"] is True and answer["missing"] == ["weights.gguf"]
    assert workshop.pictures.jobs == {}


def test_a_brush_refuses_an_engine_it_does_not_know(studio, workshop):
    status, answer = _brush(studio, workshop, engine="dall-e")

    assert status == 422 and answer["error"].startswith(NO_ENGINE)
    assert workshop.pictures.jobs == {}


# --- Skins: POST /api/skins, /api/skins/<id>/try and /build ---------------------------------

def test_a_new_skin_starts_drawing_its_front_from_the_models_own_picture(studio, workshop):
    status, answer = _new_skin(studio, workshop)

    assert status == 202
    [job] = workshop.pictures.jobs.values()
    skin = workshop.skins.get(answer["skin"]["id"])
    assert answer["picture_job"] == {"job_id": job.id, "status_url": f"/api/image/{job.id}/status",
                                     "events_url": f"/api/image/{job.id}/events"}
    assert job.reference.resolve() == (studio.output / workshop.front).resolve()
    assert job.output_path.resolve() == (studio.output / "skins" / skin.id / "front-1.png").resolve()
    assert "make the skirt gold" in job.prompt
    assert [drawn["job_id"] for drawn in skin.tries["front"]] == [job.id]
    assert _soon(lambda: studio.started == [("picture", job)])


def _no_picture_engine(monkeypatch, _tmp_path=None):
    monkeypatch.setattr(api.image_api, "BINARY", api.REPO / "nothing-here")      # REPO is a temp folder here


@pytest.mark.parametrize("in_the_way, answer", [
    (_an_add_on_tool_is_running, (409, {"error": "an add-on is running; wait for it to finish"})),
    (_a_picture_is_being_made, (409, {"error": "a picture is being made; wait for it to finish"})),
    (_no_picture_engine, (503, {"error": "The picture engine is not installed.", "needs_setup": True})),
])
def test_a_new_skin_that_cannot_start_drawing_is_not_kept(studio, workshop, monkeypatch, tmp_path, in_the_way, answer):
    # The skin was saved before anyone asked whether its first picture could be drawn. Each
    # refusal left one with nothing drawn in the Library; asked twice, two. An agent may not
    # delete a skin, so only the person could clear them out.
    kept = sorted(skin.id for skin in workshop.skins.all())
    in_the_way(monkeypatch, tmp_path)

    assert _new_skin(studio, workshop) == answer
    assert sorted(skin.id for skin in workshop.skins.all()) == kept


def test_a_new_skin_whose_first_picture_fails_to_start_is_not_kept(studio, workshop, monkeypatch):
    # Two new skins asked for at the same moment are both let through (they are the same kind
    # of job), and the picture manager then takes only one.
    def taken(*_args, **_more):
        raise RuntimeError("an image is already being generated")
    monkeypatch.setattr(api.image_api, "start_edit", taken)
    kept = sorted(skin.id for skin in workshop.skins.all())

    assert _new_skin(studio, workshop) == (409, {"error": "an image is already being generated"})
    assert sorted(skin.id for skin in workshop.skins.all()) == kept


def test_a_new_skin_nobody_could_make_is_still_told_why_before_being_told_to_wait(
        studio, workshop, monkeypatch, tmp_path):
    _an_add_on_tool_is_running(monkeypatch, tmp_path)

    assert _send(studio, "/api/skins", {"asset": workshop.paladin["id"], "instruction": "   "}) == (
        422, {"error": 'Say what to change, for example "make the skirt gold".'})


def _tries(shop):
    return shop.skins.get(shop.draft.id).tries["front"]


def test_a_skin_try_says_when_the_picture_engine_is_not_installed_and_records_no_try(
        studio, workshop, blender, monkeypatch, tmp_path):
    monkeypatch.setattr(api.image_api, "BINARY", tmp_path / "nothing-here")

    assert _try_a_skin_again(studio, workshop) == (
        503, {"error": "The picture engine is not installed.", "needs_setup": True})
    assert _tries(workshop) == [] and workshop.pictures.jobs == {}
    assert _finish_starts(studio)


def test_a_skin_try_needs_the_part_of_the_engine_that_reads_a_picture(studio, workshop, monkeypatch):
    monkeypatch.setattr(api.image_api, "resolve_vision", lambda engine=None: None)

    status, answer = _try_a_skin_again(studio, workshop)

    assert status == 503 and answer["needs_setup"] is True
    assert answer["missing"] == [api.image_api.VISION_FILE[1]]
    assert _tries(workshop) == [] and workshop.pictures.jobs == {}


def test_a_skin_try_that_has_to_wait_says_so_with_the_skin_and_records_no_try(
        studio, workshop, monkeypatch, tmp_path):
    _an_add_on_tool_is_running(monkeypatch, tmp_path)

    status, answer = _try_a_skin_again(studio, workshop)

    assert status == 409 and answer["skin"]["id"] == workshop.draft.id
    assert _tries(workshop) == []       # a try with no picture could never be approved or redrawn


def _the_picture_manager_takes_another_start_first(monkeypatch):
    # Two picture starts asked for at the same moment are both let through (they are the same
    # kind of job), and the picture manager then takes only one.
    def taken(*_args, **_more):
        raise RuntimeError("an image is already being generated")
    monkeypatch.setattr(api.image_api, "start_edit", taken)


def test_a_skin_try_whose_picture_fails_to_start_leaves_the_skin_as_it_was(studio, workshop, monkeypatch):
    # The try was recorded before the picture was started. When the start lost, a try with no
    # picture stayed, and both approved pictures had been unapproved for it.
    _the_picture_manager_takes_another_start_first(monkeypatch)
    before = workshop.skins.get(workshop.approved.id).to_dict()
    assert before["approved"]["front"] and before["approved"]["back"]

    assert _send(studio, f"/api/skins/{workshop.approved.id}/try", {"side": "front", "instruction": "matte"}) == (
        409, {"error": "an image is already being generated"})
    assert workshop.skins.get(workshop.approved.id).to_dict() == before


@pytest.mark.parametrize("first_to_start", ["loses", "wins"])
def test_two_tries_of_one_skin_at_once_keep_the_try_that_won_the_start(studio, workshop, monkeypatch, first_to_start):
    # Two tries of one skin at the same moment are both let through, and the picture manager
    # takes one. The one it refused put back the skin as that request had first read it, so
    # the try that won, recorded since, was gone while its picture was being drawn.
    real_start, starts = api.image_api.start_edit, []
    reached, let_go = threading.Event(), threading.Event()

    def taken(*_args, **_more):
        raise RuntimeError("an image is already being generated")

    def start_edit(*args, **more):
        starts.append(args)
        first = len(starts) == 1
        if first:                       # held until the other try has asked
            reached.set()
            assert let_go.wait(20), "the test never let this start go on"
        wins = first == (first_to_start == "wins")
        return (real_start if wins else taken)(*args, **more)
    monkeypatch.setattr(api.image_api, "start_edit", start_edit)
    skin_id = workshop.approved.id
    before = workshop.skins.get(skin_id).tries["front"]

    held, held_answer = _in_the_background(
        lambda: _send(studio, f"/api/skins/{skin_id}/try", {"side": "front", "instruction": "matte"}))
    assert reached.wait(20)
    other, other_answer = _in_the_background(
        lambda: _send(studio, f"/api/skins/{skin_id}/try", {"side": "front", "instruction": "gold"}))
    other.join(1.0)                     # done, or waiting its turn behind the one held
    let_go.set()

    answers = {"matte": _answered(held, held_answer), "gold": _answered(other, other_answer)}
    winner = "matte" if first_to_start == "wins" else "gold"
    loser = "gold" if winner == "matte" else "matte"
    assert answers[loser] == (409, {"error": "an image is already being generated"})
    status, answer = answers[winner]
    assert status == 202, answer
    after = workshop.skins.get(skin_id).tries["front"]
    assert after[:-1] == before
    assert after[-1]["instruction"] == winner and after[-1]["job_id"] == answer["picture_job"]["job_id"]


def test_a_new_skin_whose_picture_did_start_is_kept_even_if_noting_it_fails(studio, workshop, monkeypatch):
    # Removing the skin is right when its picture never started. It was removed on any error,
    # also one after the start: the picture was then being drawn into a folder that was gone.
    kept = {skin.id for skin in workshop.skins.all()}
    save, saves = workshop.skins.save, []

    def full_disk_on_the_last_save(skin):
        saves.append(skin.id)
        if len(saves) == 3:         # the draft, the try, then the try's job: that one fails
            raise OSError("No space left on device")
        save(skin)
    monkeypatch.setattr(workshop.skins, "save", full_disk_on_the_last_save)

    try:
        _new_skin(studio, workshop)
    except OSError:
        pass                        # how the route answers a disk error is not this test's subject

    [made] = [skin for skin in workshop.skins.all() if skin.id not in kept]
    [job] = workshop.pictures.jobs.values()
    assert job.output_path.parent.name == made.id and len(made.tries["front"]) == 1


@pytest.mark.parametrize("stopped", ["the try is refused", "the try cannot be noted"])
def test_a_new_skin_whose_first_try_stops_before_its_picture_starts_is_not_kept(
        studio, workshop, monkeypatch, stopped):
    # Only a lost start removed the new skin. Anything else that stopped its first try before
    # the picture started, a refusal or a full disk, left a skin with nothing ever drawn.
    kept = {skin.id for skin in workshop.skins.all()}
    if stopped == "the try is refused":
        def picture_try(*_args, **_more):
            raise api.skins_api.SkinError("This model has no back picture.")
        monkeypatch.setattr(workshop.skins, "picture_try", picture_try)
    else:
        save, saves = workshop.skins.save, []

        def full_disk_on_the_try(skin):
            saves.append(skin.id)
            if len(saves) == 2:     # the draft, then the try: that one fails
                raise OSError("No space left on device")
            save(skin)
        monkeypatch.setattr(workshop.skins, "save", full_disk_on_the_try)

    try:
        _new_skin(studio, workshop)
    except OSError:
        pass                        # how the route answers a disk error is not this test's subject

    assert {skin.id for skin in workshop.skins.all()} == kept
    assert not workshop.pictures.jobs


def test_a_skin_try_waits_while_a_skin_is_being_made(studio, workshop):
    _build_a_skin(studio, workshop)

    status, answer = _try_a_skin_again(studio, workshop)

    assert (status, answer["error"]) == (409, "a skin is being made; wait for it to finish")
    assert _tries(workshop) == [] and workshop.pictures.jobs == {}


def test_a_skin_with_both_pictures_approved_starts_its_repaint(studio, workshop):
    status, answer = _build_a_skin(studio, workshop)

    assert status == 202
    [job] = workshop.builds.jobs.values()
    assert (answer["job_id"], answer["status_url"]) == (job.id, f"/api/skins/build/{job.id}/status")
    assert answer["skin"]["status"] == "building"
    assert workshop.skins.get(workshop.approved.id).status == "building"
    assert _soon(lambda: studio.started == [("skin", job)])


def test_a_skin_whose_pictures_are_not_approved_is_not_built_and_gives_its_place_back(
        studio, workshop, blender):
    assert _send(studio, f"/api/skins/{workshop.draft.id}/build", {}) == (
        422, {"error": "Approve the front picture first."})
    assert workshop.builds.jobs == {}
    assert _finish_starts(studio)


def test_a_second_skin_build_is_refused_while_the_first_is_being_made(studio, workshop):
    first, _ = _build_a_skin(studio, workshop)
    second, answer = _build_a_skin(studio, workshop)

    assert (first, second) == (202, 409)
    assert answer == {"error": "a skin is already being made; wait for it to finish"}
    assert len(workshop.builds.jobs) == 1


# --- A rig rebind: POST /api/rig/rebind -----------------------------------------------------

def _rebind_folders(studio):
    folder = studio.output / "rig"
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def test_a_rebind_saves_its_three_files_and_starts_the_run(studio, workshop):
    status, answer = _post(studio, "/api/rig/rebind", files=_rebind_bundle(asset=("Rune Knight.GLB", GLB)))

    assert status == 202
    [job] = workshop.rigs.jobs.values()
    assert answer == {"job_id": job.id, "events_url": f"/api/rig/rebind/{job.id}/events",
                      "status_url": f"/api/rig/rebind/{job.id}/status"}
    assert job.directory.parent == studio.output / "rig"
    assert job.directory.name.startswith("Rune-Knight__rebind__")
    assert (job.asset_path.read_bytes(), job.scene_path.read_bytes()) == (GLB, BLEND)
    assert job.sidecar_path.read_bytes() == _rig_sidecar(GLB, BLEND)
    assert _soon(lambda: studio.started == [("rebind", job)])


@pytest.mark.parametrize("field, wrong, error", [
    ("asset", ("knight.obj", GLB), "multipart field 'asset' must be a .glb file"),
    ("scene", ("knight.blend1", BLEND), "multipart field 'scene' must be a .blend file"),
    ("sidecar", ("knight.json", b"{}"), "multipart field 'sidecar' must be a .rig.json file"),
])
def test_a_rebind_refuses_a_file_of_the_wrong_kind_and_keeps_nothing(studio, workshop, field, wrong, error):
    assert _post(studio, "/api/rig/rebind", files=_rebind_bundle(**{field: wrong})) == (422, {"error": error})
    assert workshop.rigs.jobs == {} and _rebind_folders(studio) == []


def test_a_rebind_needs_all_three_files(studio, workshop):
    bundle = _rebind_bundle()
    del bundle["scene"]

    assert _post(studio, "/api/rig/rebind", files=bundle) == (
        422, {"error": "multipart field 'scene' must be a .blend file"})
    assert workshop.rigs.jobs == {} and _rebind_folders(studio) == []


def test_a_rebind_refuses_a_rig_record_made_for_another_model_and_gives_its_place_back(
        studio, workshop, blender):
    other = ("knight.rig.json", _rig_sidecar(GLB + b"another model", BLEND))

    status, answer = _post(studio, "/api/rig/rebind", files=_rebind_bundle(sidecar=other))

    assert status == 422 and answer["error"].startswith("asset fingerprint mismatch: ")
    assert workshop.rigs.jobs == {} and _rebind_folders(studio) == []
    assert _finish_starts(studio)


@pytest.mark.parametrize("length", [0, 768 * MIB + 1])
def test_a_rebind_refuses_a_missing_or_oversized_upload_before_reading_it(studio, workshop, length):
    assert _announce(studio, "/api/rig/rebind", length) == (
        400, {"error": "rebind bundle is missing or larger than 768 MiB"})


def test_a_second_rebind_is_refused_while_the_first_runs(studio, workshop):
    first, _ = _rebind(studio, workshop)
    second, answer = _rebind(studio, workshop)

    assert (first, second) == (202, 409) and answer == {"error": "a rig rebind is already running"}
    assert len(workshop.rigs.jobs) == 1 and len(_rebind_folders(studio)) == 1


# --- Animate: POST /api/animate/rig and /api/animate/play -----------------------------------

def _rig_runs(studio):
    return sorted(path.name for path in (studio.output / "animate").iterdir())


def test_an_uploaded_model_is_rigged_from_a_copy_in_a_run_folder_of_its_own(studio, workshop):
    status, answer = _post(studio, "/api/animate/rig", files={"asset": ("Rune Knight.GLB", GLB)})

    assert status == 202
    [job] = workshop.animations.jobs.values()
    assert answer == {"job_id": job.id, "kind": "rig", "events_url": f"/api/animate/{job.id}/events",
                      "status_url": f"/api/animate/{job.id}/status"}
    assert job.directory.parent == studio.output / "animate"
    assert job.directory.name.startswith("Rune-Knight__rig__")
    assert (job.directory / "input" / "source.glb").read_bytes() == GLB
    assert job.record["input"]["source"] == "uploaded" and job.record["license"] is None
    assert job.bone_names is None
    assert _soon(lambda: studio.started == [("animate", job)])


def test_a_model_the_studio_made_is_rigged_by_name_and_keeps_its_licence(studio, workshop):
    # The rigged model is a new file: the licence of the model it came from must travel with it.
    record = {"license": {"name": "Pixal3D licence"}, "output": {"classification": "research-only"}}
    (studio.output / workshop.finished).with_suffix(".provenance.json").write_text(json.dumps(record))

    status, _ = _post(studio, "/api/animate/rig", {"model": workshop.finished, "bone_names": "numbered"})

    assert status == 202
    [job] = workshop.animations.jobs.values()
    assert (job.directory / "input" / "source.glb").read_bytes() == GLB + b"finished"
    assert job.record["input"]["source"] == "generated"
    assert job.record["license"] == {"name": "Pixal3D licence"}
    assert job.record["classification"] == "research-only" and job.record["source_record"] == record


@pytest.mark.parametrize("fields, files, error", [
    ({}, {}, "send a .glb in 'asset' or a listed 'model'"),
    ({}, {"asset": ("knight.fbx", GLB)}, "send a .glb in 'asset' or a listed 'model'"),
    ({"model": "props/sheet/sheet.glb"}, {}, "not a model this tab offers: 'props/sheet/sheet.glb'"),
    ({"model": "../secret.glb"}, {}, "not a model this tab offers: '../secret.glb'"),
    ({"bone_names": "klingon"}, {"asset": ("knight.glb", GLB)}, "bone names is one of: numbered, "),
])
def test_an_auto_rig_refuses_a_request_it_cannot_use_and_keeps_nothing(studio, workshop, fields, files, error):
    before = _rig_runs(studio)

    status, answer = _post(studio, "/api/animate/rig", {"note": "x", **fields}, files)

    assert status == 422 and answer["error"].startswith(error)
    assert workshop.animations.jobs == {} and _rig_runs(studio) == before


@pytest.mark.parametrize("length", [0, 512 * MIB + 1])
def test_an_auto_rig_refuses_a_missing_or_oversized_upload_before_reading_it(studio, workshop, length):
    assert _announce(studio, "/api/animate/rig", length) == (
        400, {"error": "request is missing or larger than 512 MiB"})


@pytest.mark.parametrize("ask", [_rig, _play])
def test_animate_says_when_the_auto_rig_is_not_installed_and_gives_its_place_back(
        studio, workshop, blender, monkeypatch, ask):
    monkeypatch.setattr(api, "autorig_installed", lambda: False)

    assert ask(studio, workshop) == (
        409, {"error": "Auto-rig is not installed. Set it up on the Setup & Status page."})
    assert workshop.animations.jobs == {}
    assert _finish_starts(studio)


@pytest.mark.parametrize("first, second", [(_rig, _play), (_play, _rig)])
def test_one_animate_job_at_a_time_whether_a_rig_or_a_move(studio, workshop, first, second):
    assert first(studio, workshop)[0] == 202

    assert second(studio, workshop) == (409, {"error": "an Animate job is already running"})
    assert len(workshop.animations.jobs) == 1


def test_a_move_is_played_on_a_rigged_model_and_saved_beside_it(studio, workshop):
    status, answer = _send(studio, "/api/animate/play", {
        "model": RIGGED, "preset": "walk", "arm_spread": 20, "speed": 1.5, "in_place": True})

    assert status == 202
    [job] = workshop.animations.jobs.values()
    assert answer == {"job_id": job.id, "kind": "animate", "events_url": f"/api/animate/{job.id}/events",
                      "status_url": f"/api/animate/{job.id}/status"}
    assert job.result == (studio.output / RIGGED).with_name("knight_walk.glb")
    motion = job.record["motion"]
    assert (motion["preset"], motion["arm_spread_degrees"], motion["speed"], motion["in_place"]) == (
        "walk", 20.0, 1.5, True)
    assert _soon(lambda: studio.started == [("animate", job)])


@pytest.mark.parametrize("request_body, error", [
    ([1, 2], "expected a JSON object"),
    (b"{not json", "expected a JSON object"),
    ({"model": RIGGED, "preset": "moonwalk"}, "no such preset: 'moonwalk'. The moves are: walk"),
    ({"model": f"finish/{FINISHED}/paladin_20k.glb", "preset": "walk"}, "pick a rigged model first"),
    ({"model": "../secret.glb", "preset": "walk"}, "not a model this tab offers: '../secret.glb'"),
    ({"model": RIGGED, "preset": "walk", "arm_spread": 90}, "arm_spread must be between 0 and 30 degrees"),
    ({"model": RIGGED, "preset": "walk", "speed": 9}, "speed must be between 0.5 and 2"),
    ({"model": RIGGED, "preset": "walk", "in_place": "yes"}, "in place is true or false"),
])
def test_a_move_refuses_a_request_it_cannot_play(studio, workshop, request_body, error):
    assert _send(studio, "/api/animate/play", request_body) == (422, {"error": error})
    assert workshop.animations.jobs == {}


# --- The other routes that read JSON: a body that is not an object --------------------------
#
# Not starts, but the same handler and the same slip as the two picture routes above: each read
# a field from whatever JSON it was sent, and a list or a number ended the connection unanswered.

OTHER_JSON_ROUTES = [
    "/api/assets/{asset}/name", "/api/assets/{asset}/hidden", "/api/assets/{asset}/delete",
    "/api/assets/{asset}/main", "/api/image/versions/use", "/api/image/versions/delete",
    "/api/bug-report", "/api/usage", "/api/reveal",
]


@pytest.mark.parametrize("request_body", NOT_OBJECTS, ids=repr)
@pytest.mark.parametrize("path", OTHER_JSON_ROUTES)
def test_every_other_json_route_answers_a_body_that_is_not_an_object(
        studio, workshop, monkeypatch, path, request_body):
    done = []       # nothing may be renamed, deleted, switched or opened on the way to the answer
    monkeypatch.setattr(api.model_info_api, "reveal", lambda *_: done.append("a folder was opened"))
    monkeypatch.setattr(api.usage_api, "set_noticed", lambda: done.append("the notice was marked seen"))
    monkeypatch.setattr(api.usage_api, "set_sharing", lambda _share: done.append("the usage box was changed"))
    library = api.list_assets(studio.output)

    assert _send(studio, path.format(asset=workshop.paladin["id"]), request_body) == (
        422, {"error": "expected a JSON object"})
    assert done == [] and api.list_assets(studio.output) == library


# --- A plan: POST /api/chains ---------------------------------------------------------------
#
# Create sends a whole plan (picture, back, 3D model, Finish, rig, a move). Each step asked its
# own route whether its engine was set up only when the run reached it, so a missing auto-rig
# showed after the picture, the model and Finish had been made.

ALL_SET_UP = "Each is on the Setup & Status page."


@pytest.fixture
def plans(studio, workshop, engine, blender, monkeypatch):
    """Everything a plan can need is set up; a test takes away what it is about. No plan runs:
    `plans` collects the ones that were let through."""
    let_through = []
    monkeypatch.setattr(api.ChainRunner, "submit", lambda self, chain, **_more: let_through.append(chain))
    return let_through


def _plan(studio, steps, files=None, **fields):
    return _post(studio, "/api/chains", {"title": "a rune knight", "steps": json.dumps(steps), **fields}, files)


def _no_auto_rig(monkeypatch):
    monkeypatch.setattr(api, "autorig_installed", lambda: False)


def _no_3d_engine(monkeypatch):
    monkeypatch.setitem(api.BACKENDS, "trellis",
                        dataclasses.replace(api.BACKENDS["trellis"], readiness=lambda: {"ready": False}))


def _no_blender(monkeypatch):
    monkeypatch.setattr(api, "find_blender", lambda: None)


def _no_back_drawer(monkeypatch):
    monkeypatch.setattr(api, "back_drawer_ready", lambda: False)


CHARACTER = ["picture", "back", "model", "finished", "rigged", "animated"]


def test_a_plan_starts_when_everything_its_steps_need_is_set_up(studio, plans):
    status, answer = _plan(studio, CHARACTER, engine="trellis", first_move="walk")

    assert status == 202 and answer["steps"] == CHARACTER
    assert [chain.steps for chain in plans] == [CHARACTER]


def test_a_plan_is_refused_up_front_naming_everything_its_steps_still_need(studio, plans, engine, monkeypatch, tmp_path):
    _no_picture_engine(monkeypatch, tmp_path)
    _no_back_drawer(monkeypatch)
    _no_3d_engine(monkeypatch)
    _no_blender(monkeypatch)
    _no_auto_rig(monkeypatch)

    status, answer = _plan(studio, CHARACTER, engine="trellis")

    assert status == 409 and plans == []
    assert answer == {"needs_setup": True, "needs_blender": True, "error": (
        "Before this can start, set up: a picture engine that can also draw a back view (for Picture and "
        f"Back picture); {engine.label} (for 3D model); Blender 4.2 or newer (for Finish); the auto-rig "
        f"(for Rig and Animate). {ALL_SET_UP}")}


@pytest.mark.parametrize("taken_away, steps, still_needed", [
    (_no_auto_rig, CHARACTER, "the auto-rig (for Rig and Animate)"),
    (_no_auto_rig, ["rigged"], "the auto-rig (for Rig)"),
    (_no_back_drawer, CHARACTER, "a picture engine that can draw a back view (for Back picture)"),
    (_no_picture_engine, ["picture", "model"], "a picture engine (for Picture)"),
])
def test_a_plan_names_the_one_thing_missing_and_the_steps_it_is_for(studio, plans, monkeypatch, taken_away, steps, still_needed):
    taken_away(monkeypatch)

    status, answer = _plan(studio, steps, engine="trellis")

    assert status == 409 and plans == []
    assert answer == {"needs_setup": True, "error": f"Before this can start, set up: {still_needed}. {ALL_SET_UP}"}


def test_a_plan_names_the_3d_engine_it_asked_for(studio, plans, engine, monkeypatch):
    _no_3d_engine(monkeypatch)

    status, answer = _plan(studio, ["picture", "model"], engine="trellis")

    assert status == 409 and plans == []
    assert answer["error"] == f"Before this can start, set up: {engine.label} (for 3D model). {ALL_SET_UP}"


def test_a_plan_names_the_picture_engine_it_asked_for(studio, plans, monkeypatch):
    def only_flux_is_here(engine=None):
        if engine is api.image_api.QWEN:
            raise api.image_api.MissingWeights(["weights.gguf"], engine)
        return {}
    monkeypatch.setattr(api.image_api, "resolve_weights", only_flux_is_here)

    status, answer = _plan(studio, ["picture"], picture_engine="qwen")

    assert status == 409 and plans == []
    assert answer["error"] == (
        f"Before this can start, set up: the Qwen-Image 2.1 picture engine (for Picture). {ALL_SET_UP}")
    assert _plan(studio, ["picture"], picture_engine="flux")[0] == 202       # the one that is here


@pytest.mark.parametrize("taken_away, steps", [
    (_no_auto_rig, ["picture", "back", "model", "finished"]),
    (_no_blender, ["picture", "model"]),
    (_no_3d_engine, ["picture"]),
    (_no_back_drawer, ["picture", "model"]),
    (_no_picture_engine, ["finished"]),
])
def test_a_plan_is_asked_only_for_what_its_own_steps_need(studio, plans, monkeypatch, taken_away, steps):
    taken_away(monkeypatch)

    assert _plan(studio, steps, engine="trellis")[0] == 202
    assert len(plans) == 1


def test_a_plan_missing_only_blender_keeps_its_own_longer_help(studio, plans, monkeypatch):
    _no_blender(monkeypatch)

    status, answer = _plan(studio, ["picture", "model", "finished"], engine="trellis")

    assert status == 409 and plans == [] and answer["needs_blender"] is True
    assert "Install Blender" in answer["error"] and "stop after the 3D model" in answer["error"]


def test_a_plan_with_your_own_back_picture_needs_nothing_to_draw_one(studio, plans, monkeypatch):
    _no_back_drawer(monkeypatch)
    monkeypatch.setitem(api.BACKENDS, "pixal3d",
                        dataclasses.replace(api.BACKENDS["pixal3d"], readiness=lambda: {"ready": True}))

    status, _ = _plan(studio, ["back", "model"], engine="pixal3d",
                      files={"file": ("knight.png", _picture()), "back_file": ("knight-back.png", _picture())})

    assert status == 202 and len(plans) == 1


@pytest.mark.parametrize("fields, error", [
    ({"picture_engine": "dall-e"}, NO_ENGINE),
    ({"engine": "no-such-engine"}, "unknown backend 'no-such-engine'"),
])
def test_a_plan_that_names_an_engine_nobody_knows_is_refused_at_once(studio, plans, fields, error):
    status, answer = _plan(studio, ["picture", "model"], **fields)

    assert status == 400 and answer["error"].startswith(error)
    assert plans == []


# --- The starts not asked mid-way above, the slot after a crash, and every kind as the one in the way ---

def _generation_waits(studio, words):
    """Whether a generation asked for now is told to wait, in these words."""
    return _post(studio, "/api/generate", {"settings": "{}"}, {"image": ("knight.png", PICTURE)}) == (
        409, {"error": f"{words}; wait for it to finish"})


# name: (how it is asked for, a step between its waiting check and its job's entry on the books,
#        what it makes a generation answer, what that step answers here if not what it really would)
MORE_STARTS = {
    "a prop sheet": (lambda studio: _post(studio, "/api/props", {"settings": "{}"}, {"asset": ("sheet.glb", GLB)}),
                     lambda: (api.PROPS_JOBS, "create"), "a prop sheet is baking", None),
    "a prop turned": (lambda studio: _send(studio, "/api/props/runs/tavern/turn", {"prop": "barrel"}),
                      lambda: (api.PROPS_JOBS, "turn"), "a prop sheet is baking", None),
    "a Finish run resumed": (lambda studio: _post(studio, "/api/finish/runs/knight__finish__1/resume"),
                             lambda: (api, "find_blender"), "a model is being finished", None),
    "an add-on tool": (lambda studio: _send(studio, "/api/addons/vehicle-kit/run/build", {"inputs": {}}),
                       lambda: (api.addons_api, "run_payload"), "an add-on is running",
                       lambda *_asked: {"job_id": "0" * 32, "status": "running"}),
    # a setup is on the books (SETUP_ACTIVE) before its run is started, so each is held one step earlier
    **{f"setup by {route}": (lambda studio, route=route: _post(studio, route),
                             lambda step=step: (api, step), "setup is running", None)
       for route, step in zip(SETUP_ROUTES, ["clean_port_build_present", "blender_install_refusal",
                                             "gltfpack_install_refusal"], strict=True)},
    **{f"an engine's {action}": (lambda studio, action=action: _post(studio, f"/api/setup/pixal3d/{action}"),
                                 lambda action=action: (api, "remove_weights" if action == "remove" else "start_download"),
                                 "setup is running", None)
       for action in ENGINE_ACTIONS},
}


@pytest.mark.parametrize("ask, held_at, words, answer", MORE_STARTS.values(), ids=MORE_STARTS)
def test_every_other_start_holds_its_place_before_its_job_is_on_the_books(
        studio, engine, blender, installers, engines, monkeypatch, ask, held_at, words, answer):
    on, step = held_at()
    reached, let_go = _hold(monkeypatch, step, answer, on=on)
    starting, answers = _in_the_background(lambda: ask(studio))
    assert reached.wait(20)
    try:
        assert _generation_waits(studio, words)
    finally:
        let_go.set()

    _answered(starting, answers)        # whatever it answers; the studio fixture checks the place came back
    assert "_run_job" not in [runner for runner, _ in studio.started]


def _nobody_planned_for_this(*_args, **_more):
    raise KeyError("something nobody planned for")


def test_a_start_that_breaks_in_a_way_nobody_planned_for_still_gives_its_place_back(
        studio, workshop, blender, monkeypatch):
    monkeypatch.setattr(workshop.animations, "create_rig", _nobody_planned_for_this)

    status, _ = _rig(studio, workshop)

    assert status == 500
    assert _finish_starts(studio)


def test_a_start_that_dies_without_answering_still_gives_its_place_back(studio, workshop, blender, monkeypatch):
    # The picture route catches only the errors it expects: anything else ends the request
    # with no answer. Its place must not go with it, or nothing else starts until a restart.
    monkeypatch.setattr(api.image_api, "start", _nobody_planned_for_this)

    try:
        _draw(studio, workshop)
    except OSError:
        pass                # the connection was dropped; how it answers is not this test's subject

    assert _finish_starts(studio)


def _on_its_books(manager):
    manager.jobs["b" * 32], manager.active = types.SimpleNamespace(status="running"), "b" * 32


IN_THE_WAY = {
    "a Finish run": (lambda patch: _on_its_books(api.FINISH_JOBS), "a model is being finished"),
    "a prop bake": (lambda patch: _on_its_books(api.PROPS_JOBS), "a prop sheet is baking"),
    "a rig rebind": (lambda patch: _on_its_books(api.RIG_JOBS), "a rig rebind is running"),
    "an auto-rig": (lambda patch: _on_its_books(api.ANIMATE_JOBS), "an auto-rig or animation is running"),
    "a skin build": (lambda patch: _on_its_books(api.skins_api.BUILDS), "a skin is being made"),
    "a setup": (lambda patch: patch.setattr(api, "SETUP_ACTIVE", "a" * 32), "setup is running"),
    "an engine download": (lambda patch: patch.setattr(api, "download_active", lambda: object()), "setup is running"),
}


@pytest.mark.parametrize("arrange, words", IN_THE_WAY.values(), ids=IN_THE_WAY)
def test_a_generation_waits_for_each_other_kind_of_job_on_its_real_books(studio, engine, monkeypatch, arrange, words):
    # The rule is tested kind by kind elsewhere with the checks replaced; here each kind's own
    # manager says it is running, the way it does in the studio.
    arrange(monkeypatch)

    assert _generation_waits(studio, words)
    assert studio.started == []


# --- A plan: the cases the first tests left out ---------------------------------------------

def test_a_plan_that_names_no_3d_engine_is_asked_for_the_one_a_generation_would_take(studio, plans, engine, monkeypatch):
    _no_3d_engine(monkeypatch)

    status, answer = _plan(studio, ["model"])

    assert status == 409 and plans == []
    assert answer["error"] == f"Before this can start, set up: {engine.label} (for 3D model). {ALL_SET_UP}"


def test_a_back_asked_of_one_picture_engine_needs_that_engine_to_be_able_to_draw_it(studio, plans, monkeypatch):
    monkeypatch.setattr(api.image_api, "resolve_vision", lambda engine=None: None)      # Qwen cannot read a picture here

    def ask(engine_id):
        return _plan(studio, ["back"], made=json.dumps({"picture": KNIGHT}), back_settings=json.dumps({"engine": engine_id}))

    assert ask("qwen") == (409, {"needs_setup": True, "error": (
        f"Before this can start, set up: a picture engine that can draw a back view (for Back picture). {ALL_SET_UP}")})
    assert ask("flux")[0] == 202        # Flux reads the picture by itself
    assert len(plans) == 1


def test_a_plan_that_is_refused_keeps_nothing_of_what_was_uploaded_with_it(studio, plans, monkeypatch):
    _no_auto_rig(monkeypatch)
    before = sorted(path.relative_to(studio.output) for path in studio.output.rglob("*"))

    status, _ = _plan(studio, ["finished", "rigged"], files={"file": ("my knight.glb", GLB)})

    assert status == 409 and plans == []
    assert sorted(path.relative_to(studio.output) for path in studio.output.rglob("*")) == before
