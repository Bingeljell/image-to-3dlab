"""The studio's three upload routes, asked the way the page asks them.

Generate, Finish and Prop sheets each take a real multipart upload over HTTP here, against the
real request handler. Only the job runners are replaced: a run costs minutes and needs a model.
Before this file nothing ran `_create_job`, `_create_finish_job` or `_create_props_job`, so a
wrong status code, a size limit that stopped working or a missing wait would have shipped.
"""

from __future__ import annotations

import dataclasses
import http.client
import http.server
import json
import threading
import types
import urllib.error
import urllib.request

import activity_api
import generate_api as api
import pytest

GLB = b"glTF\x02\x00\x00\x00" + b"\x00" * 16       # never parsed here: the runners are stubs
PICTURE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16      # nor is this, unless a test asks for a cut-out
MIB = 1024 * 1024


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """A real studio server on a free port. It writes into a temp folder and starts no job:
    `started` collects what each runner was handed instead."""
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(api, "REPO", tmp_path)
    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(api, "SETUP_ACTIVE", None)
    monkeypatch.setattr(api, "JOBS", api.JobManager())
    monkeypatch.setattr(api, "FINISH_JOBS", api.FINISH_JOBS.__class__(output / "finish", output))
    monkeypatch.setattr(api, "PROPS_JOBS", api.PROPS_JOBS.__class__(output / "props"))
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


@pytest.fixture
def engine(monkeypatch):
    """TRELLIS.2 with its real settings check, counted as installed, taking a picture as it comes."""
    spec = dataclasses.replace(api.BACKENDS["trellis"], readiness=lambda: {"ready": True},
                               requires_alpha=False, gpu_memory_wanted=None)
    monkeypatch.setitem(api.BACKENDS, "trellis", spec)
    return spec


def _post(studio, path, fields=None, files=None):
    """POST a form the way the page's FormData does; returns (status, the JSON answer)."""
    body, content_type = activity_api.multipart(fields or {}, files or {})
    request = urllib.request.Request(
        f"http://127.0.0.1:{studio.port}{path}", data=body,
        headers={"Content-Type": content_type, "X-AssetFurnace-Request": "studio"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


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
                      "output_dir": "output/Rune-Knight", "warning": None}
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
