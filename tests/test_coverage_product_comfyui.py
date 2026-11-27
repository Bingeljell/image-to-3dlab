"""Exercise workflow and file handling with a local substitute for the HTTP session."""

import json
import sys

import pytest
import requests

from image_to_3dlab import comfyui_backend as backend


class Response:
    def __init__(self, payload=None, chunks=()):
        self.payload = payload
        self.chunks = chunks

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield from self.chunks


class Session:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def _request(self, method, url, **kwargs):
        assert url.startswith("http://example.invalid/comfy/")
        route = url.removeprefix("http://example.invalid/comfy/")
        if files := kwargs.get("files"):
            filename, handle, mime = files["image"]
            kwargs["files"] = {"image": (filename, handle.read(), mime)}
        self.calls.append((method, route, kwargs))
        reply = self.replies[route].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)


@pytest.fixture
def workflow_files(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "old"}}}))
    return image, workflow, tmp_path / "assets"


def replies_for_job():
    return {
        "system_stats": [Response()],
        "upload/image": [Response({"name": "input.png", "subfolder": "/batch"})],
        "prompt": [Response({"prompt_id": "job"})],
        "history/job": [Response({"job": {"outputs": {"mesh": {"filename": "../output.GLB"}}}})],
        "view": [Response(chunks=(b"mesh", b"-data"))],
    }


def client_with(replies):
    session = Session(replies)
    return backend.ComfyUIClient("http://example.invalid/comfy///", session), session


def test_workflow_polls_and_saves_selected_mesh(workflow_files, monkeypatch):
    replies = replies_for_job()
    replies["history/job"].insert(0, Response({}))
    sleeps = []
    monkeypatch.setattr(backend.time, "sleep", sleeps.append)
    client, session = client_with(replies)

    result = client.generate(*workflow_files, image_node="1", output_node="mesh", timeout=3, poll_interval=.25)

    assert result.parent == workflow_files[2]
    assert result.name.endswith("-output.GLB")
    assert result.read_bytes() == b"mesh-data"
    assert sleeps == [.25]
    _, _, upload = session.calls[1]
    name, content, mime = upload["files"]["image"]
    assert name.endswith("_input.png")
    assert content == b"image"
    assert mime == "image/png"
    assert upload["data"] == {"type": "input", "subfolder": "image-to-3dlab", "overwrite": "false"}
    assert session.calls[2][2]["json"]["prompt"]["1"]["inputs"]["image"] == "batch/input.png"
    assert session.calls[-1][2]["params"] == {"filename": "../output.GLB", "subfolder": "", "type": "output"}


@pytest.mark.parametrize(("route", "message"), [
    ("system_stats", "not reachable"), ("upload/image", "could not upload"),
    ("prompt", "could not queue"), ("history/job", "could not read"), ("view", "could not download"),
])
def test_http_failures_name_the_failed_operation(route, message, workflow_files):
    replies = replies_for_job()
    replies[route] = [requests.ConnectionError("connection closed")]
    client, session = client_with(replies)

    with pytest.raises(RuntimeError, match=message) as err:
        client.generate(*workflow_files, None, None, 3, .25)

    assert isinstance(err.value.__cause__, requests.ConnectionError)
    assert session.calls[-1][1] == route
    assert not workflow_files[2].exists()


@pytest.mark.parametrize(("route", "payload", "message"), [
    ("prompt", {"node_errors": {"1": "bad image"}}, "bad image"),
    ("prompt", {}, "no prompt_id"),
    ("history/job", {"job": {"status": {"status_str": "error", "messages": ["crashed"]}}}, "crashed"),
    ("history/job", {"job": {"status": {"completed": False, "messages": ["stopped"]}}}, "stopped"),
])
def test_workflow_rejection_keeps_server_error(route, payload, message, workflow_files):
    replies = replies_for_job()
    replies[route] = [Response(payload)]
    client, session = client_with(replies)
    with pytest.raises(RuntimeError, match=message):
        client.generate(*workflow_files, None, None, 3, .25)
    assert session.calls[-1][1] == route


def test_workflow_timeout_stops_polling(workflow_files, monkeypatch):
    replies = replies_for_job()
    replies["history/job"] = [Response({})]
    clock = iter([10, 10, 12])
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(backend.time, "sleep", lambda _seconds: None)
    client, session = client_with(replies)
    with pytest.raises(RuntimeError, match="job job timed out after 2s"):
        client.generate(*workflow_files, None, None, 2, .25)
    assert session.calls[-1][1] == "history/job"


@pytest.mark.parametrize(("content", "timeout", "poll_interval", "message"), [
    ("{}", 0, .25, "must be positive"),
    ("{}", 3, 0, "must be positive"),
    ("[", 3, .25, "could not read workflow"),
    (None, 3, .25, "could not read workflow"),
    ("[]", 3, .25, "API-format object"),
])
def test_invalid_workflow_never_contacts_server(content, timeout, poll_interval, message, workflow_files):
    image, workflow, out = workflow_files
    if content is None:
        workflow.unlink()
    else:
        workflow.write_text(content)
    client, session = client_with({})
    with pytest.raises(ValueError, match=message):
        client.generate(image, workflow, out, None, None, timeout, poll_interval)
    assert session.calls == []


@pytest.mark.parametrize(("workflow", "node", "message"), [
    ({}, None, r"0 LoadImage nodes.*\(none\)"),
    ({"1": {"class_type": "LoadImage"}, "2": {"class_type": "LoadImage"}}, None, "1, 2"),
    ({"1": {}}, "1", "missing or has no inputs"),
    ({"1": {"inputs": {}}}, "1", "has no 'image' input"),
])
def test_invalid_image_node_is_rejected(workflow, node, message):
    with pytest.raises(ValueError, match=message):
        backend.patch_input_image(workflow, "new.png", node)


@pytest.mark.parametrize(("entry", "node", "message"), [
    ({}, "missing", "did not produce an output"),
    ({}, None, "files found: none"),
    ({"outputs": {"preview": {"filename": "preview.png"}}}, None, "files found: preview.png"),
])
def test_workflow_without_a_mesh_explains_what_is_missing(entry, node, message):
    with pytest.raises(RuntimeError, match=message):
        backend.find_asset(entry, node)


def test_upload_falls_back_to_binary_media_type_and_generated_name(tmp_path):
    image = tmp_path / "source.unknownextension"
    image.write_bytes(b"data")
    client, session = client_with({"upload/image": [Response({})]})
    remote = client._upload(image)
    assert remote.endswith("_source.unknownextension")
    assert session.calls[0][2]["files"]["image"] == (remote, b"data", "application/octet-stream")


def test_comfyui_explains_missing_http_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    with pytest.raises(RuntimeError, match="requires requests"):
        backend.ComfyUIClient("http://example.invalid")
