"""Chain plans should handle missing inputs and interrupted job requests."""

import io
import urllib.error
from unittest.mock import Mock

import activity_api as activity
import generate_api as api
import pytest


@pytest.mark.parametrize(("step", "message"), [
    ("model", "needs a picture"),
    ("back", "needs a front picture"),
    ("rigged", "needs a 3D model"),
    ("animated", "needs a rigged model and a move"),
    ("split", "needs the prop sheet"),
])
def test_chain_missing_inputs_fail_before_requesting_job(step, message, tmp_path):
    request = Mock()
    runner = activity.ChainRunner(tmp_path, request)
    chain = activity.Chain("test plan", [step])
    runner.run(chain)
    assert chain.status == "error"
    assert message in chain.error
    assert activity.read_history(tmp_path)[0]["step"] == step
    request.assert_not_called()


def test_uploaded_back_without_a_front_fails_before_any_request(tmp_path):
    request = Mock()
    runner = activity.ChainRunner(tmp_path, request)
    chain = activity.Chain("test plan", ["back"], back_upload=("back.png", b"fixture"))
    runner.run(chain)
    assert chain.status == "error"
    assert "no front picture" in chain.error
    assert chain.back_upload is None
    request.assert_not_called()


def test_chain_keeps_upload_in_memory_if_library_copy_fails(tmp_path):
    (tmp_path / "images").write_bytes(b"not a directory")
    received = []

    def request(method, path, body, content_type):
        if method == "POST":
            received.append(api.parse_multipart(content_type, body)["image"]["data"])
            return 202, {"job_id": "j"}
        return 200, {"status": "done", "model": "generated.glb"}

    chain = activity.Chain("test plan", ["model"], upload=("front.png", b"front bytes"))
    activity.ChainRunner(tmp_path, request).run(chain)
    assert chain.status == "done"
    assert chain.made["model"] == "generated.glb"
    assert received == [b"front bytes"]
    assert chain.upload is None


def test_chain_stays_cancelled_if_requested_before_first_step(tmp_path):
    request = Mock()
    runner = activity.ChainRunner(tmp_path, request)
    chain = activity.Chain("test plan", ["model"], cancel_requested=True)
    runner.run(chain)
    assert chain.status == "cancelled"
    assert chain.job is None
    request.assert_not_called()


def test_props_plan_survives_dropped_poll_and_stops_after_worker_cancellation(tmp_path):
    calls = []

    def request(method, path, body, content_type):
        calls.append((method, path))
        if method == "POST":
            assert api.parse_multipart(content_type, body)["generated"]["value"] == "props.glb"
            return 202, {"job_id": "j"}
        if len(calls) == 2:
            raise OSError("temporary disconnect")
        if len(calls) < 5:
            return 200, {"status": "running", "last_event": {"overall_pct": 35}}
        return 200, {"status": "cancelled"}

    sleep = Mock()
    runner = activity.ChainRunner(tmp_path, request, poll=0.01, sleep=sleep)
    chain = activity.Chain("test plan", ["split"], made={"model": "props.glb"})
    runner.run(chain)
    assert calls == [("POST", "/api/props"), *(("GET", "/api/props/j/status"),) * 4]
    assert sleep.call_count == 3
    assert chain.cancel_requested and chain.status == "cancelled"
    assert activity.read_history(tmp_path, limit=1)[0]["result"] == "cancelled"


def test_chain_retries_cancelling_a_child_after_a_refusal(tmp_path):
    request = Mock(side_effect=[(409, {}), (202, {"status": "cancelling"})])
    runner = activity.ChainRunner(tmp_path, request)
    chain = activity.Chain("test plan", ["split"], job=("props", "j"))
    runner.chains[chain.id] = chain
    assert runner.cancel(chain.id)
    assert chain.cancel_sent is None
    assert runner.cancel(chain.id)
    assert chain.cancel_sent == ("props", "j")
    assert runner.cancel(chain.id)
    assert request.call_count == 2


def test_chain_runs_even_if_history_cannot_be_written(tmp_path):
    (tmp_path / activity.HISTORY_NAME).mkdir()
    request = Mock(side_effect=[(202, {"job_id": "j"}), (200, {"status": "done", "picture": "wood.png"})])
    chain = activity.Chain("wood", ["picture"])
    activity.ChainRunner(tmp_path, request).run(chain)
    assert chain.status == "done" and chain.done == ["picture"]
    assert activity.read_history(tmp_path) == []


def test_plan_form_wraps_invalid_settings_json():
    with pytest.raises(ValueError, match="plan is not readable"):
        activity.chain_from_form({"steps": {"value": "[\"model\"]"}, "finish_settings": {"value": "{"}})


def test_plan_transport_sends_token_and_wraps_non_json_refusal(monkeypatch):
    def respond(request, timeout):
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer test-token"
        assert request.get_header("Content-type") == "application/json"
        assert request.get_header("X-assetfurnace-request") == "plan"
        assert request.data == b"{}"
        assert timeout == 120
        raise urllib.error.HTTPError(request.full_url, 404, "Missing", {}, io.BytesIO(b"Not found"))

    monkeypatch.setattr("urllib.request.urlopen", respond)
    post = activity.local_request("http://127.0.0.1:8600", token="test-token")
    assert post("POST", "/api/missing", b"{}", "application/json") == (
        404, {"error": "/api/missing answered 404"},
    )
