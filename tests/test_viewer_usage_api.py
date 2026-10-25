"""Anonymous usage counts: on by default, sends only its payload, forgets everything when turned off.

Every test points the state file at tmp_path and replaces the network call: nothing here
touches a real install or reaches assetfurnace.com. The suite blocks usage for every other
test (conftest sets I3D_NO_USAGE=1); tests here pass their own env, or drop that switch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

import usage_api as ua

ON = {}                       # an environment without I3D_NO_USAGE
OFF = {"I3D_NO_USAGE": "1"}
DAY = "2026-10-25"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ua, "STATE", tmp_path / ".usage.json")

    def no_network(body):
        raise AssertionError("a test reached the network")

    monkeypatch.setattr(ua, "_post", no_network)
    monkeypatch.setattr(ua, "send_soon", lambda *a, **k: None)   # sends are tested by calling send_daily
    return tmp_path / ".usage.json"


def start(kind_parts, header=None, code=202, env=ON):
    ua.note_request(kind_parts, header, "POST", code, env=env)


def test_on_by_default_and_the_first_message_carries_a_fresh_id(isolated):
    assert ua.status(env=ON)["enabled"] is True
    sent = []
    assert ua.send_daily(env=ON, today=DAY, post=sent.append) is True
    body = sent[0]
    assert set(body) == {"schema_version", "id", "day", "version", "os", "machine", "started", "finished", "agent"}
    assert len(body["id"]) == 32 and body["day"] == DAY
    assert body["started"] == {"studio": {}, "agent": {}} and body["finished"] == {} and body["agent"] is False
    assert str(Path.home()) not in json.dumps(body)


def test_counts_what_was_started_by_whom_and_what_finished(isolated):
    start(["api", "generate"])
    start(["api", "chains"], header="agent")
    start(["api", "skins", "abc", "build"], header="agent")
    start(["api", "generate"], code=422)          # refused: nothing started
    start(["api", "assets", "x", "name"])         # not work
    ua.count_done("model", env=ON)
    ua.count_done("model", env=ON)
    ua.count_done("finish", env=ON)
    sent = []
    ua.send_daily(env=ON, today=DAY, post=sent.append)
    assert sent[0]["started"] == {"studio": {"model": 1}, "agent": {"chain": 1, "skin": 1}}
    assert sent[0]["finished"] == {"model": 2, "finish": 1}
    assert sent[0]["agent"] is True


def test_any_agent_call_counts_as_agent_use_even_one_that_starts_nothing(isolated):
    ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)
    sent = []
    ua.send_daily(env=ON, today=DAY, post=sent.append)
    assert sent[0]["agent"] is True and sent[0]["started"]["agent"] == {}


def test_once_a_day_and_only_new_counts_go_the_next_day(isolated):
    ua.count_done("picture", env=ON)
    sent = []
    assert ua.send_daily(env=ON, today=DAY, post=sent.append) is True
    assert ua.send_daily(env=ON, today=DAY, post=sent.append) is False
    ua.count_done("picture", env=ON)
    assert ua.send_daily(env=ON, today="2026-10-26", post=sent.append) is True
    assert [b["finished"] for b in sent] == [{"picture": 1}, {"picture": 1}]
    assert sent[1]["id"] == sent[0]["id"] and sent[1]["agent"] is False


def test_the_about_page_shows_what_is_sent(isolated):
    ua.count_done("finish", env=ON)
    shown = ua.status(env=ON)["sends"]
    sent = []
    ua.send_daily(env=ON, today=shown["day"], post=sent.append)
    assert sent == [shown]


def test_offline_keeps_the_counts_for_another_day(isolated):
    ua.count_done("finish", env=ON)

    def offline(body):
        raise OSError("no network")

    assert ua.send_daily(env=ON, today=DAY, post=offline) is False
    sent = []
    assert ua.send_daily(env=ON, today=DAY, post=sent.append) is True
    assert sent[0]["finished"] == {"finish": 1}


def test_a_job_finished_while_sending_is_not_lost(isolated):
    ua.count_done("finish", env=ON)
    ua.send_daily(env=ON, today=DAY, post=lambda body: ua.count_done("finish", env=ON))
    assert json.loads(isolated.read_text())["finished"] == {"finish": 1}


def test_two_sends_at_once_send_the_counts_once(isolated):
    """Found end to end: a background send and a page-load send both went out with the same counts."""
    ua.count_done("model", env=ON)
    sent = []

    def slow(body):
        sent.append(body)
        assert ua.send_daily(env=ON, today=DAY, post=sent.append) is False   # a second send while one is out

    assert ua.send_daily(env=ON, today=DAY, post=slow) is True
    assert len(sent) == 1


def test_turning_it_off_forgets_the_id_and_counts_nothing(isolated):
    ua.count_done("model", env=ON)
    first = json.loads(isolated.read_text())["id"]
    assert ua.set_sharing(False, env=ON)["enabled"] is False
    assert json.loads(isolated.read_text()) == {"schema_version": ua.SCHEMA, "opted_out": True}
    ua.count_done("model", env=ON)
    start(["api", "generate"], header="agent")
    assert ua.send_daily(env=ON, today=DAY, post=lambda b: None) is False
    assert json.loads(isolated.read_text()) == {"schema_version": ua.SCHEMA, "opted_out": True}
    ua.set_sharing(True, env=ON)
    sent = []
    ua.send_daily(env=ON, today=DAY, post=sent.append)
    assert sent[0]["id"] != first and sent[0]["finished"] == {}


@pytest.mark.parametrize("env", [{"I3D_NO_USAGE": "1"}, {"DO_NOT_TRACK": "1"}])
def test_the_environment_switch_wins_over_the_box(isolated, env):
    assert ua.status(env=env) == {**ua.status(env=env), "enabled": False, "allowed": False, "sends": None}
    ua.count_done("model", env=env)
    assert ua.send_daily(env=env, today=DAY, post=lambda b: None) is False
    assert ua.set_sharing(True, env=env)["enabled"] is False
    assert not isolated.exists()


def test_a_broken_state_file_starts_afresh(isolated):
    isolated.write_text("not json")
    ua.count_done("model", env=ON)                     # must not raise
    assert json.loads(isolated.read_text())["finished"] == {"model": 1}


def test_an_unknown_kind_is_a_bug_not_a_new_count(isolated):
    with pytest.raises(ValueError):
        ua.count_done("teleport", env=ON)


def test_every_route_the_counts_name_is_a_real_studio_route():
    """A renamed route would quietly stop counting: each STARTS path must be handled."""
    source = (REPO / "viewer" / "generate_api.py").read_text()
    for route in ua.STARTS:
        if "*" in route:
            fixed = [p for p in route if p != "*"]
            assert all(f'"{p}"' in source for p in fixed), route
        else:
            assert "parts == " + json.dumps(list(route)) in source, route


def test_the_mcp_server_marks_its_calls_as_an_agents():
    from image_to_3dlab import mcp_server

    (name, value), = mcp_server.AGENT_HEADER.items()
    assert name == ua.CLIENT_HEADER and ua.source_of(value) == ua.AGENT


def test_send_soon_sends_once_off_the_request(isolated, monkeypatch):
    import importlib
    import threading

    real = importlib.reload(ua)   # the fixture stubbed send_soon; get the real one
    monkeypatch.setattr(real, "STATE", isolated)
    went = threading.Event()
    monkeypatch.setattr(real, "_post", lambda body: went.set())
    real.send_soon(env=ON)
    assert went.wait(5)
    importlib.reload(ua)


def test_the_studio_counts_through_its_real_routes(isolated, monkeypatch):
    import functools
    import http.server
    import threading
    import urllib.request

    import generate_api

    monkeypatch.delenv("I3D_NO_USAGE", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(REPO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/api/usage"
    try:
        with urllib.request.urlopen(base, timeout=10) as response:
            assert json.loads(response.read())["enabled"] is True
        agent = urllib.request.Request(base, headers={ua.CLIENT_HEADER: "agent"})
        urllib.request.urlopen(agent, timeout=10).read()
        assert json.loads(isolated.read_text())["agent_seen"] is True
        request = urllib.request.Request(base, data=json.dumps({"share": False}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert json.loads(response.read())["enabled"] is False
        bad = urllib.request.Request(base, data=b'{"share": "yes"}', method="POST",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(bad, timeout=10)
        assert refused.value.code == 422
    finally:
        server.shutdown()


def test_a_chain_started_by_an_agent_passes_the_mark_to_each_step(tmp_path):
    import activity_api

    seen = []

    def request(method, path, body, content_type, headers=None):
        seen.append(headers)
        return 202, {"job_id": "j"}

    runner = activity_api.ChainRunner(tmp_path, request=request)
    chain = activity_api.Chain(title="fox", steps=["picture"], prompt="a fox", client="agent")
    runner._start(chain, "picture")
    assert seen == [{ua.CLIENT_HEADER: "agent"}]
    runner._start(activity_api.Chain(title="fox", steps=["picture"], prompt="a fox"), "picture")
    assert seen[-1] is None


def test_a_finished_model_is_counted_by_finish(isolated, tmp_path, monkeypatch):
    """A real Finish job with a stand-in worker that writes the model: done means one more."""
    import finish_api

    monkeypatch.delenv("I3D_NO_USAGE", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    job = finish_api.FinishJob("0" * 32, tmp_path / "out" / "fox__finish__20261025-010101")
    job.settings = {**job.settings, "hero": True}

    class Worker:
        pid = 1
        returncode = 0
        stdout = iter(["[hero] done\n"])

        def __init__(self, *args, **kwargs):
            job.result_glb.parent.mkdir(parents=True, exist_ok=True)
            job.result_glb.write_bytes(b"glb")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api, "hero_command", lambda job, settings: ["worker"])
    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    finish_api.run_job(job, finish_api.FinishJobManager(tmp_path / "out", tmp_path / "out"))
    assert job.status == "done", job.events[-1]
    assert json.loads(isolated.read_text())["finished"] == {"finish": 1}
