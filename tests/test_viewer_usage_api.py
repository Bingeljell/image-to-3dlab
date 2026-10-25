"""Anonymous usage counts: on by default, sends only its payload, forgets everything when turned off.

Every test points the state file at tmp_path and replaces the network call: nothing here
touches a real install or reaches assetfurnace.com. The suite blocks usage for every other
test (conftest sets I3D_NO_USAGE=1); tests here pass their own env, or drop that switch.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

import usage_api as ua

REAL_SEND_SOON = ua.send_soon   # the fixture stubs it; one test needs the real one
ON = {}                         # an environment without I3D_NO_USAGE
OFF = {"I3D_NO_USAGE": "1"}
DAY = "2026-10-25"
NOON = 1_761_393_600.0          # a clock for send_daily; RETRY_SECONDS apart means "later"


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


def send(sent, today=DAY, at=NOON, env=ON):
    return ua.send_daily(env=env, today=today, post=sent.append, clock=lambda: at)


def seen():
    ua.set_noticed(env=ON)


def http_error(code):
    return urllib.error.HTTPError(ua.USAGE_URL, code, "no", {}, io.BytesIO(b""))


def test_on_by_default_but_nothing_goes_until_the_notice_is_seen(isolated):
    assert ua.status(env=ON)["enabled"] is True and ua.status(env=ON)["noticed"] is False
    ua.count_done("model", env=ON)
    sent = []
    assert send(sent) is False and sent == []                       # counted, not sent
    seen()
    assert send(sent) is True
    body = sent[0]
    assert set(body) == {"schema_version", "id", "day", "version", "os", "machine", "started", "finished", "agent"}
    assert len(body["id"]) == 32 and body["day"] == DAY and body["finished"] == {"model": 1}
    assert body["started"] == {"studio": {}, "agent": {}} and body["agent"] is False
    assert str(Path.home()) not in json.dumps(body)


def test_counts_what_was_started_by_whom_and_what_finished(isolated):
    seen()
    start(["api", "generate"])
    start(["api", "chains"], header="agent")
    start(["api", "skins", "abc", "build"], header="agent")
    start(["api", "finish", "runs", "fox", "resume"])
    start(["api", "generate"], code=422)          # refused: nothing started
    start(["api", "assets", "x", "name"])         # not work
    ua.count_done("model", env=ON)
    ua.count_done("model", env=ON)
    ua.count_done("finish", env=ON)
    sent = []
    send(sent)
    assert sent[0]["started"] == {"studio": {"model": 1, "finish": 1}, "agent": {"chain": 1, "skin": 1}}
    assert sent[0]["finished"] == {"model": 2, "finish": 1}
    assert sent[0]["agent"] is True


def test_any_agent_call_counts_as_agent_use_even_one_that_starts_nothing(isolated):
    seen()
    ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)
    sent = []
    send(sent)
    assert sent[0]["agent"] is True and sent[0]["started"]["agent"] == {}


def test_an_agent_polling_does_not_rewrite_the_file_each_time(isolated):
    ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)
    before = isolated.stat().st_mtime_ns
    isolated.touch()
    marked = isolated.stat().st_mtime_ns
    ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)
    assert isolated.stat().st_mtime_ns == marked and marked >= before


def test_once_a_day_and_only_new_counts_go_the_next_day(isolated):
    seen()
    ua.count_done("picture", env=ON)
    sent = []
    assert send(sent) is True
    assert send(sent, at=NOON + 2 * ua.RETRY_SECONDS) is False       # same day
    ua.count_done("picture", env=ON)
    assert send(sent, today="2026-10-26", at=NOON + 86400) is True
    assert [b["finished"] for b in sent] == [{"picture": 1}, {"picture": 1}]
    assert sent[1]["id"] == sent[0]["id"] and sent[1]["agent"] is False


def test_the_about_page_shows_what_is_sent(isolated):
    seen()
    ua.count_done("finish", env=ON)
    shown = ua.status(env=ON)["sends"]
    sent = []
    send(sent, today=shown["day"])
    assert sent == [shown]


def test_offline_keeps_the_counts_and_waits_before_trying_again(isolated):
    seen()
    ua.count_done("finish", env=ON)
    tries = []

    def offline(body):
        tries.append(body)
        raise OSError("no network")

    assert ua.send_daily(env=ON, today=DAY, post=offline, clock=lambda: NOON) is False
    assert ua.send_daily(env=ON, today=DAY, post=offline, clock=lambda: NOON + 60) is False
    assert len(tries) == 1                                            # not one try per job
    sent = []
    assert send(sent, at=NOON + ua.RETRY_SECONDS) is True
    assert sent[0]["finished"] == {"finish": 1}


def test_busy_site_keeps_the_counts_but_a_refusal_drops_them(isolated):
    seen()
    ua.count_done("finish", env=ON)

    def busy(body):
        raise http_error(429)

    assert ua.send_daily(env=ON, today=DAY, post=busy, clock=lambda: NOON) is False
    assert json.loads(isolated.read_text())["finished"] == {"finish": 1}

    def refused(body):
        raise http_error(400)

    assert ua.send_daily(env=ON, today=DAY, post=refused, clock=lambda: NOON + ua.RETRY_SECONDS) is False
    state = json.loads(isolated.read_text())
    assert state["finished"] == {} and state["sent_day"] == DAY       # never resent, never grows


def test_counts_are_capped_in_the_message_and_the_rest_waits(isolated):
    seen()
    ua.count_done("model", env=ON)
    state = json.loads(isolated.read_text())
    state["finished"]["model"] = ua.COUNT_MAX + 5
    isolated.write_text(json.dumps(state))
    sent = []
    send(sent)
    assert sent[0]["finished"] == {"model": ua.COUNT_MAX}
    assert json.loads(isolated.read_text())["finished"] == {"model": 5}


def test_a_job_finished_while_sending_is_not_lost(isolated):
    seen()
    ua.count_done("finish", env=ON)
    ua.send_daily(env=ON, today=DAY, post=lambda body: ua.count_done("finish", env=ON), clock=lambda: NOON)
    assert json.loads(isolated.read_text())["finished"] == {"finish": 1}


def test_an_agent_call_while_sending_waits_for_the_next_message(isolated):
    seen()
    ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)

    def agent_calls(body):
        ua.note_request(["api", "assets"], "agent", "GET", None, env=ON)

    ua.send_daily(env=ON, today=DAY, post=agent_calls, clock=lambda: 0.0)   # the send began before the call
    assert json.loads(isolated.read_text()).get("agent_at")


def test_two_sends_at_once_send_the_counts_once(isolated):
    """Found end to end: a background send and a page-load send both went out with the same counts."""
    seen()
    ua.count_done("model", env=ON)
    sent = []

    def slow(body):
        sent.append(body)
        assert send(sent) is False   # a second send while one is out

    assert ua.send_daily(env=ON, today=DAY, post=slow, clock=lambda: NOON) is True
    assert len(sent) == 1


def test_turning_it_off_forgets_the_id_and_counts_nothing(isolated):
    seen()
    ua.count_done("model", env=ON)
    first = json.loads(isolated.read_text())["id"]
    assert ua.set_sharing(False, env=ON)["enabled"] is False
    assert json.loads(isolated.read_text()) == {"schema_version": ua.SCHEMA, "noticed": True, "opted_out": True}
    ua.count_done("model", env=ON)
    start(["api", "generate"], header="agent")
    assert send([]) is False
    assert json.loads(isolated.read_text()) == {"schema_version": ua.SCHEMA, "noticed": True, "opted_out": True}
    ua.set_sharing(True, env=ON)
    sent = []
    send(sent)
    assert sent[0]["id"] != first and sent[0]["finished"] == {}


def test_the_first_start_notice_shows_until_dismissed_and_keeps_the_id(isolated):
    assert ua.status(env=ON)["noticed"] is False
    ua.count_done("model", env=ON)
    first = json.loads(isolated.read_text())["id"]
    assert ua.set_noticed(env=ON)["noticed"] is True
    state = json.loads(isolated.read_text())
    assert state["id"] == first and state["finished"] == {"model": 1}   # dismissing changes nothing else
    assert ua.status(env=ON)["policy"].startswith("https://assetfurnace.com/privacy")


def test_dismissing_before_anything_is_counted_survives_the_first_count(isolated):
    seen()
    ua.count_done("model", env=ON)
    assert ua.status(env=ON)["noticed"] is True


def test_a_choice_on_about_also_counts_as_seen(isolated):
    assert ua.set_sharing(True, env=ON)["noticed"] is True
    ua.count_done("model", env=ON)
    assert ua.status(env=ON)["noticed"] is True
    assert ua.set_sharing(False, env=ON)["noticed"] is True


def test_no_notice_when_sharing_is_off_by_the_environment(isolated):
    assert ua.status(env=OFF)["noticed"] is True


@pytest.mark.parametrize("env", [{"I3D_NO_USAGE": "1"}, {"DO_NOT_TRACK": "1"}, {"DO_NOT_TRACK": "true"}])
def test_the_environment_switch_wins_over_the_box(isolated, env):
    assert ua.status(env=env) == {**ua.status(env=env), "enabled": False, "allowed": False, "sends": None}
    ua.count_done("model", env=env)
    assert ua.send_daily(env=env, today=DAY, post=lambda b: None) is False
    assert ua.set_sharing(True, env=env)["enabled"] is False
    assert not isolated.exists()


@pytest.mark.parametrize("broken", ["", "not json", "[]"])
def test_a_file_that_cannot_be_read_counts_as_off_and_is_left_alone(isolated, broken):
    """An opt-out must never be lost to a half-written or damaged file."""
    isolated.write_text(broken)
    ua.count_done("model", env=ON)
    start(["api", "generate"], header="agent")
    assert ua.send_daily(env=ON, today=DAY, post=lambda b: None) is False
    assert ua.status(env=ON)["enabled"] is False
    assert isolated.read_text() == broken
    assert ua.set_sharing(True, env=ON)["enabled"] is True            # ticking the box starts afresh


@pytest.mark.parametrize("odd", [{"finished": []}, {"finished": {"model": "x"}}, {"started": "x"},
                                 {"started": {"studio": [1]}}, {"finished": {"model": True}}])
def test_a_hand_edited_file_never_raises_into_a_job(isolated, odd):
    isolated.write_text(json.dumps({"id": "a" * 32, "noticed": True, **odd}))
    ua.count_done("model", env=ON)
    start(["api", "generate"])
    ua.status(env=ON)
    sent = []
    send(sent)
    assert sent and sent[0]["finished"].get("model") == 1


def test_a_usage_fault_never_fails_the_caller(isolated, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ua, "_read", broken)
    ua.count_done("model", env=ON)
    start(["api", "generate"])
    assert ua.status(env=ON)["enabled"] is False


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


def test_send_soon_sends_off_the_request_only_once_the_notice_was_seen(isolated, monkeypatch):
    import threading

    went = threading.Event()
    monkeypatch.setattr(ua, "_post", lambda body: went.set())
    REAL_SEND_SOON(env=ON)
    assert not went.wait(0.3)          # not seen yet: nothing goes
    ua.set_noticed(env=ON)
    REAL_SEND_SOON(env=ON)
    assert went.wait(5)
    for thread in threading.enumerate():
        if thread.name == "usage-send":
            thread.join(5)             # done before the fixture puts STATE back


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

    def post(body, content_type="application/json"):
        return urllib.request.Request(base, data=body, method="POST",
                                      headers={"Content-Type": content_type, "X-AssetFurnace-Request": "studio"})

    try:
        with urllib.request.urlopen(base, timeout=10) as response:
            assert json.loads(response.read())["enabled"] is True
        agent = urllib.request.Request(base, headers={ua.CLIENT_HEADER: "agent"})
        urllib.request.urlopen(agent, timeout=10).read()
        assert json.loads(isolated.read_text())["agent_at"]
        with urllib.request.urlopen(post(b'{"noticed": true}'), timeout=10) as response:
            assert json.loads(response.read())["noticed"] is True
        with pytest.raises(urllib.error.HTTPError) as other_site:
            urllib.request.urlopen(post(b'{"share": false}', "text/plain"), timeout=10)
        assert other_site.value.code == 415
        for flip in (b'{"share": false}', b'{"share": true}'):
            by_agent = post(flip)
            by_agent.add_header(ua.CLIENT_HEADER, "agent")
            with pytest.raises(urllib.error.HTTPError) as agent_refused:
                urllib.request.urlopen(by_agent, timeout=10)
            assert agent_refused.value.code == 403   # only the person turns it on or off
        with urllib.request.urlopen(post(b'{"share": false}'), timeout=10) as response:
            assert json.loads(response.read())["enabled"] is False
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(post(b'{"share": "yes"}'), timeout=10)
        assert refused.value.code == 422
    finally:
        server.shutdown()


def test_a_chain_started_by_an_agent_passes_the_mark_to_each_step(tmp_path):
    import activity_api

    seen_headers = []

    def request(method, path, body, content_type, headers=None):
        seen_headers.append(headers)
        return 202, {"job_id": "j"}

    runner = activity_api.ChainRunner(tmp_path, request=request)
    chain = activity_api.Chain(title="fox", steps=["picture"], prompt="a fox", client="agent")
    runner._start(chain, "picture")
    assert seen_headers == [{ua.CLIENT_HEADER: "agent"}]
    runner._start(activity_api.Chain(title="fox", steps=["picture"], prompt="a fox"), "picture")
    assert seen_headers[-1] is None


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
