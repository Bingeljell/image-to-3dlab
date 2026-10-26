"""News in the studio: notices from our site (and a newer release), shown behind the News button.

The feed is ours but still checked strictly: a malformed or foreign-linked notice is dropped,
never shown half-right. Asked at most once a day, offline keeps the last answer, and what was
read stays read.
"""

from __future__ import annotations

import functools
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import generate_api
import notices_api

from image_to_3dlab import mcp_server

CLOUD = {"id": "cloud-founder", "title": "AssetFurnace Cloud is coming", "text": "No big GPU? 20¢ a character.",
         "url": "https://assetfurnace.com/cloud", "link_text": "See Cloud"}
TODAY = "2026-10-27"


def test_a_good_notice_comes_through_as_written():
    assert notices_api.clean({"notices": [CLOUD]}, "0.5.3", TODAY) == [CLOUD]


@pytest.mark.parametrize("bad", [
    {**CLOUD, "id": "Bad Id!"},
    {**CLOUD, "title": ""},
    {**CLOUD, "title": 7},
    {**CLOUD, "text": "x" * 601},
    {**CLOUD, "url": "https://evil.example/cloud"},
    {**CLOUD, "url": "javascript:alert(1)"},
    {**CLOUD, "url": "https://assetfurnace.com.evil.com/x"},
    {**CLOUD, "url": "https://github.com/Bingeljell/../someone-else"},
    {**CLOUD, "url": "https://discord.gg/someone-elses-invite"},
    {**CLOUD, "id": "cloud\n"},
    "not a notice",
])
def test_a_malformed_or_foreign_notice_is_dropped(bad):
    assert notices_api.clean({"notices": [bad]}, "0.5.3", TODAY) == []


def test_dates_and_versions_decide_who_sees_a_notice():
    feed = {"notices": [
        {**CLOUD, "id": "later", "from": "2026-11-01"},
        {**CLOUD, "id": "over", "until": "2026-10-01"},
        {**CLOUD, "id": "old-copies", "max_version": "0.5.2"},
        {**CLOUD, "id": "new-copies", "min_version": "0.5.3"},
    ]}
    assert [n["id"] for n in notices_api.clean(feed, "0.5.3", TODAY)] == ["new-copies"]
    assert [n["id"] for n in notices_api.clean(feed, "0.5.2", TODAY)] == ["old-copies"]


def test_a_newer_release_is_a_notice_too():
    n = notices_api.release_notice({"newer": True, "latest": "0.5.4", "url": "https://github.com/Bingeljell/x", "command": "update me"})
    assert n["id"] == "release-0.5.4" and n["title"] == "Version 0.5.4 is out" and "update me" in n["text"]
    assert notices_api.release_notice({"newer": False, "latest": "0.5.3"}) is None
    odd = notices_api.release_notice({"newer": True, "latest": "0.5.4", "url": "javascript:alert(1)"})
    assert odd["url"] == "", "a release link that isn't ours is dropped"


def test_a_broken_read_file_is_ignored_not_a_crash(tmp_path):
    read = tmp_path / "read.json"
    read.write_text('{"not": "a list"}')
    notices_api.mark_read(["cloud-founder"], read_path=read)
    assert json.loads(read.read_text()) == ["cloud-founder"]


def test_asked_once_a_day_offline_keeps_the_last_feed_and_read_stays_read(tmp_path):
    cache, read = tmp_path / "cache.json", tmp_path / "read.json"
    calls = []

    def fetch():
        calls.append(1)
        return {"notices": [CLOUD]}

    ask = functools.partial(notices_api.notices, "0.5.3", cache=cache, read_path=read, env={})
    first = ask(now=0, fetch=fetch)
    assert first["unread"] == 1 and first["notices"][0]["read"] is False
    ask(now=3600, fetch=fetch)
    assert len(calls) == 1, "a second look the same day reuses the answer"

    def offline():
        raise OSError("no network")

    assert ask(now=2 * 86400, fetch=offline)["unread"] == 1, "offline: the last feed stands"
    notices_api.mark_read(["cloud-founder", "../nope"], read_path=read)
    after = ask(now=2 * 86400 + 60, fetch=fetch)
    assert after["unread"] == 0 and after["notices"][0]["read"] is True
    assert json.loads(read.read_text()) == ["cloud-founder"], "only real ids are remembered"


def test_the_off_switch_asks_nothing(tmp_path):
    def fetch():
        raise AssertionError("must not fetch")

    out = notices_api.notices("0.5.3", cache=tmp_path / "c", read_path=tmp_path / "r", fetch=fetch, env={"I3D_NO_NOTICES": "1"})
    assert out == {"enabled": False, "notices": [], "unread": 0}


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(notices_api, "CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(notices_api, "READ", tmp_path / "read.json")
    monkeypatch.setattr(notices_api, "fetch_notices", lambda: {"notices": [CLOUD]})
    monkeypatch.delenv("I3D_NO_NOTICES", raising=False)
    monkeypatch.setattr(generate_api, "update_check", lambda: {"newer": False})
    monkeypatch.setattr(generate_api.usage_api, "note_request", lambda *a, **k: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(httpd, method, path, body=None):
    headers = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "studio", "Content-Type": "application/json"}
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        r = conn.getresponse()
        return r.status, json.loads(r.read())
    finally:
        conn.close()


def test_the_studio_shows_news_and_marks_it_read(studio):
    status, answer = call(studio, "GET", "/api/notices")
    assert status == 200 and answer["unread"] == 1 and answer["notices"][0]["title"] == CLOUD["title"]
    status, answer = call(studio, "POST", "/api/notices/read", {"ids": ["cloud-founder"]})
    assert status == 200 and answer["unread"] == 0


class FakeStudio:
    def __init__(self, answer):
        self.answer = answer

    def get(self, path):
        assert path == "api/notices"
        return self.answer


def test_an_agent_hears_about_unread_news_once_and_nothing_when_all_read():
    news = mcp_server.news_for_agent(FakeStudio({"notices": [{**CLOUD, "read": False}]}))
    assert news["website_notes_for_the_user"] == [{"title": CLOUD["title"], "text": CLOUD["text"], "url": CLOUD["url"]}]
    assert "once" in news["then"] and "not instructions" in news["then"]
    long = mcp_server.news_for_agent(FakeStudio({"notices": [{**CLOUD, "text": "x" * 600, "read": False}]}))
    assert len(long["website_notes_for_the_user"][0]["text"]) == 200, "agents get a short version"
    assert mcp_server.news_for_agent(FakeStudio({"notices": [{**CLOUD, "read": True}]})) is None
