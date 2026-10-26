"""A studio reachable from a network (a RunPod pod, --host) lets in only its printed link.

Unit tests for studio_auth, then the real handler over HTTP. A request that comes through a
proxy carries X-Forwarded-For, which is how these tests stand in for a stranger on the pod's
public address; a tool on the machine itself (MCP, the plan runner) sends none and gets in.
"""

from __future__ import annotations

import functools
import http.client
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import generate_api
import serve
import studio_auth

TOKEN = "s3cret-key-for-tests"
PROXY = {"X-Forwarded-For": "203.0.113.7", "X-Forwarded-Proto": "https"}


# --- studio_auth on its own -----------------------------------------------------------------
def test_only_a_studio_listening_beyond_this_machine_needs_a_key():
    assert studio_auth.needs_token("0.0.0.0") is True
    assert studio_auth.needs_token("100.64.1.2") is True
    assert studio_auth.needs_token("127.0.0.1") is False
    assert studio_auth.needs_token("localhost") is False


def test_the_key_is_random_unless_set():
    a, b = studio_auth.make_token({}), studio_auth.make_token({})
    assert a != b and len(a) >= 30
    assert studio_auth.make_token({"I3D_STUDIO_TOKEN": " mine "}) == "mine"


def test_a_loopback_studio_lets_everyone_in_as_before():
    assert studio_auth.check(token=None, client_address="203.0.113.7", path="/", headers=PROXY).ok


def test_a_link_with_the_key_sets_a_cookie_and_drops_the_key_from_the_address():
    v = studio_auth.check(token=TOKEN, client_address="10.0.0.5", path=f"/viewer/studio.html?tab=make&token={TOKEN}", headers=PROXY)
    assert v.ok and v.redirect == "/viewer/studio.html?tab=make"
    assert v.set_cookie.startswith(f"af_studio={TOKEN};") and "HttpOnly" in v.set_cookie and "Secure" in v.set_cookie


def test_the_wrong_key_or_none_is_refused():
    for path in ("/viewer/studio.html", "/viewer/studio.html?token=nope"):
        assert not studio_auth.check(token=TOKEN, client_address="10.0.0.5", path=path, headers=PROXY).ok
    assert not studio_auth.check(token=TOKEN, client_address="10.0.0.5", path="/", headers={"Cookie": "af_studio=nope"}).ok


def test_a_tool_on_this_machine_gets_in_but_a_proxied_stranger_does_not():
    assert studio_auth.check(token=TOKEN, client_address="127.0.0.1", path="/api/chains", headers={}).ok
    assert not studio_auth.check(token=TOKEN, client_address="127.0.0.1", path="/api/chains", headers=PROXY).ok
    assert studio_auth.check(token=TOKEN, client_address="10.0.0.5", path="/", headers={"Authorization": f"Bearer {TOKEN}"}).ok


def test_dotfiles_are_hidden():
    for path in ("/.env", "/.git/config", "/output/.hidden/x.glb", "/a/.DS_Store?x=1", "/%2Eenv", "/output/.assets-meta.json",
                 "/viewer/.thumbs/x.png", "/output/a/.thumbs/x.png"):
        assert studio_auth.hidden_path(path), path
    for path in ("/viewer/studio.html", "/output/.thumbs/a.png", "/output/.previews/b.png", "/output/.fix-previews/c.png"):
        assert not studio_auth.hidden_path(path), path


def test_the_pod_link_carries_the_key():
    plan = serve.launch_plan({"RUNPOD_POD_ID": "abc123"}, port=8777, token=TOKEN)
    assert f"https://abc123-8777.proxy.runpod.net/viewer/studio.html?token={TOKEN}" in plan.message
    assert "private" in plan.message


# --- the real handler ----------------------------------------------------------------------
@pytest.fixture
def studio(tmp_path, monkeypatch):
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "page.html").write_text("<p>studio</p>", encoding="utf-8")
    (tmp_path / ".env").write_text("HF_TOKEN=secret", encoding="utf-8")
    monkeypatch.setattr(generate_api.Handler, "studio_token", TOKEN)
    monkeypatch.setattr(generate_api.Handler, "_route_post", lambda self: self._send_json(299, {"reached": self.path}))
    monkeypatch.setattr(generate_api.usage_api, "note_request", lambda *a, **k: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(httpd, method, path, headers=None):
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "studio", **(headers or {})}
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=b"{}" if method == "POST" else None, headers=sent)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


def test_a_stranger_through_the_proxy_gets_nothing(studio):
    for method, path in (("GET", "/viewer/page.html"), ("GET", "/api/catalog"), ("POST", "/api/chains")):
        status, _, body = call(studio, method, path, PROXY)
        assert status == 401 and b"key" in body, (method, path)


def test_the_printed_link_lets_you_in_and_the_cookie_keeps_you_in(studio):
    status, headers, _ = call(studio, "GET", f"/viewer/page.html?token={TOKEN}", PROXY)
    assert status == 302 and headers["Location"] == "/viewer/page.html"
    cookie = headers["Set-Cookie"].split(";")[0]
    status, _, body = call(studio, "GET", "/viewer/page.html", {**PROXY, "Cookie": cookie})
    assert status == 200 and b"studio" in body
    status, _, _ = call(studio, "POST", "/api/chains", {**PROXY, "Cookie": cookie})
    assert status == 299


def test_the_plan_runner_and_mcp_on_the_pod_still_get_in(studio):
    status, _, _ = call(studio, "POST", "/api/chains")
    assert status == 299


def test_dotfiles_are_never_served_even_with_the_key(studio):
    status, _, body = call(studio, "GET", "/.env", {"Authorization": f"Bearer {TOKEN}"})
    assert status == 404 and b"secret" not in body


def test_an_agent_elsewhere_sends_the_key_when_it_has_one(monkeypatch):
    """ASSETFURNACE_URL pointing at a pod plus I3D_STUDIO_TOKEN: the MCP's requests carry the key."""
    from image_to_3dlab import mcp_server

    monkeypatch.setenv("I3D_STUDIO_TOKEN", TOKEN)
    assert mcp_server.Studio("https://pod-8777.proxy.runpod.net").http.headers["Authorization"] == f"Bearer {TOKEN}"
    monkeypatch.delenv("I3D_STUDIO_TOKEN")
    assert "Authorization" not in mcp_server.Studio().http.headers
