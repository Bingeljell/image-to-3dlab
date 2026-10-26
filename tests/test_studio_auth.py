"""A studio reachable from a network (a RunPod pod, --host, a tunnel) lets in only its key.

Unit tests for studio_auth, then the real handler over HTTP. The test server listens on
127.0.0.1, which is exactly the case that matters: a proxy on the same machine connects from
there too, so coming from this machine must not be enough. Tools on the machine (the plan
runner, the MCP) send the key, read from the file the studio writes.
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
def test_a_studio_reachable_from_a_network_or_a_tunnel_needs_a_key():
    assert studio_auth.needs_token("0.0.0.0", {}) is True
    assert studio_auth.needs_token("100.64.1.2", {}) is True
    assert studio_auth.needs_token("127.0.0.1", {}) is False
    assert studio_auth.needs_token("localhost", {}) is False
    assert studio_auth.needs_token("127.0.0.1", {"I3D_ALLOWED_HOSTS": "box.tail1234.ts.net"}) is True, "a tunnel"


def test_the_key_is_random_unless_set_and_a_set_key_must_be_safe():
    a, b = studio_auth.make_token({}), studio_auth.make_token({})
    assert a != b and len(a) >= 30
    assert studio_auth.make_token({"I3D_STUDIO_TOKEN": f" {TOKEN} "}) == TOKEN
    for bad in ("short", "has space in it 123456", "semi;colon-1234567890", "new\nline-1234567890"):
        with pytest.raises(ValueError):
            studio_auth.make_token({"I3D_STUDIO_TOKEN": bad})


def test_a_studio_without_a_key_lets_everyone_in_as_before():
    assert studio_auth.check(token=None, path="/", headers=PROXY).ok


def test_a_link_with_the_key_sets_a_cookie_and_drops_the_key_from_the_address():
    v = studio_auth.check(token=TOKEN, path=f"/viewer/studio.html?tab=make&token={TOKEN}", headers=PROXY)
    assert v.ok and v.redirect == "/viewer/studio.html?tab=make"
    assert v.set_cookie.startswith(f"af_studio={TOKEN};") and "HttpOnly" in v.set_cookie and "Secure" in v.set_cookie


def test_the_redirect_never_leaves_the_studio():
    for path in ("/\\evil.com", "//evil.com", "/\\/evil.com"):
        assert studio_auth.check(token=TOKEN, path=f"{path}?token={TOKEN}", headers={}).redirect == "/", path


def test_the_wrong_key_or_none_is_refused_even_from_this_machine():
    for path in ("/viewer/studio.html", "/viewer/studio.html?token=nope"):
        assert not studio_auth.check(token=TOKEN, path=path, headers={}).ok
    assert not studio_auth.check(token=TOKEN, path="/", headers={"Cookie": "af_studio=nope"}).ok
    assert studio_auth.check(token=TOKEN, path="/", headers={"Authorization": f"Bearer {TOKEN}"}).ok


def test_the_key_file_is_private_and_tools_read_it(tmp_path):
    key = tmp_path / ".studio-token"
    studio_auth.save_token(TOKEN, key)
    assert key.read_text() == TOKEN
    if sys.platform != "win32":
        assert key.stat().st_mode & 0o077 == 0, "readable by its owner only"
    assert studio_auth.local_token({}, key) == TOKEN
    assert studio_auth.local_token({"I3D_STUDIO_TOKEN": "other-key-123456789"}, key) == "other-key-123456789"
    studio_auth.forget_token(key)
    assert studio_auth.local_token({}, key) is None


def test_dotfiles_are_hidden():
    for path in ("/.env", "/.git/config", "/output/.hidden/x.glb", "/a/.DS_Store?x=1", "/%2Eenv", "/output/.assets-meta.json",
                 "/viewer/.thumbs/x.png", "/output/a/.thumbs/x.png", "/%2F.git/config", "//.env", "/a%2f..%2f.git%2fconfig",
                 "/output/.thumbs/../../.env", "/a\\..\\.env", "/output/.studio-token"):
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


def test_coming_from_this_machine_is_not_enough(studio):
    """A proxy on the same machine connects from 127.0.0.1 with no forwarding headers."""
    for method, path in (("GET", "/viewer/page.html"), ("POST", "/api/chains")):
        status, _, _ = call(studio, method, path)
        assert status == 401, (method, path)


def test_the_plan_runner_sends_the_key(studio):
    import activity_api
    status, _ = activity_api.local_request(f"http://127.0.0.1:{studio.server_port}", TOKEN)("POST", "/api/chains", b"{}", "application/json")
    assert status == 299
    status, _ = activity_api.local_request(f"http://127.0.0.1:{studio.server_port}")("POST", "/api/chains", b"{}", "application/json")
    assert status == 401


def test_dotfiles_are_never_served_even_with_the_key(studio):
    status, _, body = call(studio, "GET", "/.env", {"Authorization": f"Bearer {TOKEN}"})
    assert status == 404 and b"secret" not in body


def test_an_agent_sends_the_key_from_the_environment_or_the_studios_file(monkeypatch, tmp_path):
    from image_to_3dlab import mcp_server

    monkeypatch.setenv("I3D_STUDIO_TOKEN", TOKEN)
    assert mcp_server.Studio("https://pod-8777.proxy.runpod.net").http.headers["Authorization"] == f"Bearer {TOKEN}"
    monkeypatch.delenv("I3D_STUDIO_TOKEN")
    key = tmp_path / ".studio-token"
    key.write_text(TOKEN)
    assert mcp_server.studio_key({}, key) == TOKEN, "an agent on the pod reads the running studio's key"
    monkeypatch.setattr(mcp_server, "STUDIO_KEY_FILE", tmp_path / "none")
    assert mcp_server.studio_key({}, tmp_path / "none") is None
