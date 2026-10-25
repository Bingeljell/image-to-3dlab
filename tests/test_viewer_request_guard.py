"""The studio server refuses requests that a web page could forge.

A page open in the browser can POST a plain form to http://127.0.0.1:8777 without any
preflight, and DNS rebinding lets it read the answers. These tests drive the real handler
over HTTP: cross-origin and rebinding requests stop before any route runs, while the
studio's own requests, the plan runner and the MCP client still get through.
"""

from __future__ import annotations

import functools
import http.client
import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import activity_api
import generate_api
import request_utils

from image_to_3dlab import mcp_server

NODE = shutil.which("node")
PATHS = ["/api/setup/qwen-image/remove", "/api/hf/sign-in", "/api/image", "/api/chains"]


@pytest.fixture
def server(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("private notes", encoding="utf-8")
    reached = []

    # every POST route runs from _route_post; a refused request must never get there
    def record(self):
        reached.append(self.path)
        self._send_json(299, {"reached": self.path})

    monkeypatch.setattr(generate_api.Handler, "_route_post", record)
    monkeypatch.setattr(generate_api.usage_api, "note_request", lambda *a, **k: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, reached
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(server, method, path, headers=None, body=b"{}"):
    """As the studio page sends it, unless `headers` changes that (None drops a header)."""
    httpd, _ = server
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "studio",
            "Content-Type": "application/json"}
    sent.update(headers or {})
    sent = {k: v for k, v in sent.items() if v is not None}
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=body if method == "POST" else None, headers=sent)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.mark.parametrize("path", PATHS)
def test_a_cross_origin_form_post_is_refused_before_any_route_runs(server, path):
    status, _, body = call(server, "POST", path, {"Origin": "http://evil.example", "Content-Type": "text/plain",
                                                  "X-AssetFurnace-Request": None})
    assert status == 403 and b"other sites" in body
    assert server[1] == []


@pytest.mark.parametrize("headers", [
    {"Origin": "http://evil.example"},
    {"Origin": "null"},
    {"Origin": "http://evil.example", "X-AssetFurnace-Request": None, "X-AssetFurnace-Client": "agent"},
    {"X-AssetFurnace-Request": None},
    {"Origin": None, "X-AssetFurnace-Request": None},
    {"Sec-Fetch-Site": "cross-site"},
    {"Host": "evil.example"},
])
@pytest.mark.parametrize("path", PATHS)
def test_every_foreign_or_missing_header_is_refused(server, path, headers):
    status, _, _ = call(server, "POST", path, headers)
    assert status == 403
    assert server[1] == []


@pytest.mark.parametrize("path", PATHS)
def test_the_studios_own_same_origin_requests_reach_the_route(server, path):
    status, _, _ = call(server, "POST", path)
    assert status == 299
    assert server[1] == [path]


def test_the_mcp_client_reaches_the_route_without_an_origin(server):
    httpd, reached = server
    studio = mcp_server.Studio(f"http://127.0.0.1:{httpd.server_port}", start=lambda: None)
    studio._checked = True                   # skip the "is it running" probe
    assert studio.post_json("api/hf/sign-in", {}) == {"reached": "/api/hf/sign-in"}
    assert reached == ["/api/hf/sign-in"]


def test_the_plan_runner_calls_the_server_as_the_studio_does(server):
    httpd, reached = server
    request = activity_api.local_request(f"http://127.0.0.1:{httpd.server_port}")
    status, _ = request("POST", "/api/image", b"{}", "application/json")
    assert status == 299 and reached == ["/api/image"]


def test_the_pod_smoke_test_calls_the_server_as_the_studio_does(server):
    import pod_smoke_test

    httpd, reached = server
    viewer = pod_smoke_test.Viewer(f"http://127.0.0.1:{httpd.server_port}")
    status, payload = viewer.json("POST", "/api/hf/sign-in", {"token": "x"})
    assert status == 299 and payload == {"reached": "/api/hf/sign-in"}
    assert reached == ["/api/hf/sign-in"]


@pytest.mark.parametrize("path", ["/AGENTS.md", "/api/catalog", "/viewer/studio.html"])
def test_a_dns_rebinding_host_cannot_read_the_repository_or_the_api(server, path):
    status, _, body = call(server, "GET", path, {"Host": "evil.example:8777", "Origin": None})
    assert status == 403
    assert b"private notes" not in body


TUNNEL = "box.tail1234.ts.net"


def test_a_tunnel_name_is_refused_until_it_is_allowed(server, monkeypatch):
    # tailscale serve, cloudflared, ngrok, VS Code: a loopback studio under another name
    port = server[0].server_port
    page = {"Host": TUNNEL, "Origin": f"https://{TUNNEL}"}
    assert call(server, "GET", "/AGENTS.md", {"Host": TUNNEL, "Origin": None})[0] == 403
    assert call(server, "POST", "/api/image", page)[0] == 403
    monkeypatch.setenv("I3D_ALLOWED_HOSTS", f" other.example , {TUNNEL.upper()} ")
    assert call(server, "GET", "/AGENTS.md", {"Host": TUNNEL, "Origin": None})[0] == 200
    assert call(server, "POST", "/api/image", page)[0] == 299
    # a proxy that rewrites Host to the local address still sends the page's own Origin
    assert call(server, "POST", "/api/image", {"Host": f"127.0.0.1:{port}", "Origin": f"https://{TUNNEL}"})[0] == 299
    assert call(server, "POST", "/api/image", {"Host": "evil.example", "Origin": "https://evil.example"})[0] == 403
    assert call(server, "POST", "/api/image", {"Origin": "https://evil.example"})[0] == 403
    assert server[1] == ["/api/image", "/api/image"]


def test_allowed_hosts_are_names_in_any_case():
    assert request_utils.allowed_hosts({"I3D_ALLOWED_HOSTS": " A.example , ,b.example."}) == {"a.example", "b.example"}
    assert request_utils.allowed_hosts({}) == set()


@pytest.mark.parametrize("origin,host,same", [
    ("https://name", "name:443", True),
    ("http://name", "name:80", True),
    ("https://name:443", "name", True),
    ("http://name:80", "NAME", True),
    ("http://127.0.0.1:8777", "127.0.0.1:8777", True),
    ("https://name", "name:80", False),
    ("http://name", "name:443", False),
    ("http://name:8777", "name:8778", False),
    ("http://name:8777", "name", False),
    ("https://other", "name:443", False),
    ("null", "name", False),
    ("http://name:x", "name", False),
])
def test_origin_and_host_match_with_default_ports_left_out(origin, host, same, monkeypatch):
    monkeypatch.delenv("I3D_ALLOWED_HOSTS", raising=False)
    assert request_utils.same_origin(origin, {host.lower()}) is same


def test_head_is_refused_for_a_rebinding_host_too(server):
    status, _, _ = call(server, "HEAD", "/AGENTS.md", {"Host": "evil.example", "Origin": None})
    assert status == 403


def test_a_page_load_still_works_even_from_a_link_on_another_site(server):
    status, _, body = call(server, "GET", "/AGENTS.md", {"Origin": None})
    assert status == 200 and body == b"private notes"
    status, _, _ = call(server, "GET", "/AGENTS.md", {"Origin": None, "Sec-Fetch-Site": "cross-site"})
    assert status == 200


def test_the_studio_pages_load_the_request_marker_before_their_modules():
    for page in ("studio.html", "setup.html"):
        html = (REPO / "viewer" / page).read_text(encoding="utf-8")
        marker = html.index('<script src="./request-header.js"></script>')
        assert marker < html.index('<script type="module"'), page


@pytest.mark.skipif(NODE is None, reason="Node runs the browser script")
def test_the_request_marker_tags_same_origin_writes_only():
    script = (REPO / "viewer" / "request-header.js").read_text(encoding="utf-8")
    program = """
      const sent = [];
      globalThis.window = globalThis;
      globalThis.location = new URL('http://127.0.0.1:8777/viewer/studio.html');
      globalThis.fetch = async (input, init = {}) => {
        sent.push(new Headers(init.headers || (input instanceof Request ? input.headers : undefined))
                    .get('X-AssetFurnace-Request'));
      };
      eval(SCRIPT);
      await fetch('/api/image', { method: 'POST', body: '{}' });
      await fetch('/api/catalog');
      await fetch('https://huggingface.co/x', { method: 'POST' });
      await fetch('/api/x', { method: 'POST', headers: { 'X-AssetFurnace-Request': 'own' } });
      await fetch(new Request('http://127.0.0.1:8777/api/chains', { method: 'POST', body: 'x' }));
      console.log(JSON.stringify(sent));
    """.replace("SCRIPT", json.dumps(script))
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    assert json.loads(result.stdout) == ["studio", None, None, "own", "studio"]
