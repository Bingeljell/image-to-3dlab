"""Who may talk to the studio server: the studio page itself, and local tools, never a web page.

A page open in any browser tab can POST a plain form to http://127.0.0.1:8777 with no
preflight, and DNS rebinding (a site pointing its own name at 127.0.0.1) lets it read the
answers. So:

- `check_host`: a loopback server answers only loopback names (no DNS rebinding), plus the
  names in `I3D_ALLOWED_HOSTS`.
- `check_request`: on top of that, an `Origin` must be the studio's own, the browser must not
  call the fetch cross-site, and a write must carry one of `MARKER_HEADERS`. A plain form
  cannot send a custom header, and a script on another site that adds one sets off a CORS
  preflight this server never answers, so the header proves the request is not a forgery.
  The studio's pages add `X-AssetFurnace-Request` (request-header.js); the MCP server sends
  `X-AssetFurnace-Client`. Any other script that writes must send one of them too.

A loopback studio reached through a proxy under another name (tailscale serve, cloudflared,
ngrok, VS Code port forwarding) needs that name in `I3D_ALLOWED_HOSTS`: host names only, no
port, comma-separated, any case. A studio bound to a non-loopback address (`--host`, a pod)
skips the Host check: it relies on the network being trusted and is not protected against
DNS rebinding. The tests cover loopback and the allow-list; RunPod's proxy was not tested.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

# custom headers no cross-site page can send without a preflight
MARKER_HEADERS = ("X-AssetFurnace-Request", "X-AssetFurnace-Client")
ALLOWED_HOSTS_ENV = "I3D_ALLOWED_HOSTS"
DEFAULT_PORTS = {"http": 80, "https": 443}


def local_address(value: str) -> bool:
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def allowed_hosts(environ=None) -> set[str]:
    """Extra host names a loopback studio answers to, from `I3D_ALLOWED_HOSTS`."""
    raw = (os.environ if environ is None else environ).get(ALLOWED_HOSTS_ENV, "")
    return {name.strip().lower().rstrip(".") for name in raw.split(",") if name.strip()}


def check_host(handler) -> str:
    """The Host a request names. Raises PermissionError when a loopback server is asked
    under another name that `I3D_ALLOWED_HOSTS` does not list: that is a page that rebound
    its own domain to 127.0.0.1."""
    host = handler.headers.get("Host", "")
    parsed = urlsplit("//" + host)
    if not parsed.hostname or parsed.username or parsed.path or parsed.query or parsed.fragment:
        raise PermissionError("Invalid request host")
    if (local_address(handler.server.server_address[0]) and not local_address(parsed.hostname)
            and parsed.hostname.rstrip(".") not in allowed_hosts()):
        raise PermissionError("Use the local studio address, or add this name to I3D_ALLOWED_HOSTS")
    return host


def same_origin(origin: str, hosts: set[str]) -> bool:
    """Whether `origin` is the studio's own: a host in `hosts` ("name" or "name:port", as a
    Host header gives it) or a name in `I3D_ALLOWED_HOSTS`. A default port counts as no
    port, so https://name matches name:443 and http://name matches name:80."""
    source = urlsplit(origin)
    if (source.scheme not in DEFAULT_PORTS or not source.hostname or source.username
            or source.path or source.query or source.fragment):
        return False
    try:
        port = source.port or DEFAULT_PORTS[source.scheme]
    except ValueError:
        return False
    name = source.hostname.rstrip(".")
    if name in allowed_hosts():
        return True
    for host in hosts:
        parsed = urlsplit("//" + host)
        try:
            host_port = parsed.port or DEFAULT_PORTS[source.scheme]
        except ValueError:
            continue
        if (parsed.hostname or "").rstrip(".") == name and host_port == port:
            return True
    return False


def check_request(handler, *, mutation: bool) -> None:
    """Raises PermissionError unless the request comes from the studio page or a local tool.

    Reads (``mutation=False``) only need a good Host. Writes also need: an Origin, if any,
    that is the studio's own; no cross-site fetch; and one of `MARKER_HEADERS`."""
    host = check_host(handler)
    if not mutation:
        return
    allowed = {host.lower()}
    # behind a proxy (RunPod) the page's origin is the public name the proxy forwards
    forwarded = handler.headers.get("X-Forwarded-Host")
    if forwarded and not local_address(handler.server.server_address[0]):
        allowed.add(forwarded.split(",")[0].strip().lower())
    origin = handler.headers.get("Origin")
    if origin is not None and not same_origin(origin, allowed):
        raise PermissionError("Use the studio page: requests from other sites are refused")
    if (handler.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        raise PermissionError("Cross-site requests are refused")
    if not any(handler.headers.get(name) for name in MARKER_HEADERS):
        raise PermissionError("Use the studio page: this request lacks its header")
