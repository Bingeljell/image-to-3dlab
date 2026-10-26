"""A key for a studio reachable from a network (a RunPod pod, `--host` on a LAN).

On your own machine the studio listens on 127.0.0.1 and needs no key. Anywhere else, anyone
who learns the address could use it, so `serve.py` makes a random key at start-up and prints a
link carrying it (`?token=...`), like Jupyter does. Opening that link once stores the key in a
cookie; every request without the key is refused. Tools on the same machine (the MCP server,
the plan runner, the CLI) connect from 127.0.0.1 with no proxy in between, and need no key.

Pure functions, so the tests import them; generate_api.Handler calls `check`.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
from http.cookies import SimpleCookie
from typing import NamedTuple
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

COOKIE = "af_studio"
TOKEN_ENV = "I3D_STUDIO_TOKEN"
# headers a proxy adds: a request carrying one came from outside, even if it reached us via 127.0.0.1
FORWARD_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP", "Forwarded", "CF-Connecting-IP")


def loopback(address: str) -> bool:
    if address.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def make_token(environ) -> str:
    """The key: `I3D_STUDIO_TOKEN` if set (so a restart keeps the same link), else a fresh random one."""
    return (environ.get(TOKEN_ENV) or "").strip() or secrets.token_urlsafe(24)


def needs_token(bind_host: str) -> bool:
    """Only a studio listening beyond this machine needs a key."""
    return not loopback(bind_host)


def link_with_token(url: str, token: str) -> str:
    parts = urlsplit(url)
    query = [*parse_qsl(parts.query), ("token", token)]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _same(a: str | None, b: str) -> bool:
    return bool(a) and hmac.compare_digest(a.encode(), b.encode())


class Verdict(NamedTuple):
    ok: bool
    redirect: str | None = None      # the token came in the address: send them on without it
    set_cookie: str | None = None


def check(*, token: str | None, client_address: str, path: str, headers) -> Verdict:
    """May this request in? `token` is None on a loopback studio (no key needed)."""
    if token is None:
        return Verdict(True)
    if loopback(client_address) and not any(headers.get(h) for h in FORWARD_HEADERS):
        return Verdict(True)                                        # a tool on this machine
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get("Cookie") or "")
    except Exception:
        cookie = SimpleCookie()                                     # a garbled Cookie header: as if none
    if COOKIE in cookie and _same(cookie[COOKIE].value, token):
        return Verdict(True)
    auth = headers.get("Authorization") or ""
    if auth.startswith("Bearer ") and _same(auth[7:].strip(), token):
        return Verdict(True)                                        # scripts and agents elsewhere
    parts = urlsplit(path)
    query = parse_qsl(parts.query, keep_blank_values=True)
    given = [v for k, v in query if k == "token"]
    if given and _same(given[-1], token):
        clean = urlunsplit(("", "", parts.path, urlencode([(k, v) for k, v in query if k != "token"]), ""))
        secure = (headers.get("X-Forwarded-Proto") or "").lower() == "https"
        cookie_text = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000" + ("; Secure" if secure else "")
        return Verdict(True, redirect=clean or "/", set_cookie=cookie_text)
    return Verdict(False)


# the studio's own picture caches inside output/, which its pages load
SERVED_DOT_FOLDERS = {".thumbs", ".previews", ".fix-previews"}


def hidden_path(path: str) -> bool:
    """`.env`, `.git/config` and any other dotfile or dot-folder: never served. The exceptions
    are the studio's picture caches directly under output/ (output/.thumbs/...)."""
    parts = [unquote(p) for p in urlsplit(path).path.split("/") if p]
    for i, part in enumerate(parts):
        if part.startswith(".") and not (i == 1 and parts[0] == "output" and part in SERVED_DOT_FOLDERS):
            return True
    return False
