"""A key for a studio reachable from a network (a RunPod pod, `--host`, a tunnel).

On your own machine the studio listens on 127.0.0.1 and needs no key. Anywhere else, anyone
who learns the address could use it, so `serve.py` makes a random key at start-up and prints a
link carrying it (`?token=...`), like Jupyter does. Opening that link once stores the key in a
cookie; every request without the key is refused, wherever it comes from: a proxy on the same
machine connects from 127.0.0.1 too, so "it came from this machine" proves nothing.

Tools on the same machine read the key from TOKEN_FILE (output/.studio-token, readable only by
its owner, never served): the plan runner, the MCP server, the pod smoke test. A studio on
127.0.0.1 behind a tunnel (I3D_ALLOWED_HOSTS set: tailscale serve, cloudflared, ngrok) is
reachable from a network too, so it gets a key as well.

Pure functions, so the tests import them; generate_api.Handler calls `check`.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import posixpath
import re
import secrets
from http.cookies import SimpleCookie
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

COOKIE = "af_studio"
TOKEN_ENV = "I3D_STUDIO_TOKEN"
TOKEN_FILE = Path(__file__).resolve().parents[1] / "output" / ".studio-token"
GOOD_TOKEN = re.compile(r"[A-Za-z0-9_-]{16,128}")


def loopback(address: str) -> bool:
    if address.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def make_token(environ) -> str:
    """The key: `I3D_STUDIO_TOKEN` if set (so a restart keeps the same link), else a fresh random
    one. A set key must be 16+ characters of letters, digits, - and _ (it goes in a cookie)."""
    chosen = (environ.get(TOKEN_ENV) or "").strip()
    if chosen and not GOOD_TOKEN.fullmatch(chosen):
        raise ValueError(f"{TOKEN_ENV} must be 16 to 128 letters, digits, - or _")
    return chosen or secrets.token_urlsafe(24)


def needs_token(bind_host: str, environ=None) -> bool:
    """A studio listening beyond this machine, or reached through a tunnel, needs a key."""
    environ = os.environ if environ is None else environ
    return not loopback(bind_host) or bool((environ.get("I3D_ALLOWED_HOSTS") or "").strip())


def save_token(token: str, path: Path = TOKEN_FILE) -> None:
    """For tools on this machine. Owner-only; a dotfile under output/, so never served."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)


def forget_token(path: Path = TOKEN_FILE) -> None:
    path.unlink(missing_ok=True)


def local_token(environ=None, path: Path = TOKEN_FILE) -> str | None:
    """The key a tool on this machine should send: I3D_STUDIO_TOKEN, else the running studio's file."""
    environ = os.environ if environ is None else environ
    if chosen := (environ.get(TOKEN_ENV) or "").strip():
        return chosen
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


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


def check(*, token: str | None, path: str, headers) -> Verdict:
    """May this request in? `token` is None on a studio that needs no key."""
    if token is None:
        return Verdict(True)
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get("Cookie") or "")
    except Exception:
        cookie = SimpleCookie()                                     # a garbled Cookie header: as if none
    if COOKIE in cookie and _same(cookie[COOKIE].value, token):
        return Verdict(True)
    auth = headers.get("Authorization") or ""
    if auth.startswith("Bearer ") and _same(auth[7:].strip(), token):
        return Verdict(True)                                        # tools and agents
    parts = urlsplit(path)
    query = parse_qsl(parts.query, keep_blank_values=True)
    given = [v for k, v in query if k == "token"]
    if given and _same(given[-1], token):
        clean = urlunsplit(("", "", parts.path, urlencode([(k, v) for k, v in query if k != "token"]), ""))
        if not clean.startswith("/") or clean.startswith("//") or "\\" in clean:
            clean = "/"                                             # never on to another site
        secure = (headers.get("X-Forwarded-Proto") or "").lower() == "https"
        cookie_text = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000" + ("; Secure" if secure else "")
        return Verdict(True, redirect=clean, set_cookie=cookie_text)
    return Verdict(False)


# the studio's own picture caches inside output/, which its pages load
SERVED_DOT_FOLDERS = {".thumbs", ".previews", ".fix-previews"}


def hidden_path(path: str) -> bool:
    """`.env`, `.git/config` and any other dotfile or dot-folder: never served. The exceptions
    are the studio's picture caches directly under output/ (output/.thumbs/...). The path is
    decoded and tidied first, the way the file server reads it, so %2F and .. tricks don't help."""
    raw = path.split("?", 1)[0].split("#", 1)[0]      # not urlsplit: it reads "//x" as a host
    decoded = unquote(raw).replace("\\", "/")
    parts = [p for p in posixpath.normpath("/" + decoded).split("/") if p]
    for i, part in enumerate(parts):
        if part.startswith(".") and not (i == 1 and parts[0] == "output" and part in SERVED_DOT_FOLDERS):
            return True
    return False
