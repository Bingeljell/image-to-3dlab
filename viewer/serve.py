#!/usr/bin/env python3
"""Serve the repo so the mesh viewer can load GLBs by relative path.

    python viewer/serve.py                      # http://127.0.0.1:8777
    python viewer/serve.py --open A.glb B.glb   # and print/launch a compare URL

Bound to 127.0.0.1 only: this serves the whole repository, including `output/`, and has no
business being reachable from the network.

The viewer exists because judging a mesh by eye is the acceptance test in this repo, and
every such judgement previously required opening Blender. It is also driveable by browser
automation, which means the agent can look at its own output instead of asking.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path
from typing import ClassVar

REPO = Path(__file__).resolve().parents[1]


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, with the two headers a GLB viewer actually needs."""

    extensions_map: ClassVar[dict[str, str]] = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".js": "text/javascript",
    }

    def end_headers(self) -> None:
        # Meshes are large and change in place; a cached GLB would silently show the old one.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:  # quieter than the default
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def compare_url(assets: list[str], port: int, labels: list[str] | None = None) -> str:
    """A viewer URL for up to three assets, paths relative to the repo root.

    Absolute paths inside the repo are rewritten to relative; anything outside it would not
    be served, so it is reported rather than silently 404ing in the browser.
    """
    params: dict[str, str] = {}
    for index, asset in enumerate(assets[:3]):
        path = Path(asset)
        if path.is_absolute():
            try:
                path = path.relative_to(REPO)
            except ValueError as exc:
                raise ValueError(f"{asset} is outside the repo and cannot be served") from exc
        key = "abc"[index]
        params[key] = str(path)
        if labels and index < len(labels):
            params["l" + key] = labels[index]
    return f"http://127.0.0.1:{port}/viewer/index.html?" + urllib.parse.urlencode(params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--open", nargs="*", metavar="GLB", help="assets to compare")
    parser.add_argument("--labels", nargs="*", help="captions, one per asset")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = compare_url(args.open, args.port, args.labels) if args.open else \
        f"http://127.0.0.1:{args.port}/viewer/index.html"
    print(url, flush=True)

    handler = functools.partial(Handler, directory=str(REPO))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        if args.open and not args.no_browser:
            webbrowser.open(url)
        print(f"serving {REPO} — ctrl-c to stop", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
