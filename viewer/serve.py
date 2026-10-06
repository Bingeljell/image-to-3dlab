#!/usr/bin/env python3
"""Serve the repo so the mesh viewer can load GLBs by relative path.

    python viewer/serve.py                      # http://127.0.0.1:8777
    python viewer/serve.py --open A.glb B.glb   # and print/launch a compare URL
    python viewer/serve.py --host 100.x.y.z      # explicit Tailscale/LAN interface

Bound to 127.0.0.1 by default. An explicit --host exposes the whole repository, including
`output/`, on that interface; use a private interface protected by appropriate ACLs.

The viewer exists because judging a mesh by eye is the acceptance test in this repo, and
every such judgement previously required opening Blender. It is also driveable by browser
automation, which means the agent can look at its own output instead of asking.
"""

from __future__ import annotations

import argparse
import os
import http.server
import signal
import socketserver
import sys
import urllib.parse
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

# Sibling import: works when run as `python viewer/serve.py` (script dir on sys.path)
# and when loaded by the test suite via importlib (script dir not on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_api import OUTPUT_ROOT, Handler, _reconcile_orphaned_jobs, _terminate_active_job

REPO = Path(__file__).resolve().parents[1]


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Allow SSE and static assets while the one generator job runs."""

    daemon_threads = True
    allow_reuse_address = True


class StaticViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Serve viewer assets only; no repository files or generation endpoints."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **Handler.extensions_map}

    def do_POST(self) -> None:
        self.send_error(405, "Static viewer mode does not accept jobs")


def compare_url(
    assets: list[str],
    port: int,
    labels: list[str] | None = None,
    *,
    host: str = "127.0.0.1",
) -> str:
    """A viewer URL for up to three assets, paths relative to the repo root.

    Absolute paths inside the repo are rewritten to relative; anything outside it would not
    be served, so it is reported rather than silently 404ing in the browser.
    """
    params: dict[str, str] = {}
    for index, asset in enumerate(assets[:4]):
        path = Path(asset)
        if path.is_absolute():
            try:
                path = path.relative_to(REPO)
            except ValueError as exc:
                raise ValueError(f"{asset} is outside the repo and cannot be served") from exc
        key = "abcd"[index]
        params[key] = str(path)
        if labels and index < len(labels):
            params["l" + key] = labels[index]
    return f"http://{host}:{port}/viewer/index.html?" + urllib.parse.urlencode(params)


class LaunchPlan(NamedTuple):
    host: str
    open_browser: bool
    message: str


def launch_plan(env: Mapping[str, str], port: int = 8777) -> LaunchPlan:
    """How `./lab` (serve.py --auto) should listen, for a new user who just ran the installer.

    On their own machine: privately, and open the browser. On a RunPod pod: on all
    interfaces, because RunPod's HTTP proxy is the only way in, and print that link. Over
    plain SSH: still privately, with the tunnel command, because listening on a network by
    default would expose the whole repository to everyone on it.
    """
    local = f"http://127.0.0.1:{port}/viewer/studio.html"
    pod = env.get("RUNPOD_POD_ID")
    if pod:
        return LaunchPlan("0.0.0.0", False, (
            f"Open https://{pod}-{port}.proxy.runpod.net/viewer/studio.html\n"
            f"(the pod needs HTTP port {port} exposed; add it in the pod's settings if not)"))
    if env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"):
        return LaunchPlan("127.0.0.1", False, (
            f"This is a remote machine. On your own computer, run\n"
            f"    ssh -L {port}:127.0.0.1:{port} <this machine>\n"
            f"then open {local}"))
    return LaunchPlan("127.0.0.1", True, f"Opening {local}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="interface address to bind")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--open", nargs="*", metavar="GLB", help="assets to compare")
    parser.add_argument("--labels", nargs="*", help="captions, one per asset")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="pick how to listen for this machine (what ./lab uses)")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="serve only viewer assets and disable generation endpoints",
    )
    args = parser.parse_args()
    plan = launch_plan(os.environ, args.port) if args.auto else None
    if plan is not None:
        args.host = plan.host

    if args.static_only and args.open:
        parser.error("--static-only cannot serve repository paths passed with --open")
    url = compare_url(args.open, args.port, args.labels, host=args.host) if args.open else (
        f"http://{args.host}:{args.port}/index.html" if args.static_only
        else f"http://{args.host}:{args.port}/viewer/studio.html"
    )
    print(url, flush=True)

    # A prior server life may have died mid-generation (crash, closed terminal) without
    # ever recording how that job ended, and its detached child can still be running with
    # nobody tracking it -- resolve those before accepting new jobs.
    if OUTPUT_ROOT.is_dir():
        orphaned = _reconcile_orphaned_jobs(OUTPUT_ROOT)
        if orphaned:
            print(f"reconciled {len(orphaned)} orphaned job(s) from a previous session: "
                  f"{', '.join(orphaned)}", flush=True)

    # A deliberate stop (Ctrl-C, `kill`) must not orphan an in-flight job the way an
    # uncaught crash does -- SIGTERM has no default Python handler (it would just kill this
    # process outright, mid-job, with no chance to clean up), so both signals get one here.
    def _handle_shutdown_signal(signum, frame):
        _terminate_active_job()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    import functools
    handler_class = StaticViewerHandler if args.static_only else Handler
    directory = REPO / "viewer" if args.static_only else REPO
    handler = functools.partial(handler_class, directory=str(directory))
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        if args.open and not args.no_browser:
            webbrowser.open(url)
        if plan is not None:
            print(plan.message, flush=True)
            if plan.open_browser and not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{args.port}/viewer/studio.html")
        print(f"serving {REPO} — ctrl-c to stop", flush=True)
        try:
            httpd.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
