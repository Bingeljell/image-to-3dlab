#!/usr/bin/env python3
"""Serve the studio, and the repo so it can load GLBs by relative path.

    python viewer/serve.py                      # http://127.0.0.1:8777/viewer/studio.html
    python viewer/serve.py --auto               # what ./lab runs: picks how to listen, opens the browser
    python viewer/serve.py --host 100.x.y.z      # explicit Tailscale/LAN interface

Bound to 127.0.0.1 by default, with no key. Listening anywhere else (--host, a RunPod pod) it
makes a random key and prints a link carrying it (studio_auth.py): only that link gets in.
Set I3D_STUDIO_TOKEN to keep the same key across restarts. Dotfiles are never served.
Behind a proxy that keeps 127.0.0.1 (tailscale serve, cloudflared, ngrok, VS Code port
forwarding), list the proxy's host names in I3D_ALLOWED_HOSTS, e.g.
I3D_ALLOWED_HOSTS=box.tail1234.ts.net (see request_utils.py).

The viewer exists because judging a mesh by eye is the acceptance test in this repo, and
every such judgement previously required opening Blender. It is also driveable by browser
automation, which means the agent can look at its own output instead of asking.
"""

from __future__ import annotations

import argparse
import os
import signal
import socketserver
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

# Sibling import: works when run as `python viewer/serve.py` (script dir on sys.path)
# and when loaded by the test suite via importlib (script dir not on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import studio_auth
import usage_api
from generate_api import (
    OUTPUT_ROOT,
    Handler,
    _reconcile_orphaned_jobs,
    _terminate_active_job,
)

from image_to_3dlab import processes

REPO = Path(__file__).resolve().parents[1]


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Allow SSE and static assets while the one generator job runs."""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        # A browser that closes a tab or cancels a download mid-response is normal, not an
        # error: say nothing rather than print a traceback into the terminal.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class LaunchPlan(NamedTuple):
    host: str
    open_browser: bool
    message: str


def launch_plan(env: Mapping[str, str], port: int = 8777, token: str | None = None) -> LaunchPlan:
    """How `./lab` (serve.py --auto) should listen, for a new user who just ran the installer.

    On their own machine: privately, and open the browser. On a RunPod pod: on all
    interfaces, because RunPod's HTTP proxy is the only way in, and print that link. Over
    plain SSH: still privately, with the tunnel command, because listening on a network by
    default would expose the whole repository to everyone on it.
    """
    local = f"http://127.0.0.1:{port}/viewer/studio.html"
    pod = env.get("RUNPOD_POD_ID")
    if pod:
        link = f"https://{pod}-{port}.proxy.runpod.net/viewer/studio.html"
        return LaunchPlan("0.0.0.0", False, (
            f"Open {studio_auth.link_with_token(link, token) if token else link}\n"
            f"(the pod needs HTTP port {port} exposed; add it in the pod's settings if not)\n"
            f"Keep this link private: it is the key to this studio."))
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
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="pick how to listen for this machine (what ./lab uses)")
    args = parser.parse_args()
    plan = launch_plan(os.environ, args.port) if args.auto else None
    if plan is not None:
        args.host = plan.host
    # reachable from a network (or a tunnel): only the printed link (it carries a key) gets in
    try:
        token = studio_auth.make_token(os.environ) if studio_auth.needs_token(args.host) else None
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    Handler.studio_token = token
    if token:
        studio_auth.save_token(token)       # tools on this machine (MCP, plan runner) read it here
    else:
        studio_auth.forget_token()          # a key left from an earlier, keyed run
    if plan is not None and token:
        plan = launch_plan(os.environ, args.port, token)

    url = f"http://{args.host}:{args.port}/viewer/studio.html"
    print(studio_auth.link_with_token(url, token) if token else url, flush=True)
    if token and os.environ.get("I3D_ALLOWED_HOSTS"):
        print(f"Through your tunnel, open its address with ?token={token} at the end. Keep it private.", flush=True)

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
        studio_auth.forget_token()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    import functools
    handler = functools.partial(Handler, directory=str(REPO))
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        if plan is not None:
            print(plan.message, flush=True)
            if plan.open_browser and not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{args.port}/viewer/studio.html")
        # Windows: sd-cli and the other workers die with this server, also when it is killed
        # from Task Manager (TerminateProcess runs none of the handlers above). Only after the
        # browser opens: a browser this starts would be in the job and close with the studio.
        processes.kill_children_with_this_process()
        print(f"serving {REPO} — ctrl-c to stop", flush=True)
        if usage_api.sharing():
            print("sharing anonymous usage counts, once a day; turn it off on the About page "
                  "or with I3D_NO_USAGE=1", flush=True)
            usage_api.send_soon()
        try:
            httpd.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
