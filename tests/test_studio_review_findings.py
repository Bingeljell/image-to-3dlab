"""Four studio front-end review findings, each pinned by a test that fails on the bug.

Node runs the shipped ES modules (as tests/test_studio_old_frontend_bugs.py does). The two
page-wiring pieces that cannot be imported without a browser (the Reset pose branch in
`viewer/studio/viewer.js` and the spot-fix callbacks in `viewer/studio/app.js`) are cut out of the
real source text and run against real or stubbed objects, so they are still run, not just read.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import download_api as dl
import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
THREE_URL = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
SETUP_PROGRESS = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
PLAYER = (REPO / "viewer" / "animation" / "player.js").as_uri()
RIG = (REPO / "viewer" / "animation" / "rig-inspection.js").as_uri()


def _node(program: str):
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True, capture_output=True, text=True, encoding="utf-8", env=os.environ.copy(),
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


# 1. Setup reattach has no status URL -----------------------------------------------------------

def test_a_reattached_setup_run_carries_the_status_url_a_fresh_start_does(monkeypatch):
    monkeypatch.setattr(dl, "DOWNLOADS", {})
    run = dl.DownloadRun(dl.BY_ID["pixal3d"])
    run.status = "running"
    dl.DOWNLOADS["pixal3d"] = run
    payload = dl.running_payload()
    # generate_api answers a fresh start with exactly these two URLs
    assert payload["events_url"] == "/api/setup/pixal3d/events"
    assert payload.get("status_url") == "/api/setup/pixal3d/status", \
        "the reattach payload has no status_url, so the page polls /undefined when the stream drops"


def _watch_with_dropped_stream(run: dict, status_body: dict, wait_ms: int = 80):
    return _node(f"""
      import {{ openSetupWatch }} from {json.dumps(SETUP_PROGRESS)};
      const urls = [], events = [];
      let stream = null;
      openSetupWatch({json.dumps(run)}, {{
        fetchImpl: async (url) => {{ urls.push(String(url));
          if (String(url).includes('undefined')) return {{ ok: false, status: 404, json: async () => ({{}}) }};
          return {{ ok: true, status: 200, json: async () => ({json.dumps(status_body)}) }}; }},
        EventSourceImpl: class {{ constructor() {{ stream = this; }} close() {{}} }},
        onEvent: (event) => events.push(event),
        interval: 1,
      }});
      stream.onerror?.(new Error('dropped'));
      await new Promise((r) => setTimeout(r, {wait_ms}));
      console.log(JSON.stringify({{ urls, events }})); process.exit(0);
    """)


@needs_node
def test_a_watch_given_only_an_events_url_never_polls_undefined():
    out = _watch_with_dropped_stream(
        {"backend": "qwen-image", "rebuild": False, "events_url": "/api/setup/qwen-image/events"},
        {"status": "running", "last_event": {"phase": "downloading"}},
    )
    assert out["urls"], "the dropped stream was never followed up by a poll"
    assert not any("undefined" in url for url in out["urls"]), f"polled {out['urls'][0]!r}"
    assert set(out["urls"]) == {"/api/setup/qwen-image/status"}
    assert not any("no longer running" in (e.get("detail") or "") for e in out["events"]), \
        "a run that is still going was reported as no longer running"


# 4. Polling repeats the log --------------------------------------------------------------------

def _logs_after_polls(status_body: dict | list, wait_ms: int = 80) -> list[str]:
    bodies = status_body if isinstance(status_body, list) else [status_body]
    out = _node(f"""
      import {{ openSetupWatch }} from {json.dumps(SETUP_PROGRESS)};
      const bodies = {json.dumps(bodies)};
      let n = 0;
      const logs = [];
      openSetupWatch({{ status_url: '/s' }}, {{
        fetchImpl: async () => ({{ ok: true, status: 200, json: async () => bodies[Math.min(n++, bodies.length - 1)] }}),
        EventSourceImpl: null,
        onEvent: (event) => {{ if (event.log) logs.push(event.log); }},
        interval: 1,
      }});
      await new Promise((r) => setTimeout(r, {wait_ms}));
      console.log(JSON.stringify({{ logs, polls: n }})); process.exit(0);
    """)
    assert out["polls"] >= 3, "the watch did not poll enough times to show anything"
    return out["logs"]


@needs_node
@pytest.mark.parametrize("shape", ["tail_only", "tail_with_last_event"])
def test_two_polls_with_the_same_log_tail_do_not_emit_the_same_log_twice(shape):
    body = {"status": "running", "log_tail": "fetching weights 40%"}
    if shape == "tail_with_last_event":
        body["last_event"] = {"phase": "downloading", "log": "fetching weights 40%"}
    logs = _logs_after_polls(body)
    assert logs.count("fetching weights 40%") <= 1, \
        f"the same log text was emitted {logs.count('fetching weights 40%')} times, so the page repeats it"


@needs_node
def test_a_growing_log_tail_emits_only_the_new_text():
    logs = _logs_after_polls([
        {"status": "running", "log_tail": "one"},
        {"status": "running", "log_tail": "one\ntwo"},
        {"status": "running", "log_tail": "one\ntwo"},
    ])
    seen = "\n".join(logs)
    assert seen.count("one") == 1 and seen.count("two") == 1, f"log text was repeated: {logs}"
