"""Four studio front-end review findings, each pinned by a test that fails on the bug.

Node runs the shipped ES modules (as tests/test_studio_old_frontend_bugs.py does). The two
page-wiring pieces that cannot be imported without a browser (the Reset pose branch in
`viewer/studio/viewer.js` and the spot-fix callbacks in `viewer/studio/app.js`) are cut out of the
real source text and run against real or stubbed objects, so they are still run, not just read.
"""

from __future__ import annotations

import json
import os
import re
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


# 2. Reset pose ends on the clip's first frame --------------------------------------------------

def _rest_branch_source() -> str:
    source = (REPO / "viewer" / "studio" / "viewer.js").read_text(encoding="utf-8")
    start = source.index("if (act === 'rest')")
    open_brace = source.index("{", start)
    depth, i = 0, open_brace
    while True:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[open_brace + 1:i]
        i += 1


@needs_node
def test_reset_pose_leaves_the_bones_at_rest_and_the_playhead_at_zero():
    """Runs the shipped Reset branch of viewer.js with the real player, a real three.js mixer and
    a skinned mesh whose clip starts away from rest."""
    branch = _rest_branch_source()
    out = _node(f"""
      import * as THREE from {json.dumps(THREE_URL)};
      import {{ AnimationPlayer }} from {json.dumps(PLAYER)};
      import {{ inspectRig, resetRigPose }} from {json.dumps(RIG)};
      const bone = new THREE.Bone(); bone.name = 'arm';           // rest: x = 0
      const mesh = new THREE.SkinnedMesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      mesh.add(bone); mesh.bind(new THREE.Skeleton([bone]));
      const root = new THREE.Group(); root.add(mesh);
      root.updateMatrixWorld(true);
      // frame 0 puts the arm at x = 5, frame 1s at x = 9: neither is the rest pose
      const clip = new THREE.AnimationClip('wave', 1,
        [new THREE.VectorKeyframeTrack('arm.position', [0, 1], [5, 0, 0, 9, 0, 0])]);
      const player = new AnimationPlayer(new THREE.AnimationMixer(root));
      player.select(clip);
      player.action.time = 0.5; player.mixer.update(0);           // the user scrubbed mid-clip
      const view = {{ rig: inspectRig(root, [clip]), resetPose() {{ resetRigPose(this.rig); }} }};
      const before = bone.position.x;
      let renders = 0, buttons = 0;
      const run = new Function('player', 'view', 'requestRender', 'syncPlayButton',
        {json.dumps(branch)});
      run(player, view, () => renders++, () => buttons++);
      console.log(JSON.stringify({{ before, x: bone.position.x, time: player.time, playing: player.playing }}));
    """)
    assert out["before"] == pytest.approx(7), "the harness did not pose the bone mid-clip"
    assert out["time"] == 0, "the playhead is not at the start"
    assert out["x"] == pytest.approx(0), \
        f"Reset pose left the bone at x={out['x']}: the clip's first frame (x=5) was written over the rest pose"


# 3. The brush keeps the old asset --------------------------------------------------------------

def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


@needs_node
def test_after_a_model_change_with_the_brush_open_the_brush_is_off_or_patch_and_keep_use_the_new_asset():
    """Either way out is fine. (a) load() in viewer.js turns the brush off, checked in its source,
    because a page is needed to run it. (b) the app.js callbacks, run with stubs, post the file of
    the asset now shown."""
    viewer_js = (REPO / "viewer" / "studio" / "viewer.js").read_text(encoding="utf-8")
    load = _between(viewer_js, "function load(url", "/** One picture")
    brush_turned_off = bool(re.search(r"data-act=.fix.|closeBrush|endBrush|setBrush|\bfix\b", load, re.IGNORECASE))

    app = (REPO / "viewer" / "studio" / "app.js").read_text(encoding="utf-8")
    block = app[app.index("fixSession: {") + len("fixSession: "):app.index("async function postFix")]
    literal = block[:block.rindex("} })") + 1]
    out = _node(f"""
      const posts = [];
      const A = {{ id: 'a', finished: 'a/model.glb' }}, B = {{ id: 'b', finished: 'b/model.glb' }};
      const ctx = {{
        current: A, fixAsset: null, assets: [A, B], viewingStep: null, viewingPictures: false,
        canFixShown: () => true, fixBlockedNote: () => '',
        postFix: async (action, payload) => {{ posts.push({{ action, from: payload.from }}); return {{ preview: 'p', file: 'x' }}; }},
        url: (p) => p, loadLibrary: async () => {{}}, madeVersion: () => null, show: () => {{}},
      }};
      const session = new Function('ctx', 'with (ctx) {{ return (' + {json.dumps(literal)} + '); }}')(ctx);
      session.open();                 // brush on, with model A
      ctx.current = B;                // the user opens model B; the brush stays open
      await session.patch([], null);
      await session.keep('p');
      console.log(JSON.stringify({{ froms: posts.map((p) => p.from) }}));
    """)
    uses_new_asset = out["froms"] and all(f == "b/model.glb" for f in out["froms"])
    assert brush_turned_off or uses_new_asset, (
        f"model B is on screen with the brush open, and Patch and Keep post {out['froms']}: "
        "the old asset's file")


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
