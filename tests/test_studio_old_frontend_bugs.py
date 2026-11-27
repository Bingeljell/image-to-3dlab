"""Studio front-end bugs found in review (2026-10) that still showed on current code.

The browser modules run under Node, the way tests/test_studio_modules.py does it. Two checks
read the source instead, because `viewer/studio/app.js` and the Reset button cannot be loaded
without a browser page.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import generate_api as api
import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
THREE_URL = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()


def _node(program: str, env: dict | None = None):
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, **(env or {})},
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _with_three(body: str, wrap: str = "") -> str:
    """Let `import 'three'` (the bare name the browser's import map gives) find the vendored copy.

    `wrap` is extra module text appended to a wrapper around three, to swap a class out."""
    if wrap:
        source = f"export * from {json.dumps(THREE_URL)};\n{wrap}"
        target = "data:text/javascript," + quote(source, safe="")
    else:
        target = THREE_URL
    return f"""
      import {{ register }} from 'node:module';
      register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, n) {{
        return s === 'three' ? {{ url: {json.dumps(target)}, shortCircuit: true }} : n(s, c);
      }}`));
      {body}
    """


# 1. OBJ loader ---------------------------------------------------------------------------------

def _load_obj(lines: list[str]):
    module_url = (REPO / "viewer" / "IndexedOBJLoader.js").as_uri()
    program = _with_three(f"""
      const {{ IndexedOBJLoader }} = await import({json.dumps(module_url)});
      const bytes = new TextEncoder().encode({json.dumps(chr(10).join(lines))});
      globalThis.fetch = async () => ({{
        ok: true,
        headers: {{ get: () => String(bytes.byteLength) }},
        body: new ReadableStream({{ start(c) {{ c.enqueue(bytes); c.close(); }} }}),
      }});
      let mesh = null, error = null;
      await new IndexedOBJLoader().load('/m.obj', (root) => {{ mesh = root; }}, null, (e) => {{ error = String(e); }});
      const index = mesh?.geometry.index.array;
      console.log(JSON.stringify({{
        error, index: index ? Array.from(index) : null,
        indexTight: index ? index.buffer.byteLength === index.byteLength : null,
        positionTight: mesh ? mesh.geometry.attributes.position.array.buffer.byteLength
          === mesh.geometry.attributes.position.array.byteLength : null,
      }}));
    """)
    return _node(program)


@needs_node
def test_obj_loader_skips_faces_that_point_past_the_vertices():
    out = _load_obj([
        "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 1",
        "f 1 2 3",         # good
        "f 1 2 99",        # past the end
        "f 1 2 0",         # OBJ counts from 1: 0 is not a vertex
        "f 1 2 3 99",      # a quad with one bad corner
        "f -1 -2 -3",      # relative indices are fine
        "f 1 2 x",         # not a number
    ])
    assert out["error"] is None
    assert out["index"] == [0, 1, 2, 3, 2, 1]


@needs_node
def test_obj_loader_hands_over_buffers_with_no_spare_room():
    out = _load_obj(["v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"])
    assert out["positionTight"] is True and out["indexTight"] is True


# 2. Stale load and "no size" in the model viewport ---------------------------------------------

_FAKE_RENDERER = """
export class WebGLRenderer {
  constructor() { this.domElement = { addEventListener() {}, removeEventListener() {}, style: {},
    getRootNode() { return { addEventListener() {}, removeEventListener() {} }; },
    ownerDocument: { addEventListener() {}, removeEventListener() {} }, setPointerCapture() {}, releasePointerCapture() {} }; }
  setPixelRatio() {} setSize() {} render() {} dispose() {} forceContextLoss() {}
}
export class PMREMGenerator {
  constructor() {} fromScene() { return { texture: { dispose() {} } }; } dispose() {}
}
"""


def _viewport_program(body: str) -> str:
    module_url = (REPO / "viewer" / "components" / "model-viewport.js").as_uri()
    return _with_three(f"""
      globalThis.devicePixelRatio = 1;
      globalThis.window = globalThis;
      const pending = [];
      globalThis.fetch = async () => {{
        const bytes = await new Promise((resolve) => pending.push(resolve));
        return {{ ok: true, headers: {{ get: () => '0' }},
          body: new ReadableStream({{ start(c) {{ c.enqueue(bytes); c.close(); }} }}) }};
      }};
      const text = (s) => new TextEncoder().encode(s);
      const {{ createModelViewport, disposeModelViewport }} = await import({json.dumps(module_url)});
      const makePane = () => {{
        const stats = {{ textContent: 'keep' }};
        return {{ stats, appendChild() {{}}, querySelector: () => stats }};
      }};
      {body}
    """, wrap=_FAKE_RENDERER)


@needs_node
def test_a_load_that_finishes_after_its_view_was_replaced_does_not_touch_the_screen():
    out = _node(_viewport_program("""
      const pane = makePane();
      const loaded = [];
      const view = createModelViewport({ pane, spec: { ext: 'obj', url: '/slow.obj' }, onLoaded: () => loaded.push(1) });
      const sceneBefore = view.scene.children.length;
      disposeModelViewport(view);          // the user opened another model meanwhile
      pending[0](text('v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3'));
      await new Promise((r) => setTimeout(r, 50));
      console.log(JSON.stringify({
        stats: pane.stats.textContent, added: view.scene.children.length - sceneBefore,
        loaded: loaded.length, hasRoot: !!view.root,
      }));
    """))
    assert out["stats"] == "keep", "a replaced load wrote its face count over the pane"
    assert out["added"] == 0, "a replaced load put its model into the scene"
    assert out["loaded"] == 0 and out["hasRoot"] is False


@needs_node
def test_a_load_the_caller_says_is_stale_is_ignored():
    out = _node(_viewport_program("""
      const pane = makePane();
      const loaded = [];
      let current = true;
      const view = createModelViewport({
        pane, spec: { ext: 'obj', url: '/slow.obj' }, onLoaded: () => loaded.push(1), isCurrent: () => current,
      });
      const sceneBefore = view.scene.children.length;
      current = false;                     // a newer load started
      pending[0](text('v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3'));
      await new Promise((r) => setTimeout(r, 50));
      console.log(JSON.stringify({ stats: pane.stats.textContent,
        added: view.scene.children.length - sceneBefore, loaded: loaded.length }));
    """))
    assert out["stats"] == "keep"
    assert out["added"] == 0 and out["loaded"] == 0


@needs_node
def test_a_model_with_no_size_is_an_error_and_is_not_placed():
    out = _node(_viewport_program("""
      const pane = makePane();
      const errors = [], loaded = [];
      const view = createModelViewport({
        pane, spec: { ext: 'obj', url: '/empty.obj' },
        onLoaded: () => loaded.push(1), onError: (e) => errors.push(String(e.message || e)),
      });
      const sceneBefore = view.scene.children.length;
      pending[0](text('# nothing here\\n'));
      await new Promise((r) => setTimeout(r, 50));
      const scale = view.root ? view.root.scale.x : null;
      console.log(JSON.stringify({ errors, loaded: loaded.length, scale,
        added: view.scene.children.length - sceneBefore }));
    """))
    assert out["loaded"] == 0, "an empty model was placed on screen (it is scaled by 1 / 0)"
    assert out["added"] == 0
    assert len(out["errors"]) == 1


# 3. Setup polling fallback ---------------------------------------------------------------------

@needs_node
def test_setup_keeps_asking_when_the_event_stream_drops():
    module_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import * as m from {json.dumps(module_url)};
      const events = [], urls = [];
      let stream = null;
      m.openSetupWatch({{ status_url: '/api/setup/qwen-image/status', events_url: '/api/setup/qwen-image/events' }}, {{
        fetchImpl: async (url) => {{ urls.push(url); return {{ ok: true, status: 200,
          json: async () => ({{ status: 'done', last_event: {{ phase: 'done' }}, log_tail: 'ok' }}) }}; }},
        EventSourceImpl: class {{ constructor() {{ stream = this; }} close() {{}} }},
        onEvent: (event) => events.push(event.phase),
        interval: 1,
      }});
      stream.onerror?.(new Error('dropped'));       // the stream dies; the run then finishes
      await new Promise((r) => setTimeout(r, 100));
      console.log(JSON.stringify({{ events, polled: urls.length > 0 }}));
    """
    out = _node(program)
    assert out["polled"] is True
    assert "done" in out["events"]


def test_setup_page_has_a_way_to_learn_a_run_finished_without_the_stream():
    source = (REPO / "viewer" / "studio" / "setup.js").read_text(encoding="utf-8")
    assert re.search(r"status_url|openSetupWatch", source), \
        "setup.js only listens to an EventSource and never asks the status page"


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("what", ["status", "events"])
def test_a_forgotten_setup_run_answers_404_json_not_a_file_page(server, monkeypatch, what):
    monkeypatch.setattr(api, "DOWNLOADS", {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}/api/setup/qwen-image/{what}"
    with pytest.raises(urllib.error.HTTPError) as caught:
        opener.open(url, timeout=5)
    assert caught.value.code in (404, 410)
    assert caught.value.headers.get_content_type() == "application/json", \
        "the page got a static file answer, which a polling page reads as a dropped connection"
    body = json.loads(caught.value.read())
    assert body["status"] == "error" and "no longer running" in body["error"]


# 4. Spot-fix Keep ------------------------------------------------------------------------------

def test_spot_fix_keep_uses_the_asset_the_brush_was_opened_on():
    """Source check: app.js cannot be loaded without a page. The keep and patch callbacks must
    not read `current` (whichever row is selected when the button is pressed)."""
    app = (REPO / "viewer" / "studio" / "app.js").read_text(encoding="utf-8")
    block = app[app.index("fixSession: {"):app.index("async function postFix")]
    keep = block[block.index("keep: async"):block.index("discard:")]
    patch = block[block.index("patch: async"):block.index("keep: async")]
    assert "current" not in keep, "Keep files the patch onto whatever asset is selected now"
    assert "current" not in patch, "the preview is made from whatever asset is selected now"
