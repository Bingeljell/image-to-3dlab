"""Studio front-end bugs found in review (2026-10) that still showed on current code.

The browser modules run under Node, the way tests/test_studio_modules.py does it. Two checks
read the source instead, because `viewer/studio/app.js` and the Reset button cannot be loaded
without a browser page.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

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
