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
