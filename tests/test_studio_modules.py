"""The studio's pure browser modules, run under Node against the shipped files."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


KNIGHT = {"name": "rune knight", "kind": "character", "stage": "animated",
          "picture": "images/k/k.png", "model": "a/k.glb", "finished": "finish/k/k_40k.glb",
          "rigged": "animate/k/k_rigged.glb", "clips": [{"name": "dance"}, {"name": "run"}], "props": []}
FOX = {"name": "fox", "kind": "picture", "stage": "picture", "picture": "images/f/f.png",
       "model": None, "finished": None, "rigged": None, "clips": [], "props": []}
SHEET = {"name": "tavern", "kind": "prop set", "stage": "finished", "picture": None, "model": "a/s.glb",
         "finished": None, "rigged": None, "clips": [], "props": [{"name": "barrel", "file": "props/s/barrel_LOD0.glb"}]}


@needs_node
def test_progress_and_next_step_per_kind():
    out = _run("library.js", f"""
      const a = {json.dumps([KNIGHT, FOX, SHEET])};
      console.log(JSON.stringify(a.map(x => [m.doneCount(x), m.stepCount(x), m.nextStep(x)?.id ?? null])));""")
    assert out == [[5, 5, None], [1, 5, "model"], [3, 3, None]]


@needs_node
def test_status_lines_read_like_sentences():
    out = _run("library.js", f"""
      console.log(JSON.stringify({json.dumps([KNIGHT, FOX, SHEET])}.map(m.statusText)));""")
    assert out == ["Animated · 2 moves", "Picture", "1 prop, game-ready"]


@needs_node
def test_filters_and_what_the_viewer_shows():
    out = _run("library.js", f"""
      const a = {json.dumps([KNIGHT, FOX, SHEET])};
      console.log(JSON.stringify({{
        props: m.filterAssets(a, 'props').map(x => x.name),
        characters: m.filterAssets(a, 'characters').map(x => x.name),
        show: a.map(m.displayModel),
        url: m.servedUrl('/output/', 'images/a b/c#d.png'),
      }}));""")
    assert out["props"] == ["tavern"]
    assert out["characters"] == ["rune knight", "fox"]
    assert out["show"] == ["animate/k/k_rigged.glb", None, "props/s/barrel_LOD0.glb"]
    assert out["url"] == "/output/images/a%20b/c%23d.png"


@needs_node
def test_axis_map_from_the_front_puts_x_right_y_up_z_nearest():
    out = _run("gizmo.js", """
      const p = m.projectAxes([0, 0, 1]);
      const by = Object.fromEntries(p.map(a => [a.key, [Math.round(a.x * 100) / 100, Math.round(a.y * 100) / 100, Math.round(a.depth * 100) / 100]]));
      console.log(JSON.stringify({ by, nearest: p[p.length - 1].key, farthest: p[0].key }));""")
    assert out["by"]["+x"] == [1, 0, 0]
    assert out["by"]["+y"] == [0, -1, 0]          # screen y grows downward
    assert out["by"]["+z"] == [0, 0, 1]
    assert out["nearest"] == "+z" and out["farthest"] == "-z"


@needs_node
def test_axis_map_looking_straight_down_does_not_break():
    out = _run("gizmo.js", """
      const p = m.projectAxes([0, 1, 0]);
      console.log(JSON.stringify(p.every(a => Number.isFinite(a.x) && Number.isFinite(a.y))));""")
    assert out is True


def test_the_classic_viewer_opens_the_tab_named_in_its_address():
    # the studio's Setup link is index.html#setup; the hash must win after the landing tab
    app = (REPO / "viewer" / "app.js").read_text()
    landing = app.index("setMode(landingMode(")
    assert app.index("if (modes[location.hash.slice(1)]) setMode(location.hash.slice(1));") > landing


@needs_node
def test_best_ready_engine_comes_first_and_missing_weights_drop_out():
    catalog = {"backends": [
        {"id": "trellis", "kind": "3d", "rank": 3, "build_present": True, "weights": [{"present": True}], "license": {"name": "MIT"}},
        {"id": "pixal3d", "kind": "3d", "rank": 1, "build_present": True, "weights": [{"present": True}], "license": {"name": "MIT"}},
        {"id": "sf3d", "kind": "3d", "rank": 5, "build_present": True, "weights": [{"present": False}]},
        {"id": "hunyuan-cuda", "kind": "3d", "rank": 2, "supported_here": False},
        {"id": "autorig", "kind": "tool", "rank": 0},
    ]}
    out = _run("jobs.js", f"console.log(JSON.stringify(m.readyEngines({json.dumps(catalog)}).map(e => e.id)));")
    assert out == ["pixal3d", "trellis"]


@needs_node
def test_errors_are_put_in_plain_words():
    out = _run("jobs.js", """console.log(JSON.stringify([
      m.plainError('Blender: Error: Not enough memory for 4096 x 4096 bake'),
      m.plainError('Pixal3D (C++/GGML) is not installed/ready'),
      m.plainError('a generation is running; wait for it to finish'),
      m.plainError('could not map skeleton to SOMA'),
      m.plainError(''),
    ]));""")
    assert out[0].startswith("This Mac ran out of memory")
    assert out[1].startswith("This engine is not installed")
    assert out[2].startswith("Another job is running")
    assert out[3].startswith("This doesn't look like a humanoid")
    assert out[4].startswith("The step stopped without saying why")


@needs_node
def test_following_a_job_reports_progress_and_ends_on_a_terminal_status():
    out = _run("jobs.js", """
      const replies = [
        { status: 'running', last_event: { phase: 'bake', pct: 40, message: 'Baking' }, log_tail: 'a' },
        { status: 'running', last_event: { phase: 'bake', pct: 80, message: 'Baking' }, log_tail: 'ab' },
        { status: 'done', last_event: { phase: 'done', result_url: '/x.glb' }, log_tail: 'abc' },
      ];
      let i = 0;
      const fetchImpl = async () => ({ json: async () => replies[Math.min(i++, replies.length - 1)] });
      const seen = [];
      const job = m.followJob('/api/finish/1/status', { interval: 1, fetchImpl, onUpdate: (u) => seen.push([u.status, u.percent]) });
      const final = await job.done;
      console.log(JSON.stringify({ seen, final: final.status, url: final.last_event.result_url }));""")
    assert out["seen"] == [["running", 40], ["running", 80], ["done", None]]
    assert out["final"] == "done" and out["url"] == "/x.glb"


@needs_node
def test_a_job_with_no_news_for_three_minutes_is_flagged_as_stalled():
    out = _run("jobs.js", """
      let clock = 0;
      const replies = [{ status: 'running', last_event: { phase: 'shape' } }, { status: 'running', last_event: { phase: 'shape' } }, { status: 'done', last_event: {} }];
      let i = 0;
      const fetchImpl = async () => { clock += 200000; return { json: async () => replies[Math.min(i++, 2)] }; };
      const stalls = [];
      await m.followJob('/s', { interval: 1, fetchImpl, now: () => clock, onUpdate: (u) => stalls.push(u.stalled) }).done;
      console.log(JSON.stringify(stalls));""")
    assert out == [False, True, False]
