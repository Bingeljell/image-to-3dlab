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
