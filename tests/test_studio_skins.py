"""The studio's Skins card (viewer/studio/skins.js), run under Node against the shipped file.

What matters: the card appears only where a skin can be made; each state shows the one thing
to do next (wait, approve or ask again, paint, retry); the original and the new picture sit
side by side; nothing the person typed reaches the page unescaped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "skins.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\nconst url = (r) => '/output/' + r;\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


PALADIN = {"id": "a1", "name": "paladin", "model": "p__pixal3d__1/p__pixal3d__1.glb",
           "finished": "finish/p/p_20k.glb", "skins": []}
BASE = {"finished": "finish/p/p_20k.glb", "front": "images/p/p.png", "back": "images/p/p__back.png"}


def skin(**changes):
    state = {"id": "s1", "name": "gold skirt", "status": "pictures", "base": BASE, "needs_back": True,
             "tries": {"front": [{"number": 1, "picture": "skins/s1/front-1.png", "ready": True}], "back": []},
             "approved": {"front": None, "back": None}, "next_side": "front", "ready": False,
             "picture_job": {"status": "done"}, "build_job": None, "created": 1}
    state.update(changes)
    return state


@needs_node
def test_only_finished_pixal3d_models_get_the_card():
    out = _run(f"""console.log(JSON.stringify([
      m.canSkin({json.dumps(PALADIN)}),
      m.canSkin({json.dumps({**PALADIN, "finished": None})}),
      m.canSkin({json.dumps({**PALADIN, "model": "t__trellis__1/t.glb"})})]));""")
    assert out == [True, False, False]


@needs_node
def test_a_drawn_try_sits_beside_the_original_with_approve_and_ask_again():
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(skin())}, url)));")
    assert "/output/images/p/p.png" in html and "/output/skins/s1/front-1.png?v=1" in html
    assert 'data-skin-approve="front"' in html and 'data-skin-redo="front"' in html
    assert "Step 1 of 2" in html


@needs_node
def test_while_drawing_there_is_nothing_to_approve():
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(skin(picture_job={'status': 'running'}))}, url)));")
    assert "Drawing" in html and "data-skin-approve" not in html and "data-skin-redo" not in html


@needs_node
def test_both_approved_offers_to_paint():
    ready = skin(next_side=None, ready=True, approved={"front": "skins/s1/front-1.png", "back": "skins/s1/back-2.png"})
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(ready)}, url)));")
    assert "data-skin-build" in html and "/output/skins/s1/back-2.png" in html


@needs_node
def test_building_shows_progress_and_a_failure_offers_a_retry():
    building = skin(status="building", build_job={"job_id": "b", "last_event": {"overall_pct": 40, "message": "[3/6] shape"}})
    failed = skin(status="error", error="the repaint step stopped (exit 1)")
    out = _run(f"""console.log(JSON.stringify([m.draftHtml({json.dumps(building)}, url),
      m.draftHtml({json.dumps(failed)}, url), m.skinBusy({json.dumps(building)})]));""")
    assert 'style="width:40%"' in out[0] and "data-skin-cancel" in out[0]
    assert "data-skin-build" in out[1] and "Try building again" in out[1]
    assert out[2] is True


@needs_node
def test_the_card_lists_finished_skins_and_this_models_drafts_only():
    asset = {**PALADIN, "skins": [{"id": "k1", "name": "purple", "file": "skins/k1/purple_20k.glb"}]}
    other = skin(id="s2", base={**BASE, "finished": "finish/q/q_20k.glb"})
    out = _run(f"""const drafts = m.draftsFor({json.dumps(asset)}, {json.dumps([skin(), other, skin(id='s3', status='done')])});
      console.log(JSON.stringify([drafts.map(d => d.id), m.skinCardHtml({json.dumps(asset)}, drafts, 'k1', url)]));""")
    assert out[0] == ["s1"]
    assert '<option value="k1" selected>purple</option>' in out[1] and 'data-draft="s1"' in out[1]


@needs_node
def test_names_are_escaped():
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(skin(name='<img onerror=x>'))}, url)));")
    assert "<img onerror" not in html and "&lt;img onerror=x&gt;" in html


@needs_node
def test_a_try_whose_picture_never_arrived_cannot_be_approved():
    """The studio restarted while it drew: no picture, so no Approve, only Ask again."""
    lost = skin(tries={"front": [{"number": 1, "picture": "skins/s1/front-1.png", "ready": False}], "back": []},
                picture_job=None)
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(lost)}, url)));")
    assert "data-skin-approve" not in html and "data-skin-redo" in html and "Could not draw" in html


@needs_node
def test_the_drawing_bar_follows_qwens_steps():
    drawing = skin(picture_job={"status": "running", "last_event": {"percent": 25.0}})
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(drawing)}, url)));")
    assert 'style="width:25%"' in html


@needs_node
def test_a_server_that_does_not_say_ready_still_shows_the_picture():
    """2026-10-19: a newer page on an older server read a drawn picture as never arriving."""
    older = skin(tries={"front": [{"number": 1, "picture": "skins/s1/front-1.png"}], "back": []})
    html = _run(f"console.log(JSON.stringify(m.draftHtml({json.dumps(older)}, url)));")
    assert 'data-skin-approve="front"' in html


@needs_node
def test_the_big_viewer_gets_the_original_and_the_newest_drawn_try():
    out = _run(f"""console.log(JSON.stringify([
      m.largePair({json.dumps(skin())}),
      m.largePair({json.dumps(skin(picture_job={'status': 'running'}))}),
      m.largePair({json.dumps(skin(next_side=None, ready=True))})]));""")
    assert out[0]["original"] == "images/p/p.png" and out[0]["fresh"] == "skins/s1/front-1.png" and out[0]["version"] == 1
    assert "try 1" in out[0]["caption"] and out[1] is None and out[2] is None
