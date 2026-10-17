"""The agents' copy of Create's rules must match the studio's own (viewer/studio/plan.js)."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import agent_plan

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node runs the studio's plan.js")
RECIPES = json.loads((REPO / "image_to_3dlab" / "prompt_recipes.json").read_text())


def _plan_js(body: str):
    url = (REPO / "viewer" / "studio" / "plan.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


CASES = [
    {"have": have, "want": want, "finishUpload": fu, "frontBack": fb}
    for have, want, fu, fb in itertools.product(("idea", "picture", "model"), ("picture", "prop", "set", "character"),
                                                (False, True), (False, True))
]


@needs_node
def test_every_have_want_combination_plans_the_same_steps_as_the_studio():
    studio = _plan_js(f"""console.log(JSON.stringify({json.dumps(CASES)}.map(c =>
        m.planFor(c.have, c.want, {{ finishUpload: c.finishUpload, frontBack: c.frontBack }}))));""")
    ours = [agent_plan.plan_for(c["have"], c["want"], finish_upload=c["finishUpload"], front_back=c["frontBack"]) for c in CASES]
    assert ours == studio


@needs_node
def test_what_each_starting_point_allows_matches_the_studio():
    assert agent_plan.ALLOWED == _plan_js("console.log(JSON.stringify(m.ALLOWED));")
    assert set(agent_plan.HAVES) == set(_plan_js("console.log(JSON.stringify(Object.keys(m.HAVES)));"))
    assert set(agent_plan.WANTS) == set(_plan_js("console.log(JSON.stringify(Object.keys(m.WANTS)));"))


@needs_node
def test_stop_after_matches_the_studio():
    steps = ["picture", "back", "model", "finished", "rigged", "animated"]
    lasts = [None, "picture", "model", "animated", "split"]
    studio = _plan_js(f"console.log(JSON.stringify({json.dumps(lasts)}.map(l => m.stopAfter({json.dumps(steps)}, l))));")
    assert [agent_plan.stop_after(steps, last) for last in lasts] == studio


@needs_node
def test_the_prompt_sent_matches_the_studio_with_and_without_the_added_wording():
    texts = ["a knight in rusty armour", "  a red barrel,. ", ""]
    combos = [(t, goal, use) for t in texts for goal in ("character", "prop", "set", "picture", "nope") for use in (True, False)]
    studio = _plan_js(f"""const r = {json.dumps(RECIPES)};
      console.log(JSON.stringify({json.dumps(combos)}.map(([t, g, u]) => m.composePrompt(t, g, r, u))));""")
    assert [agent_plan.compose_prompt(t, g, RECIPES, u) for t, g, u in combos] == studio


def test_refusal_names_the_reason():
    assert agent_plan.refusal("idea", "character") is None
    assert agent_plan.refusal("picture", "picture") == "You already have the picture."
    assert "want must be" in agent_plan.refusal("idea", "dragon")
