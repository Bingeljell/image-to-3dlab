"""The studio's pure browser modules, run under Node against the shipped files."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

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
def test_single_props_show_under_props_and_not_under_characters():
    barrel = {"name": "barrel", "kind": "prop", "stage": "finished", "props": []}
    out = _run("library.js", f"""
      const a = {json.dumps([KNIGHT, FOX, SHEET, barrel])};
      console.log(JSON.stringify({{
        props: m.filterAssets(a, 'props').map(x => x.name),
        characters: m.filterAssets(a, 'characters').map(x => x.name),
      }}));""")
    assert out["props"] == ["tavern", "barrel"]
    assert out["characters"] == ["rune knight", "fox"]


@needs_node
def test_a_steps_time_reads_in_plain_minutes_and_seconds():
    out = _run("library.js", """console.log(JSON.stringify(
      [7, 42.4, 60, 252, 3599, 3780, null, -1].map(m.duration)));""")
    assert out == ["7s", "42s", "1m 0s", "4m 12s", "59m 59s", "1h 3m", "", ""]


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


@needs_node
def test_best_ready_engine_comes_first_and_missing_weights_drop_out():
    catalog = {"backends": [
        {"id": "trellis", "kind": "3d", "rank": 3, "state": "ready", "build_present": True, "weights": [{"present": True}], "license": {"name": "MIT"}},
        {"id": "pixal3d", "kind": "3d", "rank": 1, "state": "ready", "build_present": True, "weights": [{"present": True}], "license": {"name": "MIT"}},
        {"id": "sf3d", "kind": "3d", "rank": 5, "state": "partial", "build_present": True, "weights": [{"present": False}]},
        {"id": "hunyuan-cuda", "kind": "3d", "rank": 2, "state": "unsupported", "supported_here": False},
        {"id": "autorig", "kind": "tool", "rank": 0},
    ]}
    out = _run("jobs.js", f"console.log(JSON.stringify(m.readyEngines({json.dumps(catalog)}).map(e => e.id)));")
    assert out == ["pixal3d", "trellis"]


@needs_node
def test_engine_picker_agrees_with_setup_about_ready():
    # A user's Setup page showed TRELLIS.2 "ready", yet the Create picker hid it: Setup
    # trusts the server's state, which counts TRELLIS ready once built (its weights fetch
    # on first run, and TinyCLIP/BiRefNet are optional on a Mac). The picker must agree.
    catalog = {"backends": [
        {"id": "pixal3d", "kind": "3d", "rank": 1, "state": "ready", "build_present": True, "weights": [{"present": True}]},
        {"id": "trellis", "kind": "3d", "rank": 3, "state": "ready", "build_present": True,
         "weights": [{"present": True}, {"present": False}, {"present": False}]},
        {"id": "hunyuan_xiong", "kind": "3d", "rank": 2, "state": "missing", "build_present": False, "weights": [{"present": False}]},
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
def test_install_help_for_blender_is_shown_as_it_is():
    """2026-10-19, the first site report: a Mac with no Blender was told Blender had crashed."""
    message = "Finish needs Blender 4.2 or newer, and none was found. Install Blender from blender.org."
    out = _run("jobs.js", f"console.log(JSON.stringify(m.plainError({json.dumps(message)})));")
    assert out == message


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


@needs_node
def test_library_search_hides_and_pages():
    assets = [{"name": f"knight {i}", "kind": "character", "hidden": i % 10 == 0} for i in range(45)]
    assets.append({"name": "tavern barrel", "kind": "prop set", "hidden": False})
    out = _run("library.js", f"""
      const a = {json.dumps(assets)};
      const v = (o) => {{ const r = m.visibleAssets(a, o); return [r.rows.length, r.more, r.hiddenCount]; }};
      console.log(JSON.stringify({{
        first: v({{}}),
        withHidden: v({{ showHidden: true }}),
        search: v({{ query: 'BARREL' }}),
        twoWords: v({{ query: 'knight 4' }}),
        props: v({{ filter: 'props' }}),
      }}));""")
    assert out["first"] == [30, 11, 5]        # 46 assets, 5 hidden: 41 shown in pages of 30
    assert out["withHidden"] == [30, 16, 5]
    assert out["search"] == [1, 0, 0]
    assert out["twoWords"] == [8, 0, 1]       # "4" anywhere: 4, 14, 24, 34, 40-44 (9); 40 is hidden
    assert out["props"] == [1, 0, 0]


@needs_node
def test_the_studio_opens_on_create_unless_a_run_is_going():
    out = _run("library.js", """
      console.log(JSON.stringify([m.openingView(null), m.openingView({ id: 'r1' }), m.openingView(undefined)]));""")
    assert out == ["create", "running", "create"]


@needs_node
def test_only_a_glb_opens_in_the_viewer():
    out = _run("library.js", """
      console.log(JSON.stringify(['cart.glb', 'KNIGHT.GLB', 'scene.gltf', 'boy.png', 'glb', ''].map(m.isModelFile)));""")
    assert out == [True, True, False, False, False, False]


@needs_node
def test_a_cancelled_run_reads_as_cancelled_not_as_a_failure():
    out = _run("create.js", """
      const a = { steps: ['picture', 'model'], status: 'running', current: 'picture', done: [], made: {}, percent: 10, message: 'Drawing', full_log: 'x', stalled: false };
      console.log(JSON.stringify(m.chainEvents(a, { ...a, status: 'cancelled' })));""")
    assert out == [["cancel", "picture"]]


@needs_node
def test_a_cancelled_step_shows_no_log_and_a_failed_one_keeps_it():
    out = _run("jobs.js", """
      console.log(JSON.stringify([m.jobEnd({ status: 'cancelled', log_tail: 'noise' }), m.jobEnd({ status: 'error', last_event: { message: 'Killed' }, log_tail: 'a\\nb' })]));""")
    cancelled, failed = out
    assert cancelled == {"cancelled": True, "why": "Cancelled. Everything finished before this step is kept.", "log": ""}
    assert failed["cancelled"] is False and failed["log"] == "a\nb" and failed["why"]


@needs_node
def test_the_log_drops_the_picture_models_tokenizer_chatter():
    log = "\n".join([
        "<|im_start|>user",
        '" to 17 tokens ["<|im_start|>", "system", "\u010a", "Com", "preh", "end",',
        '"\u0120and", "\u0120analyze", "\u0120the", "\u0120provided", "\u0120prompt", ".", "',
        "<|im_end|>\", \"\u010a\", ]",
        "denoising step 2/10",
    ])
    out = _run("jobs.js", f"console.log(JSON.stringify(m.readableLog({json.dumps(log)}, 4)));")
    assert out == "denoising step 2/10"


@needs_node
def test_every_3d_try_gets_a_fresh_seed():
    out = _run("create.js", """
      const form = m.planForm({ steps: ['model'], description: 'x', modelSeed: m.freshSeed(() => 0.5) });
      console.log(JSON.stringify([m.freshSeed(() => 0.5), form.get('model_seed'), m.planForm({ steps: ['model'] }).get('model_seed')]));""")
    assert out == [1073741823, "1073741823", None]


@needs_node
def test_one_prop_and_nine_props_are_told_apart_by_name():
    out = _run("plan.js", "console.log(JSON.stringify([m.WANTS.prop.label, m.WANTS.set.label]));")
    assert out == ["One prop", "Nine props"]


@needs_node
def test_a_sheet_is_put_away_so_the_viewer_under_it_can_be_seen():
    """Create, Activity, How to use and About are opaque sheets over the viewer. Anything that
    shows a model has to take one away first, or the click loads the model behind it and looks
    like nothing happened (2026-10-09)."""
    out = _run("pages.js", """
      // a stage with sheets in it, in document order, like the real one
      const stage = { sheets: [] };
      stage.querySelector = (selector) => (selector === '.sheet' ? stage.sheets[0] ?? null : null);
      const open = (name) => stage.sheets.push({
        className: 'sheet', name,
        remove() { stage.sheets.splice(stage.sheets.indexOf(this), 1); },
      });
      const names = () => JSON.stringify(stage.sheets.map((s) => s.name));
      open('create');
      m.closeSheet(stage);
      const afterCreate = names();
      open('activity');
      m.closeSheet(stage);
      m.closeSheet(stage);                            // closing nothing is not an error
      console.log(JSON.stringify({ afterCreate, afterActivity: names(), bare: names() }));""")
    assert out["afterCreate"] == "[]"
    assert out["afterActivity"] == out["bare"] == "[]"


def test_everything_the_studio_shows_in_the_viewer_puts_a_sheet_away_first():
    """Both routes into the viewer -- picking a Library asset, and watching a run -- have to
    clear the sheet. Opening a GLB already did; the other two did not."""
    app = (REPO / "viewer" / "studio" / "app.js").read_text()

    def body(name: str) -> str:
        start = app.index(f"function {name}(")
        return app[start:app.index("\n}", start)]

    for name in ("show", "showMaking", "openCreateSheet", "openFile"):
        assert "closeSheet(" in body(name), name
    pages_import = next(line for line in app.splitlines() if line.startswith("import") and "'./pages.js'" in line)
    assert "closeSheet" in pages_import, "closeSheet is not imported from pages.js"


@needs_node
def test_the_how_to_page_covers_every_button_in_the_top_bar():
    html = (REPO / "viewer" / "studio.html").read_text()
    import re
    bar = html.split('<header class="bar">')[1].split("</header>")[0]
    labels = [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(r'class="ghost[^"]*"[^>]*>([^<]+)<', bar)]
    out = _run("pages.js", "console.log(JSON.stringify(m.GUIDE.map(([title, text]) => title + ' ' + text).join(' ')));")
    assert labels and all(label in out for label in labels if label != "How to use"), labels
    for must in ("+ Create", "Steps", "Download GLB", "Cancel", "close the tab"):
        assert must in out, must


def test_the_classic_viewer_is_gone_and_old_links_land_in_the_studio():
    viewer = REPO / "viewer"
    assert not (viewer / "app.js").exists() and not (viewer / "modes").exists()
    assert 'url=./studio.html' in (viewer / "index.html").read_text()
    studio = (viewer / "studio.html").read_text()
    assert "index.html" not in studio and "Classic" not in studio
    assert 'href="./setup.html"' in studio


def test_setup_has_a_way_back_to_the_studio_and_every_element_its_script_needs():
    import re
    page = (REPO / "viewer" / "setup.html").read_text()
    assert page.count('href="./studio.html"') >= 2  # the logo, the Back button, the Done link
    script = (REPO / "viewer" / "studio" / "setup.js").read_text()
    ids = set(re.findall(r"s\('([a-z-]+)'\)", script)) - {"huggingface"}  # that one is a card it builds
    missing = [i for i in ids if f'id="{i}"' not in page]
    assert ids and not missing, missing


def test_a_machine_with_no_route_is_not_told_that_amd_is_coming():
    """AMD ships: five routes in the catalogue list it and the README says so. The
    no-route banner used to promise it anyway -- on the one screen whose whole job is to
    tell a visitor what this machine can actually run."""
    script = (REPO / "viewer" / "studio" / "setup.js").read_text()
    assert "nothing to install yet" in script
    banner = script.split("nothing to install yet")[0].rsplit("return `", 1)[-1]
    for promise in ("coming", "not yet", "later", "soon"):
        assert promise not in banner.lower(), promise


def test_the_banner_it_prints_names_amd_as_a_supported_machine():
    """The sentence the banner builds lists what every backend needs, so it is only honest
    while AMD is in that list. This is the fact the banner above used to contradict."""
    sys.path.insert(0, str(REPO / "viewer"))
    from backend_catalog import catalog_status

    supported = catalog_status(host="other")["host"]["supported"]
    assert any("AMD" in entry for entry in supported), supported


RECIPES = json.loads((REPO / "image_to_3dlab" / "prompt_recipes.json").read_text())


def test_every_goal_has_tips_and_the_3d_goals_add_wording():
    goals = RECIPES["goals"]
    assert set(goals) >= {"character", "prop", "set", "picture", "upload"}
    assert all(goal["tips"] for goal in goals.values())
    for key in ("character", "prop", "set"):
        assert goals[key]["wording"]
    assert "A-pose" in goals["character"]["wording"] and "empty hands" in goals["character"]["wording"]


def test_character_wording_asks_for_clearly_defined_eyes():
    # Small realistic faces came out with a dark-rimmed shelf for eyes; slightly larger,
    # clearly defined eyes with eyelids fixed it (2026-10-14, viking + mage).
    wording = RECIPES["goals"]["character"]["wording"]
    assert "slightly larger, clearly defined eyes with visible eyelids" in wording


def test_the_added_wording_names_only_what_it_wants_and_leaves_fingers_alone():
    # Naming a thing summons it ("no cape" draws a cape), and asking about fingers curls or
    # doubles them (2026-10-12). The words the studio adds follow both rules.
    import re
    for key, goal in RECIPES["goals"].items():
        words = set(re.findall(r"[a-z']+", goal["wording"].lower()))
        assert not words & {"no", "not", "never", "without", "don't"}, key
        assert not words & {"finger", "fingers"}, key


@needs_node
def test_each_starting_point_gets_only_the_steps_it_needs():
    out = _run("plan.js", """console.log(JSON.stringify({
      ideaChar: m.planFor('idea', 'character'),
      pictureProp: m.planFor('picture', 'prop'),
      ideaSet: m.planFor('idea', 'set'),
      ideaPicture: m.planFor('idea', 'picture'),
      glbChar: m.planFor('model', 'character'),
      glbCharFinish: m.planFor('model', 'character', { finishUpload: true }),
      stop: m.stopAfter(m.planFor('idea', 'character'), 'model'),
      reasons: [m.ALLOWED.model.picture, m.ALLOWED.picture.picture],
    }));""")
    assert out["ideaChar"] == ["picture", "model", "finished", "rigged", "animated"]
    assert out["pictureProp"] == ["model", "finished"]
    assert out["ideaSet"] == ["picture", "model", "split"]
    assert out["ideaPicture"] == ["picture"]
    assert out["glbChar"] == ["rigged", "animated"]
    assert out["glbCharFinish"] == ["finished", "rigged", "animated"]
    assert out["stop"] == ["picture", "model"]
    assert all(isinstance(r, str) for r in out["reasons"])


@needs_node
def test_front_and_back_adds_a_back_picture_before_the_3d_model():
    out = _run("plan.js", """console.log(JSON.stringify({
      ideaChar: m.planFor('idea', 'character', { frontBack: true }),
      pictureProp: m.planFor('picture', 'prop', { frontBack: true }),
      ideaSet: m.planFor('idea', 'set', { frontBack: true }),
      ideaPicture: m.planFor('idea', 'picture', { frontBack: true }),
      glbChar: m.planFor('model', 'character', { frontBack: true }),
      off: m.planFor('idea', 'character'),
      label: m.STEP_LABELS.back,
    }));""")
    assert out["ideaChar"] == ["picture", "back", "model", "finished", "rigged", "animated"]
    assert out["pictureProp"] == ["back", "model", "finished"]
    assert out["ideaSet"] == ["picture", "model", "split"]        # a sheet of nine has no back
    assert out["ideaPicture"] == ["picture"]                       # no 3D, no back
    assert out["glbChar"] == ["rigged", "animated"]                # the model exists already
    assert out["off"] == ["picture", "model", "finished", "rigged", "animated"]
    assert out["label"] == "Back picture"


@needs_node
def test_the_prompt_gets_the_goals_wording_and_warnings_for_trouble_words():
    out = _run("plan.js", f"""
      const r = {json.dumps(RECIPES)};
      console.log(JSON.stringify({{
        added: m.composePrompt('An orc blacksmith with short tusks.', 'character', r),
        off: m.composePrompt('An orc blacksmith', 'character', r, false),
        picture: m.composePrompt('A sunset over hills', 'picture', r),
        warnCape: m.troubleFor('a knight with a red cape holding a sword', 'character', r).map(w => w.word),
        propCape: m.troubleFor('a cape on a hook', 'prop', r).length,
        notInside: m.troubleFor('a landscaper gnome', 'character', r).length,
      }}));""")
    assert out["added"].startswith("An orc blacksmith with short tusks, full body, front view, "
                                   "standing in an A-pose")
    assert out["off"] == "An orc blacksmith"
    assert out["picture"] == "A sunset over hills"
    assert out["warnCape"] == ["cape", "holding"]
    assert out["propCape"] == 0            # capes are fine on a prop
    assert out["notInside"] == 0           # "landscaper" is not "landscape"


@needs_node
def test_progress_is_read_from_whichever_field_a_step_uses():
    out = _run("jobs.js", """console.log(JSON.stringify([
      m.percentOf({ overall_pct: 55 }), m.percentOf({ percent: 40 }), m.percentOf({ progress: 0.25 }), m.percentOf({ phase: 'x' })]));""")
    assert out == [55, 40, 25, None]


@needs_node
def test_the_live_log_drops_engine_chatter():
    log = "\n".join([
        "loading model weights",
        "[VERBOSE] ggml - ggml_metal_library_compile_pipeline: loaded kernel_unary_f32_f32_4_op=100_cnt=1",
        "0x9e21eb100 | th_max = 1024 | th_width = 32",
        "",
        "sampling step 3/10",
        "[DEBUG] tensor shapes ok",
        "sampling step 4/10",
    ])
    out = _run("jobs.js", f"console.log(JSON.stringify(m.readableLog({json.dumps(log)}, 4)));")
    assert out == "loading model weights\nsampling step 3/10\nsampling step 4/10"


@needs_node
def test_the_starting_camera_looks_at_the_middle_of_a_model_on_the_floor():
    url = (REPO / "viewer" / "studio" / "floor.js").as_uri()
    three = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    # floor.js imports 'three' by bare name (the page maps it); map it the same way here
    program = f"""
      import {{ register }} from 'node:module';
      register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, n) {{ return s === 'three' ? {{ url: {json.dumps(three)}, shortCircuit: true }} : n(s, c); }}`));
      const m = await import({json.dumps(url)});
      const h = m.homeView(2);
      console.log(JSON.stringify(h));"""
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True, text=True)
    out = json.loads(result.stdout)
    assert out["target"] == [0, 1, 0]
    assert out["position"] == [1.45, 1.35, 2.45]


@needs_node
def test_a_server_run_becomes_the_progress_calls_the_view_understands():
    out = _run("create.js", """
      const a = { steps: ['picture', 'model'], status: 'running', current: 'picture', done: [], made: {}, percent: 10, message: 'Drawing', full_log: '', stalled: false };
      const b = { ...a, current: 'model', done: ['picture'], made: { picture: 'images/p.png' }, percent: null, message: 'Shaping' };
      const c = { ...b, status: 'error', error: 'This Mac ran out of memory.', full_log: 'Killed' };
      console.log(JSON.stringify([m.chainEvents(null, a).map((e) => e[0]), m.chainEvents(a, b), m.chainEvents(b, c).map((e) => e[0]),
                                  m.chainEvents(b, { ...b, status: 'done', done: ['picture', 'model'] }).map((e) => e[0])]));""")
    first, second, failed, finished = out
    assert first == ["step", "progress"]
    assert second[0] == ["done", "picture", {"picture": "images/p.png"}] and second[1] == ["step", "model"]
    assert failed == ["fail"]
    assert finished == ["done", "finish"]


@needs_node
def test_the_page_follows_a_run_the_server_carries_and_lands_on_the_result():
    out = _run("create.js", """
      const snaps = [
        { steps: ['rigged', 'animated'], status: 'running', current: 'rigged', done: [], made: {}, message: 'Rigging' },
        { steps: ['rigged', 'animated'], status: 'done', current: null, done: ['rigged', 'animated'], made: { rigged: 'animate/r.glb' } },
      ];
      const calls = [];
      let sent = null, poll = 0;
      const fetchImpl = async (url, opts) => {
        if (opts?.method === 'POST') { sent = Object.fromEntries(opts.body.entries()); return { ok: true, json: async () => ({ id: 'r1' }) }; }
        return { json: async () => snaps[Math.min(poll++, snaps.length - 1)] };
      };
      const ui = new Proxy({}, { get: (_, k) => (...args) => calls.push(k) });
      const ok = await m.runChain({ steps: ['rigged', 'animated'], description: 'sensei', firstMove: 'walk' },
        { ui, ctx: { reload: async () => {} }, fetchImpl, interval: 0 });
      console.log(JSON.stringify({ ok, calls, title: sent.title, steps: JSON.parse(sent.steps), move: sent.first_move }));""")
    assert out["ok"] is True
    assert out["title"] == "sensei" and out["steps"] == ["rigged", "animated"] and out["move"] == "walk"
    assert out["calls"] == ["begin", "cancelWith", "step", "progress", "done", "done", "finish"]


@needs_node
def test_activity_says_how_long_in_words_and_groups_by_day():
    out = _run("pages.js", """
      const recs = [{ time: '2026-10-06T23:10:00' }, { time: '2026-10-06T09:00:00' }, { time: '2026-10-05T22:00:00' }, { time: '2026-09-30T10:00:00' }];
      console.log(JSON.stringify({ d: [38, 252, 3780, 60].map(m.duration),
        g: m.byDay(recs, new Date('2026-10-06T23:30:00Z')).map((g) => [g.label, g.items.length]) }));""")
    assert out["d"] == ["38 s", "4 min 12 s", "1 h 3 min", "1 min"]
    assert out["g"] == [["Today", 2], ["Yesterday", 1], ["2026-09-30", 1]]


@needs_node
def test_a_character_is_turned_to_face_the_camera_whichever_way_its_feet_point():
    url = (REPO / "viewer" / "studio" / "floor.js").as_uri()
    three = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    program = f"""
      import {{ register }} from 'node:module';
      register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, n) {{ return s === 'three' ? {{ url: {json.dumps(three)}, shortCircuit: true }} : n(s, c); }}`));
      const m = await import({json.dumps(url)});
      // turn each forward by the angle (about +Y) and report where it ends up
      const turned = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.03, 0, -1]].map(([x, y, z]) => {{
        const a = m.facingAngle([x, y, z]);
        return [x * Math.cos(a) + z * Math.sin(a), -x * Math.sin(a) + z * Math.cos(a)].map((v) => Math.round(v * 100) / 100);
      }});
      console.log(JSON.stringify({{ turned, none: m.facingAngle(null) }}));"""
    out = json.loads(subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True).stdout)
    assert out["none"] == 0
    for x, z in out["turned"]:
        assert abs(x) < 0.01 and z > 0.99  # everyone ends up facing +Z, the camera


@needs_node
def test_a_prop_sheet_splits_from_the_studio_not_the_classic_view():
    out = _run("steps.js", """
      const sheet = { kind: 'prop set', model: 'm/sheet.glb', props: [], clips: [] };
      const ctx = { presets: [], engines: [] };
      console.log(JSON.stringify({ sheet: m.stepBody('finished', sheet, ctx), character: m.stepBody('finished', { ...sheet, kind: 'character' }, ctx) }));""")
    assert 'data-run="finished"' in out["sheet"] and "Split into props" in out["sheet"]
    assert "index.html" not in out["sheet"]
    assert "Split into props" not in out["character"]


@needs_node
def test_create_makes_a_fresh_picture_each_time_unless_a_seed_is_kept():
    out = _run("create.js", """
      const fresh = m.pictureSettings({ size: 1024, steps: 16 }, () => 0.5);
      const kept = m.pictureSettings({ newEachTime: false, seed: 7, size: 999, steps: 900 });
      const form = m.planForm({ steps: ['picture'], description: 'x', pictureSettings: kept });
      console.log(JSON.stringify({ fresh, kept, sent: JSON.parse(form.get('picture_settings')) }));""")
    assert out["fresh"] == {"width": 1024, "height": 1024, "steps": 16, "seed": 1073741823}
    assert out["kept"] == {"width": 768, "height": 768, "steps": 50, "seed": 7}  # odd sizes fall back, steps are capped
    assert out["sent"] == out["kept"]


@needs_node
def test_create_asks_for_the_same_picture_steps_the_server_defaults_to():
    """Create sends its own steps, so a default changed only on the server never reaches the studio."""
    from viewer import image_api
    out = _run("create.js", "console.log(JSON.stringify(m.pictureSettings({}, () => 0)));")
    assert out["steps"] == image_api.DEFAULTS["steps"]


@needs_node
def test_create_sends_keep_everything_only_when_ticked():
    out = _run("create.js", """
      const on = m.planForm({ steps: ['model'], keepEverything: true });
      const off = m.planForm({ steps: ['model'] });
      console.log(JSON.stringify([on.get('keep_everything'), off.get('keep_everything')]));""")
    assert out == ["1", None]


@needs_node
def test_every_finished_step_can_be_downloaded_on_its_own():
    """The raw model stays downloadable after Finish (asked for by a user who lost it)."""
    out = _run("library.js", f"""
      const k = {json.dumps(KNIGHT)};
      console.log(JSON.stringify(['model', 'finished', 'rigged', 'picture', 'animated'].map(s => m.stepFile(k, s))));""")
    assert out == ["a/k.glb", "finish/k/k_40k.glb", "animate/k/k_rigged.glb", None, None]


@needs_node
def test_clay_and_normals_swap_the_paint_out_and_put_it_back():
    """Clay and Normals replace every mesh's material; Paint restores the model's own,
    including a mesh with several materials, whose groups need one material per slot."""
    out = _run("looks.js", """
      const one = { material: 'skin', userData: {} };
      const many = { material: ['robe', 'belt'], userData: {} };
      const looks = { clay: 'CLAY', normal: 'NORMAL' };
      const seen = [];
      for (const mode of ['clay', 'normal', 'clay', 'paint']) {
        m.applyLook([one, many], mode, looks);
        seen.push([one.material, many.material]);
      }
      const entries = [{ mesh: one }, { mesh: many }, { mesh: many }];
      console.log(JSON.stringify({ seen, meshes: m.meshesOf(entries).length }));""")
    assert out["seen"] == [
        ["CLAY", ["CLAY", "CLAY"]],
        ["NORMAL", ["NORMAL", "NORMAL"]],
        ["CLAY", ["CLAY", "CLAY"]],
        ["skin", ["robe", "belt"]],
    ]
    assert out["meshes"] == 2   # a two-material mesh is swapped once, not twice


@needs_node
def test_the_face_count_reads_the_viewport_stats():
    out = _run("looks.js", """
      console.log(JSON.stringify([
        m.faceLabel('893,412 faces\\n450,000 verts (as stored)'),
        m.faceLabel('loading… 40%'),
        m.faceLabel(''),
      ]));""")
    assert out == ["893,412 faces", "", ""]


def test_the_viewer_offers_clay_and_normal_views():
    """The buttons exist and are wired to the tested swap; the how-to page names them."""
    viewer = (REPO / "viewer" / "studio" / "viewer.js").read_text()
    for act in ("clay", "normal"):
        assert f'data-act="{act}"' in viewer
    assert "applyLook(meshesOf(view.materials)" in viewer
    guide = (REPO / "viewer" / "studio" / "pages.js").read_text()
    assert "clay" in guide and "normals" in guide and "face count" in guide


@needs_node
def test_view_pictures_is_offered_only_when_a_model_covers_the_picture():
    out = _run("library.js", f"""
      const a = {json.dumps([KNIGHT, FOX, SHEET])};
      console.log(JSON.stringify(a.map(m.canViewPictures)));""")
    # the knight's model hides its picture; the fox is only a picture; the sheet kept none
    assert out == [True, False, False]


RAW_RIG = {"name": "raw rig", "kind": "character", "stage": "rigged", "picture": "images/r/r.png",
           "model": "a/r.glb", "finished": None, "rigged": "animate/r/r_rigged.glb", "clips": [], "props": []}
UPLOAD = {"name": "upload", "kind": "model", "stage": "model", "picture": None, "model": "a/u.glb",
          "finished": None, "rigged": None, "clips": [], "props": []}


@needs_node
def test_steps_unlock_by_the_files_they_need_not_by_order():
    out = _run("library.js", f"""
      const steps = ['picture', 'model', 'finished', 'rigged', 'animated'];
      const a = {json.dumps([KNIGHT, FOX, RAW_RIG, UPLOAD])};
      console.log(JSON.stringify(a.map(x => steps.map(s => m.stepState(x, s)))));""")
    assert out == [
        ["done", "done", "done", "done", "done"],
        ["done", "now", "later", "later", "later"],
        # rigged straight from the raw model: Finish was skipped and can still run
        ["done", "done", "now", "done", "now"],
        # an upload with no picture: Finish and Rig are open, Animate waits for the rig
        ["done", "done", "now", "now", "later"],
    ]


@needs_node
def test_progress_counts_skipped_steps_as_not_done():
    out = _run("library.js", f"""
      const a = {json.dumps([RAW_RIG, UPLOAD])};
      console.log(JSON.stringify(a.map(x => [m.doneCount(x), m.nextStep(x)?.id ?? null])));""")
    assert out == [[3, "finished"], [2, "finished"]]


@needs_node
def test_finish_detail_takes_a_preset_or_a_custom_count():
    out = _run("library.js", """
      console.log(JSON.stringify([m.finishFaces('40000', '1'), m.finishFaces('custom', '62500.4')]));""")
    assert out == [40000, 62500]


@needs_node
def test_each_activity_entry_is_one_line():
    out = _run("pages.js", """
      const r = (step, result, seconds) => ({ time: '2026-10-12T17:19:05', title: 'blocky toy knight', step, result, seconds });
      console.log(JSON.stringify([r('rigged', 'done', 42), r('finished', 'error', 130), r('model', 'cancelled', 5)].map(m.activityLine)));""")
    assert out == [
        "✓ Rig · blocky toy knight · 17:19 · 42 s",
        "! Finish · blocky toy knight · 17:19 · 2 min 10 s · failed",
        "■ 3D model · blocky toy knight · 17:19 · 5 s · stopped",
    ]


@needs_node
def test_the_picture_guide_renders_from_the_real_doc():
    doc = (REPO / "docs" / "prompt-guide.md").read_text()
    out = _run("guide.js", f"""
      const html = m.renderGuide({json.dumps(doc)});
      console.log(JSON.stringify({{ html, links: [
        m.isGuideLink('https://github.com/Bingeljell/image-to-3dlab/blob/main/docs/prompt-guide.md'),
        m.isGuideLink('https://github.com/Bingeljell/image-to-3dlab/blob/main/README.md')] }}));""")
    html = out["html"]
    assert html.startswith("<h2>Picking a picture")
    assert '<img src="/docs/images/prompt-guide-style-and-pose.jpg"' in html   # served by the studio
    assert "<h3>What works</h3>" in html and "<b>1. A soft 3D-render look.</b>" in html
    assert html.count("<li>") == 3                                            # the "go wrong" list
    assert "<pre class=\"log\">A [character] in A-pose" in html
    assert "**" not in html and "](" not in html                              # no raw Markdown left
    assert out["links"] == [True, False]


def test_the_tips_link_to_the_guide_the_studio_opens():
    recipes = json.loads((REPO / "image_to_3dlab" / "prompt_recipes.json").read_text())
    assert any("prompt-guide.md" in tip for tip in recipes["goals"]["character"]["tips"])
    assert (REPO / "docs" / "prompt-guide.md").is_file()


def _version(id_, family, version, updated, **extra):
    return {"id": id_, "family": family, "version": version, "updated": updated, "name": f"knight {version}",
            "kind": "character", "hidden": False, "main": False, **extra}


@needs_node
def test_a_family_is_one_row_that_shows_the_main_version_else_the_newest():
    assets = [_version("c", "f1", 3, 30), _version("a", "f1", 1, 10), _version("b", "f1", 2, 20),
              _version("x", "f2", 1, 25), _version("z", "f3", 1, 5, hidden=True)]
    out = _run("library.js", f"""
      const a = {json.dumps(assets)};
      const rows = (list, o) => m.visibleFamilies(list, o).rows.map(r => [r.key, r.shown.id, r.members.map(x => x.id).join('')]);
      const starred = a.map(x => x.id === 'a' ? {{ ...x, main: true }} : x);
      console.log(JSON.stringify({{
        plain: rows(a, {{}}),
        starred: rows(starred, {{}}),
        withHidden: rows(a, {{ showHidden: true }}),
        paged: m.visibleFamilies(a, {{ limit: 1 }}).more,
      }}));""")
    assert out["plain"] == [["f1", "c", "abc"], ["f2", "x", "x"]]     # newest shown; versions in order
    assert out["starred"][0] == ["f1", "a", "abc"]                    # the star wins over newest
    assert [r[0] for r in out["withHidden"]] == ["f1", "f2", "f3"]
    assert out["paged"] == 1


@needs_node
def test_the_version_a_step_just_made_is_the_new_id_in_the_same_family():
    before = [_version("a", "f1", 1, 10), _version("x", "f2", 1, 12)]
    after = before + [_version("b", "f1", 2, 40), _version("y", "f2", 2, 41)]
    out = _run("library.js", f"""
      const before = {json.dumps(before)}, after = {json.dumps(after)};
      console.log(JSON.stringify({{
        made: m.madeVersion(before, after, 'f1')?.id ?? null,
        none: m.madeVersion(after, after, 'f1')?.id ?? null,
      }}));""")
    assert out == {"made": "b", "none": None}      # f2's new version is someone else's


@needs_node
def test_a_finished_step_can_be_run_again_as_a_new_version():
    out = _run("steps.js", """
      const ctx = { presets: [], engines: [{ id: 'pixal3d', label: 'Pixal3D', license: 'x' }] };
      const knight = { kind: 'character', picture: 'i/k.png', prompt: 'A knight', model: 'm/k.glb', finished: 'f/k.glb', rigged: 'r/k.glb', clips: [] };
      const raw = { ...knight, finished: null };
      console.log(JSON.stringify({
        model: m.stepBody('model', knight, ctx), rig: m.stepBody('rigged', knight, ctx), rigRaw: m.stepBody('rigged', raw, ctx),
        picture: m.stepBody('picture', knight, ctx), noPrompt: m.stepBody('picture', { ...knight, prompt: null }, ctx),
        firstRig: m.stepBody('rigged', { ...raw, rigged: null }, ctx),
      }));""")
    assert "Make another 3D model" in out["model"] and "new version" in out["model"]
    assert "Re-rig" in out["rig"] and "new version" in out["rig"]
    assert 'data-k="rigsource"' in out["rig"]                       # finished and raw both exist: you choose
    assert 'data-k="rigsource"' not in out["rigRaw"]                # only the raw one: nothing to choose
    assert "Re-rig" not in out["firstRig"]                          # a first rig is just "rig it"
    assert "Draw another picture" in out["picture"] and out["noPrompt"] == ""


@needs_node
def test_a_deletion_is_described_in_plain_words_and_sizes():
    out = _run("library.js", """
      const folders = (...paths) => ({ folders: paths.map((path) => ({ path })) });
      console.log(JSON.stringify({
        rig: m.deletionWords(folders('animate/a__rig__1')),
        two: m.deletionWords(folders('finish/a__finish__1', 'animate/a__rig__1')),
        three: m.deletionWords(folders('images/p', 'commercial_conditional/m', 'finish/f')),
        none: m.deletionWords(folders()),
        sizes: [m.sizeText(0), m.sizeText(500 * 1024), m.sizeText(12.4 * 1024 ** 2), m.sizeText(1.46 * 1024 ** 3), m.sizeText('x')],
      }));""")
    assert out["rig"] == "its rig and moves" and out["two"] == "its finished model and its rig and moves"
    assert out["three"] == "its picture, its 3D model and its finished model" and out["none"] == "nothing"
    assert out["sizes"] == ["1 KB", "500 KB", "12 MB", "1.5 GB", ""]


@needs_node
def test_the_library_filters_by_step_engine_and_main_and_sorts_by_name():
    assets = [
        _version("a", "f1", 1, 30, name="Zed", stage="animated", model="x/z__pixal3d__1/z.glb"),
        _version("b", "f2", 1, 20, name="alpha", stage="model", model="x/a__trellis__1/a.glb", main=True),
        _version("c", "f3", 1, 10, name="Mid", stage="picture", model=None, finished=None),
        _version("d", "f1", 2, 5, name="Zed · v2", stage="model", model="x/z__trellis__2/z.glb"),
    ]
    out = _run("library.js", f"""
      const a = {json.dumps(assets)};
      const names = (o) => m.visibleFamilies(a, o).rows.map(r => r.shown.id).join('');
      console.log(JSON.stringify({{
        newest: names({{}}), byName: names({{ sort: 'name' }}),
        picture: names({{ stage: 'picture' }}), model: names({{ stage: 'model' }}),
        trellis: names({{ engine: 'TRELLIS.2' }}), starred: names({{ starred: true }}),
        engines: m.engineChoices(a),
      }}));""")
    assert out["newest"] == "abc" and out["byName"] == "bca"      # families: Zed (v2 inside), alpha, Mid
    assert out["picture"] == "c" and out["model"] == "bd"     # a family shows the version that matches
    assert out["trellis"] == "bd" and out["starred"] == "b"
    assert out["engines"] == ["Pixal3D", "TRELLIS.2"]


@needs_node
def test_finish_offers_three_tested_detail_levels_and_warns_about_custom():
    out = _run("steps.js", """
      const ctx = { presets: [], engines: [] };
      const html = m.stepBody('finished', { kind: 'character', picture: 'i/k.png', model: 'm/k.glb', finished: null, clips: [] }, ctx);
      const options = [...html.matchAll(/<option value="(\\w+)"( selected)?>([^<]*)<\\/option>/g)]
        .filter((o) => /^(\\d+|custom)$/.test(o[1])).map((o) => [o[1], !!o[2], o[3]]);
      console.log(JSON.stringify({ options, hint: /unexpected results/.test(html) }));""")
    values = [o[0] for o in out["options"] if o[0] != "1024"]
    detail = [o for o in out["options"] if o[0] in ("10000", "20000", "40000", "custom", "5000", "100000", "200000")]
    assert [o[0] for o in detail] == ["10000", "20000", "40000", "custom"]       # no 5k (broken), 100k or 200k
    assert [o[0] for o in detail if o[1]] == ["20000"]                           # 20,000 is the default
    assert "unexpected results" in detail[-1][2] and out["hint"] is True
    assert values


@needs_node
def test_an_agents_link_names_the_asset_to_open():
    out = _run("library.js", """console.log(JSON.stringify(
      ['#asset=03711bf99d31', 'asset=03711bf99d31', '#asset=../../x', '#asset=', '', null, '#other=1&asset=0123456789ab'].map(m.linkedAssetId)));""")
    assert out == ["03711bf99d31", "03711bf99d31", None, None, None, None, "0123456789ab"]


@needs_node
def test_finish_detail_sends_a_triangle_count_or_full_detail():
    out = _run("library.js", """console.log(JSON.stringify(
      [m.finishDetailSettings('20000'), m.finishDetailSettings('custom', '30000'), m.finishDetailSettings('full')]));""")
    assert out == [{"faces": 20000}, {"faces": 30000}, {"keep_mesh": True}]


@needs_node
def test_finish_offers_the_edge_fade_switch_on_by_default_only_with_a_picture():
    out = _run("steps.js", """
      const ctx = { presets: [], engines: [] };
      const asset = (picture) => ({ kind: 'character', picture, model: 'm/k.glb', finished: null, clips: [] });
      console.log(JSON.stringify([asset('i/k.png'), asset(null)].map((a) => m.stepBody('finished', a, ctx))
        .map((html) => (html.match(/<input type="checkbox" data-k="edgefade"( checked)?>/) || [null, null]).map(Boolean))));""")
    assert out == [[True, True], [False, False]]


@needs_node
def test_finish_offers_the_nudge_switch_on_by_default_only_with_a_picture():
    out = _run("steps.js", """
      const ctx = { presets: [], engines: [] };
      const asset = (picture) => ({ kind: 'character', picture, model: 'm/k.glb', finished: null, clips: [] });
      console.log(JSON.stringify([asset('i/k.png'), asset(null)].map((a) => m.stepBody('finished', a, ctx))
        .map((html) => (html.match(/<input type="checkbox" data-k="nudge"( checked)?>/) || [null, null]).map(Boolean))));""")
    assert out == [[True, True], [False, False]]


@needs_node
def test_the_raw_model_stays_viewable_after_later_steps():
    finished = {**KNIGHT, "rigged": None, "clips": [], "stage": "finished"}
    raw_only = {**KNIGHT, "finished": None, "rigged": None, "clips": [], "stage": "model"}
    out = _run("library.js", f"""
      const cases = {json.dumps([KNIGHT, finished, raw_only, SHEET])};
      console.log(JSON.stringify(cases.map((a) => ['model', 'finished', 'rigged'].map((s) => m.canViewStep(a, s)))));""")
    assert out == [
        [True, True, False],     # rigged: raw and finished both viewable; the rig is already on screen
        [True, False, False],    # finished: the raw model is one click away
        [False, False, False],   # raw only: it is already what the viewer shows
        [False, False, False],   # a prop set shows its props
    ]


@needs_node
def test_model_stats_read_as_short_rows():
    out = _run("library.js", """console.log(JSON.stringify({
        rigged: m.statsRows({triangles: 19981, vertices: 23130, materials: 1, bones: 52, moves: ['idle', 'walk'],
                             bytes: 4.6 * 1024 ** 2,
                             textures: [{width: 2048, height: 2048}, {width: 2048, height: 2048}, {width: 1024, height: 1024}]}),
        plain: m.statsRows({triangles: 3, vertices: 3, materials: 0, bones: 0, moves: [], bytes: 2048,
                            textures: [{width: null, height: null}]}),
        none: m.statsRows(null),
      }));""")
    assert out["rigged"] == [["Triangles", "19,981"], ["Vertices", "23,130"],
                             ["Textures", "2 × 2048², 1 × 1024²"], ["Bones", "52"], ["Moves", "2"],
                             ["File size", "5 MB"]]
    assert out["plain"] == [["Triangles", "3"], ["Vertices", "3"], ["Textures", "1"], ["File size", "2 KB"]]
    assert out["none"] == []


@needs_node
def test_show_in_folder_is_offered_only_on_the_studios_own_machine():
    out = _run("library.js", """console.log(JSON.stringify(
        ['localhost', '127.0.0.1', '[::1]', '100.64.1.2', 'my-mac.local'].map(m.canReveal)));""")
    assert out == [True, True, True, False, False]


@needs_node
def test_a_picture_on_the_clipboard_becomes_a_named_file_and_anything_else_is_ignored():
    out = _run("create.js", """
      const file = (name, type) => new File(['x'], name, { type });
      const named = m.pastedPicture({ files: [file('image.png', 'image/png')] });
      const unnamed = m.pastedPicture({ files: [file('', 'image/jpeg')] }, () => 5);
      console.log(JSON.stringify({
        named: named.name, unnamed: [unnamed.name, unnamed.type],
        gif: m.pastedPicture({ files: [file('a.gif', 'image/gif')] }),
        text: m.pastedPicture({ files: [] }), none: m.pastedPicture(null),
        second: m.pastedPicture({ files: [file('notes.txt', 'text/plain'), file('', 'image/webp')] }, () => 9).name }));""")
    assert out == {"named": "image.png", "unnamed": ["pasted-5.jpg", "image/jpeg"], "gif": None,
                   "text": None, "none": None, "second": "pasted-9.webp"}


@needs_node
def test_a_picture_file_size_reads_in_bytes_kilobytes_or_megabytes():
    out = _run("create.js", """
      console.log(JSON.stringify([0, 500, 2048, 1048575, 3145728].map(m.fileSizeText)));""")
    assert out == ["0 B", "500 B", "2 KB", "1023 KB", "3.0 MB"]


@needs_node
def test_closing_a_sheet_from_outside_runs_its_own_clean_up():
    """2026-10-19 (#141): Create's paste listener outlived a sheet closed by a Library click,
    because closeSheet only removed the element. A sheet's clean-up now always runs."""
    out = _run("pages.js", """
      const calls = [];
      const el = { onSheetClose: () => calls.push('clean'), remove: () => calls.push('remove') };
      m.closeSheet({ querySelector: () => el });
      m.closeSheet({ querySelector: () => ({ remove: () => calls.push('plain') }) });
      m.closeSheet({ querySelector: () => null });
      console.log(JSON.stringify(calls));""")
    assert out == ["clean", "remove", "plain"]


@needs_node
def test_without_blender_finish_says_so_before_anyone_clicks():
    """2026-10-19's first site report: Finish was offered on a Mac with no Blender, and failed."""
    out = _run("steps.js", """
      const model = { kind: 'character', model: 'm/k.glb', picture: 'images/k.png', props: [], clips: [] };
      const sheet = { kind: 'prop set', model: 'm/sheet.glb', props: [], clips: [] };
      const ctx = (ready) => ({ presets: [], engines: [], blenderReady: ready });
      console.log(JSON.stringify([m.stepBody('finished', model, ctx(false)), m.stepBody('finished', sheet, ctx(false)),
                                  m.stepBody('finished', model, ctx(true)), m.stepBody('finished', model, { presets: [], engines: [] })]));""")
    missing, sheet, ready, unknown = out
    assert "needs Blender" in missing and "blender.org" in missing and 'data-run="finished"' not in missing
    assert sheet.startswith("<p") and "Splitting into props" in sheet
    assert 'data-run="finished"' in ready and 'data-run="finished"' in unknown
