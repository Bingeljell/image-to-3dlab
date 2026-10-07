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


RECIPES = json.loads((REPO / "image_to_3dlab" / "prompt_recipes.json").read_text())


def test_every_goal_has_tips_and_the_3d_goals_add_wording():
    goals = RECIPES["goals"]
    assert set(goals) >= {"character", "prop", "set", "picture", "upload"}
    assert all(goal["tips"] for goal in goals.values())
    for key in ("character", "prop", "set"):
        assert goals[key]["wording"]
    assert "T-pose" in goals["character"]["wording"] and "empty hands" in goals["character"]["wording"]


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
    assert out["added"].startswith("An orc blacksmith with short tusks, full body, standing in a T-pose")
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
