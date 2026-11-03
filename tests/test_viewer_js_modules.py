"""Focused tests for dependency-free browser modules that can run under Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_rig_inspection_deduplicates_bones_and_resets_each_skeleton():
    module_url = (REPO / "viewer" / "animation" / "rig-inspection.js").as_uri()
    program = f"""
      import {{ describeBone, inspectRig, resetRigPose }} from {json.dumps(module_url)};
      const parent = {{ isBone: true, name: 'root' }};
      const child = {{ isBone: true, name: 'knee' }};
      const bone = {{
        isBone: true, name: 'hip', parent, children: [child],
        position: {{ x: 1, y: 2, z: 3 }},
        quaternion: {{ x: 0, y: 0, z: 0, w: 1 }},
        scale: {{ x: 1, y: 1, z: 1 }},
      }};
      const skeleton = {{ bones: [bone], resets: 0, pose() {{ this.resets += 1; }} }};
      const mesh = {{ isSkinnedMesh: true, skeleton }};
      const root = {{ traverse(callback) {{ callback(bone); callback(mesh); callback(mesh); }} }};
      const rig = inspectRig(root, [{{ name: 'idle' }}]);
      bone.position.x = 4;
      const description = describeBone(bone, rig);
      resetRigPose(rig);
      console.log(JSON.stringify({{
        bones: rig.bones.map((item) => item.name),
        skeletons: rig.skeletons.length,
        meshes: rig.skinnedMeshes.length,
        clips: rig.animations.map((item) => item.name),
        resets: skeleton.resets,
        description,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "bones": ["hip"],
        "skeletons": 1,
        "meshes": 2,
        "clips": ["idle"],
        "resets": 1,
        "description": {
            "name": "hip",
            "parent": "root",
            "children": ["knee"],
            "bind": {
                "position": [1, 2, 3],
                "quaternion": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
            "pose": {
                "position": [4, 2, 3],
                "quaternion": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
        },
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_animation_player_owns_transport_without_owning_threejs():
    module_url = (REPO / "viewer" / "animation" / "player.js").as_uri()
    program = f"""
      import {{ AnimationPlayer }} from {json.dumps(module_url)};
      let queued = null;
      const action = {{
        time: 0, paused: false,
        reset() {{ this.time = 0; return this; }},
        play() {{ return this; }}
      }};
      const mixer = {{
        clipAction() {{ return action; }},
        stopAllAction() {{}},
        update(delta) {{ action.time += delta; }},
        addEventListener() {{}},
        getRoot() {{ return {{}}; }},
        uncacheRoot() {{}},
      }};
      const frames = [];
      const states = [];
      const player = new AnimationPlayer(mixer, {{
        requestFrame(callback) {{ queued = callback; return 7; }},
        cancelFrame() {{ queued = null; }},
        onFrame(time, duration) {{ frames.push([time, duration]); }},
        onStateChange(playing) {{ states.push(playing); }},
      }});
      player.select({{ name: 'walk', duration: 2 }});
      player.play();
      queued(1000);
      queued(1500);
      player.pause();
      player.seek(9);
      console.log(JSON.stringify({{
        time: player.time,
        duration: player.duration,
        paused: action.paused,
        states,
        lastFrame: frames.at(-1),
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "time": 2,
        "duration": 2,
        "paused": True,
        "states": [False, False, True, False],
        "lastFrame": [2, 2],
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_camera_view_controls_snap_and_reset_without_touching_model_state():
    controls_url = (REPO / "viewer" / "components" / "camera-view-controls.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    program = f"""
      import * as THREE from {json.dumps(three_url)};
      import {{ setCameraView }} from {json.dumps(controls_url)};
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(0, 0, 4);
      const controls = {{ target: new THREE.Vector3(), updates: 0, update() {{ this.updates++; }} }};
      const untouched = {{ rigEdits: 3, pose: 'idle' }};
      setCameraView({{ camera, controls }}, 'left');
      const left = camera.position.toArray();
      setCameraView({{ camera, controls }}, 'top');
      const top = camera.position.toArray();
      const topUp = camera.up.toArray();
      setCameraView({{ camera, controls }}, 'reset');
      console.log(JSON.stringify({{
        left, top, topUp, reset: camera.position.toArray(), target: controls.target.toArray(),
        updates: controls.updates, untouched,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "left": [-4, 0, 0],
        "top": [0, 4, 0],
        "topUp": [0, 0, -1],
        "reset": [1.45, 0.6, 2.45],
        "target": [0, 0, 0],
        "updates": 3,
        "untouched": {"rigEdits": 3, "pose": "idle"},
    }


def test_rig_sidecar_schema_is_valid_json_and_versioned():
    schema = json.loads((REPO / "rigs" / "rig-sidecar.schema.json").read_text())

    assert schema["properties"]["schemaVersion"]["const"] == 1
    assert schema["properties"]["coordinateSpace"]["const"] == "armature-local"
    assert "binding" in schema["properties"]


def _run_panel(script: str) -> object:
    """Drive the shipped JobProgressPanel against a minimal DOM and read the rows back.

    The panel is the thing that actually renders "estimating…", so it is the thing under
    test — a re-implementation of its rules here would pass while the shipped file stayed
    broken, which is exactly how this bug survived (2026-09-21).
    """
    module_url = (REPO / "viewer" / "components" / "job-progress.js").as_uri()
    program = f"""
      import {{ JobProgressPanel }} from {json.dumps(module_url)};

      // Enough DOM for the panel: rows it creates, and the three text elements it writes.
      const make = () => {{
        const node = {{
          className: '', dataset: {{}}, innerHTML: '', textContent: '', style: {{}},
          children: [],
          classList: {{
            add: (c) => {{ node.className += ' ' + c; }},
            contains: (c) => node.className.split(/\\s+/).includes(c),
          }},
          appendChild: (child) => {{ node.children.push(child); return child; }},
          querySelector: (selector) => {{
            const key = selector.replace('.', '');
            node._parts = node._parts || {{}};
            node._parts[key] = node._parts[key] || make();
            return node._parts[key];
          }},
        }};
        return node;
      }};
      globalThis.document = {{ createElement: make }};

      const host = make(), bar = make(), label = make(), eta = make();
      const panel = new JobProgressPanel({{ stages: host, bar, label, eta }});
      const rowState = () => host.children.map((row) => [
        row.className.trim(), row.querySelector('.stage-detail').textContent,
      ]);
      {script}
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _run_welcome(expr: str):
    module_url = (REPO / "viewer" / "components" / "welcome-content.js").as_uri()
    program = f"import * as w from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


WELCOME = {
    "brand": {"name": "AssetFurnace"},
    "version": "0.3.0",
    "routes": [{"id": "pixal3d", "state": "missing"}, {"id": "qwen-image", "state": "ready"}],
    "news": [{"version": "0.3.0", "sections": {
        "Fixed": ["f1"], "Added": ["a1", "a2", "a3"], "Changed": ["c1"]}}],
}


def _run_banner(expr: str):
    module_url = (REPO / "viewer" / "components" / "update-banner.js").as_uri()
    program = f"import * as b from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


CHECK = {"enabled": True, "newer": True, "current": "0.2.0", "latest": "0.3.0",
         "url": "https://github.com/x/releases/tag/v0.3.0", "command": "curl … | bash"}


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_gated_warning_goes_once_access_is_confirmed_and_no_sooner():
    module_url = (REPO / "viewer" / "components" / "gated-caveat.js").as_uri()
    program = f"""
      import {{ showsCaveat }} from {json.dumps(module_url)};
      const gated = {{ caveat: 'gated', gated_repo: 'org/model' }};
      const other = {{ caveat: 'not licensed in the EU', gated_repo: null }};
      console.log(JSON.stringify([
        showsCaveat(gated, {{ 'org/model': 'yes' }}),
        showsCaveat(gated, {{ 'org/model': 'no' }}),
        showsCaveat(gated, {{ 'org/model': 'unknown' }}),
        showsCaveat(gated, {{}}),
        showsCaveat(other, {{ 'org/model': 'yes' }}),
        showsCaveat({{ caveat: null }}, {{}}),
      ]));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [False, True, True, True, True, False]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_the_setup_panel_says_a_run_finished_and_looks_like_it():
    """The title used to be written once, at the start, and never again: a 16 GB fetch that
    ended perfectly sat there saying "Downloading", at 100%, looking like a page that gave up
    halfway. The panel is now also *painted* with how the run ended, not just titled."""
    module_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ runTitle, setupVerb, panelState }} from {json.dumps(module_url)};
      const weights = {{ label: 'TRELLIS.2 (AMD)', build_present: true,
                         setup_fetches_weights: true }};
      const unbuilt = {{ label: 'TRELLIS.2 (AMD)', build_present: false,
                         setup_fetches_weights: true }};
      const logs = {{ label: 'Pixal3D', build_present: true, setup_fetches_weights: false }};
      const inFlight = ['queued', 'log', 'downloading', 'building'];
      const endings = ['done', 'error', 'cancelled'];
      console.log(JSON.stringify({{
        verbs: [setupVerb(unbuilt, 'download'), setupVerb(weights, 'download'),
                setupVerb(logs, 'download'), setupVerb(weights, 'rebuild')],
        whileRunning: inFlight.map((phase) => runTitle(weights, 'download', {{ phase }})),
        states: inFlight.map((phase) => panelState({{ phase }})),
        end: endings.map((phase) => runTitle(weights, 'download', {{ phase }})),
        endStates: endings.map((phase) => panelState({{ phase }})),
        // A resumed run whose backend the catalogue no longer lists still names something.
        unknown: runTitle({{ label: 'trellis-rocm' }}, 'download', {{ phase: 'done' }}),
      }}));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    out = json.loads(result.stdout)

    # The third card is built and its code bootstrap fetches nothing. A download
    # action on it is still a download (the Mac TRELLIS weights), not another build.
    assert out["verbs"] == ["Setting up", "Downloading", "Downloading", "Rebuilding"]
    assert out["whileRunning"] == ["Downloading TRELLIS.2 (AMD)"] * 4, (
        "a run in flight must not claim an outcome it has not reached")
    assert out["states"] == ["running"] * 4
    assert out["end"] == ["✓ Installed · TRELLIS.2 (AMD)", "! Setup failed · TRELLIS.2 (AMD)",
                          "■ Setup cancelled · TRELLIS.2 (AMD)"]
    assert out["endStates"] == ["done", "error", "cancelled"]
    assert out["unknown"] == "✓ Installed · trellis-rocm"


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_the_finished_marks_are_the_studios_own():
    """One convention: the Activity list marks a finished thing '✓', '!' and '■', and the Setup
    panel borrows them rather than inventing a second set."""
    pages_url = (REPO / "viewer" / "studio" / "pages.js").as_uri()
    setup_url = (REPO / "viewer" / "components" / "setup-progress.js").as_uri()
    program = f"""
      import {{ MARK, activityParts }} from {json.dumps(pages_url)};
      import {{ runTitle }} from {json.dumps(setup_url)};
      const backend = {{ label: 'Stable Fast 3D', build_present: true,
                         setup_fetches_weights: true }};
      console.log(JSON.stringify({{
        done: runTitle(backend, 'download', {{ phase: 'done' }}).split(' ')[0],
        error: runTitle(backend, 'download', {{ phase: 'error' }}).split(' ')[0],
        cancelled: runTitle(backend, 'download', {{ phase: 'cancelled' }}).split(' ')[0],
        activity: activityParts({{ result: 'done', step: 'rig', title: 'x', time: '' }}),
      }}));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    out = json.loads(result.stdout)

    assert (out["done"], out["error"], out["cancelled"]) == ("✓", "!", "■")
    assert out["activity"][0].split(" ")[0] == out["done"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_model_load_error_is_rendered_as_text_not_markup():
    """H1: a crafted GLB error message must not be parsed as HTML in the studio overlay."""
    module_url = (REPO / "viewer" / "studio" / "viewer.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    stub_viewport = (
        "export function createModelViewport(opts) { "
        "globalThis.__capturedOnError = opts.onError; "
        "return { dispose() {} }; "
        "}\n"
        "export function disposeModelViewport() {}"
    )
    stub_viewport_url = "data:text/javascript," + quote(stub_viewport, safe="")
    program = f"""
      import {{ register }} from 'node:module';
      register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, n) {{
        if (s === 'three') return {{ url: {json.dumps(three_url)}, shortCircuit: true }};
        if (s === '../components/model-viewport.js' && c.parentURL === {json.dumps(module_url)}) {{
          return {{ url: {json.dumps(stub_viewport_url)}, shortCircuit: true }};
        }}
        return n(s, c);
      }}`));

      const make = () => {{
        const node = {{
          nodeType: 1, className: '', dataset: {{}}, hidden: false, style: {{}}, children: [],
          _html: '', _text: '', _listeners: {{}},
          get innerHTML() {{ return node._html; }},
          set innerHTML(value) {{ node._html = value; node.children = []; node._text = ''; }},
          get textContent() {{ return node._text || node.children.map((c) => c.textContent).join(''); }},
          set textContent(value) {{ node._text = String(value); node.children = []; node._html = ''; }},
          classList: {{
            add: (c) => {{ node.className += (node.className ? ' ' : '') + c; }},
            toggle: () => {{}},
            contains: (c) => node.className.split(/\\s+/).includes(c),
          }},
          appendChild: (child) => {{ node.children.push(child); return child; }},
          querySelector: () => make(),
          querySelectorAll: () => [],
          addEventListener: (type, fn) => {{ (node._listeners[type] = node._listeners[type] || []).push(fn); }},
          setAttribute: (k, v) => {{ node[k] = String(v); }},
          getAttribute: (k) => node[k] ?? null,
          remove: () => {{}},
          closest: () => null,
        }};
        return node;
      }};

      globalThis.document = {{ createElement: make }};
      globalThis.window = globalThis;
      globalThis.addEventListener = () => {{}};
      globalThis.removeEventListener = () => {{}};
      globalThis.ResizeObserver = class {{ observe() {{}} disconnect() {{}} }};
      globalThis.requestAnimationFrame = () => 0;
      globalThis.devicePixelRatio = 1;

      const {{ createStudioViewer }} = await import({json.dumps(module_url)});
      const stage = make();
      const viewer = createStudioViewer({{ stage, logoUrl: 'logo.svg' }});
      viewer.load('test.glb');
      const payload = '<img src=x onerror=alert(1)>';
      globalThis.__capturedOnError({{ message: payload }});

      const overlay = stage.children.find((n) => n.className === 'sv-overlay');
      console.log(JSON.stringify({{
        textContains: overlay.textContent.includes(payload),
        htmlContains: overlay.innerHTML.includes(payload),
        textContent: overlay.textContent,
      }}));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True, capture_output=True, text=True,
    )
    out = json.loads(result.stdout)

    assert out["textContains"] is True
    assert out["htmlContains"] is False


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_studio_viewer_3d_tools_follow_real_picture_and_empty_mode_switches():
    """Toolbar/gizmo are inert for pictures and empty stages, but return for 3D models."""
    module_url = (REPO / "viewer" / "studio" / "viewer.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    stubs = {
        "../components/model-viewport.js": "export function createModelViewport() { return { renderer: { domElement: { remove() {} } } }; } export function disposeModelViewport() {}",
        "../components/camera-view-controls.js": "export function setCameraView() { return false; }",
        "../animation/player.js": "export class AnimationPlayer {}",
        "./gizmo.js": "export const AXES = []; export function projectAxes() { return []; }",
        "./bones.js": "export function createBoneDisplay() { return null; }",
        "./floor.js": "export function createFloor() { return {}; } export function standOnFloor() { return 1; } export function homeView() { return { target: [0, 0, 0], position: [0, 0, 1] }; } export function facingAngle() { return 0; }",
        "./looks.js": "export function applyLook() {} export function meshesOf() { return []; } export function faceLabel(s) { return s; }",
        "./zoom.js": "export function attachZoom(overlay) { overlay.zoomAttached = true; }",
    }
    stub_urls = {key: "data:text/javascript," + quote(value, safe="") for key, value in stubs.items()}
    resolve_cases = "".join(
        f"if (s === {json.dumps(spec)} && c.parentURL === {json.dumps(module_url)}) return {{ url: {json.dumps(url)}, shortCircuit: true }};"
        for spec, url in stub_urls.items()
    )
    program = f"""
      import {{ register }} from 'node:module';
      register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, n) {{
        if (s === 'three') return {{ url: {json.dumps(three_url)}, shortCircuit: true }};
        {resolve_cases}
        return n(s, c);
      }}`));
      const make = () => {{
        const node = {{ nodeType: 1, className: '', hidden: false, inert: false, style: {{}}, children: [], _html: '', zoomAttached: false,
          get innerHTML() {{ return node._html; }}, set innerHTML(value) {{ node._html = value; node.children = []; }},
          get textContent() {{ return ''; }}, set textContent(value) {{ node._text = String(value); }},
          classList: {{ add() {{}}, remove() {{}}, toggle() {{}}, contains() {{ return false; }} }},
          appendChild(child) {{ node.children.push(child); return child; }}, querySelector: () => make(), querySelectorAll: () => [],
          addEventListener() {{}}, setAttribute(k, v) {{ node[k] = String(v); }}, getAttribute(k) {{ return node[k] ?? null; }},
          remove() {{}}, closest: () => null }};
        return node;
      }};
      globalThis.document = {{ createElement: make }}; globalThis.window = globalThis;
      globalThis.addEventListener = () => {{}}; globalThis.removeEventListener = () => {{}};
      globalThis.ResizeObserver = class {{ observe() {{}} disconnect() {{}} }};
      globalThis.requestAnimationFrame = () => 0; globalThis.devicePixelRatio = 1;
      const {{ createStudioViewer, pictureOverlayTop, pictureMaxHeight }} = await import({json.dumps(module_url)});
      const stage = make(); const viewer = createStudioViewer({{ stage, logoUrl: 'logo.svg' }});
      const tools = stage.children.find((node) => node.className === 'sv-tools');
      const state = () => [tools.hidden, tools.inert];
      viewer.load('model.glb'); const model = state();
      viewer.showImage('picture.png', 'Picture'); const picture = state(); const zoom = stage.children.find((node) => node.className === 'sv-overlay').zoomAttached;
      viewer.load('model-again.glb'); const modelAgain = state();
      viewer.showEmpty('Nothing here'); const empty = state();
      console.log(JSON.stringify({{ model, picture, modelAgain, empty, zoom,
        pictureTop: pictureOverlayTop({{ top: 100 }}, {{ bottom: 180 }}),
        pictureMax: [pictureMaxHeight({{ top: 0, height: 600 }}, {{ bottom: 180 }}, 40, 20),
          pictureMaxHeight({{ top: 0, height: 100 }}, {{ bottom: 180 }}, 40, 20)] }}));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {
        "model": [False, False], "picture": [True, True], "modelAgain": [False, False], "empty": [True, True], "zoom": True,
        "pictureTop": 92,
        "pictureMax": [316, 0],
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_picture_zoom_clicks_in_at_the_pointer_and_does_not_zoom_after_drag():
    """The shipped zoom module makes its zoom-in cursor actionable without stealing pans."""
    module_url = (REPO / "viewer" / "studio" / "zoom.js").as_uri()
    program = f"""
      const {{ attachZoom }} = await import({json.dumps(module_url)});
      const listeners = {{}};
      const box = {{
        getBoundingClientRect: () => ({{ left: 0, top: 0, width: 400, height: 300 }}),
        addEventListener: (name, fn) => {{ listeners[name] = fn; }},
        setPointerCapture: () => {{}},
      }};
      const image = {{ style: {{}}, parentElement: box }};
      const root = {{ querySelectorAll: () => [image], querySelector: () => null }};
      attachZoom(root);
      listeners.click({{ clientX: 100, clientY: 50 }});
      const zoomed = [image.style.transform, image.style.cursor];
      listeners.pointerdown({{ clientX: 100, clientY: 100, pointerId: 1 }});
      listeners.pointermove({{ clientX: 130, clientY: 100 }});
      listeners.pointerup();
      listeners.click({{ clientX: 130, clientY: 100 }});
      const afterDrag = [image.style.transform, image.style.cursor];
      listeners.dblclick();
      console.log(JSON.stringify({{ zoomed, afterDrag, reset: image.style.transform }}));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {
        "zoomed": ["translate(-100px, -50px) scale(2)", "grab"],
        "afterDrag": ["translate(-70px, -50px) scale(2)", "grab"],
        "reset": "translate(0px, 0px) scale(1)",
    }


def test_picture_overlay_leaves_room_for_stage_heading():
    """Picture mode starts below the absolute title instead of drawing underneath it."""
    css = (REPO / "viewer" / "studio" / "studio.css").read_text()
    assert ".stage:has(.sv-picture) .sv-overlay" in css
    assert "align-items: start; padding: var(--sv-picture-top, 24px) 24px 24px" in css
    assert "var(--sv-picture-max-height, 78vh)" in css


def viewer_scripts() -> list[Path]:
    """Every first-party viewer script; vendored libraries are someone else's to parse."""
    viewer = REPO / "viewer"
    return sorted(p for p in viewer.rglob("*.js")
                  if "vendor" not in p.relative_to(viewer).parts
                  and "node_modules" not in p.parts)


@pytest.mark.skipif(NODE is None, reason="Node is required to parse browser ES modules")
@pytest.mark.parametrize("script", viewer_scripts(), ids=lambda p: str(p.relative_to(REPO)))
def test_every_viewer_script_parses(script):
    # One syntax error in any module stops the whole app loading: no tabs, no drop, no
    # browse, and dropped files download instead. 0.3.5 shipped exactly that.
    result = subprocess.run(
        [NODE, "--input-type=module", "--check"],
        stdin=script.open("rb"),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()

