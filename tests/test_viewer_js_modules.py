"""Focused tests for dependency-free browser modules that can run under Node."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

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

