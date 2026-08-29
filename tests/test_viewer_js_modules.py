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
def test_job_progress_formats_durations_from_the_shipped_module():
    module_url = (REPO / "viewer" / "components" / "job-progress.js").as_uri()
    program = (
        f"import {{ formatDuration }} from {json.dumps(module_url)};"
        "console.log(JSON.stringify(["
        "formatDuration(null), formatDuration(18.4), formatDuration(120), formatDuration(3720)"
        "]));"
    )

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == ["estimating…", "18s", "2 min", "1h 2m"]


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
