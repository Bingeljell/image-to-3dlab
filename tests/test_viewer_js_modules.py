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


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_bone_picker_maps_a_viewport_hit_back_to_the_bone():
    picker_url = (REPO / "viewer" / "animation" / "bone-picker.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    program = f"""
      import * as THREE from {json.dumps(three_url)};
      import {{ BonePicker }} from {json.dumps(picker_url)};
      const listeners = new Map();
      const canvas = {{
        addEventListener(name, callback) {{ listeners.set(name, callback); }},
        removeEventListener(name) {{ listeners.delete(name); }},
        getBoundingClientRect() {{ return {{ left: 0, top: 0, width: 100, height: 100 }}; }},
      }};
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(0, 0, 3);
      camera.updateMatrixWorld(true);
      const bone = new THREE.Bone();
      bone.name = 'DEF-hip';
      const child = new THREE.Bone();
      child.name = 'DEF-knee';
      child.position.y = 0.5;
      bone.add(child);
      scene.add(bone);
      scene.updateMatrixWorld(true);
      let selected = null;
      const picker = new BonePicker({{
        scene, camera, canvas, bones: [bone, child],
        onSelect(value) {{ selected = value?.name || null; }},
      }});
      scene.updateMatrixWorld(true);
      picker.update();
      const hit = picker.pick(50, 50);
      const beforeDispose = scene.children.includes(picker.markers);
      const bodyCount = picker.bodyBones.length;
      picker.dispose();
      console.log(JSON.stringify({{
        hit: hit?.name,
        selected,
        beforeDispose,
        bodyCount,
        afterDispose: scene.children.includes(picker.markers),
        listeners: listeners.size,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "hit": "DEF-hip",
        "selected": "DEF-hip",
        "beforeDispose": True,
        "bodyCount": 1,
        "afterDispose": False,
        "listeners": 0,
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_rig_sidecar_parser_validates_references_corrections_and_asset_hash():
    module_url = (REPO / "viewer" / "rig" / "rig-sidecar.js").as_uri()
    program = f"""
      import {{ fingerprintAsset, parseRigSidecar }} from {json.dumps(module_url)};
      import {{ webcrypto }} from 'node:crypto';
      const hash = await fingerprintAsset(new TextEncoder().encode('abc'), webcrypto);
      const sidecar = parseRigSidecar({{
        schemaVersion: 1,
        rigProfile: 'rigify.quadruped.v1',
        assetFingerprint: hash,
        coordinateSpace: 'armature-local',
        mirror: {{ axis: 'X', origin: 0 }},
        joints: {{
          chest: {{ label: 'Chest', position: [0, 1, 0], sourceBone: 'DEF-spine' }},
          shoulder: {{
            label: 'Left shoulder', position: [0.2, 1, 0], sourceBone: 'DEF-upper_arm.L',
            parent: 'chest'
          }},
        }},
        corrections: {{
          shoulder: {{
            sourcePosition: [0.2, 1, 0], targetPosition: [0.25, 1.1, 0],
            delta: [0.05, 0.1, 0], mirrored: false
          }}
        }}
      }});
      let rejected = '';
      try {{
        parseRigSidecar({{ ...sidecar, joints: {{ shoulder: sidecar.joints.shoulder }} }});
      }} catch (error) {{ rejected = error.message; }}
      console.log(JSON.stringify({{
        hash,
        profile: sidecar.rigProfile,
        target: sidecar.corrections.shoulder.targetPosition,
        rejected,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "hash": "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "profile": "rigify.quadruped.v1",
        "target": [0.25, 1.1, 0],
        "rejected": "Rig sidecar: joints.shoulder.parent references unknown joint",
    }


def test_rig_sidecar_schema_is_valid_json_and_versioned():
    schema = json.loads((REPO / "rigs" / "rig-sidecar.schema.json").read_text())

    assert schema["properties"]["schemaVersion"]["const"] == 1
    assert schema["properties"]["coordinateSpace"]["const"] == "armature-local"
