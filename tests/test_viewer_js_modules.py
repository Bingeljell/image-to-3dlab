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
def test_trellis_input_advice_is_explicitly_non_blocking():
    module_url = (REPO / "viewer" / "components" / "trellis-input-advice.js").as_uri()
    program = f"""
      import {{ presentTrellisAdvice }} from {json.dumps(module_url)};
      console.log(JSON.stringify([
        presentTrellisAdvice({{ verdict: 'likely_flat', flat_risk: 0.91 }}),
        presentTrellisAdvice({{ verdict: 'uncertain', flat_risk: 0.51 }}),
        presentTrellisAdvice({{ verdict: 'likely_dimensional', flat_risk: 0.08 }}),
      ]));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(result.stdout)
    assert [value["tone"] for value in values] == ["risk", "uncertain", "suitable"]
    assert all(value["blocking"] is False for value in values)
    assert "91%" in values[0]["message"]


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
        style: {{}},
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


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_rig_edit_status_shares_pending_metadata_without_sharing_corrections():
    state_url = (REPO / "viewer" / "core" / "rig-edit-state.js").as_uri()
    program = f"""
      import {{ getRigEditState, setRigEditState, subscribeRigEditState }} from {json.dumps(state_url)};
      const observed = [];
      const unsubscribe = subscribeRigEditState((state) => observed.push(state));
      setRigEditState({{ pendingCount: 2.8, assetLabel: 'fox.glb', corrections: {{ secret: true }} }});
      unsubscribe();
      setRigEditState({{ pendingCount: 9, assetLabel: 'ignored.glb' }});
      console.log(JSON.stringify({{ observed, current: getRigEditState() }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "observed": [
            {"pendingCount": 0, "assetLabel": None},
            {"pendingCount": 2, "assetLabel": "fox.glb"},
        ],
        "current": {"pendingCount": 9, "assetLabel": "ignored.glb"},
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
        }},
        binding: {{
          adapter: 'rigify.basic-quadruped.blender-5.2.v1',
          sceneFingerprint: hash,
          metarigObjectId: 'metarig-uuid',
          joints: {{
            shoulder: {{
              targets: [{{ boneId: 'bone-uuid', boneName: 'front_thigh.L', endpoint: 'head' }}]
            }}
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
        adapter: sidecar.binding.adapter,
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
        "adapter": "rigify.basic-quadruped.blender-5.2.v1",
        "rejected": "Rig sidecar: joints.shoulder.parent references unknown joint",
    }


def test_rig_sidecar_schema_is_valid_json_and_versioned():
    schema = json.loads((REPO / "rigs" / "rig-sidecar.schema.json").read_text())

    assert schema["properties"]["schemaVersion"]["const"] == 1
    assert schema["properties"]["coordinateSpace"]["const"] == "armature-local"
    assert "binding" in schema["properties"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_rig_correction_session_mirrors_undoes_and_exports_atomic_edits():
    module_url = (REPO / "viewer" / "rig" / "correction-session.js").as_uri()
    program = f"""
      import {{ RigCorrectionSession }} from {json.dumps(module_url)};
      const sidecar = {{
        schemaVersion: 1, rigProfile: 'quadruped', assetFingerprint: 'sha256:test',
        coordinateSpace: 'armature-local', mirror: {{ axis: 'X', origin: 0 }},
        joints: {{
          left: {{ label: 'Left', position: [1, 2, 3], sourceBone: 'left', mirrorOf: 'right' }},
          right: {{ label: 'Right', position: [-1, 2, 3], sourceBone: 'right', mirrorOf: null }},
        }}, corrections: {{}},
      }};
      const session = new RigCorrectionSession(sidecar);
      const partner = session.setTarget('left', [1.25, 2.5, 3], {{ mirror: true }});
      const edited = session.toSidecar();
      const undoWorked = session.undo();
      const afterUndo = session.toSidecar();
      const redoWorked = session.redo();
      session.reset('left', {{ mirror: true }});
      console.log(JSON.stringify({{
        partner,
        left: edited.corrections.left,
        right: edited.corrections.right,
        undoWorked, undoCount: Object.keys(afterUndo.corrections).length,
        redoWorked, resetCount: Object.keys(session.toSidecar().corrections).length,
        sourceUntouched: sidecar.corrections,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "partner": "right",
        "left": {
            "sourcePosition": [1, 2, 3],
            "targetPosition": [1.25, 2.5, 3],
            "delta": [0.25, 0.5, 0],
            "mirrored": False,
        },
        "right": {
            "sourcePosition": [-1, 2, 3],
            "targetPosition": [-1.25, 2.5, 3],
            "delta": [-0.25, 0.5, 0],
            "mirrored": True,
        },
        "undoWorked": True,
        "undoCount": 0,
        "redoWorked": True,
        "resetCount": 0,
        "sourceUntouched": {},
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_fit_skeleton_overlay_maps_armature_local_joints_and_deform_bones():
    overlay_url = (REPO / "viewer" / "rig" / "fit-skeleton-overlay.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    program = f"""
      import * as THREE from {json.dumps(three_url)};
      import {{ FitSkeletonOverlay }} from {json.dumps(overlay_url)};
      const scene = new THREE.Scene();
      const listeners = new Map();
      const canvas = {{
        style: {{}},
        addEventListener(name, callback) {{ listeners.set(name, callback); }},
        removeEventListener(name) {{ listeners.delete(name); }},
        getBoundingClientRect() {{ return {{ left: 0, top: 0, width: 100, height: 100 }}; }},
      }};
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(1, 1, 3);
      camera.updateMatrixWorld(true);
      const armature = new THREE.Group();
      armature.position.set(1, 0, 0);
      const bone = new THREE.Bone();
      bone.name = 'DEF-shoulder.L';
      armature.add(bone);
      scene.add(armature);
      scene.updateMatrixWorld(true);
      const sidecar = {{
        joints: {{
          chest: {{ position: [0, 1, 0], sourceBone: 'DEF-shoulder.L', parent: null }},
          shoulder: {{ position: [0.2, 1, 0], sourceBone: 'DEF-shoulder.L', parent: 'chest' }},
        }},
        corrections: {{}},
      }};
      let picked = null;
      const overlay = new FitSkeletonOverlay({{
        scene, sidecar, bones: [bone], camera, canvas,
        onSelect(id) {{ picked = id; }},
      }});
      overlay.selectJoint('shoulder');
      overlay.setXray(false);
      overlay.setJointLocalPosition('shoulder', [0.3, 1.1, 0]);
      const position = overlay.positions.get('shoulder').toArray();
      const roundTrip = overlay.worldToSidecar(overlay.positions.get('shoulder'));
          const mapped = overlay.jointsForBone('DEF-shoulder.L');
          const gizmoPosition = overlay.gizmo.position.toArray();
          const gizmoVisible = overlay.gizmo.visible;
      const listenerCount = listeners.size;
      const sceneCount = scene.children.length;
      overlay.dispose();
      console.log(JSON.stringify({{
        position, roundTrip, mapped, selected: overlay.selectedId, sceneCount,
        remaining: scene.children.length,
            depthTest: overlay.markerMaterial.depthTest, listenerCount, gizmoPosition, gizmoVisible,
        listenersAfterDispose: listeners.size, picked,
      }}));
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload.pop("position") == pytest.approx([1.3, 0, -1.1])
    assert payload.pop("roundTrip") == pytest.approx([0.3, 1.1, 0])
    assert payload.pop("gizmoPosition") == pytest.approx([1.3, 0, -1.1])
    assert payload == {
        "mapped": ["chest", "shoulder"],
        "selected": "shoulder",
        "sceneCount": 4,
        "remaining": 1,
        "depthTest": True,
        "listenerCount": 4,
        "listenersAfterDispose": 0,
        "picked": None,
        "gizmoVisible": True,
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_fit_skeleton_overlay_selects_bones_and_drags_joint_in_camera_plane():
    overlay_url = (REPO / "viewer" / "rig" / "fit-skeleton-overlay.js").as_uri()
    three_url = (REPO / "viewer" / "vendor" / "three.module.js").as_uri()
    program = f"""
      import * as THREE from {json.dumps(three_url)};
      import {{ FitSkeletonOverlay }} from {json.dumps(overlay_url)};
      const canvas = {{
        style: {{}}, addEventListener() {{}}, removeEventListener() {{}},
        setPointerCapture() {{}}, releasePointerCapture() {{}},
        getBoundingClientRect() {{ return {{ left: 0, top: 0, width: 200, height: 200 }}; }},
      }};
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      camera.position.set(0, 0, 3);
      camera.updateMatrixWorld(true);
      const scene = new THREE.Scene();
      const armature = new THREE.Group();
      const bone = new THREE.Bone();
      bone.name = 'DEF-limb';
      armature.add(bone);
      scene.add(armature);
      scene.updateMatrixWorld(true);
      let overlay;
      let finished = null;
      overlay = new FitSkeletonOverlay({{
        scene,
        sidecar: {{
          joints: {{
            root: {{ position: [0, 0, 0], sourceBone: 'DEF-limb', parent: null }},
            tip: {{ position: [0.4, 0, 0], sourceBone: 'DEF-limb', parent: 'root' }},
          }},
          corrections: {{}},
        }},
        bones: [bone], camera, canvas,
        onSelect(id) {{ overlay.selectJoint(id); }},
        onMove(id, position, done) {{ if (done) finished = {{ id, position }}; }},
      }});
      scene.updateMatrixWorld(true);
      const toScreen = (position) => {{
        const projected = position.clone().project(camera);
        return {{ x: (projected.x + 1) * 100, y: (1 - projected.y) * 100 }};
      }};
      const midpoint = toScreen(new THREE.Vector3(0.2, 0, 0));
      const boneJoint = overlay.pickBone(midpoint.x, midpoint.y);
      const start = toScreen(overlay.positions.get('tip'));
      const event = (x, y) => ({{
        button: 0, pointerId: 1, clientX: x, clientY: y,
        preventDefault() {{}}, stopImmediatePropagation() {{}},
      }});
      overlay.handlePointerDown(event(start.x, start.y));
      overlay.handlePointerMove(event(start.x + 20, start.y));
      overlay.handlePointerUp(event(start.x + 20, start.y));
      console.log(JSON.stringify({{
        boneJoint, selected: overlay.selectedId, finished,
        moved: overlay.positions.get('tip').x > 0.4,
        boneToneMapped: overlay.boneMaterial.toneMapped,
        markerToneMapped: overlay.markerMaterial.toneMapped,
      }}));
      overlay.dispose();
    """

    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["boneJoint"] == "tip"
    assert payload["selected"] == "tip"
    assert payload["finished"]["id"] == "tip"
    assert payload["moved"] is True
    assert payload["boneToneMapped"] is False
    assert payload["markerToneMapped"] is False


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


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_the_final_stage_is_ticked_when_the_job_reports_done():
    # The reported bug: a finished run whose "Compress textures" row still read
    # "estimating…". A stage is only ticked when a *later* stage starts, and the last
    # stage has none -- so the job's own terminal event has to finish the list.
    rows = _run_panel("""
      panel.configure({ stages: ['retopologise', 'repaint', 'compress'], stage_labels: {} });
      panel.apply({ phase: 'compress', overall_pct: 97, message: 'Compressing' });
      const midRun = rowState();
      panel.apply({ phase: 'done', overall_pct: 100, message: 'Finished at 4.3 MB',
                    elapsed_seconds: 333, result_url: '/x.glb' });
      console.log(JSON.stringify({ midRun, afterDone: rowState(), bar: bar.style.width,
                                   label: label.textContent }));
    """)
    assert rows["midRun"][2][0] == "stage-row active"
    assert [state for state, _ in rows["afterDone"]] == ["stage-row done"] * 3
    assert [detail for _, detail in rows["afterDone"]] == ["done"] * 3
    assert rows["bar"] == "100%"
    assert rows["label"] == "Finished at 4.3 MB"


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_failed_job_marks_the_running_stage_rather_than_completing_it():
    rows = _run_panel("""
      panel.configure({ stages: ['retopologise', 'repaint'], stage_labels: {} });
      panel.apply({ phase: 'repaint', overall_pct: 20, message: 'Repainting' });
      panel.apply({ phase: 'error', message: 'worker exited with code 1' });
      console.log(JSON.stringify(rowState()));
    """)
    assert rows[0][0] == "stage-row done"          # it really did finish
    assert rows[1] == ["stage-row failed", "failed"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_stage_reporting_no_sub_progress_says_running_not_estimating():
    # "0% · ~estimating…" on a stage that never reports a percentage reads as a stall.
    rows = _run_panel("""
      panel.configure({ stages: ['retopologise', 'repaint'], stage_labels: {} });
      panel.apply({ phase: 'retopologise', overall_pct: 0, message: 'Retopologising' });
      const bare = rowState()[0];
      panel.apply({ phase: 'repaint', overall_pct: 30, message: 'Denoising step 6/15',
                    step: 6, total: 15, stage_pct: 27, stage_eta_seconds: 227 });
      console.log(JSON.stringify({ bare, measured: rowState()[1] }));
    """)
    assert rows["bare"] == ["stage-row active", "running"]
    assert rows["measured"] == ["stage-row active", "6/15 · ~4 min"]


def _run_welcome(expr: str):
    module_url = (REPO / "viewer" / "components" / "welcome-content.js").as_uri()
    program = f"import * as w from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


WELCOME = {
    "brand": {"name": "Bingeljell's AssetFurnace"},
    "version": "0.3.0",
    "routes": [{"id": "pixal3d", "state": "missing"}, {"id": "qwen-image", "state": "ready"}],
    "news": [{"version": "0.3.0", "sections": {
        "Fixed": ["f1"], "Added": ["a1", "a2", "a3"], "Changed": ["c1"]}}],
}


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_welcome_shows_on_first_run_and_after_an_update_only():
    payload = json.dumps(WELCOME)
    quiet = json.dumps({**WELCOME, "news": []})
    assert _run_welcome(
        f"[w.shouldShow(null, {payload}), w.shouldShow('0.2.0', {payload}),"
        f" w.shouldShow('0.3.0', {quiet})]"
    ) == [True, True, False]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_welcome_greets_first_visits_updates_and_reopens_differently():
    payload = json.dumps(WELCOME)
    first, update, reopen = _run_welcome(
        f"[w.greeting(null, {payload}), w.greeting('0.2.0', {payload}),"
        f" w.greeting('0.3.0', {payload}, true)]")
    assert first["title"] == "Welcome to Bingeljell's AssetFurnace"
    assert update["title"].startswith("Welcome back") and "0.3.0" in update["kicker"]
    assert "0.3.0" in reopen["kicker"] and reopen["title"].startswith("Welcome to")


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_welcome_news_lists_added_first_and_counts_the_rest():
    release = json.dumps(WELCOME["news"][0])
    assert _run_welcome(f"w.newsItems({release}, 3)") == {
        "items": ["a1", "a2", "a3"], "more": 2}


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_welcome_escapes_html_and_renders_backticks_as_code():
    assert _run_welcome("w.inlineCode('run `a<b>` & go')") == (
        "run <code>a&lt;b&gt;</code> &amp; go")


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_welcome_points_at_setup_until_something_is_installed():
    nothing = json.dumps({**WELCOME, "routes": [{"id": "pixal3d", "state": "missing"}]})
    image_ready = json.dumps(WELCOME)
    unsupported = json.dumps({**WELCOME, "routes": []})
    assert _run_welcome(
        f"[w.primaryAction({nothing}), w.primaryAction({image_ready}),"
        f" w.primaryAction({unsupported})]"
    ) == [{"label": "Set up a route", "mode": "setup"},
          {"label": "Make something", "mode": "generate-image"},
          None]


def _run_banner(expr: str):
    module_url = (REPO / "viewer" / "components" / "update-banner.js").as_uri()
    program = f"import * as b from {json.dumps(module_url)}; console.log(JSON.stringify({expr}));"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


CHECK = {"enabled": True, "newer": True, "current": "0.2.0", "latest": "0.3.0",
         "url": "https://github.com/x/releases/tag/v0.3.0", "command": "curl … | bash"}


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_update_banner_shows_for_a_newer_release_until_dismissed():
    check = json.dumps(CHECK)
    shown, dismissed, next_one = _run_banner(
        f"[b.bannerFor({check}, null), b.bannerFor({check}, '0.3.0'),"
        f" b.bannerFor({check}, '0.2.9')]")
    assert "0.3.0 is out" in shown["text"] and shown["command"] == "curl … | bash"
    assert dismissed is None
    assert next_one is not None  # dismissing an older release does not hide a newer one


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_update_banner_stays_hidden_when_off_or_up_to_date():
    off = json.dumps({**CHECK, "enabled": False})
    same = json.dumps({**CHECK, "newer": False})
    assert _run_banner(f"[b.bannerFor({off}, null), b.bannerFor({same}, null),"
                       " b.bannerFor(null, null)]") == [None, None, None]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_alpha_badge_only_asks_for_rembg_when_the_backend_needs_a_cutout():
    """Issue #36: Pixal3D cuts out the background itself, yet the badge told a Windows
    tester to 'enable rembg to continue', an option only TRELLIS has."""
    module_url = (REPO / "viewer" / "components" / "alpha-badge.js").as_uri()
    program = (
        f"import {{ alphaBadge }} from {json.dumps(module_url)};"
        "console.log(JSON.stringify(["
        "alphaBadge(true, true), alphaBadge(false, true), alphaBadge(false, false)"
        "]));"
    )
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    has_cutout, trellis_without, pixal_without = json.loads(result.stdout)

    assert has_cutout["tone"] == "alpha-good"
    assert trellis_without["tone"] == "alpha-bad" and "rembg" in trellis_without["text"]
    assert pixal_without["tone"] == "alpha-good"
    assert "rembg" not in pixal_without["text"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_embedded_preview_lands_on_the_bare_3d_view_not_the_app():
    module_url = (REPO / "viewer" / "core" / "embed.js").as_uri()
    program = f"""
      import {{ isEmbedded, landingMode }} from {json.dumps(module_url)};
      console.log(JSON.stringify([
        isEmbedded('?a=x.glb&restricted=1'), isEmbedded(''),
        landingMode({{ embedded: true, skipSetup: false }}),
        landingMode({{ embedded: true, skipSetup: true }}),
        landingMode({{ embedded: false, skipSetup: true }}),
        landingMode({{ embedded: false, skipSetup: false }}),
      ]));
    """
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [True, False, "compare", "compare", "generate", "setup"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_cancel_works_before_the_job_exists_and_never_reaches_an_older_one():
    """The Props tab showed Cancel during an upload but wired it only once the job came
    back: a click did nothing, or cancelled the job before."""
    module_url = (REPO / "viewer" / "core" / "early-cancel.js").as_uri()
    program = f"""
      import {{ earlyCancel }} from {json.dumps(module_url)};
      const sent = [];
      const early = earlyCancel((id) => sent.push(id));
      const clicked = early.click();          // while the upload is still going
      const before = [...sent];
      early.started('job-2');                 // the response arrives
      const late = earlyCancel((id) => sent.push(id));
      late.started('job-3');
      const direct = late.click();
      const quiet = earlyCancel((id) => sent.push(id));
      quiet.started('job-4');                 // never clicked
      console.log(JSON.stringify({{ clicked, before, direct, sent }}));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {
        "clicked": "queued", "before": [], "direct": "sent", "sent": ["job-2", "job-3"],
    }


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_link_to_a_model_lands_on_compare_with_the_app_around_it():
    """`serve.py --open a.glb` writes ?a=... without restricted=1; it used to land on Setup."""
    module_url = (REPO / "viewer" / "core" / "embed.js").as_uri()
    program = f"""
      import {{ isEmbedded, landingMode, linksAModel }} from {json.dumps(module_url)};
      const search = '?a=output%2Fowl.glb&la=Owl';
      console.log(JSON.stringify([
        isEmbedded(search), linksAModel(search), linksAModel(''),
        landingMode({{ embedded: false, linked: true, skipSetup: false }}),
        landingMode({{ embedded: false, linked: false, skipSetup: false }}),
      ]));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [False, True, False, "compare", "setup"]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_mouse_wheel_scrolls_the_menu_sideways_only_while_it_has_more_to_show():
    """The bar hides its scrollbar to stay one row; a plain wheel never reached About."""
    module_url = (REPO / "viewer" / "core" / "sideways-wheel.js").as_uri()
    program = f"""
      import {{ sidewaysScroll }} from {json.dumps(module_url)};
      const bar = (scrollLeft) => ({{ scrollLeft, scrollWidth: 900, clientWidth: 600 }});
      console.log(JSON.stringify([
        sidewaysScroll({{ deltaX: 0, deltaY: 120 }}, bar(0)),           // wheel down: right
        sidewaysScroll({{ deltaX: 0, deltaY: -120 }}, bar(200)),        // wheel up: left
        sidewaysScroll({{ deltaX: 0, deltaY: 500 }}, bar(250)),         // stops at the end
        sidewaysScroll({{ deltaX: 0, deltaY: 120 }}, bar(300)),         // already there
        sidewaysScroll({{ deltaX: 40, deltaY: 5 }}, bar(0)),            // a sideways swipe
        sidewaysScroll({{ deltaX: 0, deltaY: 3, deltaMode: 1 }}, bar(0)),  // wheel in lines
        sidewaysScroll({{ deltaX: 0, deltaY: 120 }},
                       {{ scrollLeft: 0, scrollWidth: 600, clientWidth: 600 }}),  // it all fits
      ]));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [120, 80, 300, None, None, 48, None]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_any_pane_compare_loads_counts_as_a_link_and_an_empty_one_does_not():
    module_url = (REPO / "viewer" / "core" / "embed.js").as_uri()
    searches = ["?b=out.glb&lb=After", "?d=x.glb", "?a=", "?a=%20", "?la=Owl", "?e=x.glb"]
    program = f"""
      import {{ linksAModel }} from {json.dumps(module_url)};
      console.log(JSON.stringify({json.dumps(searches)}.map(linksAModel)));
    """
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [True, True, False, False, False, False]


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_linked_visit_keeps_the_model_and_the_welcome_waits():
    payload = json.dumps(WELCOME)
    assert _run_welcome(
        f"[w.shouldShow(null, {payload}, true), w.shouldShow('0.2.0', {payload}, true),"
        f" w.shouldShow(null, {payload}, false)]"
    ) == [False, False, True]


def test_generate_preview_iframe_asks_for_the_embedded_view():
    generate = (REPO / "viewer" / "modes" / "generate.js").read_text()
    app = (REPO / "viewer" / "app.js").read_text()
    css = (REPO / "viewer" / "styles" / "base.css").read_text()
    assert "restricted=1" in generate
    assert "isEmbedded(location.search)" in app
    assert ".embedded #mode-switch" in css


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
