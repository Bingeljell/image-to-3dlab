import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from '../../viewer/vendor/three.module.js';
import { AnimationPlayer } from '../../viewer/animation/player.js';
import { describeBone, inspectRig, resetRigPose } from '../../viewer/animation/rig-inspection.js';
import { createCameraViewControls, setCameraView } from '../../viewer/components/camera-view-controls.js';
import { projectAxes } from '../../viewer/studio/gizmo.js';
import { installViewerBrowser } from './viewer-browser.js';
import { mapViewerThree } from './viewer-three.js';

const hooks = mapViewerThree();
const { createBoneDisplay } = await import('../../viewer/studio/bones.js');
const { createFloor, facingAngle, standOnFloor } = await import('../../viewer/studio/floor.js');
hooks.deregister();

test('animation player follows timestamps and a real mixer completion event', (t) => {
  const { frame } = installViewerBrowser(t);
  const root = new THREE.Group();
  const mixer = new THREE.AnimationMixer(root);
  const clip = new THREE.AnimationClip('rise', 2, [new THREE.NumberKeyframeTrack('.position[y]', [0, 2], [0, 4])]);
  const player = new AnimationPlayer(mixer);
  assert.equal(player.play(), false);
  player.seek(1);
  player.tick(1234);
  const action = player.select(clip);
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = true;
  assert.equal(player.duration, 2);
  assert.equal(player.time, 0);
  assert.equal(player.toggle(), true);
  assert.equal(player.play(), false);
  frame(1000);
  frame(900);
  assert.equal(root.position.y, 0);
  frame(1400);
  assert.equal(root.position.y, 1);
  assert.equal(player.toggle(), false);
  frame(2400);
  assert.equal(root.position.y, 1);
  player.seek('not a number');
  assert.equal(root.position.y, 0);
  player.seek(5);
  assert.equal(root.position.y, 4);
  player.seek(-5);
  assert.equal(root.position.y, 0);
  player.play();
  frame(2500);
  frame(4500);
  assert.equal(player.playing, false);
  assert.equal(player.time, 2);
  player.dispose();
  assert.equal(player.duration, 0);
  assert.equal(player.time, 0);
  assert.equal(mixer.existingAction(clip), null);
});

function makeRig() {
  const root = new THREE.Group();
  const parent = new THREE.Bone();
  const child = new THREE.Bone();
  child.name = 'Hand';
  child.position.y = 0.5;
  const coincident = new THREE.Bone();
  parent.add(child, coincident, new THREE.Group());
  root.add(parent);
  root.updateMatrixWorld(true);
  const onlyInSkeleton = new THREE.Bone();
  const skeleton = new THREE.Skeleton([parent, child, coincident, onlyInSkeleton]);
  const firstMesh = new THREE.SkinnedMesh();
  firstMesh.bind(skeleton);
  const secondMesh = new THREE.SkinnedMesh();
  secondMesh.bind(skeleton);
  root.add(firstMesh, secondMesh, new THREE.SkinnedMesh());
  return { root, parent, child, coincident, firstMesh, secondMesh, onlyInSkeleton };
}

test('rig inspection deduplicates shared skeletons and separates current pose from bind pose', () => {
  const { root, parent, child, firstMesh, secondMesh, onlyInSkeleton } = makeRig();
  const clip = new THREE.AnimationClip('test', 1, []);
  const animations = [clip];
  const rig = inspectRig(root, animations);
  assert.notEqual(rig.animations, animations);
  assert.deepEqual(rig.animations, [clip]);
  assert.equal(rig.skinnedMeshes.length, 3);
  assert.equal(rig.skinnedMeshes[1], secondMesh);
  assert.deepEqual(rig.skeletons, [firstMesh.skeleton]);
  assert.equal(rig.bones.length, 4);
  assert.ok(rig.bones.includes(onlyInSkeleton));
  assert.equal(describeBone(null, rig), null);
  assert.equal(describeBone(parent, rig).name, '(unnamed bone)');
  assert.deepEqual(describeBone(parent, rig).children, ['Hand', '(unnamed bone)']);
  assert.equal(describeBone(parent, rig).parent, null);
  assert.equal(describeBone(child, rig).parent, '(unnamed bone)');
  child.position.y = 1.25;
  assert.deepEqual(describeBone(child, rig).pose.position, [0, 1.25, 0]);
  assert.deepEqual(describeBone(child, rig).bind.position, [0, 0.5, 0]);
  resetRigPose(rig);
  assert.equal(child.position.y, 0.5);
  parent.name = 'Arm';
  assert.equal(describeBone(child, rig).parent, 'Arm');

  const partial = { name: 'Loose', position: { x: '3', z: -1 } };
  assert.deepEqual(describeBone(partial, rig).bind, {
    position: [3, 0, -1], quaternion: [0, 0, 0, 1], scale: [1, 1, 1],
  });
  assert.equal(inspectRig(null).bones.length, 0);
});

test('skeleton display places real links on their bones and hides collapsed links', () => {
  const { root, parent, child, coincident } = makeRig();
  root.position.x = 2;
  const display = createBoneDisplay([parent, child, coincident]);
  const scene = new THREE.Scene();
  scene.add(display.group);
  display.update();
  const [arm, collapsed, rootDot, tipDot] = display.group.children;
  assert.equal(collapsed.visible, false);
  assert.equal(arm.visible, true);
  assert.equal(arm.scale.y, 0.5);
  assert.deepEqual(rootDot.position.toArray(), [2, 0, 0]);
  assert.deepEqual(tipDot.position.toArray(), [2, 0.5, 0]);
  child.position.set(1, 0, 0);
  display.update();
  assert.equal(arm.scale.y, 1);
  assert.ok(new THREE.Vector3(0, 1, 0).applyQuaternion(arm.quaternion).x > 0.99);
  const freed = [];
  for (const resource of [arm.geometry, arm.material, rootDot.geometry, rootDot.material]) {
    resource.addEventListener('dispose', () => freed.push(resource));
  }
  display.dispose();
  assert.equal(freed.length, 4);
  assert.equal(scene.children.length, 0);
});

test('camera toolbar handles absent views and snaps all axes without changing its target', (t) => {
  const { document } = installViewerBrowser(t, '<main></main>');
  const container = document.querySelector('main');
  let view = null;
  const changes = [];
  let controlUpdates = 0;
  const ctrl = createCameraViewControls({ container, getView: () => view, onChange: (name) => changes.push(name) });
  const find = (name) => container.querySelector(`[data-camera-view="${name}"]`);
  assert.equal(find('top').disabled, true);
  ctrl.setEnabled(true);
  find('top').click();
  assert.deepEqual(changes, []);
  const camera = new THREE.PerspectiveCamera();
  assert.equal(setCameraView({ camera }, 'front'), false);
  const target = new THREE.Vector3(0, 1, 0);
  camera.position.copy(target);
  view = { camera, controls: { target, update: () => controlUpdates++ } };
  assert.equal(setCameraView(view, 'unknown'), false);
  for (const [name, axis] of Object.entries({
    front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0],
    right: [1, 0, 0], top: [0, 1, 0], bottom: [0, -1, 0],
  })) {
    find(name).click();
    assert.deepEqual(camera.position.clone().sub(target).normalize().toArray(), axis);
    assert.deepEqual(target.toArray(), [0, 1, 0]);
  }
  find('reset').click();
  assert.deepEqual(camera.position.toArray(), [1.45, 0.6, 2.45]);
  assert.equal(controlUpdates, 7);
  assert.deepEqual(changes, ['front', 'back', 'left', 'right', 'top', 'bottom', 'reset']);
  ctrl.dispose();
  assert.equal(container.childElementCount, 0);
});

test('floor geometry spans its configured radius and empty models have a neutral height', () => {
  const empty = new THREE.Group();
  assert.equal(standOnFloor(empty), 1);
  const floor = createFloor({ radius: 10, cell: 0.5 });
  assert.equal(floor.children[0].material.uniforms.cell.value, 0.5);
  assert.equal(floor.children[0].material.uniforms.fadeTo.value, 6);
  assert.equal(facingAngle([1]), 0);
  assert.equal(facingAngle([0, 1, 0]), 0);
  assert.equal(facingAngle([1, 0, 0]), -Math.PI / 2);
  const top = projectAxes([0, 1, 0]);
  assert.equal(top.at(-1).key, '+y');
  assert.equal(top.find(({ key }) => key === '+x').x, 1);
});
