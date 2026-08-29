import * as THREE from 'three';
import { AnimationPlayer } from '../animation/player.js';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { specsFromFiles } from '../core/asset-files.js';

const element = (id) => document.getElementById(id);
const viewport = element('animate-viewport');
const empty = element('animate-empty');
const fileInput = element('animate-file');
const drop = element('animate-drop');
const clipSelect = element('animate-clip');
const playButton = element('animate-play');
const timeline = element('animate-timeline');
const timeLabel = element('animate-time');
const loopToggle = element('animate-loop');
const skeletonToggle = element('animate-skeleton');
const resetButton = element('animate-reset');
const rigSummary = element('animate-rig-summary');
const status = element('animate-status');

let view = null;
let player = null;
let skeletonHelper = null;
let renderPending = false;

const formatTime = (seconds) => {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = (value % 60).toFixed(2).padStart(5, '0');
  return `${minutes}:${remainder}`;
};

function requestRender() {
  if (renderPending || !view) return;
  renderPending = true;
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  renderPending = false;
  if (!view) return;
  const moving = view.controls.update();
  view.renderer.render(view.scene, view.camera);
  if (moving) requestRender();
}

function resizeViewport() {
  if (!view) return;
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (!width || !height) return;
  view.renderer.setSize(width, height);
  view.camera.aspect = width / height;
  view.camera.updateProjectionMatrix();
  requestRender();
}

function updateTransport(time = 0, duration = 0) {
  timeline.max = String(duration || 0);
  timeline.value = String(Math.min(time, duration || 0));
  timeLabel.textContent = `${formatTime(time)} / ${formatTime(duration)}`;
}

function updatePlaying(playing) {
  playButton.textContent = playing ? 'Pause' : 'Play';
  playButton.classList.toggle('on', playing);
}

function configureLoop() {
  if (!player?.action) return;
  if (loopToggle.checked) {
    player.action.setLoop(THREE.LoopRepeat, Infinity);
    player.action.clampWhenFinished = false;
  } else {
    player.action.setLoop(THREE.LoopOnce, 1);
    player.action.clampWhenFinished = true;
  }
}

function selectClip(index) {
  if (!player || !view) return;
  player.clear();
  view.resetPose();
  const clip = view.rig.animations[index];
  if (!clip) {
    playButton.disabled = true;
    timeline.disabled = true;
    updateTransport();
    requestRender();
    return;
  }
  player.select(clip);
  configureLoop();
  playButton.disabled = false;
  timeline.disabled = false;
  timeline.step = String(1 / 60);
  requestRender();
}

function buildClipMenu() {
  clipSelect.replaceChildren();
  const bindPose = document.createElement('option');
  bindPose.value = '-1';
  bindPose.textContent = 'Bind pose';
  clipSelect.appendChild(bindPose);
  view.rig.animations.forEach((clip, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = clip.name || `Clip ${index + 1}`;
    clipSelect.appendChild(option);
  });
  clipSelect.value = '-1';
  clipSelect.disabled = view.rig.animations.length === 0;
}

function disposeCurrent() {
  player?.dispose();
  player = null;
  if (skeletonHelper) {
    skeletonHelper.removeFromParent();
    skeletonHelper.geometry?.dispose();
    skeletonHelper.material?.dispose();
    skeletonHelper = null;
  }
  if (view) {
    disposeModelViewport(view);
    view.renderer.domElement.remove();
    for (const url of new Set(view.spec.revoke || [])) URL.revokeObjectURL(url);
    view = null;
  }
  renderPending = false;
}

function loaded(loadedView) {
  if (view !== loadedView) return;
  empty.hidden = true;
  view.camera.position.set(1.45, 0.6, 2.45);
  view.controls.target.set(0, 0, 0);
  view.controls.update();

  player = new AnimationPlayer(new THREE.AnimationMixer(view.modelRoot), {
    onFrame: (time, duration) => { updateTransport(time, duration); requestRender(); },
    onStateChange: updatePlaying,
  });

  if (view.rig.bones.length) {
    skeletonHelper = new THREE.SkeletonHelper(view.modelRoot);
    skeletonHelper.visible = skeletonToggle.checked;
    view.scene.add(skeletonHelper);
  }

  buildClipMenu();
  skeletonToggle.disabled = view.rig.bones.length === 0;
  resetButton.disabled = view.rig.skeletons.length === 0;
  rigSummary.textContent = `${view.rig.bones.length} bones · ` +
    `${view.rig.skinnedMeshes.length} skinned mesh${view.rig.skinnedMeshes.length === 1 ? '' : 'es'} · ` +
    `${view.rig.animations.length} clip${view.rig.animations.length === 1 ? '' : 's'}`;
  status.textContent = view.rig.bones.length
    ? 'Rig loaded. Choose a clip or inspect the bind pose.'
    : 'Model loaded, but it does not contain a skinned skeleton.';
  resizeViewport();
}

function loadFiles(fileList) {
  status.textContent = '';
  let spec;
  try {
    const specs = specsFromFiles(fileList);
    spec = specs.find((candidate) => candidate.kind === 'model');
    if (!spec) {
      for (const url of new Set(specs.flatMap((candidate) => candidate.revoke || []))) {
        URL.revokeObjectURL(url);
      }
      throw new Error('Choose a GLB, GLTF, or OBJ model. Animation requires a rigged GLB/GLTF.');
    }
  } catch (error) {
    status.textContent = error.message || String(error);
    return;
  }

  disposeCurrent();
  empty.hidden = false;
  empty.textContent = `Loading ${spec.label}…`;
  rigSummary.textContent = 'Inspecting skeleton and clips…';
  clipSelect.disabled = true;
  playButton.disabled = true;
  timeline.disabled = true;
  skeletonToggle.disabled = true;
  resetButton.disabled = true;
  updateTransport();

  view = createModelViewport({
    pane: viewport,
    spec,
    onChange: requestRender,
    onLoaded: loaded,
    onError: (error) => {
      status.textContent = `${spec.label}: ${error.message || error}`;
      rigSummary.textContent = 'Load failed';
    },
  });
  resizeViewport();
}

drop.onclick = () => fileInput.click();
drop.onkeydown = (input) => {
  if (input.key === 'Enter' || input.key === ' ') {
    input.preventDefault();
    fileInput.click();
  }
};
fileInput.onchange = () => {
  if (fileInput.files.length) loadFiles(fileInput.files);
  fileInput.value = '';
};
for (const event of ['dragenter', 'dragover']) {
  drop.addEventListener(event, (input) => {
    input.preventDefault();
    input.stopPropagation();
    drop.classList.add('drag');
  });
}
for (const event of ['dragleave', 'drop']) {
  drop.addEventListener(event, (input) => {
    input.preventDefault();
    input.stopPropagation();
    drop.classList.remove('drag');
  });
}
drop.addEventListener('drop', (input) => {
  if (input.dataTransfer.files.length) loadFiles(input.dataTransfer.files);
});

clipSelect.onchange = () => selectClip(Number(clipSelect.value));
playButton.onclick = () => player?.toggle();
timeline.oninput = () => {
  player?.pause();
  player?.seek(Number(timeline.value));
};
loopToggle.onchange = configureLoop;
skeletonToggle.onchange = () => {
  if (skeletonHelper) skeletonHelper.visible = skeletonToggle.checked;
  requestRender();
};
resetButton.onclick = () => {
  clipSelect.value = '-1';
  selectClip(-1);
  status.textContent = 'Restored the GLB bind pose.';
};

addEventListener('resize', resizeViewport);
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'animate') requestAnimationFrame(resizeViewport);
});
