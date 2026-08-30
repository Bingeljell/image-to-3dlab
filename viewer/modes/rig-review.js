import { BonePicker } from '../animation/bone-picker.js';
import { describeBone } from '../animation/rig-inspection.js';
import { createCameraViewControls } from '../components/camera-view-controls.js';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { specsFromFiles } from '../core/asset-files.js';
import { FitSkeletonOverlay } from '../rig/fit-skeleton-overlay.js';
import { RigCorrectionSession } from '../rig/correction-session.js';
import {
  findRigSidecarFile,
  fingerprintAsset,
  parseRigSidecar,
} from '../rig/rig-sidecar.js';

const element = (id) => document.getElementById(id);
const viewport = element('rig-viewport');
const empty = element('rig-empty');
const drop = element('rig-drop');
const fileInput = element('rig-file');
const status = element('rig-status');
const summary = element('rig-summary');
const sidecarSummary = element('rig-sidecar-summary');
const opacity = element('rig-opacity');
const opacityValue = element('rig-opacity-value');
const xray = element('rig-xray');
const showDeformBones = element('rig-show-deform-bones');
const showDeformJoints = element('rig-show-deform-joints');
const showFitBones = element('rig-show-fit-bones');
const showFitJoints = element('rig-show-fit-joints');
const hierarchySearch = element('rig-hierarchy-search');
const hierarchy = element('rig-hierarchy');
const selectionEmpty = element('rig-selection-empty');
const selection = element('rig-selection');
const boneName = element('rig-bone-name');
const boneParent = element('rig-bone-parent');
const boneChildren = element('rig-bone-children');
const mappedJoints = element('rig-mapped-joints');
const clearBone = element('rig-clear-bone');
const jointEditor = element('rig-joint-editor');
const jointName = element('rig-joint-name');
const jointInputs = ['x', 'y', 'z'].map((axis) => element(`rig-joint-${axis}`));
const mirrorEdit = element('rig-mirror-edit');
const resetJoint = element('rig-reset-joint');
const correctionActions = element('rig-correction-actions');
const undo = element('rig-undo');
const redo = element('rig-redo');
const resetAll = element('rig-reset-all');
const exportCorrections = element('rig-export');
const correctionSummary = element('rig-correction-summary');
const rebind = element('rig-rebind');
const cancelRebind = element('rig-cancel-rebind');
const rigJob = element('rig-job');
const rigJobBar = element('rig-job-bar');
const rigJobLabel = element('rig-job-label');
const rigArtifacts = element('rig-artifacts');
const artifactLinks = {
  result_url: element('rig-result-glb'),
  scene_url: element('rig-result-blend'),
  sidecar_url: element('rig-result-sidecar'),
  report_url: element('rig-result-report'),
};

let view = null;
let bonePicker = null;
let fitOverlay = null;
let sidecar = null;
let selectedBone = null;
let selectedJointId = null;
let correctionSession = null;
let sidecarFilename = 'rig-corrections.rig.json';
let materialStates = new Map();
let hierarchyRows = [];
let renderPending = false;
let loadToken = 0;
let selectedModelFile = null;
let selectedSceneFile = null;
let rebindRunning = false;
let rebindSource = null;
let rebindPoll = null;
const cameraViews = createCameraViewControls({
  container: viewport,
  getView: () => view,
  onChange: requestRender,
});

function requestRender() {
  if (!view || renderPending) return;
  renderPending = true;
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  renderPending = false;
  if (!view) return;
  const moving = view.controls.update();
  bonePicker?.update();
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

function selectBone(bone, index = -1, preferredJointId = null) {
  selectedBone = bone;
  selectionEmpty.hidden = !!bone;
  selection.hidden = !bone;
  clearBone.disabled = !bone;
  hierarchyRows.forEach((row) => row.classList.toggle('on', Number(row.dataset.index) === index));
  if (!bone || !view) {
    selectedJointId = null;
    fitOverlay?.selectJoint(null);
    jointEditor.hidden = true;
    requestRender();
    return;
  }
  const description = describeBone(bone, view.rig);
  boneName.textContent = description.name;
  boneParent.textContent = description.parent || 'armature root';
  boneChildren.textContent = description.children.length ? description.children.join(', ') : '—';
  const jointIds = fitOverlay?.jointsForBone(bone.name) || [];
  mappedJoints.textContent = jointIds.length
    ? jointIds.map((id) => sidecar.joints[id].label).join(', ')
    : 'No mapped fit joint';
  selectedJointId = jointIds.includes(preferredJointId) ? preferredJointId : (jointIds[0] || null);
  fitOverlay?.selectJoint(selectedJointId);
  updateJointEditor();
  requestRender();
}

function selectFitJoint(id) {
  if (!sidecar?.joints[id] || !view) return;
  const index = view.rig.bones.findIndex((bone) => bone.name === sidecar.joints[id].sourceBone);
  if (index >= 0) {
    bonePicker.select(index);
    selectBone(view.rig.bones[index], index, id);
  }
}

function updateJointEditor() {
  jointEditor.hidden = !selectedJointId || !correctionSession;
  if (jointEditor.hidden) return;
  jointName.textContent = sidecar.joints[selectedJointId].label;
  correctionSession.target(selectedJointId).forEach((value, axis) => {
    jointInputs[axis].value = Number(value.toFixed(6));
  });
  mirrorEdit.disabled = !correctionSession.mirrorPartner(selectedJointId);
  if (mirrorEdit.disabled) mirrorEdit.checked = false;
}

function refreshCorrections() {
  if (!correctionSession || !fitOverlay) return;
  for (const id of Object.keys(sidecar.joints)) {
    fitOverlay.setJointLocalPosition(id, correctionSession.target(id));
  }
  const count = Object.keys(correctionSession.corrections).length;
  correctionSummary.textContent = count
    ? `${count} corrected joint${count === 1 ? '' : 's'} · rebind required`
    : 'No corrections.';
  undo.disabled = correctionSession.undoStack.length === 0;
  redo.disabled = correctionSession.redoStack.length === 0;
  resetAll.disabled = count === 0;
  exportCorrections.disabled = false;
  updateRebindAvailability();
  updateJointEditor();
  requestRender();
}

function updateRebindAvailability() {
  const count = correctionSession ? Object.keys(correctionSession.corrections).length : 0;
  rebind.disabled = rebindRunning || !count || !selectedModelFile ||
    !selectedModelFile.name.toLowerCase().endsWith('.glb') || !selectedSceneFile ||
    !sidecar?.binding;
}

function applyJointInputs() {
  if (!selectedJointId || !correctionSession) return;
  const target = jointInputs.map((input) => Number(input.value));
  if (target.some((value) => !Number.isFinite(value))) {
    status.textContent = 'Joint position must contain three finite numbers.';
    return;
  }
  correctionSession.setTarget(selectedJointId, target, { mirror: mirrorEdit.checked });
  status.textContent = 'Fit correction recorded. Download the sidecar before rebinding.';
  refreshCorrections();
}

function previewJointMove(id, target, finished) {
  selectedJointId = id;
  target.forEach((value, axis) => { jointInputs[axis].value = Number(value.toFixed(6)); });
  if (finished) {
    correctionSession.setTarget(id, target, { mirror: mirrorEdit.checked });
    status.textContent = 'Fit correction recorded. Download the sidecar before rebinding.';
    refreshCorrections();
  } else {
    requestRender();
  }
}

function boneDepth(bone) {
  let depth = 0;
  let parent = bone.parent;
  while (parent?.isBone) { depth++; parent = parent.parent; }
  return depth;
}

function buildHierarchy() {
  hierarchy.replaceChildren();
  hierarchyRows = view.rig.bones.map((bone, index) => {
    const row = document.createElement('button');
    row.className = 'rig-bone-row';
    row.dataset.index = String(index);
    row.dataset.search = (bone.name || '').toLowerCase();
    row.style.paddingLeft = `${8 + boneDepth(bone) * 10}px`;
    row.textContent = bone.name || '(unnamed bone)';
    row.onclick = () => bonePicker.select(index);
    hierarchy.appendChild(row);
    return row;
  });
  filterHierarchy();
}

function filterHierarchy() {
  const query = hierarchySearch.value.trim().toLowerCase();
  hierarchyRows.forEach((row) => { row.hidden = !!query && !row.dataset.search.includes(query); });
}

function applyMeshOpacity() {
  const amount = Number(opacity.value) / 100;
  opacityValue.textContent = `${opacity.value}%`;
  for (const [material, initial] of materialStates) {
    material.transparent = initial.transparent || amount < 0.999;
    material.opacity = initial.opacity * amount;
    material.depthWrite = amount < 0.999 ? false : initial.depthWrite;
    material.needsUpdate = true;
  }
  requestRender();
}

function captureMaterials() {
  materialStates = new Map();
  for (const record of view.materials) {
    const material = record.original;
    if (!material || materialStates.has(material)) continue;
    materialStates.set(material, {
      opacity: material.opacity,
      transparent: material.transparent,
      depthWrite: material.depthWrite,
    });
  }
  applyMeshOpacity();
}

function disposeCurrent() {
  cameraViews.setEnabled(false);
  bonePicker?.dispose();
  fitOverlay?.dispose();
  bonePicker = null;
  fitOverlay = null;
  selectBone(null);
  hierarchy.replaceChildren();
  hierarchyRows = [];
  materialStates = new Map();
  if (view) {
    disposeModelViewport(view);
    view.renderer.domElement.remove();
    for (const url of new Set(view.spec.revoke || [])) URL.revokeObjectURL(url);
    view = null;
  }
  sidecar = null;
  correctionSession = null;
  selectedJointId = null;
  jointEditor.hidden = true;
  correctionActions.hidden = true;
  selectedModelFile = null;
  selectedSceneFile = null;
  rebind.disabled = true;
  renderPending = false;
}

function loaded(loadedView, sidecarMessage) {
  if (view !== loadedView) return;
  empty.hidden = true;
  view.camera.position.set(1.45, 0.6, 2.45);
  view.controls.target.set(0, 0, 0);
  view.controls.update();
  cameraViews.setEnabled(true);

  bonePicker = new BonePicker({
    scene: view.scene,
    camera: view.camera,
    canvas: view.renderer.domElement,
    bones: view.rig.bones,
    onSelect: selectBone,
  });
  bonePicker.setXray(xray.checked);
  bonePicker.setBonesVisible(showDeformBones.checked);
  bonePicker.setJointsVisible(showDeformJoints.checked);

  if (sidecar) {
    try {
      correctionSession = new RigCorrectionSession(sidecar);
      fitOverlay = new FitSkeletonOverlay({
        scene: view.scene,
        sidecar: correctionSession.toSidecar(),
        bones: view.rig.bones,
        camera: view.camera,
        canvas: view.renderer.domElement,
        onSelect: selectFitJoint,
        onMove: previewJointMove,
        onDragChange: (dragging) => { view.controls.enabled = !dragging; },
      });
      fitOverlay.setXray(xray.checked);
      fitOverlay.setBonesVisible(showFitBones.checked);
      fitOverlay.setJointsVisible(showFitJoints.checked);
    } catch (error) {
      sidecarMessage = `Sidecar verified, but cannot map to this skeleton: ${error.message}`;
      sidecar = null;
      correctionSession = null;
    }
  }

  captureMaterials();
  buildHierarchy();
  showDeformBones.disabled = view.rig.bones.length === 0;
  showDeformJoints.disabled = view.rig.bones.length === 0;
  showFitBones.disabled = !fitOverlay;
  showFitJoints.disabled = !fitOverlay;
  hierarchySearch.disabled = view.rig.bones.length === 0;
  correctionActions.hidden = !correctionSession;
  summary.textContent = `${view.rig.bones.length} deform bones · ` +
    `${view.rig.skinnedMeshes.length} skinned mesh${view.rig.skinnedMeshes.length === 1 ? '' : 'es'}`;
  sidecarSummary.textContent = sidecarMessage;
  refreshCorrections();
  status.textContent = view.rig.bones.length
    ? 'Rig loaded. Select a bone body, joint marker, or hierarchy row.'
    : 'Model loaded without a skinned deform skeleton.';
  resizeViewport();
}

async function prepareSidecar(files, modelFile) {
  const sidecarFile = findRigSidecarFile(files);
  if (!sidecarFile) return {
    data: null, filename: null, file: null, sceneFile: null,
    message: 'No .rig.json sidecar · deform inspection only',
  };
  try {
    const data = parseRigSidecar(await sidecarFile.text());
    const fingerprint = await fingerprintAsset(modelFile);
    if (fingerprint !== data.assetFingerprint) {
      throw new Error('asset fingerprint does not match the selected model');
    }
    const sceneFiles = files.filter((file) => file.name.toLowerCase().endsWith('.blend'));
    if (sceneFiles.length > 1) throw new Error('choose only one prepared .blend scene');
    const sceneFile = sceneFiles[0] || null;
    if (sceneFile && !data.binding) throw new Error('sidecar has no Blender binding manifest');
    if (sceneFile) {
      const sceneFingerprint = await fingerprintAsset(sceneFile);
      if (sceneFingerprint !== data.binding.sceneFingerprint) {
        throw new Error('scene fingerprint does not match the selected .blend');
      }
    }
    return {
      data,
      filename: sidecarFile.name,
      file: sidecarFile,
      sceneFile,
      message: `Verified ${sidecarFile.name}\n${data.rigProfile} · ` +
        `${Object.keys(data.joints).length} fit joints\n` +
        (sceneFile ? `Verified ${sceneFile.name} · Rebind ready` : 'No prepared .blend · export only'),
    };
  } catch (error) {
    return {
      data: null, filename: null, file: null, sceneFile: null,
      message: `Sidecar rejected: ${error.message}`,
    };
  }
}

async function loadFiles(fileList) {
  const token = ++loadToken;
  const files = [...fileList];
  status.textContent = 'Preparing Rig Edit…';
  let spec;
  try {
    spec = specsFromFiles(files).find((candidate) => candidate.kind === 'model');
    if (!spec) throw new Error('Choose a GLB or GLTF model, optionally with one .rig.json sidecar.');
  } catch (error) {
    status.textContent = error.message || String(error);
    return;
  }
  const modelFile = files.find((file) => file.name === spec.label);
  const prepared = await prepareSidecar(files, modelFile);
  if (token !== loadToken) {
    for (const url of new Set(spec.revoke || [])) URL.revokeObjectURL(url);
    return;
  }

  disposeCurrent();
  sidecar = prepared.data;
  selectedModelFile = modelFile;
  selectedSceneFile = prepared.sceneFile;
  sidecarFilename = prepared.filename || `${spec.label.replace(/\.[^.]+$/, '')}.rig.json`;
  empty.hidden = false;
  empty.textContent = `Loading ${spec.label}…`;
  summary.textContent = 'Inspecting deform skeleton…';
  sidecarSummary.textContent = prepared.message;
  hierarchySearch.disabled = true;
  showDeformBones.disabled = true;
  showDeformJoints.disabled = true;
  showFitBones.disabled = true;
  showFitJoints.disabled = true;

  view = createModelViewport({
    pane: viewport,
    spec,
    onChange: requestRender,
    onLoaded: (loadedView) => loaded(loadedView, prepared.message),
    onError: (error) => {
      status.textContent = `${spec.label}: ${error.message || error}`;
      summary.textContent = 'Load failed';
    },
  });
  resizeViewport();
}

function setRebindRunning(running) {
  rebindRunning = running;
  cancelRebind.hidden = !running;
  updateRebindAvailability();
}

function stopRebindStreams() {
  rebindSource?.close();
  rebindSource = null;
  if (rebindPoll) clearInterval(rebindPoll);
  rebindPoll = null;
}

function applyRebindEvent(event) {
  rigJob.hidden = false;
  rigJobBar.style.width = `${Math.max(0, Math.min(100, event.overall_pct || 0))}%`;
  rigJobLabel.textContent = event.message || event.phase;
  if (event.phase === 'done') {
    stopRebindStreams();
    setRebindRunning(false);
    loadRebindResult(event).catch((error) => {
      status.textContent = `Rebind finished, but result loading failed: ${error.message}`;
    });
  } else if (event.phase === 'error') {
    stopRebindStreams();
    setRebindRunning(false);
    status.textContent = event.message || 'Rebind failed';
  }
}

function startRebindPolling(jobId) {
  if (rebindPoll) return;
  rebindPoll = setInterval(async () => {
    try {
      const response = await fetch(`/api/rig/rebind/${jobId}/status`);
      if (!response.ok) return;
      const payload = await response.json();
      if (payload.last_event) applyRebindEvent(payload.last_event);
    } catch (_) { /* the next poll or SSE reconnect can recover */ }
  }, 2000);
}

async function submitRebind() {
  if (rebind.disabled || !correctionSession) return;
  const form = new FormData();
  form.append('asset', selectedModelFile, selectedModelFile.name);
  form.append('scene', selectedSceneFile, selectedSceneFile.name);
  form.append('sidecar', new Blob([
    JSON.stringify(correctionSession.toSidecar(), null, 2) + '\n',
  ], { type: 'application/json' }), sidecarFilename);
  setRebindRunning(true);
  rigArtifacts.hidden = true;
  rigJob.hidden = false;
  rigJobBar.style.width = '2%';
  rigJobLabel.textContent = 'Uploading verified rig bundle…';
  try {
    const response = await fetch('/api/rig/rebind', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    rebindSource = new EventSource(payload.events_url);
    rebindSource.onmessage = (message) => applyRebindEvent(JSON.parse(message.data));
    rebindSource.onerror = () => startRebindPolling(payload.job_id);
    cancelRebind.onclick = async () => {
      await fetch(`/api/rig/rebind/${payload.job_id}/cancel`, { method: 'POST' });
    };
  } catch (error) {
    setRebindRunning(false);
    rigJobLabel.textContent = `Rebind failed: ${error.message}`;
    status.textContent = rigJobLabel.textContent;
  }
}

async function loadRebindResult(event) {
  for (const [field, link] of Object.entries(artifactLinks)) link.href = event[field];
  rigArtifacts.hidden = false;
  const [assetResponse, sceneResponse, sidecarResponse] = await Promise.all([
    fetch(event.result_url), fetch(event.scene_url), fetch(event.sidecar_url),
  ]);
  if (![assetResponse, sceneResponse, sidecarResponse].every((response) => response.ok)) {
    throw new Error('one or more result artifacts could not be downloaded');
  }
  const base = selectedModelFile.name.replace(/\.[^.]+$/, '');
  const files = [
    new File([await assetResponse.blob()], `${base}-rebound.glb`, { type: 'model/gltf-binary' }),
    new File([await sceneResponse.blob()], `${base}-rebound.blend`),
    new File([await sidecarResponse.blob()], `${base}-rebound.rig.json`, {
      type: 'application/json',
    }),
  ];
  await loadFiles(files);
  status.textContent = 'Rebind complete. Inspect the replacement rig or download its artifacts.';
}

drop.onclick = () => fileInput.click();
drop.onkeydown = (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
};
fileInput.onchange = () => {
  if (fileInput.files.length) loadFiles(fileInput.files);
  fileInput.value = '';
};
for (const name of ['dragenter', 'dragover', 'dragleave', 'drop']) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    event.stopPropagation();
    drop.classList.toggle('drag', name === 'dragenter' || name === 'dragover');
  });
}
drop.addEventListener('drop', (event) => {
  if (event.dataTransfer.files.length) loadFiles(event.dataTransfer.files);
});

opacity.oninput = applyMeshOpacity;
xray.onchange = () => {
  bonePicker?.setXray(xray.checked);
  fitOverlay?.setXray(xray.checked);
  requestRender();
};
showDeformBones.onchange = () => {
  bonePicker?.setBonesVisible(showDeformBones.checked); requestRender();
};
showDeformJoints.onchange = () => {
  bonePicker?.setJointsVisible(showDeformJoints.checked); requestRender();
};
showFitBones.onchange = () => {
  fitOverlay?.setBonesVisible(showFitBones.checked); requestRender();
};
showFitJoints.onchange = () => {
  fitOverlay?.setJointsVisible(showFitJoints.checked); requestRender();
};
hierarchySearch.oninput = filterHierarchy;
clearBone.onclick = () => bonePicker?.clear();
jointInputs.forEach((input) => input.addEventListener('change', applyJointInputs));
resetJoint.onclick = () => {
  if (!selectedJointId || !correctionSession) return;
  correctionSession.reset(selectedJointId, { mirror: mirrorEdit.checked });
  refreshCorrections();
};
undo.onclick = () => { if (correctionSession?.undo()) refreshCorrections(); };
redo.onclick = () => { if (correctionSession?.redo()) refreshCorrections(); };
resetAll.onclick = () => { correctionSession?.resetAll(); refreshCorrections(); };
exportCorrections.onclick = () => {
  if (!correctionSession) return;
  const blob = new Blob([JSON.stringify(correctionSession.toSidecar(), null, 2) + '\n'], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = sidecarFilename;
  link.click();
  URL.revokeObjectURL(url);
};
rebind.onclick = submitRebind;
addEventListener('resize', resizeViewport);
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'rig') requestAnimationFrame(resizeViewport);
});
