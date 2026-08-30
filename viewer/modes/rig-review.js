import { BonePicker } from '../animation/bone-picker.js';
import { describeBone } from '../animation/rig-inspection.js';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { specsFromFiles } from '../core/asset-files.js';
import { FitSkeletonOverlay } from '../rig/fit-skeleton-overlay.js';
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
const showDeform = element('rig-show-deform');
const showFit = element('rig-show-fit');
const hierarchySearch = element('rig-hierarchy-search');
const hierarchy = element('rig-hierarchy');
const selectionEmpty = element('rig-selection-empty');
const selection = element('rig-selection');
const boneName = element('rig-bone-name');
const boneParent = element('rig-bone-parent');
const boneChildren = element('rig-bone-children');
const mappedJoints = element('rig-mapped-joints');
const clearBone = element('rig-clear-bone');

let view = null;
let bonePicker = null;
let fitOverlay = null;
let sidecar = null;
let selectedBone = null;
let materialStates = new Map();
let hierarchyRows = [];
let renderPending = false;
let loadToken = 0;

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

function selectBone(bone, index = -1) {
  selectedBone = bone;
  selectionEmpty.hidden = !!bone;
  selection.hidden = !bone;
  clearBone.disabled = !bone;
  hierarchyRows.forEach((row) => row.classList.toggle('on', Number(row.dataset.index) === index));
  if (!bone || !view) {
    fitOverlay?.selectJoint(null);
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
  fitOverlay?.selectJoint(jointIds[0] || null);
  requestRender();
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
  renderPending = false;
}

function loaded(loadedView, sidecarMessage) {
  if (view !== loadedView) return;
  empty.hidden = true;
  view.camera.position.set(1.45, 0.6, 2.45);
  view.controls.target.set(0, 0, 0);
  view.controls.update();

  bonePicker = new BonePicker({
    scene: view.scene,
    camera: view.camera,
    canvas: view.renderer.domElement,
    bones: view.rig.bones,
    onSelect: selectBone,
  });
  bonePicker.setXray(xray.checked);
  bonePicker.setVisible(showDeform.checked);

  if (sidecar) {
    try {
      fitOverlay = new FitSkeletonOverlay({ scene: view.scene, sidecar, bones: view.rig.bones });
      fitOverlay.setXray(xray.checked);
      fitOverlay.setVisible(showFit.checked);
    } catch (error) {
      sidecarMessage = `Sidecar verified, but cannot map to this skeleton: ${error.message}`;
      sidecar = null;
    }
  }

  captureMaterials();
  buildHierarchy();
  showDeform.disabled = view.rig.bones.length === 0;
  showFit.disabled = !fitOverlay;
  hierarchySearch.disabled = view.rig.bones.length === 0;
  summary.textContent = `${view.rig.bones.length} deform bones · ` +
    `${view.rig.skinnedMeshes.length} skinned mesh${view.rig.skinnedMeshes.length === 1 ? '' : 'es'}`;
  sidecarSummary.textContent = sidecarMessage;
  status.textContent = view.rig.bones.length
    ? 'Rig loaded. Select a bone body, joint marker, or hierarchy row.'
    : 'Model loaded without a skinned deform skeleton.';
  resizeViewport();
}

async function prepareSidecar(files, modelFile) {
  const sidecarFile = findRigSidecarFile(files);
  if (!sidecarFile) return { data: null, message: 'No .rig.json sidecar · deform inspection only' };
  try {
    const data = parseRigSidecar(await sidecarFile.text());
    const fingerprint = await fingerprintAsset(modelFile);
    if (fingerprint !== data.assetFingerprint) {
      throw new Error('asset fingerprint does not match the selected model');
    }
    return {
      data,
      message: `Verified ${sidecarFile.name}\n${data.rigProfile} · ${Object.keys(data.joints).length} fit joints`,
    };
  } catch (error) {
    return { data: null, message: `Sidecar rejected: ${error.message}` };
  }
}

async function loadFiles(fileList) {
  const token = ++loadToken;
  const files = [...fileList];
  status.textContent = 'Preparing rig review…';
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
  empty.hidden = false;
  empty.textContent = `Loading ${spec.label}…`;
  summary.textContent = 'Inspecting deform skeleton…';
  sidecarSummary.textContent = prepared.message;
  hierarchySearch.disabled = true;
  showDeform.disabled = true;
  showFit.disabled = true;

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
showDeform.onchange = () => { bonePicker?.setVisible(showDeform.checked); requestRender(); };
showFit.onchange = () => { fitOverlay?.setVisible(showFit.checked); requestRender(); };
hierarchySearch.oninput = filterHierarchy;
clearBone.onclick = () => bonePicker?.clear();
addEventListener('resize', resizeViewport);
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'rig') requestAnimationFrame(resizeViewport);
});
