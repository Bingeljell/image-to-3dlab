import * as THREE from 'three';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import {
  IMAGE_EXTS,
  extOf,
  isImageFile,
  isModelFile,
  specsFromFiles,
} from '../core/asset-files.js';

const state = {
  cull: true,          // default ON: a double-sided preview cannot show a hollow mesh
  wire: false,
  mode: 'textured',    // textured | flat | normals
  spin: false,
  sync: true,
};

// Up to three slots. Each holds a model view, an image view, or null. `views` is the
// compacted list of *model* views — the only ones the 3D render/frame/sync loops touch;
// image panes are static. Both are kept in sync by rebuildViews().
const MAX = 3;
// ?restricted=1 -- set when this page is loaded inside the Generate tab's embedded preview
// iframe. That panel is sized for one model; a second/third pane crammed into it makes every
// pane too small to see anything. Full multi-model comparison belongs in the dedicated
// Compare tab (the whole screen), not this embedded view, so hide the add-pane affordances.
const RESTRICTED = new URLSearchParams(location.search).get('restricted') === '1';
const slots = [null, null, null];
let views = [];
const rebuildViews = () => { views = slots.filter((s) => s && s.kind === 'model'); updateChrome(); };

const panesEl = document.getElementById('panes');
const dropEl = document.getElementById('drop');
let ghostEl = null;     // the "＋ add pane" placeholder, built once in the wiring section
const errEl = document.getElementById('err');
const setErr = (msg) => { errEl.textContent = msg || ''; };

// Fully event-driven rendering. Nothing repaints unless something asked it to: a moved
// camera, a state change, or an active spin. When idle there is no requestAnimationFrame
// scheduled at all -- zero CPU, zero GPU. This matters beyond battery: a free-running
// 60fps loop tightens the macOS GPU watchdog, the same watchdog that kills the remesh
// kernel mid-dispatch and, on 2026-08-13, took the machine down. `renderFrame` is a
// hoisted declaration defined with the render loop below.
let renderPending = false;
function requestRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(renderFrame);
}
const invalidate = requestRender;

// --- turning dropped files into slot specs -------------------------------------------

function firstEmpty() { return slots.findIndex((s) => s === null); }
function occupied() { return slots.filter(Boolean).length; }

// Load specs into slots. `targetSlot` (or null) forces the first spec to replace a
// specific pane; the rest fill empty slots in order, up to MAX.
function loadSpecs(specList, targetSlot = null) {
  let placed = 0;
  for (const spec of specList) {
    const idx = (targetSlot != null && placed === 0) ? targetSlot : firstEmpty();
    if (idx < 0) {
      setErr(`only ${MAX} panes — "${spec.label}" and any after it were skipped`);
      break;
    }
    mountSlot(idx, spec);
    placed++;
  }
}

function handleFiles(fileList, targetSlot = null) {
  setErr('');
  const files = [...fileList];
  // An image-only drop onto a loaded model pane means "align to this", not "replace".
  if (targetSlot != null && slots[targetSlot]?.kind === 'model'
      && !files.some(isModelFile) && files.some(isImageFile)) {
    setOverlay(slots[targetSlot], files.find(isImageFile));
    return;
  }
  try {
    loadSpecs(specsFromFiles(files), targetSlot);
  } catch (e) {
    setErr(e.message || String(e));
  }
}

// --- slot lifecycle ------------------------------------------------------------------

function mountSlot(slotIndex, spec) {
  if (slots[slotIndex]) clearSlot(slotIndex);
  if (spec.kind === 'image') mountImage(slotIndex, spec);
  else mountModel(slotIndex, spec);
}

// Shared pane chrome: caption, control cluster (top-right), stats. Returns the pane and a
// hook the caller fills with kind-specific controls.
function makePane(slotIndex, spec, kindLabel, ctlHtml) {
  const pane = document.createElement('div');
  pane.className = `pane ${spec.kind}`;
  pane.style.order = String(slotIndex);   // keep visual order == slot order
  pane.innerHTML =
    `<div class="cap"><span class="kind">${kindLabel}</span><b>${spec.label}</b></div>` +
    `<div class="ctl">${ctlHtml}</div>` +
    `<div class="stats">loading…</div>`;
  panesEl.appendChild(pane);
  pane.querySelector('.clear').onclick = () => clearSlot(slotIndex);
  wirePaneDrop(pane, slotIndex);
  return pane;
}

function mountModel(slotIndex, spec) {
  const pane = makePane(slotIndex, spec, 'model',
    `<button class="ovl" title="overlay a source image ON TOP of this model, to align it">⧉ overlay</button>` +
    `<button class="clear" title="remove">✕</button>`);

  const view = createModelViewport({
    pane,
    spec,
    slotIndex,
    onChange: invalidate,
    onLoaded: () => { applyState(); frame(); },
    onError: (error) => setErr(`${spec.label}: ${error.message || error}`),
  });
  view.ovlBtn = pane.querySelector('.ovl');
  view.ovlBtn.onclick = () => { pendingOverlay = view; ovlInput.click(); };

  slots[slotIndex] = view;
  rebuildViews();
  resize();
}

function mountImage(slotIndex, spec) {
  const pane = makePane(slotIndex, spec, 'image',
    `<button class="clear" title="remove">✕</button>`);

  const wrap = document.createElement('div');
  wrap.className = 'imgwrap';
  const img = document.createElement('img');
  img.src = spec.url;
  wrap.appendChild(img);
  pane.appendChild(wrap);

  const view = { kind: 'image', spec, slotIndex, pane, wrap, img,
                 stats: pane.querySelector('.stats') };

  img.onload = () => { updateImageStats(view); };
  img.onerror = () => { view.stats.textContent = 'FAILED'; setErr(`${spec.label}: could not decode image`); };
  wireImagePanZoom(view);

  slots[slotIndex] = view;
  rebuildViews();
}

// Wheel to zoom, drag to pan, double-click to reset — so a source image can be inspected
// at the same detail as the mesh beside it.
function wireImagePanZoom(view) {
  let scale = 1, tx = 0, ty = 0;
  view.zoom = () => scale;
  const apply = () => {
    view.img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    updateImageStats(view);
  };
  view.wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    scale = Math.max(0.2, Math.min(20, scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
    apply();
  }, { passive: false });
  let drag = null;
  view.wrap.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX - tx, y: e.clientY - ty };
    view.wrap.classList.add('grabbing');
    view.wrap.setPointerCapture(e.pointerId);
  });
  view.wrap.addEventListener('pointermove', (e) => {
    if (!drag) return;
    tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply();
  });
  const end = () => { drag = null; view.wrap.classList.remove('grabbing'); };
  view.wrap.addEventListener('pointerup', end);
  view.wrap.addEventListener('pointercancel', end);
  view.wrap.addEventListener('dblclick', () => { scale = 1; tx = 0; ty = 0; apply(); });
}

function updateImageStats(view) {
  const z = view.zoom ? Math.round(view.zoom() * 100) : 100;
  view.stats.textContent = `${view.img.naturalWidth}×${view.img.naturalHeight}px\n${z}% zoom`;
}

function clearSlot(slotIndex) {
  const view = slots[slotIndex];
  if (!view) return;
  if (view.kind === 'model') {
    removeOverlay(view);
    disposeModelViewport(view);
  }
  view.pane.remove();
  for (const u of (view.spec.revoke || [])) {
    // Only revoke once the last view sharing this blob set is gone (model bundles share one).
    const stillUsed = slots.some((s) => s && s !== view && s.spec.revoke === view.spec.revoke);
    if (!stillUsed) URL.revokeObjectURL(u);
  }
  slots[slotIndex] = null;
  rebuildViews();
  invalidate();
}

// --- alignment overlay (onion-skin) --------------------------------------------------

let pendingOverlay = null;   // model view awaiting an image from the overlay picker

function setOverlay(view, file) {
  if (view.kind !== 'model') return;
  removeOverlay(view);
  const url = URL.createObjectURL(file);
  const img = document.createElement('img');
  img.className = 'model-overlay';
  img.src = url;
  img.style.opacity = '0.6';
  view.pane.appendChild(img);

  const bar = document.createElement('div');
  bar.className = 'ovlbar';
  const label = document.createElement('span');
  label.className = 'lbl';
  label.textContent = 'overlay';
  const range = document.createElement('input');
  range.type = 'range'; range.min = '0'; range.max = '100'; range.value = '60';
  const pct = document.createElement('span');
  pct.textContent = '60%';
  range.oninput = () => { img.style.opacity = String(range.value / 100); pct.textContent = `${range.value}%`; };
  const rm = document.createElement('button');
  rm.textContent = '✕'; rm.title = 'remove overlay';
  rm.onclick = () => removeOverlay(view);
  bar.append(label, range, pct, rm);
  view.pane.appendChild(bar);

  view.overlay = { img, url, bar };
  view.ovlBtn?.classList.add('on');
  invalidate();
}

function removeOverlay(view) {
  if (!view.overlay) return;
  URL.revokeObjectURL(view.overlay.url);
  view.overlay.img.remove();
  view.overlay.bar.remove();
  view.overlay = null;
  view.ovlBtn?.classList.remove('on');
}

// --- shared 3D render state ----------------------------------------------------------

function applyState() {
  for (const view of views) {
    for (const rec of view.materials) {
      let mat = rec.original;
      if (state.mode === 'flat') {
        rec.flat ||= new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.85 });
        mat = rec.flat;
      } else if (state.mode === 'normals') {
        rec.normal ||= new THREE.MeshNormalMaterial();
        mat = rec.normal;
      }
      mat.side = state.cull ? THREE.FrontSide : THREE.DoubleSide;
      mat.wireframe = state.wire;
      // flatShading computes normals from screen-space derivatives (dFdx/dFdy), which are
      // undefined for GL_LINES -- so a flatShading material goes solid black in wireframe
      // mode even with valid vertex normals now guaranteed at load time. Force it off for
      // wireframe, restored from the material's own original value otherwise (rec.flat/
      // rec.normal never had flatShading:true to begin with, so this only ever matters for
      // rec.original).
      if ('flatShading' in mat) {
        mat.flatShading = state.wire ? false : (mat === rec.original && rec.origFlatShading);
      }
      mat.needsUpdate = true;
      rec.mesh.material = mat;
    }
  }
  invalidate();
}

let azimuth = 20, elevation = 0.18;

function fitDistance(camera) {
  // Model is normalised into a unit box centred on the origin, so its bounding sphere
  // radius is at most sqrt(3)/2. Fit that, with a small margin, in the narrower of the
  // two FOVs -- otherwise a tall pane crops the subject horizontally.
  const radius = Math.sqrt(3) / 2;
  const vFov = camera.fov * Math.PI / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  return 1.25 * radius / Math.sin(Math.min(vFov, hFov) / 2);
}

function frame() {
  const a = azimuth * Math.PI / 180;
  for (const view of views) {
    const d = fitDistance(view.camera);
    view.camera.position.set(
      Math.sin(a) * d * Math.cos(elevation),
      Math.sin(elevation) * d,
      Math.cos(a) * d * Math.cos(elevation),
    );
    view.controls.target.set(0, 0, 0);
    view.controls.update();
  }
  invalidate();
}

function setAzimuth(deg) { azimuth = deg; frame(); }

// --- chrome (drop zone, add button) --------------------------------------------------

function updateChrome() {
  const n = occupied();
  // Fully empty -> the big centred drop zone. One or two loaded -> a ghost "add pane" in
  // the remaining space so loading side by side is obvious. Full -> neither.
  dropEl.classList.toggle('hidden', n > 0);
  if (ghostEl) ghostEl.style.display = (!RESTRICTED && n > 0 && n < MAX) ? 'flex' : 'none';
  document.getElementById('addcount').textContent = `(${n}/${MAX})`;
  document.getElementById('add').disabled = RESTRICTED || n >= MAX;
  if (RESTRICTED) document.getElementById('add').style.display = 'none';
}

// --- wiring --------------------------------------------------------------------------

const btn = (id) => document.getElementById(id);
btn('cull').onclick = (e) => {
  state.cull = !state.cull;
  e.target.classList.toggle('on', state.cull);
  e.target.textContent = state.cull ? 'backface ON' : 'DOUBLE-SIDED';
  applyState();
};
btn('wire').onclick = (e) => { state.wire = !state.wire; e.target.classList.toggle('on', state.wire); applyState(); };
btn('flat').onclick = (e) => {
  state.mode = state.mode === 'flat' ? 'textured' : 'flat';
  e.target.classList.toggle('on', state.mode === 'flat');
  btn('norm').classList.remove('on');
  applyState();
};
btn('norm').onclick = (e) => {
  state.mode = state.mode === 'normals' ? 'textured' : 'normals';
  e.target.classList.toggle('on', state.mode === 'normals');
  btn('flat').classList.remove('on');
  applyState();
};
btn('spin').onclick = (e) => { state.spin = !state.spin; e.target.classList.toggle('on', state.spin); if (state.spin) invalidate(); };
btn('sync').onclick = (e) => { state.sync = !state.sync; e.target.classList.toggle('on', state.sync); };
for (const b of document.querySelectorAll('[data-az]')) {
  b.onclick = () => setAzimuth(Number(b.dataset.az));
}

// File pickers. The ＋ button and drop-zone "browse" share one input; overlay images use
// a second so they never fill a new slot by mistake.
const fileInput = btn('file');
const openPicker = () => fileInput.click();
btn('add').onclick = openPicker;
btn('browse').onclick = openPicker;
fileInput.onchange = () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
  fileInput.value = '';        // let the same file be re-picked later
};
const ovlInput = btn('ovlfile');
ovlInput.onchange = () => {
  if (pendingOverlay && ovlInput.files[0]) setOverlay(pendingOverlay, ovlInput.files[0]);
  pendingOverlay = null;
  ovlInput.value = '';
};

// The in-layout "＋ add pane" placeholder. It fills the empty space beside loaded panes;
// clicking or dropping onto it loads the next model or source image side by side. `order`
// keeps it visually last, after the real panes (which order by slot index).
ghostEl = document.createElement('div');
ghostEl.className = 'pane ghost';
ghostEl.style.order = '99';
ghostEl.style.display = 'none';
ghostEl.innerHTML =
  `<div class="inner"><div class="plus">＋</div>` +
  `<div class="t">Add a model or source image</div>` +
  `<div class="s">click to browse · or drop a file here</div></div>`;
ghostEl.onclick = openPicker;
ghostEl.addEventListener('dragover', (e) => { e.preventDefault(); ghostEl.classList.add('dragtarget'); });
ghostEl.addEventListener('dragleave', () => ghostEl.classList.remove('dragtarget'));
ghostEl.addEventListener('drop', (e) => {
  e.preventDefault(); e.stopPropagation();
  ghostEl.classList.remove('dragtarget');
  document.body.classList.remove('dragging');
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
panesEl.appendChild(ghostEl);

// Drag-and-drop. Dropping onto a pane targets that pane (replace, or overlay an image on a
// model); dropping elsewhere fills the next empty slots. dragover must preventDefault or
// the browser navigates to the file.
function wirePaneDrop(pane, slotIndex) {
  pane.addEventListener('dragover', (e) => { e.preventDefault(); pane.classList.add('dragtarget'); });
  pane.addEventListener('dragleave', () => pane.classList.remove('dragtarget'));
  pane.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation();
    pane.classList.remove('dragtarget');
    document.body.classList.remove('dragging');
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files, slotIndex);
  });
}
document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
document.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

// --- initial load from the query string ----------------------------------------------
// Assets come from ?a=path&b=path&c=path (+ la/lb/lc labels), relative to the repo root
// that serve.py serves. An image extension makes an image pane; anything else, a model.
// This keeps `serve.py --open` and browser automation working; drops layer on top.
const q = new URLSearchParams(location.search);
for (const key of ['a', 'b', 'c']) {
  const url = q.get(key);
  if (!url) continue;
  // Repo-relative: anchor at the server root, not at /viewer/ where this page lives.
  const resolved = /^(https?:|\/|blob:)/.test(url) ? url : '/' + url;
  const ext = extOf(new URL(resolved, location.href).pathname);
  const label = q.get('l' + key) || url.split('/').pop();
  const idx = firstEmpty();
  if (idx < 0) break;
  if (IMAGE_EXTS.has(ext)) mountSlot(idx, { kind: 'image', url: resolved, label });
  else mountSlot(idx, { kind: 'model', url: resolved, label, ext });
}
updateChrome();

// --- sizing + render loop ------------------------------------------------------------

function resize() {
  for (const view of views) {
    const w = view.pane.clientWidth, h = view.pane.clientHeight;
    // updateStyle must stay ON. With it off, three.js leaves the canvas CSS size equal to
    // the drawing buffer, so at devicePixelRatio 2 the canvas is twice the pane and you see
    // the top-left quadrant of a correctly framed scene.
    view.renderer.setSize(w, h);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
  }
}
addEventListener('resize', () => { resize(); frame(); });

// One rendered frame. Advances spin, lets OrbitControls damping settle, mirrors the lead
// camera when synced, draws every model pane, and reschedules itself *only* while
// something is still moving. When nothing moves it simply stops -- no rAF stays queued, so
// an idle viewer costs nothing. Image panes are static DOM and never enter this loop.
function renderFrame() {
  renderPending = false;
  let moving = false;

  if (state.spin) {
    let spun = false;
    for (const v of views) if (v.root) { v.root.rotation.y += 0.004; spun = true; }
    if (spun) moving = true;      // never reschedule forever with nothing to spin
  }
  // Damping keeps easing for a few frames after input stops; honour it, then settle.
  for (const v of views) if (v.controls.update()) moving = true;

  // One camera drives the rest, so a side-by-side comparison stays honest.
  if (state.sync && views.length > 1) {
    const lead = views[0];
    for (const v of views) {
      if (v === lead) continue;
      v.camera.position.copy(lead.camera.position);
      v.camera.quaternion.copy(lead.camera.quaternion);
      v.controls.target.copy(lead.controls.target);
    }
  }
  for (const v of views) v.renderer.render(v.scene, v.camera);

  if (moving) requestRender();
}
window.__viewerInvalidate = invalidate;   // so automation can force a repaint
window.__viewerReady = true;
