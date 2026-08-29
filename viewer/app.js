import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { OrbitControls } from './vendor/controls/OrbitControls.js';
import { IndexedOBJLoader } from './IndexedOBJLoader.js';
import { RoomEnvironment } from './vendor/environments/RoomEnvironment.js';

// Built once (pure geometry, no GPU resources) and baked per-renderer below —
// this is what stands in for Blender's Material Preview studio HDRI.
const roomEnvironment = new RoomEnvironment();

// What a slot can hold. A model root drives the 3D pipeline; an image becomes a source
// pane (or, dropped onto a model, an alignment overlay). Adding a model format is one line
// here plus a vendored loader import; anything unlisted is rejected in the UI.
const MODEL_EXTS = new Set(['glb', 'gltf', 'obj']);
const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'avif']);
const extOf = (name) => name.split('.').pop().toLowerCase();
const isModelFile = (f) => MODEL_EXTS.has(extOf(f.name));
const isImageFile = (f) => IMAGE_EXTS.has(extOf(f.name));

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

// A "bundle" is one root file plus any sibling resources it references (a .gltf's .bin and
// textures). Local files carry no server path, so we hand the loader blob URLs and a
// LoadingManager that resolves each referenced resource by basename.
function makeManager(blobByName) {
  const m = new THREE.LoadingManager();
  m.setURLModifier((url) => {
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    const base = decodeURIComponent(url.split('/').pop().split('?')[0]);
    return blobByName.get(base) || url;
  });
  return m;
}

// A drop that contains a model file loads the model (images ride along as its textures).
// A drop with no model file treats each image as its own source-image pane. This is the
// rule that keeps "gltf + textures" and "a bare reference photo" from colliding.
function specsFromFiles(fileList) {
  const files = [...fileList];
  const models = files.filter(isModelFile);
  if (models.length) {
    const blobByName = new Map();
    const urls = [];
    for (const f of files) {
      const u = URL.createObjectURL(f);
      urls.push(u);
      blobByName.set(f.name, u);
    }
    const manager = makeManager(blobByName);
    return models.map((f) => ({
      kind: 'model', url: blobByName.get(f.name), label: f.name,
      ext: extOf(f.name), manager, revoke: urls,
    }));
  }
  const images = files.filter(isImageFile);
  if (images.length) {
    return images.map((f) => {
      const u = URL.createObjectURL(f);
      return { kind: 'image', url: u, label: f.name, revoke: [u] };
    });
  }
  throw new Error('nothing loadable — expected a .glb/.gltf/.obj model or a PNG/JPG image');
}

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

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  pane.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161a);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(0, 0.3, 3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  // Studio HDRI environment, baked once per renderer with PMREMGenerator — this is
  // what Blender's Material Preview/LookDev viewport actually uses instead of lamps,
  // and why it reads as evenly lit from every side. See vendor/environments/RoomEnvironment.js.
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(roomEnvironment, 0.04).texture;
  pmremGenerator.dispose();
  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x30302a, 0.15));

  const view = { kind: 'model', spec, slotIndex, pane, renderer, scene, camera, controls,
                 root: null, stats: pane.querySelector('.stats'), materials: [], overlay: null };

  // Render on demand: a moving camera is the only reason to repaint. See renderFrame()'s
  // note on the GPU watchdog for why this viewer never runs a free-running 60fps loop.
  controls.addEventListener('change', invalidate);
  view.ovlBtn = pane.querySelector('.ovl');
  view.ovlBtn.onclick = () => { pendingOverlay = view; ovlInput.click(); };

  slots[slotIndex] = view;
  rebuildViews();
  resize();
  loadModel(view);
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
    view.controls.dispose();
    view.renderer.dispose();
    view.renderer.forceContextLoss();
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

function loadModel(view) {
  const spec = view.spec;
  const onLoad = (root) => {
    // Normalise to a unit box centred on the origin, so a size difference cannot
    // masquerade as a quality difference between panes.
    //
    // Wrapped in a group rather than transforming `root` directly: Box3.setFromObject
    // reads world matrices, so measuring and then mutating the same object needs a
    // matrix flush between each step and silently mis-frames if you forget one.
    const pivot = new THREE.Group();
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const s = 1 / Math.max(size.x, size.y, size.z);
    root.position.sub(centre);          // centre in local space, before scaling
    pivot.scale.setScalar(s);
    pivot.add(root);
    pivot.updateMatrixWorld(true);

    let faces = 0, verts = 0;
    root.traverse((o) => {
      if (!o.isMesh) return;
      const g = o.geometry;
      // Some exports (e.g. a bare shape-stage mesh.export() with no baked material) ship
      // no NORMAL attribute at all. The default material's flatShading:true papers over
      // that by computing per-face normals from screen-space derivatives (dFdx/dFdy) --
      // but flat/normals mode swap in materials that read the vertex attribute directly,
      // and wireframe draws GL_LINES where those derivatives are undefined anyway. Either
      // way a missing normal reads as a zero vector -> zero lighting -> solid black.
      // Compute real vertex normals once at load time so every mode has something to read.
      if (!g.attributes.normal) g.computeVertexNormals();
      verts += g.attributes.position.count;
      faces += g.index ? g.index.count / 3 : g.attributes.position.count / 3;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        view.materials.push({ mesh: o, original: m, flat: null, normal: null,
                              origFlatShading: !!m.flatShading });
      }
    });
    view.root = pivot;
    view.scene.add(pivot);
    view.stats.textContent =
      `${faces.toLocaleString()} faces\n${verts.toLocaleString()} verts (as stored)`;
    applyState();
    frame();
  };
  const onProgress = (e) => {
    if (!e.lengthComputable) return;
    view.stats.textContent = `loading… ${Math.round(e.loaded / e.total * 100)}%`;
  };
  const onError = (e) => {
    view.stats.textContent = 'FAILED';
    setErr(`${spec.label}: ${e.message || e}`);
  };

  if (spec.ext === 'obj') {
    new IndexedOBJLoader().load(spec.url, (root) => {
      // o_voxel's GLB exporter maps native TRELLIS coordinates as
      // (x, y, z) -> (x, z, -y). Apply the identical view-only transform so a raw OBJ
      // and its processed GLB share an up axis and camera angle.
      root.rotation.x = -Math.PI / 2;
      onLoad(root);
    }, onProgress, onError);
  } else {
    const loader = spec.manager ? new GLTFLoader(spec.manager) : new GLTFLoader();
    loader.load(spec.url, (gltf) => onLoad(gltf.scene), onProgress, onError);
  }
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

// --- Generate mode -------------------------------------------------------------------
// Kept in this same shell so the generated result can open the exact Compare viewer (including
// its backface-cull acceptance toggle) without changing any of Compare's loading/render code.
const generateView = document.getElementById('generate-view');
const compareView = document.getElementById('compare-view');
// Backend metadata (stages, labels, requires_alpha) is server-owned truth (viewer/generate_api.py
// BACKENDS registry) -- fetched once so the frontend never hardcodes a second copy that can drift.
let backendMeta = {
  trellis: {
    requires_alpha: true,
    stages: ['load', 'sparse_structure', 'shape_slat_coarse', 'shape_slat_fine', 'texture_slat', 'decode', 'bake'],
    stage_labels: { load: 'Load', sparse_structure: 'Sparse structure', shape_slat_coarse: 'Shape SLat coarse',
      shape_slat_fine: 'Shape SLat fine', texture_slat: 'Texture SLat', decode: 'Decode', bake: 'Bake / remesh' },
  },
}; // placeholder until /api/backends resolves; keeps the page usable if that fetch is slow/fails
let generateStages = [];
const stageRows = new Map();
const generateStagesEl = document.getElementById('generate-stages');
function buildStageRows(backendId) {
  const meta = backendMeta[backendId] || { stages: ['running'], stage_labels: { running: 'Running' } };
  generateStages = meta.stages.map((phase) => [phase, meta.stage_labels[phase] || phase]);
  stageRows.clear();
  generateStagesEl.innerHTML = '';
  for (const [phase, label] of generateStages) {
    const row = document.createElement('div');
    row.className = 'stage-row'; row.dataset.phase = phase;
    row.innerHTML = `<span class="stage-dot">○</span><span>${label}</span><span class="stage-detail">queued</span>`;
    generateStagesEl.appendChild(row);
    stageRows.set(phase, row);
  }
}
buildStageRows('trellis');
const gen = {
  file: null, objectUrl: null, hasAlpha: false, jobId: null, source: null,
  running: false, outputDir: null, pollTimer: null,
};
const setupState = { ready: false };
function currentBackend() { return g('generate-backend').value; }
function backendRequiresAlpha() {
  const meta = backendMeta[currentBackend()];
  return meta ? meta.requires_alpha : true; // fail conservative if metadata hasn't loaded yet
}
const g = (id) => document.getElementById(id);
const formatDuration = (seconds) => {
  if (seconds == null || !Number.isFinite(Number(seconds))) return 'estimating…';
  const n = Math.max(0, Math.round(Number(seconds)));
  if (n < 60) return `${n}s`;
  const m = Math.round(n / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)}h ${m % 60}m`;
};
const resetStageRows = () => {
  for (const row of stageRows.values()) {
    row.className = 'stage-row'; row.querySelector('.stage-dot').textContent = '○';
    row.querySelector('.stage-detail').textContent = 'queued';
  }
};
const updateGenerateButton = () => {
  const needsRembg = backendRequiresAlpha() && !gen.hasAlpha && !g('generate-rembg').checked;
  g('generate-submit').disabled = !gen.file || gen.running || needsRembg || !setupState.ready;
};
const creditsView = document.getElementById('credits-view');
const setGenerateMode = (mode) => {
  compareView.classList.toggle('hidden', mode !== 'compare');
  generateView.hidden = mode !== 'generate';
  creditsView.hidden = mode !== 'credits';
  g('mode-compare').classList.toggle('on', mode === 'compare');
  g('mode-generate').classList.toggle('on', mode === 'generate');
  g('mode-credits').classList.toggle('on', mode === 'credits');
};
g('mode-compare').onclick = () => setGenerateMode('compare');
g('mode-generate').onclick = () => setGenerateMode('generate');
g('mode-credits').onclick = () => setGenerateMode('credits');

async function inspectAlpha(file) {
  // JPEG and opaque formats cannot carry a useful foreground alpha. For PNG/WebP, inspect the
  // decoded pixels in-browser so the guardrail is visible before a long job is submitted.
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const pixels = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    let min = 255;
    for (let i = 3; i < pixels.length; i += 4) { if (pixels[i] < min) min = pixels[i]; if (min === 0) break; }
    bitmap.close();
    return file.type === 'image/png' || file.type === 'image/webp' ? min < 255 : false;
  } catch (_) { return false; }
}
function setGenerateFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  if (gen.objectUrl) URL.revokeObjectURL(gen.objectUrl);
  gen.file = file; gen.objectUrl = URL.createObjectURL(file); gen.hasAlpha = false;
  g('generate-preview').src = gen.objectUrl;
  g('generate-drop').classList.add('has-image');
  const badge = g('generate-alpha'); badge.hidden = false; badge.className = 'alpha-badge';
  badge.textContent = 'checking alpha…';
  inspectAlpha(file).then((hasAlpha) => {
    if (gen.file !== file) return;
    gen.hasAlpha = hasAlpha;
    badge.className = `alpha-badge ${hasAlpha ? 'alpha-good' : 'alpha-bad'}`;
    badge.textContent = hasAlpha ? 'transparent foreground ✓' : 'no alpha — enable rembg to continue';
    updateGenerateButton();
  });
  updateGenerateButton();
  g('generate-status').textContent = '';
}
const generateFile = g('generate-file');
const generateDrop = g('generate-drop');
generateDrop.onclick = () => generateFile.click();
generateFile.onchange = () => { if (generateFile.files[0]) setGenerateFile(generateFile.files[0]); generateFile.value = ''; };
for (const event of ['dragenter', 'dragover']) generateDrop.addEventListener(event, (e) => { e.preventDefault(); generateDrop.classList.add('drag'); });
for (const event of ['dragleave', 'drop']) generateDrop.addEventListener(event, (e) => { e.preventDefault(); generateDrop.classList.remove('drag'); });
generateDrop.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) setGenerateFile(e.dataTransfer.files[0]); });
g('generate-rembg').onchange = updateGenerateButton;

function applyGenerateProgress(event) {
  const phase = event.phase;
  if (stageRows.has(phase)) {
    const current = generateStages.findIndex(([name]) => name === phase);
    for (let i = 0; i < current; i++) {
      const row = stageRows.get(generateStages[i][0]);
      row.className = 'stage-row done'; row.querySelector('.stage-dot').textContent = '✓';
      row.querySelector('.stage-detail').textContent = 'done';
    }
    const row = stageRows.get(phase);
    const pct = event.stage_pct == null ? '' : `${event.stage_pct}%`;
    const steps = event.step != null ? `${event.step}/${event.total}` : pct;
    row.className = event.stage_pct === 100 ? 'stage-row done' : 'stage-row active';
    row.querySelector('.stage-dot').textContent = event.stage_pct === 100 ? '✓' : '●';
    row.querySelector('.stage-detail').textContent = event.stage_pct === 100
      ? 'done' : `${steps} · ~${formatDuration(event.stage_eta_seconds)}`;
    g('generate-overall-bar').style.width = `${Math.max(0, Math.min(100, event.overall_pct || 0))}%`;
    g('generate-overall-label').textContent = event.message || phase;
    g('generate-overall-eta').textContent = event.total_eta_seconds == null
      ? 'Total still estimating' : `Total ~${formatDuration(event.total_eta_seconds)}`;
  }
  if (phase === 'done') {
    g('generate-overall-bar').style.width = '100%';
    g('generate-overall-label').textContent = 'Generation complete';
    g('generate-overall-eta').textContent = '';
    g('generate-status').textContent = 'Done — the generated GLB is ready below.';
    const durationEl = g('generate-duration');
    durationEl.style.display = 'block';
    durationEl.textContent = `Generated in ${formatDuration(event.elapsed_seconds)}`;
    g('generate-progress-box').style.display = 'none';
    g('generate-viewer').style.display = 'block';
    g('generate-downloads').style.display = 'flex';
    g('generate-summary').style.display = 'block';
    const src = `${event.result_url}?t=${Date.now()}`;
    // No add-pane inside the embedded preview -- a Compare-capable iframe crammed into this
    // sidebar-sized panel makes every pane too small to see once a second model is added.
    // Full multi-model comparison belongs in the dedicated Compare tab, which gets the whole
    // screen.
    g('generate-frame').src = `/viewer/index.html?a=${encodeURIComponent(src)}&la=Generated&restricted=1`;
    g('generate-glb').href = event.result_url;
    const savedNote = g('generate-saved-note');
    if (gen.outputDir) {
      savedNote.style.display = 'block';
      savedNote.textContent = `Already saved to ${gen.outputDir}/ — these buttons are just an extra browser-download option.`;
    }
    g('generate-manifest').hidden = !event.manifest_url;
    if (event.manifest_url) {
      g('generate-manifest').href = event.manifest_url;
      fetch(event.manifest_url).then((response) => response.json()).then((manifest) => {
        // Manifest shape differs per backend (TRELLIS: run_stages_1_3/to_glb; Hunyuan:
        // shape/remesh/paint; SF3D writes none at all) -- render whatever stage timings exist
        // generically rather than hardcoding one backend's field names.
        const timings = manifest.timings_seconds || {};
        const parts = Object.entries(timings)
          .filter(([name]) => name !== 'total')
          .map(([name, value]) => `${name} ${formatDuration(value)}`);
        g('generate-summary').textContent = `${manifest.backend || currentBackend()} manifest · ` +
          `total ${formatDuration(timings.total)}` + (parts.length ? ` · ${parts.join(' · ')}` : '');
      }).catch(() => {});
    } else {
      g('generate-summary').textContent = '';
    }
  } else if (phase === 'error') {
    g('generate-status').textContent = event.message || 'Generation failed';
  }
}
function stopStatusPoll() { if (gen.pollTimer) { clearInterval(gen.pollTimer); gen.pollTimer = null; } }
// Fallback for when the SSE stream (source.onerror, below) drops and EventSource's own
// auto-reconnect doesn't recover in time -- e.g. a sub-minute SF3D run that finished
// server-side while the browser's connection was down, with no other way to find out.
// Polls the same single-shot snapshot the stream itself is built from, so a job that
// already finished is discovered on the very next tick instead of leaving the progress
// bar stuck indefinitely.
function startStatusPoll(jobId) {
  if (gen.pollTimer) return; // already polling
  gen.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/generate/${jobId}/status`);
      if (!res.ok) return; // server restarted and lost this job; keep waiting/reconnecting
      const data = await res.json();
      if (data.last_event) applyGenerateProgress(data.last_event);
      if (data.status === 'done' || data.status === 'error' || data.status === 'cancelled') {
        gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
        stopGenerateStream();
      }
    } catch (e) { /* still offline; next tick will retry */ }
  }, 4000);
}
function stopGenerateStream() {
  if (gen.source) { gen.source.close(); gen.source = null; }
  stopStatusPoll();
}
function startGenerateStream(jobId) {
  stopGenerateStream();
  const source = new EventSource(`/api/generate/${jobId}/events`); gen.source = source;
  source.onmessage = (message) => {
    const event = JSON.parse(message.data); applyGenerateProgress(event);
    if (event.phase === 'done' || event.phase === 'error') {
      gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
      source.close();
    }
  };
  source.onopen = () => {
    stopStatusPoll();
    if (gen.running) g('generate-status').textContent = 'Generating…';
  };
  source.onerror = () => {
    if (gen.running) {
      g('generate-status').textContent = 'Progress connection lost; reconnecting…';
      startStatusPoll(jobId);
    }
  };
}
g('generate-submit').onclick = async () => {
  if (!gen.file || gen.running) return;
  gen.running = true; resetStageRows();
  g('generate-progress-box').style.display = 'block'; g('generate-viewer').style.display = 'none';
  g('generate-downloads').style.display = 'none'; g('generate-summary').style.display = 'none';
  g('generate-duration').style.display = 'none';
  g('generate-status').textContent = 'Submitting job…';
  g('generate-cancel').style.display = 'block'; updateGenerateButton();
  const backendId = currentBackend();
  const perBackendSettings = {
    trellis: () => ({
      resolution: g('generate-resolution').value,
      seed: Number(g('generate-seed').value),
      decimation_target: Number(g('generate-decimation').value),
      texture_size: Number(g('generate-texture').value),
      allow_rembg: g('generate-rembg').checked,
    }),
    sf3d: () => ({
      texture_resolution: Number(g('sf3d-texture').value),
      foreground_ratio: Number(g('sf3d-foreground').value),
      remesh: g('sf3d-remesh').value,
      target_vertices: Number(g('sf3d-vertices').value),
    }),
    'hunyuan-mlx': () => ({
      octree_resolution: Number(g('hunyuan-octree').value),
      seed: Number(g('hunyuan-seed').value),
      decimation_target: Number(g('hunyuan-decimation').value),
      paint_seed: Number(g('hunyuan-paint-seed').value),
      paint_res: Number(g('hunyuan-paint-res').value),
      paint_steps: Number(g('hunyuan-paint-steps').value),
      paint_tex: Number(g('hunyuan-paint-tex').value),
    }),
    'hunyuan-mlx-xiong': () => ({
      model: g('xiong-model').value,
      octree_resolution: Number(g('xiong-octree').value),
      seed: Number(g('xiong-seed').value),
      quantize: Number(g('xiong-quantize').value),
      decimation_target: Number(g('xiong-decimation').value),
      paint_seed: Number(g('xiong-paint-seed').value),
      paint_res: Number(g('xiong-paint-res').value),
      paint_steps: Number(g('xiong-paint-steps').value),
      paint_tex: Number(g('xiong-paint-tex').value),
    }),
  };
  const settings = {
    backend: backendId,
    output_dir: g('generate-output-dir').value.trim() || undefined,
    output_name: g('generate-output-name').value.trim() || undefined,
    debug: g('generate-debug').checked,
    ...perBackendSettings[backendId](),
  };
  const body = new FormData(); body.append('image', gen.file); body.append('settings', JSON.stringify(settings));
  try {
    const response = await fetch('/api/generate', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `request failed (${response.status})`);
    gen.jobId = payload.job_id; gen.outputDir = payload.output_dir;
    g('generate-status').textContent = 'Job started. MPS is serialized; this may take a while.';
    startGenerateStream(gen.jobId);
  } catch (error) {
    gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
    g('generate-status').textContent = error.message || String(error);
  }
};
g('generate-cancel').onclick = async () => {
  if (!gen.jobId) return;
  try { await fetch(`/api/generate/${gen.jobId}/cancel`, { method: 'POST' }); }
  catch (_) { /* the SSE stream will surface the process outcome */ }
};

async function refreshSetup() {
  const el = g('setup-checks');
  const backendId = currentBackend();
  try {
    const res = await fetch('/api/setup?backend=' + encodeURIComponent(backendId));
    const s = await res.json();
    setupState.ready = !!s.ready;
    const rows = [];
    const buildLabel = backendMeta[backendId]?.label || backendId;
    if (s.build && s.build.present) {
      const interp = s.build.interpreter ? '<code>' + s.build.interpreter + '</code>' : '';
      rows.push('<div class="setup-check"><span class="ok">✓</span><span>' + buildLabel + ' build</span>' + interp + '</div>');
    } else {
      rows.push('<div class="setup-check"><span class="bad">✗</span><span>' + buildLabel + ' build missing</span></div>');
      if (s.build && s.build.hint) rows.push('<div class="setup-check"><span class="hint">→</span><span class="hint">' + s.build.hint + '</span></div>');
    }
    for (const repo of Object.values(s.weights || {})) {
      rows.push(repo.present
        ? '<div class="setup-check"><span class="ok">✓</span><span>' + repo.label + '</span><code>' + repo.human + ' on disk</code></div>'
        : '<div class="setup-check"><span class="warn">⚠</span><span>' + repo.label + '</span><code>not on disk — first run downloads</code></div>');
    }
    el.innerHTML = rows.join('');
    // Only TRELLIS has an automated bootstrap script (scripts/bootstrap_trellis_space_macos.py);
    // other backends' setup is manual, so the run-setup button only ever applies there.
    const runBox = g('setup-run');
    runBox.hidden = backendId !== 'trellis' || !!(s.build && s.build.present);
    if (!runBox.hidden) g('setup-run-btn').disabled = false;
  } catch (e) {
    el.innerHTML = '<div class="setup-check"><span class="bad">✗</span><span>Setup check failed: ' + e + '</span></div>';
  }
  updateGenerateButton();
}
async function loadBackendMeta() {
  try {
    const res = await fetch('/api/backends');
    const data = await res.json();
    const merged = {};
    for (const b of data.backends) merged[b.id] = b;
    backendMeta = merged;
  } catch (e) { /* keep the trellis-only placeholder; page stays usable */ }
  buildStageRows(currentBackend());
  refreshSetup();
}
g('generate-backend').onchange = () => {
  const backendId = currentBackend();
  for (const block of document.querySelectorAll('.backend-fields')) {
    block.hidden = block.dataset.backend !== backendId;
  }
  buildStageRows(backendId);
  resetStageRows();
  gen.hasAlpha = false;
  g('generate-alpha').hidden = true;
  refreshSetup();
};
g('setup-run-btn').onclick = async () => {
  const btn = g('setup-run-btn'), log = g('setup-log');
  btn.disabled = true; log.style.display = 'block'; log.textContent = 'starting bootstrap…\n';
  try {
    const res = await fetch('/api/setup/run', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) { log.textContent += (data.error || res.statusText) + '\n'; btn.disabled = false; return; }
    const source = new EventSource(data.events_url);
    source.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.message) { log.textContent += ev.message + '\n'; log.scrollTop = log.scrollHeight; }
      if (ev.phase === 'setup_done') { source.close(); refreshSetup(); }
    };
  } catch (e) { log.textContent += e.message + '\n'; btn.disabled = false; }
};
loadBackendMeta();
