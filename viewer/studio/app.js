// AssetFurnace studio: Library | one viewer | Steps.
// Steps run here through the server's job APIs, prop-sheet splitting included.

import { STEPS, stepCount, stepDone, stepState, statusText, displayModel, stepFile, canViewStep, servedUrl, visibleFamilies, madeVersion, engineOf, engineChoices, openingView, isModelFile, duration, canViewPictures, sizeText, deletionWords, linkedAssetId, statsRows, canReveal } from './library.js';
import { createStudioViewer } from './viewer.js';
import { readyEngines, readableLog } from './jobs.js';
import { stepBody, wireStep } from './steps.js';
import { openCreate, runChain } from './create.js';
import { openActivity, openAbout, openGuide, openReport, closeSheet } from './pages.js';
import { STEP_LABELS } from './plan.js';

const $ = (id) => document.getElementById(id);
const LOGO = './studio/mark.svg';

let base = '/output/';
let assets = [];
let filter = 'all', query = '', showHidden = false, limit = 30;
let stageFilter = 'all', engineFilter = 'all', starredOnly = false, sortBy = 'newest';
let current = null;
let engines = [], presets = [], recipes = null, frontBack = null, busy = false, playNext = null;
let blenderReady = true;  // Finish and Split need Blender; the Finish step says so before anyone clicks
let making = null, viewingMaking = false;  // the asset Create is making right now
let viewingPictures = false;  // the viewer shows the asset's pictures instead of its model
let viewingStep = null;       // a step id when the viewer shows that step's own model (the raw one, say)
const openFamilies = new Set();  // families whose versions are listed under their row

const viewer = createStudioViewer({ stage: $('stage'), logoUrl: LOGO, onClipChange: (clip) => setDownload(clip.file) });

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (relative) => servedUrl(base, relative);
const fileName = (relative) => relative?.split('/').pop() ?? '';

function assetButton(asset, { sub = false } = {}) {
  const pips = STEPS.slice(0, stepCount(asset)).map((step) => `<i class="${stepDone(asset, step.id) ? 'done' : ''}"></i>`).join('');
  // a small server-made copy, not the full picture (often a megabyte each)
  const thumb = asset.picture ? `<img src="/api/thumb?path=${encodeURIComponent(asset.picture)}" alt="" loading="lazy">` : `<img class="mark" src="${LOGO}" alt="">`;
  const when = new Date(asset.updated * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  const name = sub ? `v${asset.version} · ${statusText(asset)}` : asset.name;
  return `<button class="asset${sub ? ' sub' : ''}${asset.hidden ? ' is-hidden' : ''}" data-id="${escape(asset.id)}" title="${escape(when)}" aria-current="${!viewingMaking && asset === current}">
      <span class="thumb">${thumb}</span>
      <span style="min-width:0"><span class="nm">${asset.main ? '<b class="star" title="Main version">★</b> ' : ''}${escape(name)}</span><span class="st"><span class="pips" aria-hidden="true">${pips}</span>${sub ? escape(when) : escape(statusText(asset))}</span></span>
    </button>`;
}

function renderEngineChoices() {
  const select = $('engineFilter'), choices = engineChoices(assets);
  if (engineFilter !== 'all' && !choices.includes(engineFilter)) engineFilter = 'all';
  select.innerHTML = `<option value="all">Any engine</option>${choices.map((c) => `<option value="${escape(c)}">${escape(c)}</option>`).join('')}`;
  select.value = engineFilter;
}

function renderLibrary() {
  renderEngineChoices();
  const { rows: list, more, hiddenCount } = visibleFamilies(assets, { filter, query, showHidden, limit, stage: stageFilter, engine: engineFilter, starred: starredOnly, sort: sortBy });
  $('assetList').innerHTML = makingRow() + (list.length ? list.map(({ key, shown, members }) => {
    if (members.length < 2) return `<li>${assetButton(shown)}</li>`;
    const open = openFamilies.has(key) || members.includes(current);
    return `<li class="family">${assetButton(shown)}
      <button type="button" class="vers" data-family="${escape(key)}" aria-expanded="${open}">${members.length} versions <span aria-hidden="true">${open ? '▾' : '▸'}</span></button>
      ${open ? `<ul class="versions">${members.map((m) => `<li>${assetButton(m, { sub: true })}</li>`).join('')}</ul>` : ''}</li>`;
  }).join('') : `<li class="empty-lib">${assets.length ? 'Nothing matches.' : 'No assets yet. Press + Create to make your first.'}</li>`)
    + (more ? `<li><button class="ghost more" id="showMore">Show ${Math.min(more, 30)} more of ${more}</button></li>` : '');
  const shown = assets.filter((a) => !a.hidden).length;
  $('libFoot').textContent = `${shown} asset${shown === 1 ? '' : 's'} · saved in output/ on this machine`;
  document.querySelector('.hidden-toggle').hidden = !hiddenCount && !showHidden;
  $('hiddenLabel').textContent = `Show hidden (${assets.filter((a) => a.hidden).length})`;
  $('showMore')?.addEventListener('click', () => { limit += 30; renderLibrary(); });
}

function summaries(asset) {
  const faces = fileName(asset.finished).match(/_(\d+k)\.glb$/)?.[1];
  return {
    picture: asset.picture ? 'Done' : 'No picture kept (model uploaded or older run)',
    model: asset.model ? engineOf(asset.model) : asset.finished ? 'Made outside the app' : '',
    finished: asset.kind === 'prop set' ? `${asset.props.length} props cut out, each with detail levels` : faces ? `${faces.replace('k', ',000')} triangles` : /_full\.glb$/.test(fileName(asset.finished)) ? 'Full detail · every triangle, for renders' : 'Done',
    rigged: 'Auto-rig',
    animated: asset.clips.map((c) => c.name.replace(/_/g, ' ')).join(', '),
  };
}

function renderSteps(asset) {
  $('sideKind').textContent = { 'prop set': 'Prop set', prop: 'Prop', character: 'Character', model: '3D model', picture: 'Picture' }[asset.kind] || '';
  const total = stepCount(asset), sum = summaries(asset);
  $('steps').innerHTML = STEPS.map((step, i) => {
    if (i >= total) return `<li class="step na"><header><span class="dot">–</span><span><h3>${step.label}</h3><div class="sum">Not needed for props</div></span></header></li>`;
    const state = stepState(asset, step.id);
    const dot = state === 'done' ? '✓' : i + 1;
    // a finished model can be finished again at another detail level; a rigged one gets more moves
    const redo = state === 'done' && ({
      picture: !!asset.prompt, model: !!asset.picture && !!engines.length, finished: asset.kind !== 'prop set',
      rigged: asset.kind !== 'prop set', animated: true,
    }[step.id]);
    const body = state === 'now' || redo ? `<div class="body">${stepBody(step.id, asset, ctx)}</div>` : '';
    return `<li class="step ${state}" data-step="${step.id}"><header><span class="dot">${dot}</span><span style="min-width:0"><h3>${step.label}</h3>${state === 'done' ? `<div class="sum">${escape(sum[step.id])}${stepLink(asset, step.id)}</div>` : ''}</span>${state === 'done' && asset.timings?.[step.id] ? `<span class="time">${duration(asset.timings[step.id])}</span>` : ''}</header>${body}</li>`;
  }).join('');
  $('steps').querySelectorAll('.step').forEach((card) => {
    if (card.querySelector('.body [data-run]')) wireStep(card, card.dataset.step, asset, ctx);
    card.querySelector('[data-newpicture]')?.addEventListener('click', () => openCreateSheet({ text: asset.prompt, want: asset.kind === 'prop' ? 'prop' : 'character' }));
  });
  $('steps').querySelector('[data-pictures]')?.addEventListener('click', () => {
    if (viewingPictures) { show(asset); return; }
    viewingPictures = true;
    viewingStep = null;
    viewer.setClips([]);
    viewer.showImage(url(asset.picture), 'What the model was made from.', asset.back ? url(asset.back) : null);
    renderSteps(asset);
  });
  $('steps').querySelectorAll('[data-viewstep]').forEach((button) => button.addEventListener('click', () => {
    const stepId = button.dataset.viewstep;
    if (viewingStep === stepId) { show(asset); return; }
    const file = stepFile(asset, stepId);
    viewingStep = stepId;
    viewingPictures = false;
    viewer.setFacing(null);
    viewer.setClips([]);
    viewer.load(url(file), { label: `${asset.name} · ${STEPS.find((s) => s.id === stepId)?.label ?? stepId}` });
    renderSteps(asset);
    setDownload(file);
  }));
  setDownload(viewingStep ? stepFile(asset, viewingStep) : displayModel(asset));
}

function stepLink(asset, stepId) {
  if (stepId === 'picture' && canViewPictures(asset)) {
    return ` · <button type="button" class="linkish" data-pictures>${viewingPictures ? 'Show model' : `View picture${asset.back ? 's' : ''}`}</button>`;
  }
  const file = stepFile(asset, stepId);
  const view = canViewStep(asset, stepId)
    ? ` · <button type="button" class="linkish" data-viewstep="${stepId}">${viewingStep === stepId ? 'Show latest' : 'View'}</button>` : '';
  return file ? `${view} · <a class="step-dl" href="${escape(url(file))}" download="${escape(fileName(file))}">Download GLB</a>` : '';
}

function setDownload(file) {
  const download = $('download');
  download.hidden = !file;
  if (file) { download.href = url(file); download.setAttribute('download', fileName(file)); }
  showModelInfo(file);
}

// The stats and path of the file the panel's Download button points at. A newer file wins a
// race: the answer for one the user has already moved on from is dropped.
let infoFor = null;
async function showModelInfo(file) {
  infoFor = file;
  const stats = $('modelStats');
  const actions = $('fileActions');
  $('fileCard').hidden = true;
  if (!file) return;
  const response = await fetch(`/api/model-info?file=${encodeURIComponent(file)}`).catch(() => null);
  const info = response?.ok ? await response.json() : null;
  if (infoFor !== file || !info) return;
  const tips = { Triangles: 'GLB files store triangles only, so there is no quad count.' };
  stats.innerHTML = statsRows(info.stats).map(([label, value]) => {
    const classes = [tips[label] && 'tip-left', label === 'Textures' && 'wide'].filter(Boolean).join(' ');
    return `<div${classes ? ` class="${classes}"` : ''}${tips[label] ? ` data-tip="${escape(tips[label])}"` : ''}><dt>${escape(label)}</dt><dd>${escape(value)}</dd></div>`;
  }).join('');
  stats.hidden = !info.stats;
  $('revealBtn').hidden = !canReveal(location.hostname);
  $('fileCard').hidden = false;
  actions.dataset.file = info.file;
  actions.dataset.path = info.path;
}

$('revealBtn').addEventListener('click', () => {
  fetch('/api/reveal', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file: $('fileActions').dataset.file }) });
});

$('copyPathBtn').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText($('fileActions').dataset.path);
    button.textContent = 'Copied';
  } catch {
    button.textContent = 'Copy failed';
  }
  setTimeout(() => { button.textContent = 'Copy path'; }, 1500);
});

// what the step actions need from the page
const ctx = {
  get engines() { return engines; },
  get presets() { return presets; },
  get blenderReady() { return blenderReady; },
  url: (relative) => url(relative),
  asset: () => current,
  onBusy: (on) => { busy = on; document.body.classList.toggle('busy', on); },
  // after a new move, open the asset on that move
  onDone: async (stepId, { preset } = {}) => {
    playNext = preset;
    const before = assets, keepId = current?.id, family = current?.family || keepId;
    await loadLibrary(null, { pick: false });
    // a redo makes a new version: open that one, not the one it branched from
    show(madeVersion(before, assets, family) ?? assets.find((a) => a.id === keepId) ?? assets[0] ?? null);
  },
};

function show(asset) {
  closeSheet($('stage'));   // Create/Activity/How to use/About cover the viewer; get them out of the way
  current = asset;
  viewingMaking = false;
  viewingPictures = false;
  viewingStep = null;
  $('progress').hidden = true;
  renderLibrary();
  $('vTitle').textContent = asset?.name ?? '';
  $('vTitle').classList.toggle('long', (asset?.name ?? '').length > 60);
  $('vMeta').textContent = asset ? statusText(asset) : '';
  if (!asset) {
    viewer.showEmpty('Nothing to show yet. Pick an asset on the left.');
    $('steps').innerHTML = '';
    return;
  }
  renderSteps(asset);
  renderHideButton();
  // only rigged files are turned: the skeleton is what says which way the feet point
  viewer.setFacing(asset.rigged && (asset.clips.length || displayModel(asset) === asset.rigged) ? asset.facing : null);
  if (asset.kind === 'character' && asset.clips.length) {
    // open on the move just made, else a calm one (idle, then walk), never alphabetically first ("death")
    const prefer = [playNext, 'idle', 'walk'].map((name) => asset.clips.findIndex((c) => c.name === name)).find((i) => i >= 0);
    const index = prefer ?? 0;
    playNext = null;
    viewer.setClips(asset.clips.map((c) => ({ name: c.name.replace(/_/g, ' '), url: url(c.file), file: c.file })), { restUrl: url(asset.rigged), index });
  } else if (asset.kind === 'prop set' && asset.props.length) {
    viewer.setClips(asset.props.map((p) => ({ name: p.name.replace(/[_-]/g, ' '), url: url(p.file), file: p.file })), { still: true });
  } else {
    viewer.setClips([]);
    const model = displayModel(asset);
    if (model) viewer.load(url(model), { label: asset.name });
    else if (asset.picture) {
      const inRun = making && !making.finished && !making.failed && making.made?.picture === asset.picture;
      viewer.showImage(url(asset.picture), inRun ? 'Being made into 3D now. Its progress is above.'
        : 'A picture so far. Make it 3D from the Steps panel.', asset.back ? url(asset.back) : null);
    }
    else viewer.showEmpty('Nothing to show for this asset yet.');
  }
}

$('assetList').addEventListener('click', (event) => {
  const toggle = event.target.closest('[data-family]');
  if (toggle) {
    const key = toggle.dataset.family;
    if (openFamilies.has(key)) openFamilies.delete(key); else openFamilies.add(key);
    renderLibrary();
    return;
  }
  const button = event.target.closest('.asset');
  if (!button) return;  // the "Show more" row has its own handler
  if (button.dataset.making != null) { showMaking(); return; }
  const asset = assets.find((a) => a.id === button.dataset.id);
  if (asset && (asset !== current || viewingMaking)) show(asset);
});

$('search').addEventListener('input', (event) => { query = event.target.value; limit = 30; renderLibrary(); });
$('stageFilter').addEventListener('change', (event) => { stageFilter = event.target.value; limit = 30; renderLibrary(); });
$('engineFilter').addEventListener('change', (event) => { engineFilter = event.target.value; limit = 30; renderLibrary(); });
$('sortBy').addEventListener('change', (event) => { sortBy = event.target.value; renderLibrary(); });
$('starredOnly').addEventListener('change', (event) => { starredOnly = event.target.checked; limit = 30; renderLibrary(); });
$('showHidden').addEventListener('change', (event) => { showHidden = event.target.checked; renderLibrary(); });
$('hideBtn').addEventListener('click', async () => {
  if (!current) return;
  const hidden = !current.hidden;
  const response = await fetch(`/api/assets/${current.id}/hidden`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hidden }) });
  if (!response.ok) return;
  current.hidden = hidden;
  renderLibrary();
  renderHideButton();
});

$('mainBtn').addEventListener('click', async () => {
  if (!current) return;
  const main = !current.main;
  const response = await fetch(`/api/assets/${current.id}/main`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ main }) });
  if (!response.ok) return;
  await refreshAssets();
  renderHideButton();
});

$('deleteBtn').addEventListener('click', async () => {
  if (!current) return;
  const bar = $('deleteBar');
  const response = await fetch(`/api/assets/${current.id}/delete-plan`);
  const plan = await response.json().catch(() => ({}));
  const close = () => { bar.hidden = true; bar.innerHTML = ''; };
  bar.hidden = false;
  if (!response.ok || plan.refused) {
    bar.innerHTML = `<p>${escape(plan.refused || plan.error || 'This could not be checked.')}</p><div class="row end"><button class="ghost small" data-close>OK</button></div>`;
    bar.querySelector('[data-close]').onclick = close;
    return;
  }
  bar.innerHTML = `<p><b>Delete this version?</b> This removes ${escape(deletionWords(plan))} from your disk and frees ${escape(sizeText(plan.bytes))}. It cannot be undone.</p>
    <div class="row end"><button class="ghost small" data-close>Keep it</button><button class="danger small" data-delete>Delete</button></div>`;
  bar.querySelector('[data-close]').onclick = close;
  bar.querySelector('[data-delete]').onclick = async (event) => {
    event.target.disabled = true;
    const family = current.family;
    const done = await fetch(`/api/assets/${current.id}/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: true }) });
    if (!done.ok) { bar.innerHTML = `<p>${escape((await done.json().catch(() => ({}))).error || 'Could not delete.')}</p><div class="row end"><button class="ghost small" data-close>OK</button></div>`; bar.querySelector('[data-close]').onclick = close; return; }
    close();
    await loadLibrary(null, { pick: false });
    show(assets.find((x) => x.family === family) ?? assets[0] ?? null);
  };
});

function renderHideButton() {
  $('deleteBtn').hidden = !current;
  $('deleteBar').hidden = true;
  const star = $('mainBtn');
  star.hidden = !current || (current.versions || 1) < 2;   // a lone version needs no star
  star.textContent = current?.main ? '★ Main' : '☆ Main';
  star.dataset.tip = current?.main ? 'This is the main version. Click to clear it.' : 'Mark this version as the main one: the one to export and use in your game.';

  const button = $('hideBtn');
  button.hidden = !current;
  button.textContent = current?.hidden ? 'Unhide' : 'Hide';
}

document.querySelectorAll('.seg [data-filter]').forEach((button) => {
  button.onclick = () => {
    filter = button.dataset.filter;
    document.querySelectorAll('.seg [data-filter]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    renderLibrary();
  };
});

async function loadLibrary(keepId = null, { pick = true } = {}) {
  try {
    const response = await fetch('/api/assets');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    base = payload.base;
    assets = payload.assets;
    show(pick ? assets.find((a) => a.id === keepId) ?? assets[0] ?? null : null);
  } catch (error) {
    $('libFoot').textContent = `The Library didn't load: ${error.message}`;
    viewer.showEmpty('The Library did not load. Is the viewer server running?');
  }
}

async function refreshAssets() {
  const response = await fetch('/api/assets');
  const payload = await response.json();
  if (!response.ok) return;
  base = payload.base;
  assets = payload.assets;
  if (current) current = assets.find((a) => a.id === current.id) ?? current;
  renderLibrary();
}

async function loadTools() {
  const [catalog, animate, finishCaps] = await Promise.all([
    fetch('/api/catalog').then((r) => r.json()).catch(() => null),
    fetch('/api/animate/models').then((r) => r.json()).catch(() => null),
    fetch('/api/finish/capabilities').then((r) => r.json()).catch(() => null),
  ]);
  blenderReady = finishCaps ? !!finishCaps.ready : true;   // unknown: let the server say
  engines = readyEngines(catalog);
  frontBack = catalog?.front_back || null;
  presets = animate?.installed ? animate.presets : [];
  recipes = await fetch('/api/recipes').then((r) => r.json()).catch(() => null);
}

// ------------------------------------------------------------------ Create
function openCreateSheet(prefill = null) {
  closeSheet($('stage'));
  openCreate({ stage: $('stage'), recipes, engines, presets, frontBack, prefill, onStart: (plan) => startChain(plan) });
}

$('createBtn').addEventListener('click', () => {
  if (making && !making.failed && !making.finished) { showMaking(); return; }
  openCreateSheet();
});

const clock = (ms) => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

function makingRow() {
  if (!making) return '';
  const n = making.plan.steps.length, done = making.done.length;
  const pips = making.plan.steps.map((s) => `<i class="${making.done.includes(s) ? 'done' : s === making.current && !making.cancelled ? (making.failed ? 'stop' : 'run') : ''}"></i>`).join('');
  const thumb = making.made.picture ? `<img src="${url(making.made.picture)}" alt="">` : `<img class="mark glow" src="${LOGO}" alt="">`;
  const status = making.cancelled ? 'Cancelled' : making.failed ? `Stopped at ${STEP_LABELS[making.current]}` : making.finished ? 'Done' : `${STEP_LABELS[making.current] || 'Starting'}… step ${Math.min(done + 1, n)} of ${n}`;
  return `<li><button class="asset making" data-making aria-current="${viewingMaking}">
    <span class="thumb">${thumb}</span>
    <span style="min-width:0"><span class="nm">${escape(making.title)}</span><span class="st"><span class="pips" aria-hidden="true">${pips}</span>${escape(status)}</span></span>
  </button></li>`;
}

/** The viewer, Steps and Library while Create is making something. */
function showMaking() {
  if (!making) return;
  closeSheet($('stage'));
  viewingMaking = true;
  renderLibrary();
  $('vTitle').textContent = making.title;
  $('vTitle').classList.toggle('long', making.title.length > 60);
  $('vMeta').textContent = making.cancelled ? 'Cancelled' : making.failed ? 'Stopped' : making.finished ? 'Done' : 'Making…';
  $('hideBtn').hidden = true;
  viewer.setFacing(null);  // a run in progress shows its files as they come
  setDownload(null);
  $('sideKind').textContent = { character: 'Character', prop: 'Prop', set: 'Prop set', picture: 'Picture' }[making.plan.want] || '';
  $('steps').innerHTML = making.plan.steps.map((step, i) => {
    const state = making.done.includes(step) ? 'done' : step === making.current ? 'now' : 'later';
    const dot = state === 'done' ? '✓' : making.cancelled && step === making.current ? '–' : making.failed && step === making.current ? '■' : i + 1;
    const sum = state === 'done' ? 'Done' : state === 'now' ? (making.cancelled ? 'Cancelled' : making.failed ? 'Stopped. The reason is in the viewer.' : 'Running now') : making.cancelled ? 'Not run' : 'Waiting';
    const took = state === 'done' ? duration(making.seconds?.[step]) : '';
    return `<li class="step ${state}"><header><span class="dot">${dot}</span><span><h3>${STEP_LABELS[step]}</h3><div class="sum">${sum}</div></span>${took ? `<span class="time">${took}</span>` : ''}</header></li>`;
  }).join('');
  // show the newest result while the next step runs
  const m = making.made;
  viewer.setClips([]);
  const shown = m.rigged || m.finished || m.model;
  if (shown && making.shown !== shown) { viewer.load(url(shown), { label: making.title }); making.shown = shown; }
  else if (!shown && m.picture && making.shown !== `${m.picture}|${m.back || ''}`) {
    viewer.showImage(url(m.picture), '', m.back ? url(m.back) : null);   // front + back side by side
    making.shown = `${m.picture}|${m.back || ''}`;
  }
  else if (!shown && !m.picture && making.shown !== 'empty') { viewer.showEmpty(''); making.shown = 'empty'; }
  drawProgress();
}

function drawProgress() {
  const box = $('progress');
  if (!making || !viewingMaking) { box.hidden = true; return; }
  const compact = !!(making.made.picture || making.made.model) && !making.failed && !making.finished;
  box.hidden = false;
  box.className = `progress${compact ? ' compact' : ''}${making.stalled ? ' stalled' : ''}${making.failed && !making.cancelled ? ' failed' : ''}`;
  const list = making.plan.steps.map((s) => `<li class="${making.done.includes(s) ? 'done' : s === making.current && !making.cancelled ? (making.failed ? 'bad' : 'cur') : ''}"><b>${making.done.includes(s) ? '✓' : s === making.current ? (making.cancelled ? '–' : making.failed ? '■' : '●') : '○'}</b>${STEP_LABELS[s]}</li>`).join('');
  const width = making.finished ? 100 : making.percent ?? 0;
  const message = making.failed ? making.why : making.finished ? 'All done. It is in your Library.'
    : `${making.message || 'Starting…'} · ${clock(Date.now() - making.stepStarted)}`;
  const stall = making.stalled && !making.failed ? '<p class="stallnote">No news from this step for a few minutes. It may be stuck: keep waiting, or cancel.</p>' : '';
  const buttons = making.cancelled ? '<button class="ghost" data-close>Close</button>'
    : making.failed ? '<button class="ghost" data-report>Report this</button><button class="ghost" data-copy>Copy details</button><button class="ghost" data-close>Close</button>'
    : making.finished ? '<button class="ghost" data-close>Close</button>' : '<button class="ghost" data-x>Cancel</button>';
  box.innerHTML = compact
    ? `<div class="pc"><span class="meta">${escape(STEP_LABELS[making.current] || '')}</span><div class="bigbar"><span class="${making.percent == null ? 'busy' : ''}" style="width:${width}%"></span></div><span class="job-msg">${escape(message)}</span>${buttons}</div>${stall}`
    : `<div class="pcard">${making.failed || making.finished ? '' : `<img class="glow" src="${LOGO}" alt="">`}
        <ol class="chain-steps">${list}</ol>
        <div class="bigbar"><span class="${making.percent == null && !making.finished ? 'busy' : ''}" style="width:${width}%"></span></div>
        <p class="job-msg">${escape(message)}</p>${stall}
        ${making.log ? `<pre class="log">${escape(making.log)}</pre>` : ''}
        <div class="row end">${buttons}</div></div>`;
  box.querySelector('[data-x]')?.addEventListener('click', (e) => { e.target.disabled = true; making.cancel?.(); });
  box.querySelector('[data-report]')?.addEventListener('click', () => openReport($('stage')));   // the newest failure is this one
  box.querySelector('[data-copy]')?.addEventListener('click', () => navigator.clipboard?.writeText(`${making.why}\n\n${making.fullLog || ''}`));
  box.querySelector('[data-close]')?.addEventListener('click', () => { const keep = making.result; making = null; show(keep ?? current ?? assets[0] ?? null); });
}

async function startChain(plan, attach = null) {
  making = { plan, title: plan.description || plan.file?.name || 'New asset', done: [], current: null, made: {},
    percent: null, message: '', log: '', stepStarted: Date.now(), failed: false, finished: false, why: '', shown: null };
  ctx.onBusy(true);
  showMaking();
  const timer = setInterval(() => { if (viewingMaking && !making?.failed && !making?.finished) drawProgress(); }, 1000);
  const ui = {
    begin: () => {},
    step: (step) => { Object.assign(making, { current: step, percent: null, message: '', log: '', stepStarted: Date.now(), stalled: false }); renderLibrary(); if (viewingMaking) showMaking(); },
    cancelWith: (fn) => { making.cancel = fn; },
    progress: ({ event, percent, log, stalled }) => {
      Object.assign(making, { percent, message: event.message || event.phase || 'Working…', log: readableLog(log), fullLog: log, stalled });
      if (viewingMaking) drawProgress();
    },
    done: (step, made, seconds) => { making.done.push(step); making.made = made; making.seconds = seconds || {}; renderLibrary(); if (viewingMaking) showMaking(); },
    cancelled: () => { Object.assign(making, { failed: true, cancelled: true, why: 'Cancelled. Everything finished before this step is kept.', log: '', fullLog: '' }); renderLibrary(); if (viewingMaking) showMaking(); },
    fail: (step, why, log) => { Object.assign(making, { failed: true, why, log: readableLog(log, 12), fullLog: log }); renderLibrary(); if (viewingMaking) showMaking(); },
    finish: (mine) => {
      making.finished = true;
      making.result = assets.find(mine) ?? null;
      // land on the finished asset, playing its first move if it has one
      if (viewingMaking && making.result) { playNext = making.plan.firstMove; const result = making.result; making = null; show(result); }
      else renderLibrary();
    },
  };
  try {
    await runChain(plan, { ui, attach, ctx: { url, reload: async () => { await refreshAssets(); } } });
  } finally {
    clearInterval(timer);
    ctx.onBusy(false);
  }
}

// a run keeps going on the server after its tab closes: a reopened studio picks it back up
async function runningChain() {
  try { return (await (await fetch('/api/chains')).json()).running ?? null; }
  catch { return null; }  // an older server without runs: nothing to resume
}

// open on a run still going, else on Create; the Library beside it holds the recent work
async function boot() {
  await loadTools();
  await loadLibrary(null, { pick: false });
  const running = await runningChain();
  if (openingView(running) === 'running') {
    if (!making) startChain({ steps: running.steps, description: running.title, firstMove: running.first_move, want: running.want }, running.id);
  } else if (!showLinked()) openCreateSheet();
}

// an agent's "open in the studio" link: studio.html#asset=<id>
function showLinked() {
  const id = linkedAssetId(location.hash);
  const asset = id && assets.find((a) => a.id === id);
  if (!asset) return false;
  show(asset);
  return true;
}
window.addEventListener('hashchange', async () => {
  if (linkedAssetId(location.hash) && !assets.some((a) => a.id === linkedAssetId(location.hash))) await refreshAssets();
  showLinked();
});

// rename: click the title, type, Enter (Escape keeps the old name). Only the Library's label
// changes; files keep their names.
$('vTitle').title = 'Click to rename';
$('vTitle').addEventListener('click', () => {
  if (!current || viewingMaking || $('vTitle').querySelector('input')) return;
  const asset = current;
  const input = document.createElement('input');
  input.className = 'rename';
  input.value = asset.name;
  input.maxLength = 80;
  input.setAttribute('aria-label', 'Name');
  $('vTitle').replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    $('vTitle').textContent = asset.name;
    if (!save || name === asset.name) return;
    try {
      const response = await fetch(`/api/assets/${asset.id}/name`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
      if (!response.ok) throw new Error((await response.json()).error);
      await refreshAssets();
      if (current?.id === asset.id) $('vTitle').textContent = current.name;
    } catch { $('vTitle').textContent = asset.name; }
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false); });
  input.addEventListener('blur', () => finish(true));
});

// ------------------------------------------------------------------ Open a GLB
// look at any GLB from this computer: view only, nothing is copied into the Library
let openedUrl = null;

function openFile(file) {
  if (!file) return;
  closeSheet($('stage'));
  current = null;
  viewingMaking = false;
  $('progress').hidden = true;
  renderLibrary();
  $('hideBtn').hidden = true;
  setDownload(null);
  $('sideKind').textContent = '';
  $('vTitle').textContent = file.name;
  if (!isModelFile(file.name)) {
    $('vMeta').textContent = '';
    $('steps').innerHTML = '';
    viewer.setClips([]);
    viewer.showEmpty('Only GLB files open here. To make a picture into 3D, use + Create.');
    return;
  }
  $('vMeta').textContent = 'Opened from your computer · not in your Library';
  $('steps').innerHTML = '<li class="hint opened-note">Just looking. To rig, animate or finish this model, use + Create and start from "3D model".</li>';
  if (openedUrl) URL.revokeObjectURL(openedUrl);
  openedUrl = URL.createObjectURL(file);
  viewer.setFacing(null);
  viewer.setClips([]);
  viewer.load(openedUrl, { label: file.name, autoplay: true });
}

$('openBtn').addEventListener('click', () => $('openFile').click());
$('openFile').addEventListener('change', (event) => { openFile(event.target.files[0]); event.target.value = ''; });
// a file dropped anywhere opens in the viewer, instead of the browser downloading it.
// Create's sheet takes its own drops first (and marks them handled).
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => {
  if (event.defaultPrevented) return;
  event.preventDefault();
  openFile(event.dataTransfer?.files?.[0]);
});

$('guideBtn').addEventListener('click', () => openGuide($('stage')));
$('activityBtn').addEventListener('click', () => openActivity($('stage')));
$('aboutBtn').addEventListener('click', () => openAbout($('stage')));
$('reportBtn').addEventListener('click', () => openReport($('stage')));

boot();
