// AssetFurnace studio: Library | one viewer | Steps.
// Steps run here through the classic tabs' own job APIs; prop sheets still split in the classic Props tab.

import { STEPS, stepCount, doneCount, statusText, displayModel, servedUrl, visibleAssets } from './library.js';
import { createStudioViewer } from './viewer.js';
import { readyEngines, readableLog } from './jobs.js';
import { stepBody, wireStep } from './steps.js';
import { openCreate, runChain } from './create.js';
import { openActivity, openAbout } from './pages.js';
import { STEP_LABELS } from './plan.js';

const $ = (id) => document.getElementById(id);
const LOGO = './studio/mark.svg';
const CLASSIC_TAB = { model: 'generate', finished: 'finish', rigged: 'animate', animated: 'animate', picture: 'generate-image' };

let base = '/output/';
let assets = [];
let filter = 'all', query = '', showHidden = false, limit = 30;
let current = null;
let engines = [], presets = [], recipes = null, busy = false, playNext = null;
let making = null, viewingMaking = false;  // the asset Create is making right now

const viewer = createStudioViewer({ stage: $('stage'), logoUrl: LOGO, onClipChange: (clip) => setDownload(clip.file) });

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (relative) => servedUrl(base, relative);
const fileName = (relative) => relative?.split('/').pop() ?? '';

function renderLibrary() {
  const { rows: list, more, hiddenCount } = visibleAssets(assets, { filter, query, showHidden, limit });
  $('assetList').innerHTML = makingRow() + (list.length ? list.map((asset) => {
    const pips = Array.from({ length: stepCount(asset) }, (_, i) => `<i class="${i < doneCount(asset) ? 'done' : ''}"></i>`).join('');
    const thumb = asset.picture ? `<img src="${url(asset.picture)}" alt="" loading="lazy">` : `<img class="mark" src="${LOGO}" alt="">`;
    return `<li><button class="asset${asset.hidden ? ' is-hidden' : ''}" data-id="${escape(asset.id)}" aria-current="${!viewingMaking && asset === current}">
      <span class="thumb">${thumb}</span>
      <span style="min-width:0"><span class="nm">${escape(asset.name)}</span><span class="st"><span class="pips" aria-hidden="true">${pips}</span>${escape(statusText(asset))}</span></span>
    </button></li>`;
  }).join('') : `<li class="empty-lib">${assets.length ? 'Nothing matches.' : 'No assets yet. Make one in the classic view, and it shows up here.'}</li>`)
    + (more ? `<li><button class="ghost more" id="showMore">Show ${Math.min(more, 30)} more of ${more}</button></li>` : '');
  const shown = assets.filter((a) => !a.hidden).length;
  $('libFoot').textContent = `${shown} asset${shown === 1 ? '' : 's'} · saved in output/ on this machine`;
  document.querySelector('.hidden-toggle').hidden = !hiddenCount && !showHidden;
  $('hiddenLabel').textContent = `Show hidden (${assets.filter((a) => a.hidden).length})`;
  $('showMore')?.addEventListener('click', () => { limit += 30; renderLibrary(); });
}

function engineOf(path) {
  const run = path?.split('/').slice(-2, -1)[0] || '';
  const match = run.match(/__(pixal3d|trellis|hunyuan[a-z-]*|sf3d)__/);
  return match ? { pixal3d: 'Pixal3D', trellis: 'TRELLIS.2', sf3d: 'Stable Fast 3D' }[match[1]] || 'Hunyuan3D' : 'made outside the app';
}

function summaries(asset) {
  const faces = fileName(asset.finished).match(/_(\d+k)\.glb$/)?.[1];
  return {
    picture: asset.picture ? 'Done' : 'No picture kept (model uploaded or older run)',
    model: asset.model ? engineOf(asset.model) : asset.finished ? 'Made outside the app' : '',
    finished: asset.kind === 'prop set' ? `${asset.props.length} props cut out, each with detail levels` : faces ? `${faces.replace('k', ',000')} triangles` : 'Done',
    rigged: 'Auto-rig',
    animated: asset.clips.map((c) => c.name.replace(/_/g, ' ')).join(', '),
  };
}

function renderSteps(asset) {
  $('sideKind').textContent = { 'prop set': 'Prop set', character: 'Character', model: '3D model', picture: 'Picture' }[asset.kind] || '';
  const done = doneCount(asset), total = stepCount(asset), sum = summaries(asset);
  $('steps').innerHTML = STEPS.map((step, i) => {
    if (i >= total) return `<li class="step na"><header><span class="dot">–</span><span><h3>${step.label}</h3><div class="sum">Not needed for props</div></span></header></li>`;
    const state = i < done ? 'done' : i === done ? 'now' : 'later';
    const dot = state === 'done' ? '✓' : i + 1;
    const runsHere = !(asset.kind === 'prop set' && step.id === 'finished');
    const showBody = state === 'now' || (step.id === 'animated' && state === 'done');
    const body = !showBody ? '' : runsHere
      ? `<div class="body">${stepBody(step.id, asset, ctx)}</div>`
      : `<div class="body"><p class="hint">Splitting a prop sheet runs in the classic view for now.</p><a href="./index.html#${CLASSIC_TAB[step.id]}">Open Props</a></div>`;
    return `<li class="step ${state}" data-step="${step.id}"><header><span class="dot">${dot}</span><span style="min-width:0"><h3>${step.label}</h3>${state === 'done' ? `<div class="sum">${escape(sum[step.id])}</div>` : ''}</span></header>${body}</li>`;
  }).join('');
  $('steps').querySelectorAll('.step').forEach((card) => {
    if (card.querySelector('.body [data-run]')) wireStep(card, card.dataset.step, asset, ctx);
  });
  setDownload(displayModel(asset));
}

function setDownload(file) {
  const download = $('download');
  download.hidden = !file;
  if (file) { download.href = url(file); download.setAttribute('download', fileName(file)); }
}

// what the step actions need from the page
const ctx = {
  get engines() { return engines; },
  get presets() { return presets; },
  url: (relative) => url(relative),
  asset: () => current,
  onBusy: (on) => { busy = on; document.body.classList.toggle('busy', on); },
  // after a new move, open the asset on that move
  onDone: async (stepId, { preset } = {}) => { playNext = preset; await loadLibrary(current?.id); },
};

function show(asset) {
  current = asset;
  viewingMaking = false;
  $('progress').hidden = true;
  renderLibrary();
  $('vTitle').textContent = asset?.name ?? '';
  $('vMeta').textContent = asset ? statusText(asset) : '';
  if (!asset) {
    viewer.showEmpty('Nothing to show yet. Pick an asset on the left.');
    $('steps').innerHTML = '';
    return;
  }
  renderSteps(asset);
  renderHideButton();
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
    else if (asset.picture) viewer.showImage(url(asset.picture), 'A picture so far. Make it 3D from the Steps panel.');
    else viewer.showEmpty('Nothing to show for this asset yet.');
  }
}

$('assetList').addEventListener('click', (event) => {
  const button = event.target.closest('.asset');
  if (!button) return;  // the "Show more" row has its own handler
  if (button.dataset.making != null) { showMaking(); return; }
  const asset = assets.find((a) => a.id === button.dataset.id);
  if (asset && (asset !== current || viewingMaking)) show(asset);
});

$('search').addEventListener('input', (event) => { query = event.target.value; limit = 30; renderLibrary(); });
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

function renderHideButton() {
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

async function loadLibrary(keepId = null) {
  try {
    const response = await fetch('/api/assets');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    base = payload.base;
    assets = payload.assets;
    show(assets.find((a) => a.id === keepId) ?? assets[0] ?? null);
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
  const [catalog, animate] = await Promise.all([
    fetch('/api/catalog').then((r) => r.json()).catch(() => null),
    fetch('/api/animate/models').then((r) => r.json()).catch(() => null),
  ]);
  engines = readyEngines(catalog);
  presets = animate?.installed ? animate.presets : [];
  recipes = await fetch('/api/recipes').then((r) => r.json()).catch(() => null);
}

// ------------------------------------------------------------------ Create
$('createBtn').addEventListener('click', () => {
  if (making && !making.failed && !making.finished) { showMaking(); return; }
  openCreate({ stage: $('stage'), recipes, engines, presets, onStart: (plan) => startChain(plan) });
});

const clock = (ms) => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

function makingRow() {
  if (!making) return '';
  const n = making.plan.steps.length, done = making.done.length;
  const pips = making.plan.steps.map((s) => `<i class="${making.done.includes(s) ? 'done' : s === making.current ? (making.failed ? 'stop' : 'run') : ''}"></i>`).join('');
  const thumb = making.made.picture ? `<img src="${url(making.made.picture)}" alt="">` : `<img class="mark glow" src="${LOGO}" alt="">`;
  const status = making.failed ? `Stopped at ${STEP_LABELS[making.current]}` : making.finished ? 'Done' : `${STEP_LABELS[making.current] || 'Starting'}… step ${Math.min(done + 1, n)} of ${n}`;
  return `<li><button class="asset making" data-making aria-current="${viewingMaking}">
    <span class="thumb">${thumb}</span>
    <span style="min-width:0"><span class="nm">${escape(making.title)}</span><span class="st"><span class="pips" aria-hidden="true">${pips}</span>${escape(status)}</span></span>
  </button></li>`;
}

/** The viewer, Steps and Library while Create is making something. */
function showMaking() {
  if (!making) return;
  viewingMaking = true;
  renderLibrary();
  $('vTitle').textContent = making.title;
  $('vMeta').textContent = making.failed ? 'Stopped' : making.finished ? 'Done' : 'Making…';
  $('hideBtn').hidden = true;
  $('download').hidden = true;
  $('sideKind').textContent = { character: 'Character', prop: 'Prop', set: 'Prop set', picture: 'Picture' }[making.plan.want] || '';
  $('steps').innerHTML = making.plan.steps.map((step, i) => {
    const state = making.done.includes(step) ? 'done' : step === making.current ? 'now' : 'later';
    const dot = state === 'done' ? '✓' : making.failed && step === making.current ? '■' : i + 1;
    const sum = state === 'done' ? 'Done' : state === 'now' ? (making.failed ? 'Stopped. The reason is in the viewer.' : 'Running now') : 'Waiting';
    return `<li class="step ${state}"><header><span class="dot">${dot}</span><span><h3>${STEP_LABELS[step]}</h3><div class="sum">${sum}</div></span></header></li>`;
  }).join('');
  // show the newest result while the next step runs
  const m = making.made;
  viewer.setClips([]);
  const shown = m.rigged || m.finished || m.model;
  if (shown && making.shown !== shown) { viewer.load(url(shown), { label: making.title }); making.shown = shown; }
  else if (!shown && m.picture && making.shown !== m.picture) { viewer.showImage(url(m.picture), ''); making.shown = m.picture; }
  else if (!shown && !m.picture && making.shown !== 'empty') { viewer.showEmpty(''); making.shown = 'empty'; }
  drawProgress();
}

function drawProgress() {
  const box = $('progress');
  if (!making || !viewingMaking) { box.hidden = true; return; }
  const compact = !!(making.made.picture || making.made.model) && !making.failed && !making.finished;
  box.hidden = false;
  box.className = `progress${compact ? ' compact' : ''}${making.stalled ? ' stalled' : ''}${making.failed ? ' failed' : ''}`;
  const list = making.plan.steps.map((s) => `<li class="${making.done.includes(s) ? 'done' : s === making.current ? (making.failed ? 'bad' : 'cur') : ''}"><b>${making.done.includes(s) ? '✓' : s === making.current ? (making.failed ? '■' : '●') : '○'}</b>${STEP_LABELS[s]}</li>`).join('');
  const width = making.finished ? 100 : making.percent ?? 0;
  const message = making.failed ? making.why : making.finished ? 'All done. It is in your Library.'
    : `${making.message || 'Starting…'} · ${clock(Date.now() - making.stepStarted)}`;
  const stall = making.stalled && !making.failed ? '<p class="stallnote">No news from this step for a few minutes. It may be stuck: keep waiting, or cancel.</p>' : '';
  const buttons = making.failed ? '<button class="ghost" data-copy>Copy details</button><button class="ghost" data-close>Close</button>'
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
    done: (step, made) => { making.done.push(step); making.made = made; renderLibrary(); if (viewingMaking) showMaking(); },
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
async function resumeRunning() {
  try {
    const { running } = await (await fetch('/api/chains')).json();
    if (!running || making) return;
    startChain({ steps: running.steps, description: running.title, firstMove: running.first_move, want: running.want }, running.id);
  } catch { /* an older server without runs: nothing to resume */ }
}

$('activityBtn').addEventListener('click', () => openActivity($('stage')));
$('aboutBtn').addEventListener('click', () => openAbout($('stage')));

loadTools().then(() => loadLibrary()).then(() => resumeRunning());
