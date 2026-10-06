// AssetFurnace studio: Library | one viewer | Steps.
// Steps run here through the classic tabs' own job APIs; prop sheets still split in the classic Props tab.

import { STEPS, stepCount, doneCount, statusText, displayModel, servedUrl, visibleAssets } from './library.js';
import { createStudioViewer } from './viewer.js';
import { readyEngines } from './jobs.js';
import { stepBody, wireStep } from './steps.js';
import { openCreate, runChain } from './create.js';
import { STEP_LABELS } from './plan.js';

const $ = (id) => document.getElementById(id);
const LOGO = './studio/mark.svg';
const CLASSIC_TAB = { model: 'generate', finished: 'finish', rigged: 'animate', animated: 'animate', picture: 'generate-image' };

let base = '/output/';
let assets = [];
let filter = 'all', query = '', showHidden = false, limit = 30;
let current = null;
let engines = [], presets = [], recipes = null, busy = false, playNext = null, chainRunning = false;

const viewer = createStudioViewer({ stage: $('stage'), logoUrl: LOGO, onClipChange: (clip) => setDownload(clip.file) });

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (relative) => servedUrl(base, relative);
const fileName = (relative) => relative?.split('/').pop() ?? '';

function renderLibrary() {
  const { rows: list, more, hiddenCount } = visibleAssets(assets, { filter, query, showHidden, limit });
  $('assetList').innerHTML = (list.length ? list.map((asset) => {
    const pips = Array.from({ length: stepCount(asset) }, (_, i) => `<i class="${i < doneCount(asset) ? 'done' : ''}"></i>`).join('');
    const thumb = asset.picture ? `<img src="${url(asset.picture)}" alt="" loading="lazy">` : `<img class="mark" src="${LOGO}" alt="">`;
    return `<li><button class="asset${asset.hidden ? ' is-hidden' : ''}" data-id="${escape(asset.id)}" aria-current="${asset === current}">
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
    const index = Math.max(0, asset.clips.findIndex((c) => c.name === playNext));
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
  const asset = assets.find((a) => a.id === button.dataset.id);
  if (asset && asset !== current) show(asset);
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
  if (chainRunning) { $('chain').scrollIntoView({ behavior: 'smooth' }); return; }
  openCreate({ stage: $('stage'), recipes, engines, presets, onStart: (plan) => startChain(plan) });
});

async function startChain(plan) {
  chainRunning = true;
  ctx.onBusy(true);
  const box = $('chain');
  let cancel = null, startedAt = Date.now(), timer = null;
  const title = plan.description || plan.file?.name || 'New asset';
  const draw = (state = {}) => {
    box.hidden = false;
    box.innerHTML = `<header><span class="meta">MAKING</span><b>${escape(title)}</b></header>
      <ol class="chain-steps">${plan.steps.map((s) => `<li class="${state.done?.includes(s) ? 'done' : s === state.current ? (state.failed ? 'bad' : 'cur') : ''}"><b>${state.done?.includes(s) ? '✓' : s === state.current ? (state.failed ? '■' : '●') : '○'}</b>${STEP_LABELS[s]}</li>`).join('')}</ol>
      <div class="bigbar"><span></span></div><p class="job-msg"></p><pre class="log"></pre>
      <div class="row end"><button class="ghost" data-x>Cancel</button></div>`;
    box.querySelector('[data-x]').onclick = (e) => { e.target.disabled = true; cancel?.(); };
  };
  const state = { done: [], current: null, failed: false };
  const tick = () => {
    const s = Math.floor((Date.now() - startedAt) / 1000);
    const msg = box.querySelector('.job-msg');
    if (msg && !state.failed) msg.dataset.time = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  timer = setInterval(tick, 1000);
  const ui = {
    begin: () => draw(state),
    step: (step) => { state.current = step; startedAt = Date.now(); draw(state); },
    cancelWith: (fn) => { cancel = fn; },
    progress: ({ event, percent, log, stalled }) => {
      const bar = box.querySelector('.bigbar span'), msg = box.querySelector('.job-msg');
      if (percent != null) { bar.classList.remove('busy'); bar.style.width = `${percent}%`; } else bar.classList.add('busy');
      msg.textContent = `${event.message || event.phase || 'Working…'}${msg.dataset.time ? ` · ${msg.dataset.time}` : ''}`;
      box.querySelector('.log').textContent = (log || '').split('\n').slice(-5).join('\n');
      box.classList.toggle('stalled', !!stalled);
    },
    done: (step) => { state.done.push(step); },
    fail: (step, why, log) => {
      state.failed = true; draw(state);
      box.querySelector('.job-msg').textContent = why;
      box.querySelector('.log').textContent = (log || '').split('\n').slice(-12).join('\n');
      const row = box.querySelector('.row');
      row.innerHTML = '<button class="ghost" data-copy>Copy details</button><button class="ghost" data-close>Close</button>';
      row.querySelector('[data-copy]').onclick = () => navigator.clipboard?.writeText(`${why}\n\n${log || ''}`);
      row.querySelector('[data-close]').onclick = () => { box.hidden = true; };
    },
    finish: () => {
      state.current = null; draw(state);
      box.querySelector('.job-msg').textContent = 'All done.';
      box.querySelector('.bigbar span').style.width = '100%';
      box.querySelector('.log').remove();
      box.querySelector('.row').innerHTML = '<button class="ghost" data-close>Close</button>';
      box.querySelector('[data-close]').onclick = () => { box.hidden = true; };
    },
  };
  const reload = async (match) => {
    const keep = assets.find(match)?.id;
    await loadLibrary(keep);
    const found = assets.find(match);
    if (found && found !== current) show(found);
  };
  try {
    await runChain(plan, { ui, ctx: { url, reload } });
  } finally {
    clearInterval(timer);
    chainRunning = false;
    ctx.onBusy(false);
  }
}

loadTools().then(() => loadLibrary());
