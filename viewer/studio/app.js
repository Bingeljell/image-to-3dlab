// AssetFurnace studio: Library | one viewer | Steps.
// Phase 1: reads the Library and shows any asset. Steps run from the classic tabs for now.

import { STEPS, stepCount, doneCount, statusText, filterAssets, displayModel, servedUrl } from './library.js';
import { createStudioViewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const LOGO = './studio/mark.svg';
const CLASSIC_TAB = { model: 'generate', finished: 'finish', rigged: 'animate', animated: 'animate', picture: 'generate-image' };

let base = '/output/';
let assets = [];
let filter = 'all';
let current = null;

const viewer = createStudioViewer({ stage: $('stage'), logoUrl: LOGO });

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (relative) => servedUrl(base, relative);
const fileName = (relative) => relative?.split('/').pop() ?? '';

function renderLibrary() {
  const list = filterAssets(assets, filter);
  $('assetList').innerHTML = list.length ? list.map((asset) => {
    const pips = Array.from({ length: stepCount(asset) }, (_, i) => `<i class="${i < doneCount(asset) ? 'done' : ''}"></i>`).join('');
    const thumb = asset.picture ? `<img src="${url(asset.picture)}" alt="" loading="lazy">` : `<img class="mark" src="${LOGO}" alt="">`;
    return `<li><button class="asset" data-id="${escape(asset.id)}" aria-current="${asset === current}">
      <span class="thumb">${thumb}</span>
      <span style="min-width:0"><span class="nm">${escape(asset.name)}</span><span class="st"><span class="pips" aria-hidden="true">${pips}</span>${escape(statusText(asset))}</span></span>
    </button></li>`;
  }).join('') : `<li class="empty-lib">${assets.length ? 'Nothing here with this filter.' : 'No assets yet. Make one in the classic view, and it shows up here.'}</li>`;
  $('libFoot').textContent = `${assets.length} asset${assets.length === 1 ? '' : 's'} · saved in output/ on this machine`;
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
    const body = state === 'now'
      ? `<div class="body">This step runs in the classic view for now. <a href="./index.html#${CLASSIC_TAB[step.id]}">Open ${step.label}</a></div>` : '';
    return `<li class="step ${state}"><header><span class="dot">${dot}</span><span style="min-width:0"><h3>${step.label}</h3>${state === 'done' ? `<div class="sum">${escape(sum[step.id])}</div>` : ''}</span></header>${body}</li>`;
  }).join('');
  const file = displayModel(asset);
  const download = $('download');
  download.hidden = !file;
  if (file) { download.href = url(file); download.setAttribute('download', fileName(file)); }
}

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
  if (asset.kind === 'character' && asset.clips.length) {
    viewer.setClips(asset.clips.map((c) => ({ name: c.name.replace(/_/g, ' '), url: url(c.file) })), { restUrl: url(asset.rigged) });
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
  if (!button) return;
  const asset = assets.find((a) => a.id === button.dataset.id);
  if (asset && asset !== current) show(asset);
});

document.querySelectorAll('.seg [data-filter]').forEach((button) => {
  button.onclick = () => {
    filter = button.dataset.filter;
    document.querySelectorAll('.seg [data-filter]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    renderLibrary();
  };
});

async function loadLibrary() {
  try {
    const response = await fetch('/api/assets');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    base = payload.base;
    assets = payload.assets;
    show(assets[0] ?? null);
  } catch (error) {
    $('libFoot').textContent = `The Library didn't load: ${error.message}`;
    viewer.showEmpty('The Library did not load. Is the viewer server running?');
  }
}

loadLibrary();
