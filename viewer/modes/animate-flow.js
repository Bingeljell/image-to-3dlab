// The Animate tab's left column: pick a model, auto-rig it, play a preset on it, export.
// The player beside it (animate.js) only displays; every file here comes from the server.
import { groupModels } from '../components/model-groups.js';
import { showModel } from './animate.js';

const el = (id) => document.getElementById(id);
const modelSelect = el('anim-model');
const rigButton = el('anim-rig');
const presetSelect = el('anim-preset');
const playButton = el('anim-play-preset');
const spread = el('anim-spread');
const spreadValue = el('anim-spread-value');
const speed = el('anim-speed');
const speedValue = el('anim-speed-value');
const upload = el('anim-upload');
const job = el('anim-job');
const barFill = el('anim-bar-fill');
const jobText = el('anim-job-text');
const cancelButton = el('anim-cancel');
const exportLink = el('anim-export');
const setupNotice = el('anim-setup');
const status = el('animate-status');

const state = { models: [], presets: [], installed: false, running: null, source: null };

const selected = () => state.models.find((m) => m.path === modelSelect.value) || null;
const servedUrl = (model) => `/output/${model.path.split('/').map(encodeURIComponent).join('/')}`;

function fillModels(keep) {
  modelSelect.length = 1;
  for (const group of groupModels(state.models)) {
    const optgroup = document.createElement('optgroup');
    optgroup.label = group.label;
    for (const model of group.items) {
      const option = document.createElement('option');
      option.value = model.path;
      const when = new Date(model.modified * 1000)
        .toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
      option.textContent = `${model.name} · ${when}`;
      optgroup.appendChild(option);
    }
    modelSelect.appendChild(optgroup);
  }
  modelSelect.value = state.models.some((m) => m.path === keep) ? keep : '';
}

function fillPresets() {
  presetSelect.replaceChildren(...state.presets.map((preset) => {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.seconds ? `${preset.label} (${preset.seconds}s)` : preset.label;
    return option;
  }));
}

function refreshButtons() {
  const model = selected();
  const busy = !!state.running;
  const ready = state.installed && !busy;
  rigButton.disabled = !ready || !(state.source || (model && model.kind !== 'rigged'));
  rigButton.textContent = model?.kind === 'rigged' ? 'Already rigged' : 'Auto-rig';
  const rigged = model?.kind === 'rigged';
  presetSelect.disabled = !rigged || busy || !state.presets.length;
  playButton.disabled = !ready || !rigged || !state.presets.length;
  modelSelect.disabled = busy;
}

async function loadList(keep = modelSelect.value) {
  try {
    const response = await fetch('/api/animate/models');
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = await response.json();
    state.models = payload.models;
    state.presets = payload.presets;
    state.installed = payload.installed;
    fillModels(keep);
    fillPresets();
    presetSelect.onchange();
    setupNotice.hidden = state.installed;
    if (!state.installed) {
      setupNotice.innerHTML = 'Auto-rig is not installed yet (~1.6 GB). '
        + '<a href="#" id="anim-go-setup">Set it up on Setup &amp; Status</a>.';
      el('anim-go-setup').onclick = (event) => {
        event.preventDefault();
        document.dispatchEvent(new CustomEvent('viewer:navigate', { detail: { mode: 'setup' } }));
      };
    }
  } catch (error) {
    status.textContent = `Could not list models: ${error.message}`;
  }
  refreshButtons();
}

function showProgress(pct, text) {
  job.hidden = false;
  barFill.style.width = `${pct}%`;
  jobText.textContent = text;
}

function follow(started, onDone) {
  state.running = started.job_id;
  exportLink.hidden = true;
  refreshButtons();
  showProgress(0, 'Starting…');
  const events = new EventSource(started.events_url);
  events.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.phase === 'done') {
      events.close();
      state.running = null;
      job.hidden = true;
      onDone(event);
    } else if (event.phase === 'error') {
      events.close();
      state.running = null;
      job.hidden = true;
      status.textContent = event.message;
      if (event.log_tail) console.warn(event.log_tail);
      refreshButtons();
    } else {
      const seconds = Math.round(event.elapsed_seconds || 0);
      showProgress(event.overall_pct || 0, `${event.message || 'Working'}… ${seconds}s`);
    }
  };
  events.onerror = () => {
    if (!state.running) events.close();
  };
}

async function post(url, init) {
  const response = await fetch(url, { method: 'POST', ...init });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

async function startRig() {
  status.textContent = '';
  const form = new FormData();
  if (state.source) form.append('asset', state.source, state.source.name);
  else form.append('model', modelSelect.value);
  try {
    follow(await post('/api/animate/rig', { body: form }), async (event) => {
      state.source = null;
      status.textContent = 'Rigged. Pick a preset and press Animate.';
      await loadList(event.path);
      showModel(event.result_url, 'rigged model');
      offerExport(event);
    });
  } catch (error) {
    status.textContent = error.message;
    refreshButtons();
  }
}

async function startPreset() {
  status.textContent = '';
  try {
    const body = JSON.stringify({ model: modelSelect.value, preset: presetSelect.value,
                                  arm_spread: Number(spread.value), speed: Number(speed.value) });
    follow(await post('/api/animate/play', { body, headers: { 'Content-Type': 'application/json' } }),
      (event) => {
        status.textContent = 'Done. It is playing on the right; Download keeps it.';
        showModel(event.result_url, presetSelect.selectedOptions[0]?.textContent || 'animation',
          { autoplay: true });
        offerExport(event);
        refreshButtons();
      });
  } catch (error) {
    status.textContent = error.message;
    refreshButtons();
  }
}

function offerExport(event) {
  exportLink.href = event.result_url;
  exportLink.download = event.result_url.split('/').pop();
  exportLink.textContent = `Download ${exportLink.download}`;
  exportLink.hidden = false;
}

modelSelect.onchange = () => {
  state.source = null;
  exportLink.hidden = true;
  const model = selected();
  if (model) showModel(servedUrl(model), model.name);
  status.textContent = model?.kind === 'rigged'
    ? 'Rigged already. Pick a preset.'
    : model ? 'Check it stands in a T-pose, then press Auto-rig.' : '';
  refreshButtons();
};
el('anim-upload-button').onclick = () => upload.click();
upload.onchange = () => {
  const file = upload.files[0];
  upload.value = '';
  if (!file) return;
  state.source = file;
  modelSelect.value = '';
  showModel(URL.createObjectURL(file), file.name);
  status.textContent = `${file.name}: check it stands in a T-pose, then press Auto-rig.`;
  refreshButtons();
};
el('anim-refresh').onclick = () => loadList();
rigButton.onclick = startRig;
playButton.onclick = startPreset;
spread.oninput = () => { spreadValue.textContent = `${spread.value}°`; };
const showSpeed = () => { speedValue.textContent = `${Number(speed.value).toFixed(2)}×`; };
speed.oninput = showSpeed;
// Each preset carries its own default pace (strikes ship faster); picking one resets to it.
presetSelect.onchange = () => {
  const preset = state.presets.find((p) => p.id === presetSelect.value);
  speed.value = String(preset?.speed ?? 1);
  showSpeed();
};
cancelButton.onclick = () => {
  if (state.running) post(`/api/animate/${state.running}/cancel`).catch(() => {});
};

document.addEventListener('viewer:modechange', (event) => {
  if (event.detail?.mode === 'animate' && !state.running) loadList();
});
