// --- Props mode ----------------------------------------------------------------------
// Split a prop-sheet GLB into separate props and bake each one's LODs. Drives
// POST /api/props (viewer/props_api.py), a sibling of the Finish job, so the
// SSE-with-polling-fallback shape here is the one modes/finish.js uses.
//
// The results come from the run directory, not from this page's memory: the server
// describes a run from disk, and the run list below re-opens any of them.
import { JobProgressPanel, formatDuration } from '../components/job-progress.js';
import { earlyCancel } from '../core/early-cancel.js';

const f = (id) => document.getElementById(id);

const progress = new JobProgressPanel({
  stages: f('props-stages'),
  bar: f('props-overall-bar'),
  label: f('props-overall-label'),
  eta: f('props-overall-eta'),
});
const SPLIT_ONLY = { stages: ['split'], stage_labels: { split: 'Split the sheet' } };

const state = { asset: null, running: false, source: null, poll: null, tools: {}, shown: null, cancel: null };

const kb = (bytes) => (bytes == null ? '' : bytes >= 1048576
  ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`);

function updateSubmit() {
  const source = state.asset || f('props-generated').value;
  f('props-submit').disabled = !source || state.running || !state.tools.blender;
}

function setRunning(running) {
  state.running = running;
  f('props-cancel').hidden = !running;
  f('props-runs-refresh').disabled = running;
  f('props-turn').disabled = running;
  updateSubmit();
}

function stopStreams() {
  if (state.source) { state.source.close(); state.source = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

function endJob(message) {
  stopStreams();
  setRunning(false);
  f('props-status').textContent = message;
  loadRuns();
}

function applyEvent(event) {
  if (!event) return;
  // The prop rows only exist once the split has said which props there are.
  if (event.stages) progress.configure({ stages: event.stages, stage_labels: event.stage_labels });
  progress.apply(event);
  if (event.message) f('props-status').textContent = event.message;

  // Only the server's own terminal event carries the run, so only it ends the watch.
  if (event.phase === 'done' && event.run) {
    endJob(`${event.message} in ${formatDuration(event.elapsed_seconds)}`);
    showRun(event.run);
  } else if (event.phase === 'error' || event.phase === 'cancelled') {
    endJob(event.message || 'Failed');
    if (event.log_tail) console.warn('[props]', event.log_tail);
  }
}

function startPolling(jobId) {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    try {
      const response = await fetch(`/api/props/${jobId}/status`);
      if (response.status === 404) {
        // Jobs live in the server's memory; after a restart it no longer knows this one,
        // and polling on would leave the tab stuck "running" until a reload.
        endJob('The viewer restarted and lost track of this job. What it finished is under Runs on disk.');
        return;
      }
      if (!response.ok) return;
      applyEvent((await response.json()).last_event);
    } catch (_) { /* the next poll or an SSE reconnect recovers */ }
  }, 2000);
}

async function cancelJob(jobId) {
  try {
    await fetch(`/api/props/${jobId}/cancel`, { method: 'POST' });
  } catch (_) { /* the job's own events say how it ended */ }
}

function watch(payload) {
  state.source = new EventSource(payload.events_url);
  state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
  state.source.onerror = () => startPolling(payload.job_id);
  state.cancel.started(payload.job_id);
}

// Watch the job a POST started; `failure` says what could not be done if it did not.
async function start(request, failure) {
  try {
    const response = await request;
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    watch(payload);
  } catch (error) {
    setRunning(false);
    f('props-status').textContent = `${failure}: ${error.message}`;
  }
}

function begin(message) {
  setRunning(true);
  // Wired before the request goes out, not once it returns: a click during an upload
  // did nothing, or cancelled the job before this one.
  state.cancel = earlyCancel(cancelJob);
  f('props-cancel').onclick = () => {
    if (state.cancel.click() === 'queued') {
      f('props-status').textContent = 'Cancelling as soon as the job starts…';
    }
  };
  progress.configure(SPLIT_ONLY);
  progress.reset();
  f('props-empty').hidden = true;
  f('props-progress-box').hidden = false;
  f('props-status').textContent = message;
}

// --- Results ---------------------------------------------------------------------------
// A chip per prop over one detail panel: the 3D view, the turn and the downloads sit next
// to what was clicked, instead of under a list of every prop.

function preview(prop) {
  if (!prop.preview_url) {
    f('props-frame').src = 'about:blank';   // not the last prop's model under this one's name
    return;
  }
  // The Compare page, one model and no menu, as the Generate tab embeds it.
  const src = `${prop.preview_url}?t=${Date.now()}`;
  f('props-frame').src =
    `/viewer/index.html?a=${encodeURIComponent(src)}&la=${encodeURIComponent(prop.name)}&restricted=1`;
}

function downloadLink(url, bytes) {
  if (!url) return '—';
  const link = document.createElement('a');
  link.href = url;
  link.download = '';
  link.textContent = `⬇ ${kb(bytes)}`;
  return link;
}

function showProp(run, prop) {
  state.shown = prop.name;
  for (const chip of f('props-list').children) {
    chip.setAttribute('aria-pressed', String(chip.dataset.name === prop.name));
  }
  f('props-detail-name').textContent = prop.name;
  const size = prop.size ? prop.size.map((s) => s.toFixed(2)).join(' × ') : '';
  const turned = prop.extra_turn_degrees ? ` · turned ${prop.extra_turn_degrees}° by hand` : '';
  f('props-detail-meta').textContent =
    `${(prop.faces || 0).toLocaleString()} faces from the sheet` +
    `${size ? ` · ${size}` : ''}${turned}`;

  // A turn this close to 45° is a coin toss between the front and the side: the test
  // sheet's chest came back with its lock facing sideways.
  const tie = prop.yaw_tie && !prop.extra_turn_degrees;
  f('props-tie').hidden = !tie;
  f('props-tie').textContent = tie
    ? `Squared up by ${prop.yaw_degrees}°, close to 45°, so it may face sideways. ` +
      'Check the view, and use Turn 90° if its front is on the side.'
    : '';
  f('props-turn').classList.toggle('suggested', tie);
  f('props-turn').disabled = state.running;
  f('props-turn').onclick = () => turnProp(run.directory, prop.name);

  const body = f('props-lods-body');
  body.innerHTML = '';
  const targets = (run.settings && run.settings.lods) || [];
  for (const lod of prop.lods) {
    const row = document.createElement('tr');
    // What the file holds; the target it was baked for if the file could not be read.
    const triangles = lod.triangles ?? targets[lod.index];
    const cells = [`LOD${lod.index}`, triangles ? triangles.toLocaleString() : '',
      downloadLink(lod.url, lod.bytes), downloadLink(lod.web_url, lod.web_bytes)];
    for (const value of cells) {
      const cell = document.createElement('td');
      cell.append(value);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  if (!prop.lods.length) {
    body.innerHTML = '<tr><td colspan="4">No LODs yet.</td></tr>';
  }
  preview(prop);
}

function showRun(run) {
  f('props-empty').hidden = true;
  f('props-result').hidden = false;
  f('props-result-title').textContent =
    `${run.props.length} props · output/props/${run.directory}/`;
  const licence = run.licence || {};
  f('props-licence').textContent = licence.name
    ? `Licence from the source model: ${licence.name}` +
      (licence.classification ? ` (${licence.classification})` : '') + '. The props inherit it.'
    : licence.source === 'generated'
      ? 'Made on this machine, but no licence record was kept beside the model, so check ' +
        'the licence of what made it before shipping the props.'
      : 'Uploaded model: its licence is unknown here, so check it before shipping the props.';
  f('props-blend').hidden = !run.blend_url;
  if (run.blend_url) f('props-blend').href = run.blend_url;
  const list = f('props-list');
  list.innerHTML = '';
  for (const prop of run.props) {
    const chip = document.createElement('button');
    chip.textContent = prop.name;
    chip.dataset.name = prop.name;
    chip.setAttribute('aria-pressed', 'false');
    if (prop.yaw_tie && !prop.extra_turn_degrees) {
      chip.classList.add('tie');
      chip.title = 'Close to 45°: check which way it faces';
    }
    chip.onclick = () => showProp(run, prop);
    list.appendChild(chip);
  }
  const chosen = run.props.find((prop) => prop.name === state.shown) || run.props[0];
  if (chosen) showProp(run, chosen);
  // Stacked on a narrow screen, the results are below the form; bring them into view.
  if (matchMedia('(max-width: 900px)').matches) {
    f('props-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

async function turnProp(directory, name) {
  if (state.running) return;
  begin(`Turning ${name} and re-baking it…`);
  state.shown = name;
  await start(fetch(`/api/props/runs/${encodeURIComponent(directory)}/turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prop: name, degrees: 90 }),
  }), `Could not turn ${name}`);
}

// --- Runs on disk, generated models and tools --------------------------------------------

function showTools(tools) {
  const host = f('props-tools');
  host.innerHTML = '';
  const line = (cls, html) => {
    const div = document.createElement('div');
    div.className = cls;
    div.innerHTML = html;
    host.appendChild(div);
    return div;
  };
  // The server's words, the ones the Finish tab shows: where to get it and which version.
  if (!tools.blender) line('bad', '').textContent = tools.blender_problem;
  if (!tools.gltfpack) {
    line('warn', 'gltfpack not found, so LODs stay uncompressed. Put a native release ' +
      '(github.com/zeux/meshoptimizer/releases) on PATH or at ' +
      '<code>vendor/gltfpack/gltfpack</code>; the npm build cannot write WebP.');
  }
  // Refreshed on every visit, so only force it off; a choice to skip it stays made.
  f('props-compress').disabled = !tools.gltfpack;
  if (!tools.gltfpack) f('props-compress').checked = false;
}

function showGenerated(models) {
  const select = f('props-generated');
  const chosen = select.value;
  select.length = 1;
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = `${model.name} (${kb(model.bytes)})`;
    select.appendChild(option);
  }
  select.value = [...select.options].some((o) => o.value === chosen) ? chosen : '';
}

function runRow(run) {
  const row = document.createElement('div');
  row.className = run.finished ? 'props-run done' : 'props-run';
  row.innerHTML = `<span><code></code><small>${run.props.length} props` +
    `${run.finished ? '' : ', not finished'}</small></span>`;
  row.querySelector('code').textContent = run.directory;
  row.querySelector('code').title = run.directory;
  if (run.props.length) {
    const open = document.createElement('button');
    open.textContent = 'Open';
    open.onclick = () => { state.shown = null; showRun(run); };
    row.appendChild(open);
  }
  return row;
}

async function loadRuns() {
  const host = f('props-runs');
  try {
    const response = await fetch('/api/props/runs');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { runs, generated, tools } = await response.json();
    state.tools = tools;
    showTools(tools);
    showGenerated(generated);
    host.innerHTML = '';
    if (!runs.length) host.innerHTML = '<small style="opacity:.7">No prop-sheet runs yet.</small>';
    for (const run of runs) host.appendChild(runRow(run));
  } catch (error) {
    host.innerHTML = `<small style="opacity:.7">Could not list runs: ${error.message}</small>`;
  }
  updateSubmit();
}

// --- Inputs ------------------------------------------------------------------------------

f('props-asset').onchange = (event) => {
  state.asset = event.target.files[0] || null;
  f('props-asset-name').textContent = state.asset
    ? `${state.asset.name} (${kb(state.asset.size)})` : 'An upload wins over the generated model.';
  updateSubmit();
};
f('props-generated').onchange = updateSubmit;
f('props-runs-refresh').onclick = loadRuns;

f('props-submit').onclick = async () => {
  if (state.running) return;
  begin('Uploading…');
  f('props-result').hidden = true;
  state.shown = null;

  const settings = {
    names: f('props-names').value,
    lods: f('props-lods').value,
    atlas: Number(f('props-atlas').value),
    metallic: Number(f('props-metallic').value),
    roughness: Number(f('props-roughness').value),
    ior: Number(f('props-ior').value),
    compress: f('props-compress').checked,
  };
  const form = new FormData();
  if (state.asset) form.append('asset', state.asset, state.asset.name);
  else form.append('generated', f('props-generated').value);
  form.append('settings', JSON.stringify(settings));

  await start(fetch('/api/props', { method: 'POST', body: form }), 'Could not start');
};

// Listed on arrival, not on load: every preview iframe loads this page too and never
// shows this tab, and listing the runs reads every LOD on disk. Arriving also refreshes
// the generated-model list, which goes stale while the page is open.
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail?.mode === 'props' && !state.running) loadRuns();
});

progress.configure(SPLIT_ONLY);
updateSubmit();
