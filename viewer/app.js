import './modes/compare.js';

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
