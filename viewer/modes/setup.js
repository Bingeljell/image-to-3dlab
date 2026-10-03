// --- Setup & Status -------------------------------------------------------------------
// What is installed, what it would cost to install the rest, and the only place a download
// can be started. Reads GET /api/catalog (viewer/backend_catalog.py).
//
// `AGENTS.md` forbids fetching weights before the user has confirmed which pipeline and
// which route, so the download button here states the size, the source and the licence
// and waits for an answer. Nobody should find 50 GB on their disk from a button that said
// "Run setup".
//
// This page is also where the lab's other tools will report themselves as they land, which
// is why it is "Setup & Status" and not "Install": it is the machine-state page, and the
// natural home later for account and preferences.

import { showsCaveat } from '../components/gated-caveat.js';

const s = (id) => document.getElementById(id);
const SKIP_KEY = 'i2l.setup.skip';

// `panelFor` is the backend whose download the progress panel shows. It outlives the run
// so the finished result stays under the card that started it.
const state = { catalog: null, source: null, running: null, panelFor: null, tab: null,
  renderToken: 0, hfAccess: {} };

/** Put the progress panel right under the card it belongs to. It used to sit below every
 * card, so on a long list the bar was off-screen and a click on Set up looked like it had
 * done nothing (issue #36). */
function placeRunPanel() {
  const card = state.panelFor
    && s('setup-backends').querySelector(`[data-backend="${CSS.escape(state.panelFor)}"]`);
  if (card) card.after(s('setup-run'));
}

/** Whether the user asked not to land here. Browser storage can throw; never block on it. */
export function skipRequested() {
  try {
    return localStorage.getItem(SKIP_KEY) === '1';
  } catch (_) {
    return false;
  }
}

function setSkip(value) {
  try {
    localStorage.setItem(SKIP_KEY, value ? '1' : '0');
  } catch (_) { /* private window: the preference simply does not persist */ }
}

const STATE_META = {
  ready: { dot: '●', cls: 'ok', text: 'installed' },
  partial: { dot: '◐', cls: 'warn', text: 'partly downloaded' },
  missing: { dot: '○', cls: 'off', text: 'not installed' },
  unsupported: { dot: '–', cls: 'off', text: 'not available on this machine' },
};

function backendCard(backend, { readOnly = false } = {}) {
  // readOnly: a card on another machine's tab. What it is and what it costs, no state
  // (this disk says nothing about that machine) and no buttons.
  const meta = readOnly ? { cls: 'off', dot: '·', text: 'on that machine' }
    : STATE_META[backend.state] || STATE_META.missing;
  const card = document.createElement('div');
  card.className = `setup-card ${readOnly ? 'elsewhere' : backend.state}`;
  card.dataset.backend = backend.id;

  const size = !readOnly && backend.state === 'partial'
    ? `${backend.human_present} of ${backend.human_expected}`
    : backend.human_expected;
  // Say where the weights are when they are not here yet but the backend still works.
  const later = !readOnly && backend.action === 'none' && backend.bytes_present === 0
    ? ' · weights download on your first run' : '';

  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${meta.cls}">${meta.dot}</span>
      <div class="setup-card-title">
        <strong>${backend.label}</strong>
        ${backend.recommended ? '<span class="setup-pill">start here</span>' : ''}
        <div class="setup-card-state">${meta.text} · ${size}${later}</div>
      </div>
      <div class="setup-card-action"></div>
    </div>
    <p class="setup-card-best">${backend.best_for}</p>
    <p class="setup-card-trade">${backend.tradeoff}</p>
    ${!readOnly && backend.platform_note
      ? `<p class="setup-card-caveat">${backend.platform_note}</p>` : ''}
    ${showsCaveat(backend, readOnly ? {} : state.hfAccess)
      ? `<p class="setup-card-caveat">${backend.caveat}</p>` : ''}
    <details class="setup-card-detail">
      <summary>What gets downloaded</summary>
      <ul>${backend.weights.map((w) => `
        <li>
          <a href="${w.source_url}" target="_blank" rel="noopener">${w.source}</a>
          — ${w.human_expected}${w.present ? ` (${w.human_present} on disk)` : ''}
          ${w.note ? `<br><small>${w.note}</small>` : ''}
        </li>`).join('')}
      </ul>
      ${backend.extra_steps.length
        ? `<p class="setup-card-extra">${backend.extra_steps.join('<br>')}</p>` : ''}
      <p class="setup-card-licence">
        Licence: ${backend.license.name}.
        <a href="${backend.license.url}" target="_blank" rel="noopener">Read it here</a>.
      </p>
    </details>`;

  const action = card.querySelector('.setup-card-action');
  if (readOnly) return card;
  // Driven by the server's `action`, not inferred from state here. Build and weights
  // are independent: a built TRELLIS with no weights is usable and must not be offered a
  // Set up button, because re-running a finished bootstrap fails on its own patches.
  const LABELS = { build: 'Set up', download: 'Download', resume: 'Resume download' };
  // A backend that cannot run here gets no button at all. Offering one would spend
  // gigabytes of someone's bandwidth on a build that fails partway through.
  if (backend.supported_here === false) {
    // On this machine's own tab, so only an OS exclusion lands here (TRELLIS.2 on Windows);
    // the platform note under the card says why.
    action.innerHTML = '<span class="setup-unavailable">not on this OS yet</span>';
  } else if (backend.action === 'none') {
    action.innerHTML = '<span class="setup-ready">✓ ready</span>';
  } else if (backend.action === 'manual') {
    // Some routes the viewer cannot install for you: SF3D wants a shell bootstrap, and
    // the dgrauet Hunyuan shape stage is a deliberate manual clone because it is
    // Tencent-licensed code rather than just weights. Showing the command is honest; a
    // button that throws is not.
    const span = document.createElement('span');
    span.className = 'setup-manual';
    span.textContent = 'run ';
    const code = document.createElement('code');
    code.textContent = backend.install;
    span.appendChild(code);
    action.replaceChildren(span);
  } else {
    const button = document.createElement('button');
    button.textContent = LABELS[backend.action];
    button.onclick = () => confirmDownload(backend, button);
    action.appendChild(button);
  }
  // An installed build that predates one of this repo's patches (Pixal3D's 8 steps).
  // Recompiling downloads nothing, so it gets a short question, not the size dialog.
  if (backend.rebuild_reason && backend.supported_here !== false) {
    const rebuild = document.createElement('button');
    rebuild.textContent = 'Rebuild';
    rebuild.title = 'Recompile with this repo\'s latest fixes. Downloads nothing.';
    rebuild.onclick = () => confirmRebuild(backend, rebuild);
    action.appendChild(rebuild);
  }
  // Reclaiming is about bytes on disk, not about whether the backend works.
  // Only what Remove would free: files another route shares (the background remover) stay.
  if ((backend.bytes_removable ?? backend.bytes_present) > 0) {
    action.appendChild(removeButton(backend));
  }
  return card;
}

/** Reclaiming the disk, kept separate from cancelling a download.
 *
 * Cancelling leaves partial files that Hugging Face resumes from, so clearing them
 * automatically would turn a pause into a restart. This is the deliberate version, for
 * when the space is wanted back or a download came down corrupt.
 */
function removeButton(backend) {
  const button = document.createElement('button');
  button.className = 'ghost setup-remove';
  button.textContent = 'Remove';
  button.title = `Delete ${formatBytes(backend.bytes_removable ?? backend.bytes_present)} of weights from disk`;
  button.onclick = async () => {
    const shared = backend.weights.some((w) => w.path.includes('huggingface'));
    // Typed confirmation, not a yes/no. Three Remove buttons sit in one column and each
    // one deletes several gigabytes that take minutes to hours to replace; a misplaced
    // click plus a reflexive OK is a real way to lose an afternoon. Typing the backend's
    // id proves both that it was meant and *which* one was meant, which a bare "delete"
    // would not.
    // eslint-disable-next-line no-alert -- deliberate, and the strongest gate available.
    const typed = window.prompt([
      `Delete ${formatBytes(backend.bytes_removable ?? backend.bytes_present)} of ${backend.label} weights?`,
      '',
      ...backend.weights.filter((w) => w.present).map((w) => `  ${w.path}`),
      '',
      shared
        ? 'These live in the shared Hugging Face cache, so other tools on this machine'
          + ' may be using them.'
        : 'This cannot be undone from here.',
      'They can be downloaded again.',
      '',
      `Type  ${backend.id}  to confirm:`,
    ].join('\n'), '');
    if (typed === null) return;
    if (typed.trim().toLowerCase() !== backend.id.toLowerCase()) {
      window.alert(`Not deleted. You typed "${typed}", expected "${backend.id}".`);
      return;
    }
    button.disabled = true;
    try {
      const response = await fetch(`/api/setup/${encodeURIComponent(backend.id)}/remove`,
        { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      s('setup-summary').textContent = `Removed ${payload.freed}.`;
    } catch (error) {
      button.disabled = false;
      window.alert(`Could not remove: ${error.message}`);
    }
    load();
  };
  return button;
}

/** State the cost, then ask. The confirmation is the point of the whole page. */
function confirmDownload(backend, button) {
  const remaining = backend.bytes_expected - backend.bytes_present;
  // Say what this particular button actually does. TRELLIS's setup builds the Metal port
  // and fetches nothing; its weights arrive on the first generation run, and promising a
  // download here would be a lie the progress bar then has to keep.
  const body = backend.setup_fetches_weights
    ? [
      // NVIDIA TRELLIS.2 compiles first, then downloads; say both, not just the bytes.
      ...(backend.build_present || backend.install === 'viewer' ? []
        : ['This builds the code for this machine first, then:', '']),
      `About ${formatBytes(remaining)} will be downloaded from Hugging Face`,
      `into ${backend.weights[0].path.replace(/\/[^/]*$/, '/')}`,
      '',
      'Some sources need you to be signed in to Hugging Face and to have accepted their',
      'terms; the log will say so if the download is refused.',
    ]
    : [
      'This builds the Metal port first',
      'and downloads no weights.',
      '',
      `The ${backend.human_expected} of weights are fetched on your first generation run,`,
      'not now.',
    ];
  const lines = [
    `${backend.label}`,
    '',
    ...body,
    '',
    `Licence: ${backend.license.name}`,
    backend.caveat ? `\n${backend.caveat}\n` : '',
    '',
    backend.setup_fetches_weights ? 'Start the download?' : 'Start the build?',
  ].filter(Boolean);

  // eslint-disable-next-line no-alert -- a deliberate, blocking confirmation: this is the
  // one action on the page that spends the user's disk and bandwidth.
  if (!window.confirm(lines.join('\n'))) return;
  startDownload(backend, button);
}

/** Recompile an installed build. Nothing is downloaded, so the question is short. */
function confirmRebuild(backend, button) {
  const lines = [
    `Rebuild ${backend.label}?`,
    '',
    'This recompiles the copy you already have so it picks up this repo\'s latest fixes',
    '(for Pixal3D: 8 sampling steps instead of 12, so runs are faster).',
    '',
    'It downloads nothing and takes a few minutes.',
  ];
  // eslint-disable-next-line no-alert -- same deliberate confirmation as a download
  if (!window.confirm(lines.join('\n'))) return;
  startDownload(backend, button, 'rebuild');
}

function formatBytes(value) {
  if (value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

async function startDownload(backend, button, action = 'download') {
  button.disabled = true;
  s('setup-run').hidden = false;
  s('setup-run-cancel').hidden = false;
  // A route with no build yet compiles before it downloads anything.
  const verb = action === 'rebuild' ? 'Rebuilding'
    : !backend.build_present ? 'Setting up'
    : backend.setup_fetches_weights ? 'Downloading' : 'Building';
  s('setup-run-title').textContent = `${verb} ${backend.label}`;
  s('setup-run-detail').textContent = 'starting…';
  s('setup-log').textContent = '';
  s('setup-run-bar').style.width = '0%';
  state.panelFor = backend.id;
  placeRunPanel();
  s('setup-run').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  try {
    const response = await fetch(`/api/setup/${encodeURIComponent(backend.id)}/${action}`, {
      method: 'POST',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.running = backend.id;
    watch(payload);
  } catch (error) {
    button.disabled = false;
    s('setup-run-detail').textContent = `Could not start: ${error.message}`;
  }
}

function resume(run) {
  const backend = state.catalog.backends.find((b) => b.id === run.backend);
  const label = backend ? backend.label : run.backend;
  const verb = run.rebuild ? 'Rebuilding'
    : backend && !backend.build_present ? 'Setting up'
    : backend && backend.setup_fetches_weights ? 'Downloading' : 'Building';
  s('setup-run').hidden = false;
  s('setup-run-cancel').hidden = false;
  s('setup-run-title').textContent = `${verb} ${label}`;
  s('setup-run-detail').textContent = 'reconnecting…';
  s('setup-log').textContent = '';
  state.running = run.backend;
  state.panelFor = run.backend;
  placeRunPanel();
  watch(run);
}

function watch(payload) {
  state.source?.close();
  state.source = new EventSource(payload.events_url);
  state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
  s('setup-run-cancel').onclick = async () => {
    await fetch(`/api/setup/${encodeURIComponent(state.running)}/cancel`, { method: 'POST' });
  };
}

function applyEvent(event) {
  if (!event) return;
  if (typeof event.overall_pct === 'number') {
    s('setup-run-bar').style.width = `${event.overall_pct}%`;
  }
  // A compile has no percentage to show; an empty bar for half an hour reads as stuck.
  s('setup-run-bar').parentElement.classList.toggle('busy',
    event.phase === 'building' || event.phase === 'queued');
  if (event.detail) s('setup-run-detail').textContent = event.detail;
  if (event.log) {
    const log = s('setup-log');
    log.textContent += `${event.log}\n`;
    log.scrollTop = log.scrollHeight;
  }
  if (event.phase === 'done' || event.phase === 'error' || event.phase === 'cancelled') {
    state.source?.close();
    state.source = null;
    state.running = null;
    s('setup-run-cancel').hidden = true;
    load();
  }
}

/** This machine's backends: its own tab's list, or every backend on an unknown machine. */
function hereBackends(catalog) {
  return (catalog.views && catalog.views[catalog.host.id]) || catalog.backends;
}

/* One tab per machine family. This machine's tab opens first and is the only one with
 * buttons; the others say what that machine would get, so a Mac card never sits on an
 * NVIDIA page saying "use the NVIDIA one instead". */
function renderTabs(catalog) {
  const bar = s('setup-tabs');
  const here = catalog.host.id;
  if (!state.tab) {
    state.tab = catalog.platforms.some((p) => p.id === here) ? here : catalog.platforms[0].id;
  }
  bar.replaceChildren(...catalog.platforms.map((platform) => {
    const tab = document.createElement('button');
    tab.className = 'setup-tab';
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', String(platform.id === state.tab));
    tab.textContent = platform.label + (platform.id === here ? ' · this machine' : '');
    tab.onclick = () => {
      state.tab = platform.id;
      renderTabs(catalog);
      s('setup-summary').innerHTML = summarise(catalog);
      renderBackends(catalog);
    };
    return tab;
  }));
}

async function renderBackends(catalog) {
  // Renders can overlap (startup and opening the tab both load). Ask everything first,
  // then write in one go, and only if no newer render has started since.
  const token = ++state.renderToken;
  const readOnly = state.tab !== catalog.host.id;
  const blender = readOnly ? null : await blenderCard();
  const gltfpack = readOnly ? null : await gltfpackCard();
  const hf = readOnly ? null : await hfCard();
  if (token !== state.renderToken) return;
  const host = s('setup-backends');
  host.after(s('setup-run')); // park it outside the cards before they are rebuilt
  host.innerHTML = '';
  const platform = catalog.platforms.find((p) => p.id === state.tab);
  if (platform && platform.coming) {
    host.innerHTML = `<div class="setup-card elsewhere"><p class="setup-card-best">
      <strong>${platform.label} support is coming.</strong> No backend runs on it yet.
      Watch the releases on GitHub, or open an issue to say you want it.</p></div>`;
    return;
  }
  if (hf) host.appendChild(hf);
  const list = readOnly ? catalog.views[state.tab] : hereBackends(catalog);
  for (const backend of list) host.appendChild(backendCard(backend, { readOnly }));
  if (blender) host.appendChild(blender);
  if (gltfpack) host.appendChild(gltfpack);
  placeRunPanel();
}

function summarise(catalog) {
  // Another machine's tab: say so first, or "7 of 7 installed" reads as that machine's.
  if (state.tab && catalog.platforms && state.tab !== catalog.host.id) {
    const shown = catalog.platforms.find((p) => p.id === state.tab);
    return `<strong>Showing what runs on ${shown ? shown.label : state.tab}.</strong> `
      + `This machine: ${catalog.host.label}. Its own tab has the Set up buttons.`;
  }
  const mine = hereBackends(catalog);
  const ready = mine.filter((b) => b.state === 'ready');
  const onDisk = mine.reduce((total, b) => total + b.bytes_present, 0);
  // Said before anything else, because "nothing is installed" and "nothing can be
  // installed here" look the same in a list of cards, and only one of them is fixable
  // by clicking a button. Someone on the wrong machine deserves to know on arrival.
  if (catalog.host && !catalog.host.any_backend_runs_here) {
    return `<strong>This machine is ${catalog.host.label}.</strong> `
      + `Every backend here needs ${catalog.host.supported.join(' or ')}, so there is `
      + `nothing to install yet. AMD support is coming (see the AMD tab); the other `
      + `tabs show what each machine gets.`;
  }
  // A route that only needs a top-up (a newer release added a small file) is not
  // "nothing installed": that read as losing a working setup after an update.
  const partial = mine.filter((b) => b.state === 'partial');
  if (!ready.length && partial.length) {
    return `<strong>Nearly ready.</strong> ${partial.map((b) => b.label).join(', ')} `
      + `${partial.length === 1 ? 'is' : 'are'} partly downloaded: press `
      + `<em>Resume download</em> on ${partial.length === 1 ? 'its card' : 'their cards'}.`;
  }
  if (!ready.length) {
    return `<strong>No backend installed yet.</strong> Pick one below. `
      + `The recommended one is about ${mine[0].human_expected}.`;
  }
  return `<strong>${ready.length} of ${mine.length} backends installed</strong>`
    + ` · ${formatBytes(onDisk)} of model weights on disk`;
}

/* Hugging Face sign-in. Some models are gated (DINOv3 for TRELLIS.2, Stable Fast 3D), and
 * `hf auth login` in a terminal was the one step left between the install command and a
 * working setup. The token goes to the lab, which checks it with Hugging Face and saves it
 * where `hf auth login` would; it is never shown again. */
async function hfCard() {
  const card = document.createElement('div');
  card.dataset.backend = 'huggingface';
  let st = null;
  try {
    const response = await fetch('/api/hf/status');
    if (response.ok) st = await response.json();
  } catch { /* shown as unknown below */ }
  const signedIn = Boolean(st && st.signed_in);
  // The backend cards render after this one and read it to drop gated warnings already met.
  state.hfAccess = Object.fromEntries((st ? st.repos : []).map((r) => [r.repo, r.access]));
  card.className = `setup-card ${signedIn ? 'ready' : 'missing'}`;
  const rows = (st ? st.repos : []).map((r) => {
    const mark = r.access === 'yes' ? '<span class="setup-ready">✓ access</span>'
      : r.access === 'no'
        ? `<a href="${r.request_url}" target="_blank" rel="noopener">Request access</a> (approved by hand)`
        : '<span class="setup-card-state">sign in to check</span>';
    return `<li><code>${r.repo}</code> for ${r.for}: ${mark}</li>`;
  }).join('');
  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${signedIn ? 'ok' : 'off'}">${signedIn ? '●' : '○'}</span>
      <div class="setup-card-title">
        <strong>Hugging Face sign-in (for gated models)</strong>
        <div class="setup-card-state">${signedIn ? `signed in as ${st.user}` : 'not signed in'}</div>
      </div>
      <div class="setup-card-action"></div>
    </div>
    <p class="setup-card-trade">Some models ask you to accept their licence on Hugging Face
      first. Sign in once here; nothing is downloaded until you press Set up.</p>
    <ul class="setup-hf-repos">${rows}</ul>`;
  if (!signedIn) {
    const form = document.createElement('div');
    form.className = 'setup-hf-form';
    const input = document.createElement('input');
    input.type = 'password';
    input.placeholder = 'hf_… (a Read token)';
    input.autocomplete = 'off';
    const button = document.createElement('button');
    button.textContent = 'Sign in';
    const note = document.createElement('p');
    note.className = 'setup-card-trade';
    note.innerHTML = 'Make a token at <a href="https://huggingface.co/settings/tokens" '
      + 'target="_blank" rel="noopener">huggingface.co/settings/tokens</a> (type: Read).';
    button.onclick = async () => {
      button.disabled = true;
      note.textContent = 'checking with Hugging Face…';
      try {
        const response = await fetch('/api/hf/sign-in', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: input.value }),
        });
        const result = await response.json();
        input.value = '';
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
        load();
      } catch (error) {
        note.textContent = error.message;
        button.disabled = false;
      }
    };
    form.append(input, button);
    card.append(form, note);
  }
  return card;
}

/* Blender is the one requirement the viewer never installs: it is a ~400 MB application
 * people choose for themselves. Finish and the rig tools need it, so its state belongs on
 * the page that owns install state, before anyone clicks Finish and finds out. */
async function blenderCard() {
  const card = document.createElement('div');
  let caps = null;
  try {
    const response = await fetch('/api/finish/capabilities');
    if (response.ok) caps = await response.json();
  } catch { /* shown as unknown below */ }
  const found = Boolean(caps && caps.blender && !caps.blender_problem);
  card.className = `setup-card ${found ? 'ready' : 'missing'}`;
  card.dataset.backend = 'blender';
  const where = found
    ? `found · Blender ${caps.blender_version || ''} at <code>${caps.blender}</code>`
    : 'not found';
  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${found ? 'ok' : 'off'}">${found ? '●' : '○'}</span>
      <div class="setup-card-title">
        <strong>Blender (for Finish and rigging)</strong>
        <div class="setup-card-state">${where}</div>
      </div>
      <div class="setup-card-action">${found
        ? '<span class="setup-ready">✓ ready</span>'
        : '<a href="https://www.blender.org/download/" target="_blank" rel="noopener">Get Blender</a>'}</div>
    </div>
    <p class="setup-card-best">Finish (low-poly clean-up and Pixel Match) runs Blender in the background.</p>
    ${!found && caps && caps.blender_problem && !caps.blender_installable
      ? `<p class="setup-card-caveat">${caps.blender_problem}</p>` : ''}`;
  // Linux: install blender.org's LTS build from here, instead of leaving the viewer to
  // download and unpack it by hand (scripts/bootstrap_blender.py).
  if (caps && caps.blender_installable) {
    const action = card.querySelector('.setup-card-action');
    const button = document.createElement('button');
    button.textContent = 'Install Blender';
    const progress = document.createElement('p');
    progress.className = 'setup-card-trade';
    progress.textContent = 'Blender 4.2 LTS from blender.org: about 380 MB to download, '
      + 'about 1 GB unpacked in your home folder (~/blender-lts). No admin rights needed.';
    button.onclick = async () => {
      // eslint-disable-next-line no-alert -- same deliberate confirmation as a download
      if (!window.confirm('Install Blender 4.2 LTS (GPL) from blender.org?\n\n'
        + 'About 380 MB to download, about 1 GB unpacked into ~/blender-lts.')) return;
      button.disabled = true;
      try {
        const response = await fetch('/api/blender/install', { method: 'POST' });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        const source = new EventSource(payload.events_url);
        source.onmessage = (message) => {
          const event = JSON.parse(message.data);
          if (event.message) progress.textContent = event.message;
          if (event.phase === 'setup_done') { source.close(); load(); }
        };
      } catch (error) {
        progress.textContent = `Could not start: ${error.message}`;
        button.disabled = false;
      }
    };
    action.replaceChildren(button);
    card.appendChild(progress);
  }
  return card;
}

/* gltfpack shrinks the Props tab's LODs (WebP textures, GPU-ordered meshes). Optional:
 * without it props still split and bake, just bigger. A small MIT tool, so Setup offers
 * it on every machine with a build (scripts/bootstrap_gltfpack.py). */
async function gltfpackCard() {
  let tools = null;
  try {
    const response = await fetch('/api/props/tools');
    if (response.ok) tools = await response.json();
  } catch { /* shown as unknown below */ }
  const found = Boolean(tools && tools.gltfpack);
  const card = document.createElement('div');
  card.className = `setup-card ${found ? 'ready' : 'missing'}`;
  card.dataset.backend = 'gltfpack';
  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${found ? 'ok' : 'off'}">${found ? '●' : '○'}</span>
      <div class="setup-card-title">
        <strong>gltfpack (optional, for Props)</strong>
        <div class="setup-card-state">${found ? `found at <code>${tools.gltfpack}</code>` : 'not found'}</div>
      </div>
      <div class="setup-card-action">${found
        ? '<span class="setup-ready">✓ ready</span>'
        : '<a href="https://github.com/zeux/meshoptimizer/releases" target="_blank" rel="noopener">Get gltfpack</a>'}</div>
    </div>
    <p class="setup-card-best">Makes the Props tab's files much smaller, ready for web games.
      Without it, props still work, just bigger.</p>`;
  if (tools && tools.gltfpack_installable) {
    const action = card.querySelector('.setup-card-action');
    const button = document.createElement('button');
    button.textContent = 'Install gltfpack';
    const progress = document.createElement('p');
    progress.className = 'setup-card-trade';
    progress.textContent = 'gltfpack 1.3 (MIT) from GitHub: under 2 MB, into vendor/gltfpack/ '
      + 'in this folder. No admin rights needed.';
    button.onclick = async () => {
      // eslint-disable-next-line no-alert -- same deliberate confirmation as a download
      if (!window.confirm('Install gltfpack 1.3 (MIT) from GitHub?\n\n'
        + 'Under 2 MB, into vendor/gltfpack/ in this folder.')) return;
      button.disabled = true;
      try {
        const response = await fetch('/api/gltfpack/install', { method: 'POST' });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        const source = new EventSource(payload.events_url);
        source.onmessage = (message) => {
          const event = JSON.parse(message.data);
          if (event.message) progress.textContent = event.message;
          if (event.phase === 'setup_done') { source.close(); load(); }
        };
      } catch (error) {
        progress.textContent = `Could not start: ${error.message}`;
        button.disabled = false;
      }
    };
    action.replaceChildren(button);
    card.appendChild(progress);
  }
  return card;
}

export async function load() {
  const host = s('setup-backends');
  try {
    const response = await fetch('/api/catalog');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.catalog = await response.json();
    if (state.catalog.platforms) renderTabs(state.catalog);
    s('setup-summary').innerHTML = summarise(state.catalog);
    if (state.catalog.platforms) await renderBackends(state.catalog);
    else {
      host.after(s('setup-run'));
      host.innerHTML = '';
      for (const backend of state.catalog.backends) host.appendChild(backendCard(backend));
      host.appendChild(await blenderCard());
      placeRunPanel();
    }
    // A setup runs for up to an hour. Reloaded or reopened mid-run, pick it back up; the
    // event stream replays from the start, so the bar and the log come back whole.
    if (state.catalog.running_setup && !state.source) resume(state.catalog.running_setup);
    document.dispatchEvent(new CustomEvent('viewer:catalog', { detail: state.catalog }));
  } catch (error) {
    s('setup-summary').textContent = `Could not read machine status: ${error.message}`;
  }
}

s('setup-skip').checked = skipRequested();
s('setup-skip').onchange = (event) => setSkip(event.target.checked);
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'setup') load();
});

load();
