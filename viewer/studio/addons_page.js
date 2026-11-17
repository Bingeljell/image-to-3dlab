// --- Add-ons page ---------------------------------------------------------------------
// Lists every add-on, installed first. Each installed add-on can expand its tools
// and run them; runs show a log and refresh the Library when they finish.

const s = (id) => document.getElementById(id);

let activeJobId = null;
let statusTimer = null;

export function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function authorLine(addon) {
  const author = addon.author?.name || 'unknown';
  const url = addon.author?.url;
  let html = `by ${url ? `<a href="${url}" target="_blank" rel="noopener">${escapeHtml(author)}</a>` : escapeHtml(author)}`;
  if (addon.links) {
    if (addon.links.repo) {
      html += ` · <a href="${addon.links.repo}" target="_blank" rel="noopener">Source</a>`;
    }
    if (addon.links.site) {
      html += ` · <a href="${addon.links.site}" target="_blank" rel="noopener">Site page</a>`;
    }
  }
  return html;
}

export function tierBadge(addon) {
  if (addon.no_longer_listed) {
    return '<span class="addon-badge addon-unlisted">No longer listed</span>';
  }
  if (addon.tier === 'verified') return '<span class="addon-badge addon-verified">Verified</span>';
  if (addon.tier === 'listed') return '<span class="addon-badge addon-listed">Listed</span>';
  return '<span class="addon-badge addon-unknown">Unknown</span>';
}

export function runsBadge(addon) {
  return addon.runs === 'cloud'
    ? '<span class="addon-badge addon-cloud">Cloud</span>'
    : '<span class="addon-badge addon-local">On this machine</span>';
}

export function ratingLine(addon) {
  if (addon.rating == null) return '';
  const count = addon.ratings_count ? ` (${addon.ratings_count})` : '';
  return `<span class="addon-rating" aria-label="rating ${addon.rating} of 5">★ ${addon.rating}${count}</span>`;
}

export function inputControl(input) {
  const id = `addon-input-${input.id}`;
  if (input.type === 'choice') {
    const options = input.choices.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
    return `<label for="${id}">${escapeHtml(input.label)}</label>
      <select id="${id}" data-input="${escapeHtml(input.id)}">${options}</select>`;
  }
  if (input.type === 'toggle') {
    return `<label class="addon-toggle"><input type="checkbox" id="${id}" data-input="${escapeHtml(input.id)}">
      <span>${escapeHtml(input.label)}</span></label>`;
  }
  if (input.type === 'number') {
    const min = input.min != null ? `min="${input.min}"` : '';
    const max = input.max != null ? `max="${input.max}"` : '';
    return `<label for="${id}">${escapeHtml(input.label)}</label>
      <input type="number" id="${id}" data-input="${escapeHtml(input.id)}" ${min} ${max}>`;
  }
  const placeholder = (input.type === 'picture' || input.type === 'model') ? 'Library or file path' : '';
  return `<label for="${id}">${escapeHtml(input.label)}</label>
    <input type="text" id="${id}" data-input="${escapeHtml(input.id)}" placeholder="${placeholder}">`;
}

function gatherInputs(form) {
  const inputs = {};
  form.querySelectorAll('[data-input]').forEach((el) => {
    const key = el.dataset.input;
    inputs[key] = el.type === 'checkbox' ? el.checked : el.value;
  });
  return inputs;
}

function toolCard(addon, tool) {
  const card = document.createElement('div');
  card.className = 'setup-card addon-tool-card';
  card.innerHTML = `
    <div class="setup-card-title"><strong>${escapeHtml(tool.label)}</strong></div>
    <p class="setup-card-best">${escapeHtml(tool.hint)}</p>
    <form class="addon-tool-form">
      ${tool.inputs.map(inputControl).join('')}
      <button type="submit" class="addon-run">Run</button>
    </form>
    <pre class="addon-log"></pre>`;
  card.querySelector('form').addEventListener('submit', (e) => {
    e.preventDefault();
    runTool(addon.id, tool.id, card);
  });
  return card;
}

function showMessage(card, message, type = 'info') {
  let el = card.querySelector('.addon-message');
  if (!el) {
    el = document.createElement('div');
    el.className = 'addon-message';
    card.insertBefore(el, card.firstChild);
  }
  el.className = `addon-message addon-message-${type}`;
  el.textContent = message;
}

function clearMessage(card) {
  const el = card.querySelector('.addon-message');
  if (el) el.remove();
}

async function runTool(addonId, toolId, card) {
  if (activeJobId) {
    showMessage(card, 'Another add-on run is already in progress.', 'info');
    return;
  }
  clearMessage(card);
  const log = card.querySelector('.addon-log');
  log.textContent = '';
  const button = card.querySelector('.addon-run');
  button.disabled = true;
  try {
    const response = await fetch(`/api/addons/${encodeURIComponent(addonId)}/run/${encodeURIComponent(toolId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: gatherInputs(card) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'unknown error');
    activeJobId = result.job_id;
    pollJob(card);
  } catch (error) {
    button.disabled = false;
    log.textContent = `Error: ${error.message}`;
  }
}

async function pollJob(card) {
  const log = card.querySelector('.addon-log');
  const button = card.querySelector('.addon-run');
  if (!activeJobId) {
    button.disabled = false;
    return;
  }
  try {
    const response = await fetch(`/api/addons/jobs/${encodeURIComponent(activeJobId)}/status`);
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || 'unknown error');
    if (status.log_tail) log.textContent = status.log_tail;
    if (status.status === 'done') {
      activeJobId = null;
      button.disabled = false;
      log.textContent += `\nDone: ${status.result}`;
      return;
    }
    if (status.status === 'error') {
      activeJobId = null;
      button.disabled = false;
      log.textContent += `\nError: ${status.error || 'run failed'}`;
      return;
    }
    statusTimer = setTimeout(() => pollJob(card), 1000);
  } catch (error) {
    activeJobId = null;
    button.disabled = false;
    log.textContent = `Error: ${error.message}`;
  }
}

function showPlan(card, plan) {
  const box = document.createElement('div');
  box.className = 'addon-plan-box';
  const models = plan.models && plan.models.length
    ? `<ul>${plan.models.map((m) => `<li>${escapeHtml(m.name)} — ${escapeHtml(m.licence)}, ${escapeHtml(m.size)}</li>`).join('')}</ul>`
    : '';
  const version = plan.version ? ` · <strong>Version:</strong> ${escapeHtml(plan.version)}` : '';
  box.innerHTML = `
    <p><strong>Author:</strong> ${escapeHtml(plan.author)} · <strong>Licence:</strong> ${escapeHtml(plan.licence)}${version}</p>
    ${models}
    <p class="addon-warning">${escapeHtml(plan.warning)}</p>
    <button type="button" class="addon-confirm">Install ${escapeHtml(plan.name)}</button>`;
  box.querySelector('.addon-confirm').addEventListener('click', () => confirmInstall(plan.id, card, box));
  card.querySelector('.addon-plan').replaceChildren(box);
}

async function promptInstall(addonId, card) {
  clearMessage(card);
  const button = card.querySelector('.addon-install');
  button.disabled = true;
  try {
    const response = await fetch(`/api/addons/${encodeURIComponent(addonId)}/install`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const plan = await response.json();
    if (!response.ok) throw new Error(plan.error || 'unknown error');
    showPlan(card, plan);
  } catch (error) {
    button.disabled = false;
    showMessage(card, `Could not load install plan: ${error.message}`, 'error');
  }
}

async function confirmInstall(addonId, card, box) {
  clearMessage(card);
  const button = box.querySelector('.addon-confirm');
  button.disabled = true;
  try {
    const response = await fetch(`/api/addons/${encodeURIComponent(addonId)}/install`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirm: true }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'unknown error');
    await load();
  } catch (error) {
    showMessage(card, `Install failed: ${error.message}`, 'error');
  }
}

function confirmRemove(addonId, card) {
  clearMessage(card);
  const plan = card.querySelector('.addon-plan');
  const box = document.createElement('div');
  box.className = 'addon-plan-box';
  box.innerHTML = `
    <p>Remove <strong>${escapeHtml(addonId)}</strong>? This deletes its folder and cannot be undone.</p>
    <button type="button" class="addon-confirm-remove">Remove</button>
    <button type="button" class="addon-cancel-remove">Cancel</button>`;
  box.querySelector('.addon-confirm-remove').addEventListener('click', () => doRemove(addonId, card, box));
  box.querySelector('.addon-cancel-remove').addEventListener('click', () => box.remove());
  plan.replaceChildren(box);
}

async function doRemove(addonId, card, box) {
  box.querySelector('.addon-confirm-remove').disabled = true;
  try {
    const response = await fetch(`/api/addons/${encodeURIComponent(addonId)}/remove`, { method: 'POST' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'unknown error');
    await load();
  } catch (error) {
    showMessage(card, `Remove failed: ${error.message}`, 'error');
  }
}

function addonCard(addon) {
  const card = document.createElement('div');
  card.className = 'setup-card addon-card';
  card.dataset.addon = addon.id;
  const installed = addon.installed;
  const action = installed
    ? `<button type="button" class="addon-remove" data-addon="${addon.id}">Remove</button>`
    : `<button type="button" class="addon-install" data-addon="${addon.id}">Install</button>`;
  const tools = installed && addon.tools
    ? `<button type="button" class="addon-tools-toggle">Tools (${addon.tools.length})</button>
       <div class="addon-tools" hidden></div>`
    : '';
  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${installed ? 'ok' : 'off'}">${installed ? '●' : '○'}</span>
      <div class="setup-card-title">
        <strong>${escapeHtml(addon.name)}</strong>
        ${tierBadge(addon)}${runsBadge(addon)}
        <div class="setup-card-state addon-author">${authorLine(addon)} ${ratingLine(addon)}</div>
      </div>
      <div class="setup-card-action">${action}</div>
    </div>
    <p class="setup-card-best">${escapeHtml(addon.summary || '')}</p>
    ${tools}
    <div class="addon-plan"></div>`;

  const installBtn = card.querySelector('.addon-install');
  if (installBtn) installBtn.addEventListener('click', () => promptInstall(addon.id, card));
  const removeBtn = card.querySelector('.addon-remove');
  if (removeBtn) removeBtn.addEventListener('click', () => confirmRemove(addon.id, card));
  const toggle = card.querySelector('.addon-tools-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const panel = card.querySelector('.addon-tools');
      if (panel.hidden) {
        panel.innerHTML = '';
        for (const tool of addon.tools) panel.appendChild(toolCard(addon, tool));
      }
      panel.hidden = !panel.hidden;
      toggle.textContent = panel.hidden ? `Tools (${addon.tools.length})` : 'Hide tools';
    });
  }
  return card;
}

async function load() {
  const summary = s('addons-summary');
  const list = s('addons-list');
  try {
    const response = await fetch('/api/addons');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const addons = payload.addons.slice().sort((a, b) => Number(b.installed) - Number(a.installed));
    if (!payload.online) {
      summary.hidden = false;
      summary.textContent = payload.error || 'The add-on list is unreachable.';
    } else {
      summary.hidden = true;
    }
    list.innerHTML = '';
    if (!addons.length) {
      list.innerHTML = '<p class="setup-card-best">No add-ons listed yet.</p>';
      return;
    }
    for (const addon of addons) list.appendChild(addonCard(addon));
  } catch (error) {
    summary.hidden = false;
    summary.textContent = `Could not load add-ons: ${error.message}`;
  }
}

if (typeof document !== 'undefined') load();
