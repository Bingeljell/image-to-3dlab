// --- Add-ons sheet ---------------------------------------------------------------------
// One card per installed tool. Run → job → result opens in the Library.

import { closeSheet } from './pages.js';

const s = (id) => document.getElementById(id);

let addonTools = [];
let activeJobId = null;
let statusTimer = null;

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function inputControl(input) {
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
  const placeholder = (input.type === 'picture' || input.type === 'model')
    ? 'Library or file path'
    : '';
  return `<label for="${id}">${escapeHtml(input.label)}</label>
    <input type="text" id="${id}" data-input="${escapeHtml(input.id)}" placeholder="${placeholder}">`;
}

function toolCard(addon, tool) {
  const inputs = tool.inputs.map(inputControl).join('');
  const card = document.createElement('div');
  card.className = 'setup-card addon-tool-card';
  card.dataset.addon = addon.id;
  card.dataset.tool = tool.id;
  card.innerHTML = `
    <div class="setup-card-head">
      <div class="setup-card-title">
        <strong>${escapeHtml(tool.label)}</strong>
        <div class="setup-card-state">${escapeHtml(addon.name)} by ${escapeHtml(addon.author?.name || 'unknown')}</div>
      </div>
    </div>
    <p class="setup-card-best">${escapeHtml(tool.hint)}</p>
    <form class="addon-tool-form">
      ${inputs}
      <button type="submit" class="addon-run">Run</button>
    </form>
    <pre class="addon-log"></pre>`;
  card.querySelector('form').addEventListener('submit', (e) => {
    e.preventDefault();
    runTool(addon.id, tool.id, card);
  });
  return card;
}

function gatherInputs(card) {
  const inputs = {};
  card.querySelectorAll('[data-input]').forEach((el) => {
    const key = el.dataset.input;
    if (el.type === 'checkbox') {
      inputs[key] = el.checked;
    } else {
      inputs[key] = el.value;
    }
  });
  return inputs;
}

async function runTool(addonId, toolId, card) {
  if (activeJobId) {
    alert('Another add-on run is already in progress.');
    return;
  }
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
      // Tell the Library to refresh.
      document.dispatchEvent(new CustomEvent('viewer:refresh-library'));
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

async function installedTools() {
  try {
    const response = await fetch('/api/addons');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const tools = [];
    for (const addon of payload.addons) {
      if (!addon.installed || !addon.tools) continue;
      for (const tool of addon.tools) tools.push({ addon, tool });
    }
    return tools;
  } catch (error) {
    return [];
  }
}

export async function openAddons(stage) {
  if (statusTimer) clearTimeout(statusTimer);
  activeJobId = null;
  const tools = await installedTools();
  const inner = document.createElement('div');
  inner.className = 'sheet-card';
  inner.innerHTML = `
    <div class="sheet-top"><h2>Add-ons</h2><button type="button" class="ghost" data-close>Close</button></div>
    <div class="addon-sheet-list"></div>`;
  const list = inner.querySelector('.addon-sheet-list');
  if (!tools.length) {
    list.innerHTML = '<p class="setup-card-best">No add-ons installed. Install them from Setup & Status.</p>';
  } else {
    for (const { addon, tool } of tools) list.appendChild(toolCard(addon, tool));
  }
  const el = document.createElement('div');
  el.className = 'sheet';
  el.appendChild(inner);
  stage.appendChild(el);
  const close = () => {
    if (statusTimer) clearTimeout(statusTimer);
    el.remove();
  };
  inner.querySelector('[data-close]').addEventListener('click', close);
  el.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}
