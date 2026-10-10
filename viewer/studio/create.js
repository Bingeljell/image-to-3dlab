// Create: one sheet that asks "start from" and "I want", then a runner that carries one
// thing through every step it needs, using the same job APIs as the Steps panel.

import { HAVES, WANTS, ALLOWED, STEP_LABELS, planFor, stopAfter, composePrompt, troubleFor } from './plan.js';
import { plainError } from './jobs.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const PICTURE_TYPES = /\.(png|jpe?g|webp)$/i;

/**
 * The picture job's settings from Create's choices. A new seed each time by default, so the
 * same prompt gives fresh tries; a fixed seed gives the same picture back.
 */
export function pictureSettings({ size = 768, steps = 8, newEachTime = true, seed = 42 } = {}, random = Math.random) {
  const side = [512, 768, 1024].includes(Number(size)) ? Number(size) : 768;
  const count = Math.max(1, Math.min(50, Math.round(Number(steps)) || 10));
  const fixed = Number.isFinite(Number(seed)) ? Math.trunc(Number(seed)) : 42;
  return { width: side, height: side, steps: count, seed: newEachTime ? freshSeed(random) : fixed };
}

/** A new random seed, so trying again gives a different result. */
export function freshSeed(random = Math.random) {
  return Math.floor(random() * 2147483647);
}

/** Show the Create sheet over the viewer. `onStart(plan)` runs it. */
export function openCreate({ stage, recipes, engines, presets, onStart }) {
  const sheet = document.createElement('div');
  sheet.className = 'sheet';
  stage.appendChild(sheet);
  const state = { have: 'idea', want: 'character', stop: null, tips: false, useRecipe: true, text: '', file: null,
    engine: engines[0]?.id, firstMove: presets.find((p) => !p.take)?.id,
    pic: { size: 768, steps: 8, newEachTime: true, seed: 42 } };
  const close = () => sheet.remove();

  const render = () => {
    if (ALLOWED[state.have][state.want] !== true) state.want = Object.keys(ALLOWED[state.have]).find((k) => ALLOWED[state.have][k] === true);
    const plan = planFor(state.have, state.want);
    if (state.stop && !plan.includes(state.stop)) state.stop = null;
    const active = stopAfter(plan, state.stop);
    const tipKey = state.have === 'idea' ? state.want : state.have === 'picture' ? 'upload' : null;
    const recipe = recipes?.goals?.[state.want];
    const quick = [['character', 'idea', 'Make a character'], ['prop', 'idea', 'Make props'], ['picture', 'idea', 'Make a picture'], ['character', 'model', 'Rig and animate my model']];
    const source = state.have === 'idea'
      ? `<label class="field">Describe it<textarea data-k="text" rows="3" placeholder="${state.want === 'character' ? 'A stylised orc blacksmith with short tusks, a braided beard and heavy boots' : state.want === 'picture' ? 'Concept art of an orc blacksmith at a glowing forge' : 'A weathered wooden tavern barrel with iron bands'}">${esc(state.text)}</textarea></label>
         ${recipe?.wording ? `<label class="check addline"><input type="checkbox" data-k="recipe" ${state.useRecipe ? 'checked' : ''}><span><b>We add:</b> ${esc(recipe.wording)}</span></label>` : ''}
         <div class="warnings" aria-live="polite"></div>`
      : `<div class="drop" data-drop tabindex="0" role="button">${state.file ? `<b>${esc(state.file.name)}</b><br><span class="hint">${(state.file.size / 1048576).toFixed(1)} MB · click or drop another to replace</span>` : `Drop ${state.have === 'picture' ? 'a picture (PNG, JPG, WebP)' : 'a 3D model (GLB)'} here, or click to choose`}</div>`;
    sheet.innerHTML = `<form class="sheet-card" novalidate>
      <div class="sheet-top"><h2>What should we <span>make?</span></h2><button type="button" class="ghost" data-close>Cancel</button></div>
      <div class="quick">${quick.map(([w, h, t]) => `<button type="button" class="qs${w === state.want && h === state.have ? ' on' : ''}" data-w="${w}" data-h="${h}">${t}</button>`).join('')}</div>
      <div class="q"><span class="qn">1</span><b>Start from</b>${tipKey ? `<button type="button" class="tipsbtn" data-tips aria-expanded="${state.tips}">Tips for best results</button>` : ''}</div>
      ${tipKey && state.tips ? `<div class="tipsbox"><b>${esc(recipes?.goals?.[tipKey]?.label || '')}</b><ul>${(recipes?.goals?.[tipKey]?.tips || []).map((t) => `<li>${t}</li>`).join('')}</ul></div>` : ''}
      <div class="seg haves">${Object.entries(HAVES).map(([k, v]) => `<button type="button" aria-pressed="${k === state.have}" data-have="${k}">${v}</button>`).join('')}</div>
      ${source}
      <input type="file" data-file hidden accept="${state.have === 'model' ? '.glb' : '.png,.jpg,.jpeg,.webp'}">
      <div class="q"><span class="qn">2</span><b>I want</b></div>
      <div class="wants">${Object.entries(WANTS).map(([k, v]) => { const ok = ALLOWED[state.have][k]; return `<button type="button" class="kind${ok === true ? '' : ' off'}" data-want="${k}" aria-pressed="${k === state.want}" ${ok === true ? '' : `disabled data-tip="${esc(ok)}"`}><b>${v.label}</b><small>${v.hint}</small></button>`; }).join('')}</div>
      <div class="q"><span class="qn">3</span><b>Steps</b><span class="hint">click a step to stop there</span></div>
      <div class="path">${plan.map((s, i) => `${i ? '<span class="arr">→</span>' : ''}<button type="button" class="pstep${active.includes(s) ? '' : ' skip'}" data-stop="${s}">${STEP_LABELS[s]}</button>`).join('')}</div>
      ${active.includes('animated') ? `<label class="field">First move<select data-k="move">${presets.filter((p) => !p.take).map((p) => `<option value="${esc(p.id)}" ${p.id === state.firstMove ? 'selected' : ''}>${esc(p.label)}</option>`).join('')}</select></label>` : ''}
      ${active.includes('picture') ? `<details class="adv"><summary>Advanced: picture settings</summary>
        <div class="picset">
          <label class="field">Size<select data-pic="size">${[[512, '512 (fastest)'], [768, '768 (recommended)'], [1024, '1024 (best detail, slowest)']].map(([v, t]) => `<option value="${v}" ${v === state.pic.size ? 'selected' : ''}>${t}</option>`).join('')}</select></label>
          <label class="field">Steps<input type="number" data-pic="steps" min="1" max="50" value="${state.pic.steps}"><span class="hint">10 is a good start; each extra step adds the same time again.</span></label>
          <label class="check"><input type="checkbox" data-pic="newEachTime" ${state.pic.newEachTime ? 'checked' : ''}><span>New picture each time <span class="hint">untick to reuse a seed and get the same picture back</span></span></label>
          ${state.pic.newEachTime ? '' : `<label class="field">Seed<input type="number" data-pic="seed" step="1" value="${state.pic.seed}"></label>`}
        </div></details>` : ''}
      ${active.includes('model') && engines.length > 1 ? `<details class="adv"><summary>Advanced: 3D engine</summary><label class="field">Engine<select data-k="engine">${engines.map((e, i) => `<option value="${esc(e.id)}" ${e.id === state.engine ? 'selected' : ''}>${i ? '' : 'Best quality: '}${esc(e.label)}</option>`).join('')}</select></label></details>` : ''}
      <p class="error" data-error hidden></p>
      <div class="row end"><span class="hint">${active.length} step${active.length === 1 ? '' : 's'}${active.includes('picture') ? ' · the picture uses Qwen-Image, run on this machine' : ''}</span><button type="submit" class="primary">Make it</button></div>
    </form>`;
    wire(active);
  };

  const warn = () => {
    const box = sheet.querySelector('.warnings');
    if (!box) return;
    box.innerHTML = troubleFor(state.text, state.want, recipes).map((w) => `<p class="warning">“${esc(w.word)}”: ${esc(w.message)}</p>`).join('');
  };

  const takeFile = (file) => {
    if (!file) return;
    state.file = file;
    state.have = /\.glb$/i.test(file.name) ? 'model' : PICTURE_TYPES.test(file.name) ? 'picture' : state.have;
    render();
  };

  function wire(active) {
    const q = (sel) => sheet.querySelector(sel);
    q('[data-close]').onclick = close;
    sheet.querySelectorAll('[data-have]').forEach((b) => b.onclick = () => { state.have = b.dataset.have; state.file = null; state.stop = null; render(); });
    sheet.querySelectorAll('[data-want]').forEach((b) => b.onclick = () => { state.want = b.dataset.want; state.stop = null; render(); });
    sheet.querySelectorAll('[data-stop]').forEach((b) => b.onclick = () => { state.stop = state.stop === b.dataset.stop ? null : b.dataset.stop; render(); });
    sheet.querySelectorAll('.qs').forEach((b) => b.onclick = () => { state.want = b.dataset.w; state.have = b.dataset.h; state.file = null; state.stop = null; render(); });
    q('[data-tips]')?.addEventListener('click', () => { state.tips = !state.tips; render(); });
    const text = q('[data-k="text"]');
    if (text) { text.oninput = () => { state.text = text.value; warn(); }; warn(); }
    q('[data-k="recipe"]')?.addEventListener('change', (e) => { state.useRecipe = e.target.checked; });
    q('[data-k="move"]')?.addEventListener('change', (e) => { state.firstMove = e.target.value; });
    q('[data-k="engine"]')?.addEventListener('change', (e) => { state.engine = e.target.value; });
    sheet.querySelectorAll('[data-pic]').forEach((el) => el.addEventListener('change', () => {
      const key = el.dataset.pic;
      state.pic[key] = el.type === 'checkbox' ? el.checked : Number(el.value);
      if (key === 'newEachTime') { render(); sheet.querySelector('.picset').closest('details').open = true; }  // show or hide the seed box
    }));
    const picker = q('[data-file]'), drop = q('[data-drop]');
    if (drop) {
      drop.onclick = () => picker.click();
      drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); picker.click(); } };
    }
    picker.onchange = () => takeFile(picker.files[0]);
    q('form').onsubmit = (event) => {
      event.preventDefault();
      const error = q('[data-error]');
      const fail = (message) => { error.textContent = message; error.hidden = false; };
      if (state.have === 'idea' && !state.text.trim()) return fail('Describe what you want first.');
      if (state.have !== 'idea' && !state.file) return fail(`Choose ${state.have === 'model' ? 'a GLB' : 'a picture'} first.`);
      close();
      onStart({ have: state.have, want: state.want, steps: active, file: state.file, engine: state.engine, firstMove: state.firstMove,
        description: state.text.trim(), prompt: state.have === 'idea' ? composePrompt(state.text, state.want, recipes, state.useRecipe) : null,
        pictureSettings: active.includes('picture') ? pictureSettings(state.pic) : null,
        modelSeed: active.includes('model') ? freshSeed() : null });  // a new 3D try each time, like the picture
    };
    text?.focus();
  }

  // drop a file anywhere on the sheet
  sheet.addEventListener('dragover', (e) => e.preventDefault());
  sheet.addEventListener('drop', (e) => { e.preventDefault(); takeFile(e.dataTransfer.files[0]); });
  sheet.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  render();
}

/**
 * What changed between two snapshots of a server-side run, as the calls the progress view
 * understands: ['step', s], ['progress', u], ['done', s, made], ['fail', s, why, log], ['finish'].
 */
export function chainEvents(before, after) {
  const out = [];
  const seen = new Set(before?.done || []);
  for (const step of after.done) {
    if (!seen.has(step)) { out.push(['done', step, after.made]); seen.add(step); }
  }
  const running = after.status === 'running';
  if (after.current && after.current !== before?.current && running) out.push(['step', after.current]);
  if (running && after.current) {
    out.push(['progress', { event: { message: after.message }, percent: after.percent, log: after.full_log, stalled: after.stalled }]);
  }
  if (after.status === 'cancelled') out.push(['cancel', after.current]);
  if (after.status === 'error') out.push(['fail', after.current, after.error, after.full_log]);
  if (after.status === 'done') out.push(['finish']);
  return out;
}

/** The form a plan travels to the server in. */
export function planForm(plan, made = {}) {
  const form = new FormData();
  form.append('title', plan.description || plan.file?.name || 'New asset');
  form.append('steps', JSON.stringify(plan.steps));
  form.append('want', plan.want || 'character');
  if (plan.prompt) form.append('prompt', plan.prompt);
  if (plan.engine) form.append('engine', plan.engine);
  if (plan.firstMove) form.append('first_move', plan.firstMove);
  if (plan.pictureSettings) form.append('picture_settings', JSON.stringify(plan.pictureSettings));
  if (plan.modelSeed != null) form.append('model_seed', String(plan.modelSeed));
  form.append('made', JSON.stringify(made));
  if (plan.file) form.append('file', plan.file);
  return form;
}

/**
 * Carry one thing through its steps. The run itself happens on the server (closing the tab
 * does not stop it); this follows it. `ui` draws progress; `ctx.reload(match)` refreshes the
 * Library and selects the asset `match(asset)` finds. Pass `attach` (a run id) to pick up a
 * run that is already going, as a reopened tab does.
 */
export async function runChain(plan, { ui, ctx, attach = null, fetchImpl = fetch, interval = 1200 }) {
  let made = {};
  const mine = (asset) => (made.picture && asset.picture === made.picture) || (made.model && asset.model === made.model)
    || (made.finished && asset.finished === made.finished) || (made.rigged && asset.rigged === made.rigged);
  ui.begin(plan);
  let id = attach;
  if (!id) {
    try {
      const response = await fetchImpl('/api/chains', { method: 'POST', body: planForm(plan) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `the server answered ${response.status}`);
      id = payload.id;
    } catch (error) {
      ui.fail(plan.steps[0], plainError(error.message), '');
      return false;
    }
  }
  ui.cancelWith(() => fetchImpl(`/api/chains/${id}/cancel`, { method: 'POST' }).catch(() => {}));
  let before = null;
  for (;;) {
    let after;
    try {
      after = await (await fetchImpl(`/api/chains/${id}`)).json();
    } catch {
      await new Promise((r) => setTimeout(r, interval));  // a missed poll is not a failure
      continue;
    }
    if (!after?.steps) { ui.fail(plan.steps[0], 'This run is no longer known to the server (it was restarted). Anything finished is in your Library.', ''); return false; }
    made = after.made || {};
    for (const [kind, ...args] of chainEvents(before, after)) {
      if (kind === 'done') { await ctx.reload(mine); ui.done(args[0], { ...made }); }
      else if (kind === 'fail') { ui.fail(args[0], args[1], args[2] || ''); return false; }
      else if (kind === 'cancel') { ui.cancelled(args[0]); return false; }
      else if (kind === 'finish') { await ctx.reload(mine); ui.finish(mine); return true; }
      else ui[kind](...args);
    }
    before = after;
    await new Promise((r) => setTimeout(r, interval));
  }
}
