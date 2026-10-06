// Create: one sheet that asks "start from" and "I want", then a runner that carries one
// thing through every step it needs, using the same job APIs as the Steps panel.

import { HAVES, WANTS, ALLOWED, STEP_LABELS, planFor, stopAfter, composePrompt, troubleFor } from './plan.js';
import { followJob, plainError, statusUrlFor } from './jobs.js';
import { fileFrom, postStart } from './steps.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const PICTURE_TYPES = /\.(png|jpe?g|webp)$/i;

/** Show the Create sheet over the viewer. `onStart(plan)` runs it. */
export function openCreate({ stage, recipes, engines, presets, onStart }) {
  const sheet = document.createElement('div');
  sheet.className = 'sheet';
  stage.appendChild(sheet);
  const state = { have: 'idea', want: 'character', stop: null, tips: false, useRecipe: true, text: '', file: null,
    engine: engines[0]?.id, firstMove: presets.find((p) => !p.take)?.id };
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
        description: state.text.trim(), prompt: state.have === 'idea' ? composePrompt(state.text, state.want, recipes, state.useRecipe) : null });
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
 * Carry one thing through its steps. `ui` draws progress; `ctx.reload(match)` refreshes the
 * Library and selects the asset `match(asset)` finds. Stops at the first failure.
 */
export async function runChain(plan, { ui, ctx }) {
  const made = { picture: null, model: null, finished: null, rigged: null };
  const url = (rel) => ctx.url(rel);
  const file = (rel) => fileFrom(url(rel), rel.split('/').pop());
  const steps = {
    picture: async () => ({ kind: 'image', start: await postStart('/api/image', JSON.stringify({ prompt: plan.prompt, settings: {} })) }),
    model: async () => {
      const form = new FormData();
      form.append('image', made.picture ? await file(made.picture) : plan.file);
      form.append('settings', JSON.stringify({ backend: plan.engine }));
      return { kind: 'generate', start: await postStart('/api/generate', form) };
    },
    finished: async () => {
      const form = new FormData();
      form.append('asset', await file(made.model));
      form.append('image', made.picture ? await file(made.picture) : plan.file);
      form.append('settings', JSON.stringify({}));
      return { kind: 'finish', start: await postStart('/api/finish', form) };
    },
    rigged: async () => {
      const form = new FormData();
      const source = made.finished || made.model;
      if (source) form.append('model', source); else form.append('asset', plan.file);
      return { kind: 'animate', start: await postStart('/api/animate/rig', form) };
    },
    animated: async () => ({ kind: 'animate', start: await postStart('/api/animate/play', JSON.stringify({ model: made.rigged, preset: plan.firstMove })) }),
    split: async () => {
      const form = new FormData();
      form.append('generated', made.model);
      form.append('settings', JSON.stringify({}));
      return { kind: 'props', start: await postStart('/api/props', form) };
    },
  };
  // what each finished step leaves behind, for the next one
  const record = (step, final) => {
    const event = final.last_event || {};
    if (step === 'picture') made.picture = final.picture;
    if (step === 'model') made.model = final.model;
    if (step === 'finished') made.finished = (event.result_url || '').replace(/^\/output\//, '') || null;
    if (step === 'rigged') made.rigged = event.path || null;
  };
  const mine = (asset) => (made.picture && asset.picture === made.picture) || (made.model && asset.model === made.model)
    || (made.finished && asset.finished === made.finished) || (made.rigged && asset.rigged === made.rigged);

  ui.begin(plan);
  for (const step of plan.steps) {
    ui.step(step);
    let started;
    try {
      started = await steps[step]();
    } catch (error) {
      ui.fail(step, plainError(error.message), '');
      return false;
    }
    ui.cancelWith(() => fetch(`/api/${started.kind}/${started.start.job_id}/cancel`, { method: 'POST' }).catch(() => {}));
    const job = followJob(statusUrlFor(started.kind, started.start), { onUpdate: (u) => ui.progress(u) });
    const final = await job.done;
    if (final.status !== 'done') {
      ui.fail(step, final.status === 'cancelled' ? 'Cancelled. Everything finished before this step is kept.' : plainError(final.error || final.last_event?.message || final.log_tail || ''), final.log_tail || '');
      return false;
    }
    record(step, final);
    await ctx.reload(mine);
    ui.done(step, { ...made });
  }
  ui.finish(mine);
  return true;
}
