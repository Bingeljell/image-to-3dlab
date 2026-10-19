// The Steps panel's actions: each step's settings, its start call, and its progress card.
// The job APIs live on the server; the studio sends them the asset's files. Prop sheets split here too.

import { followJob, plainError, statusUrlFor, jobEnd } from './jobs.js';
import { freshSeed } from './create.js';
import { finishDetailSettings } from './library.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fileName = (path) => path?.split('/').pop() ?? '';

export async function fileFrom(url, name) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not read ${name} (${response.status})`);
  return new File([await response.blob()], name);
}

export async function postStart(url, body) {
  const response = await fetch(url, { method: 'POST', body, ...(typeof body === 'string' ? { headers: { 'Content-Type': 'application/json' } } : {}) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

/** The body for the step the asset can take next. `ctx` has engines, presets, url(). */
export function stepBody(stepId, asset, ctx) {
  if (stepId === 'picture') {
    return asset.prompt
      ? `<details class="adv redo"><summary>Draw another picture</summary><p class="hint">Opens Create with the same description. A new picture starts a new item in the Library.</p>
          <button class="primary" data-newpicture>Open Create</button></details>` : '';
  }
  if (stepId === 'model') {
    if (!asset.picture) return '<p class="hint">This asset has no picture on disk to build from.</p>';
    const [best, ...others] = ctx.engines;
    if (!best) return '<p class="hint">No 3D engine is installed yet. <a href="./setup.html">Install one in Setup</a>.</p>';
    const body = `<p class="hint">Builds a textured 3D model from the picture. Takes a few minutes.</p>
      <details class="adv"><summary>Advanced</summary>
        <label class="field">Engine<select data-k="engine">${[best, ...others].map((e, i) => `<option value="${esc(e.id)}">${i ? '' : 'Best quality: '}${esc(e.label)}</option>`).join('')}</select></label>
        <p class="hint" data-k="licence">${esc(best.license)}</p>
      </details>
      <button class="primary" data-run="model">${asset.model ? 'Make another 3D model' : 'Make 3D model'}</button>`;
    return asset.model ? redoBox('Make another 3D model', 'Builds a new 3D model from the same picture. It shows up as a new version; the current one is kept.', body) : body;
  }
  if ((stepId === 'finished') && ctx.blenderReady === false) {
    return `<p class="hint">${asset.kind === 'prop set' ? 'Splitting into props' : 'Finish'} needs Blender, a free app,
      and it isn't installed. <a href="https://www.blender.org/download/" target="_blank" rel="noopener">Get Blender</a>
      (drag it into Applications), then reopen this page. <a href="./setup.html">Setup</a> shows what's installed.</p>`;
  }
  if (stepId === 'finished' && asset.kind === 'prop set') {
    return `<p class="hint">Cuts the sheet into one model per prop, each cleaned up with lighter versions for far away. A few minutes.</p>
      <button class="primary" data-run="finished">Split into props</button>`;
  }
  if (stepId === 'finished') {
    const photo = asset.picture
      ? '<label class="check"><input type="checkbox" data-k="photo" checked><span>Pixel Match <span class="hint">keeps your picture\'s sharp details on the front</span></span></label>'
      : '<p class="hint">No picture on disk for this model, so Pixel Match is skipped: Finish cleans up the mesh and re-bakes its own texture.</p>';
    const skipped = !asset.finished && asset.rigged
      ? '<p class="hint">Skipped so far. Finishing now makes a new finished model; rig it again to use it.</p>' : '';
    const controls = `${skipped}${finishDetail()}${photo}
      <details class="adv"><summary>Advanced</summary>
        ${asset.picture ? '<label class="check"><input type="checkbox" data-k="edgefade" checked><span>Edge fade <span class="hint">Pixel Match fades out where one part sits in front of another, like an ear in front of a head. Turn it off if hair gets coloured specks or small eyes go dark, most often on Full detail.</span></span></label><label class="check"><input type="checkbox" data-k="nudge" checked><span>Nudge <span class="hint">Pixel Match first fits your picture onto the model. Turn it off if a feature shows up twice (a ghost eye) or slides off its place.</span></span></label>' : ''}
        <label class="field">Texture size<select data-k="texture"><option value="1024">1024 px</option><option value="2048" selected>2048 px</option><option value="4096">4096 px</option></select></label>
      </details>
      <button class="primary" data-run="finished">${asset.finished ? 'Finish again' : 'Finish model'}</button>`;
    return asset.finished
      ? `<details class="adv redo"><summary>Redo Finish</summary><p class="hint">Makes a new finished model from the raw one, say at another detail level. It shows up as a new version; the current one is kept.</p>${controls}</details>`
      : controls;
  }
  if (stepId === 'rigged') {
    const both = asset.finished && asset.model;
    const body = `<p class="hint">Adds a skeleton so it can move. About a minute. Works best on a character standing in an A-pose (a T-pose works too) with empty hands.</p>
      ${both ? '<label class="field">Rig which model<select data-k="rigsource"><option value="finished">The finished model</option><option value="model">The raw model (skips Finish)</option></select></label>' : ''}
      <p class="ask">Is this a humanoid: two arms, two legs, standing upright?</p>
      <div class="row"><button class="primary" data-run="rigged" data-humanoid="yes">Yes, rig it</button>
      <button class="ghost" data-run="rigged" data-humanoid="no" data-tip="Four-legged and other creatures can be rigged but not animated with the preset moves, and the rig may be wonky.">No, rig it anyway</button></div>`;
    return asset.rigged ? redoBox('Re-rig', 'Makes a new rig from the model. It shows up as a new version; the current rig and its moves are kept.', body) : body;
  }
  if (stepId === 'animated' && asset.fits_moves === false) {
    return `<p class="hint">Rigged. Preset moves are for humanoids only for now: four-legged and other creatures can be rigged,
      not animated, and the rig may be wonky. Download it to animate in Blender.</p>`;
  }
  if (stepId === 'animated') return movePicker(asset, ctx, asset.clips.length ? 'Add this move' : 'Animate');
  return '';
}

/** A step that is already done, offered again: folded away, and its result is a new version. */
function redoBox(title, note, body) {
  return `<details class="adv redo"><summary>${title}</summary><p class="hint">${note}</p>${body}</details>`;
}

// Detail: three tested triangle counts, or a custom one (the server takes 1,000 to 200,000). Tested on
// five very different models: 10k and 20k stay clean, 40k adds rough patches on some, 5k breaks.
function finishDetail() {
  return `<label class="field">Detail<select data-k="faces">
      <option value="10000">10,000 triangles · light</option>
      <option value="20000" selected>20,000 triangles · game-ready</option>
      <option value="40000">40,000 triangles · more detail</option>
      <option value="full">Full detail · every triangle, for renders (not games)</option>
      <option value="custom">Custom… (unexpected results)</option></select></label>
    <p class="hint" data-full hidden>Keeps the 3D model's own mesh (often near a million triangles) and paints it with
      Pixel Match: the sharpest look, in under a minute. Too heavy for a game; rig and animate a lighter version.</p>
    <div data-custom hidden>
      <label class="field">Triangles<input type="number" data-k="customFaces" min="1000" max="200000" step="1000" value="30000"></label>
      <p class="hint">Unexpected results: very low counts break up shapes and very high ones add rough patches. Try 10,000 to 40,000 first.</p>
    </div>`;
}

function movePicker(asset, ctx, cta) {
  const presets = ctx.presets.filter((p) => !p.take);
  if (!presets.length) return '<p class="hint">No preset moves are installed.</p>';
  const have = new Set(asset.clips.map((c) => c.name));
  return `<div class="moves" role="group" aria-label="Moves">${presets.map((p, i) => `<button class="chip" data-preset="${esc(p.id)}" aria-pressed="${i === 0}" ${have.has(p.id) ? 'data-tip="Already made: making it again replaces it"' : ''}>${esc(p.label)}${have.has(p.id) ? ' ✓' : ''}</button>`).join('')}</div>
    <details class="adv"><summary>Advanced</summary>
      <label class="field">Speed <output data-o="speed">1.00×</output><input type="range" data-k="speed" min="0.5" max="2" step="0.05" value="1"></label>
      <label class="field" data-tip="Bulky characters (armour, big shoulders) can hide their hands in their hips. This moves hanging arms outward a little; raised arms are left alone.">Arms away from body <output data-o="spread">0°</output><input type="range" data-k="spread" min="0" max="25" step="1" value="0"></label>
    </details>
    <button class="primary" data-run="animated">${cta}</button>`;
}

/** Wire a step card's controls. `ctx.onDone(asset)` reloads the Library afterwards. */
export function wireStep(card, stepId, asset, ctx) {
  const k = (key) => card.querySelector(`[data-k="${key}"]`);
  card.querySelectorAll('[data-preset]').forEach((chip) => chip.onclick = () => {
    card.querySelectorAll('[data-preset]').forEach((c) => c.setAttribute('aria-pressed', String(c === chip)));
  });
  const speed = k('speed'), spread = k('spread'), engine = k('engine');
  if (speed) speed.oninput = () => { card.querySelector('[data-o="speed"]').textContent = `${Number(speed.value).toFixed(2)}×`; };
  if (spread) spread.oninput = () => { card.querySelector('[data-o="spread"]').textContent = `${spread.value}°`; };
  const faces = k('faces');
  if (faces) faces.onchange = () => {
    card.querySelector('[data-custom]').hidden = faces.value !== 'custom';
    card.querySelector('[data-full]').hidden = faces.value !== 'full';
  };
  if (engine) engine.onchange = () => { card.querySelector('[data-k="licence"]').textContent = ctx.engines.find((e) => e.id === engine.value)?.license || ''; };

  card.querySelectorAll('[data-run]').forEach((button) => button.onclick = async () => {
    // read every setting now: the progress card replaces these controls as soon as it starts
    const settings = {
      backend: engine?.value || ctx.engines[0]?.id,
      detail: finishDetailSettings(faces?.value, k('customFaces')?.value), photo: !!k('photo')?.checked, edgeFade: k('edgefade')?.checked !== false, nudge: k('nudge')?.checked !== false, texture: Number(k('texture')?.value),
      preset: card.querySelector('[data-preset][aria-pressed="true"]')?.dataset.preset,
      speed: Number(speed?.value || 1), spread: Number(spread?.value || 0),
      rigSource: k('rigsource')?.value || 'finished',
    };
    card.dataset.lastPreset = settings.preset || '';
    const run = { model: () => startModel(asset, ctx, settings.backend),
      finished: () => (asset.kind === 'prop set' ? startSplit(asset)
        : startFinish(asset, ctx, { ...settings.detail, skip_photo: !settings.photo, skip_edge_fade: !settings.edgeFade, skip_nudge: !settings.nudge, texture_size: settings.texture })),
      rigged: () => startRig(asset, settings.rigSource),
      animated: () => startAnimate(asset, settings.preset, settings.speed, settings.spread) }[stepId];
    const note = button.dataset.humanoid === 'no'
      ? 'Rigging anyway. Preset moves only fit humanoids, so this one can be downloaded and animated in Blender. The rig may be wonky.' : '';
    await runInCard(card, run, ctx, stepId, note);
  });
}

async function startModel(asset, ctx, backend) {
  const form = new FormData();
  form.append('image', await fileFrom(ctx.url(asset.picture), fileName(asset.picture)));
  form.append('settings', JSON.stringify({ backend, seed: freshSeed() }));  // each try differs
  return { kind: 'generate', start: await postStart('/api/generate', form) };
}

async function startFinish(asset, ctx, settings) {
  const model = asset.model || asset.finished;
  const form = new FormData();
  form.append('asset', await fileFrom(ctx.url(model), fileName(model)));
  // no picture: the server skips Pixel Match and finishes the mesh alone
  if (asset.picture) form.append('image', await fileFrom(ctx.url(asset.picture), fileName(asset.picture)));
  form.append('settings', JSON.stringify(settings));
  return { kind: 'finish', start: await postStart('/api/finish', form) };
}

async function startSplit(asset) {
  const form = new FormData();
  form.append('generated', asset.model);
  form.append('settings', JSON.stringify({}));
  return { kind: 'props', start: await postStart('/api/props', form) };
}

async function startRig(asset, source = 'finished') {
  const form = new FormData();
  form.append('model', (source === 'model' ? asset.model : null) || asset.finished || asset.model);
  return { kind: 'animate', start: await postStart('/api/animate/rig', form) };
}

async function startAnimate(asset, preset, speed, arm_spread) {
  if (!preset) throw new Error('Pick a move first.');
  return { kind: 'animate', start: await postStart('/api/animate/play', JSON.stringify({ model: asset.rigged, preset, speed, arm_spread })) };
}

const settingsOf = (card) => card.dataset.lastPreset || null;

async function runInCard(card, starter, ctx, stepId, note) {
  const body = card.querySelector('.body');
  const keep = body.innerHTML;
  const show = (html) => { body.innerHTML = html; };
  show(`<div class="job"><div class="bigbar"><span></span></div><p class="job-msg">Starting…</p>${note ? `<p class="hint">${esc(note)}</p>` : ''}
    <pre class="log" aria-label="Live log"></pre><div class="row end"><button class="ghost" data-cancel>Cancel</button></div></div>`);
  ctx.onBusy?.(true);
  let started;
  try {
    started = await starter();
  } catch (error) {
    ctx.onBusy?.(false);
    show(`<div class="job failed"><p class="job-msg">${esc(plainError(error.message))}</p><div class="row end"><button class="ghost" data-back>Back</button></div></div>`);
    body.querySelector('[data-back]').onclick = () => { body.innerHTML = keep; wireStep(card, stepId, ctx.asset(), ctx); };
    return;
  }
  const { kind, start } = started;
  const statusUrl = statusUrlFor(kind, start);
  const bar = body.querySelector('.bigbar span'), msg = body.querySelector('.job-msg'), log = body.querySelector('.log');
  const startedAt = Date.now();
  body.querySelector('[data-cancel]').onclick = async (event) => {
    event.target.disabled = true;
    await fetch(`/api/${kind}/${start.job_id}/cancel`, { method: 'POST' }).catch(() => {});
  };
  const job = followJob(statusUrl, {
    onUpdate: ({ event, percent, log: tail, stalled }) => {
      const minutes = Math.floor((Date.now() - startedAt) / 60000), seconds = Math.floor((Date.now() - startedAt) / 1000) % 60;
      msg.textContent = `${event.message || event.phase || 'Working…'} · ${minutes}:${String(seconds).padStart(2, '0')}`;
      if (percent != null) bar.style.width = `${percent}%`;
      else bar.classList.add('busy');
      log.textContent = tail.split('\n').slice(-5).join('\n');
      card.classList.toggle('stalled', stalled);
      if (stalled && !body.querySelector('.stallnote')) msg.insertAdjacentHTML('afterend', '<p class="stallnote">No news from this step for a few minutes. It may be stuck: you can keep waiting or cancel.</p>');
    },
  });
  const final = await job.done;
  ctx.onBusy?.(false);
  if (final.status === 'done') { ctx.onDone?.(stepId, { preset: stepId === 'animated' ? settingsOf(card) : null }); return; }
  const { cancelled, why, log: tail } = jobEnd(final);
  show(cancelled
    ? `<div class="job"><p class="job-msg">${esc(why)}</p><div class="row end"><button class="ghost" data-back>Back</button></div></div>`
    : `<div class="job failed"><p class="job-msg">${esc(why)}</p>
    <details><summary class="hint">Show log</summary><pre class="log">${esc(tail.split('\n').slice(-30).join('\n'))}</pre></details>
    <div class="row end"><button class="ghost" data-copy>Copy details</button><button class="ghost" data-back>Back</button></div></div>`);
  body.querySelector('[data-back]').onclick = () => { body.innerHTML = keep; wireStep(card, stepId, ctx.asset(), ctx); };
  body.querySelector('[data-copy]')?.addEventListener('click', () => navigator.clipboard?.writeText(`${why}\n\n${tail}`));
}
