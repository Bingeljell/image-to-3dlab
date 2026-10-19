// Skins: new paint on a finished Pixal3D model, same mesh underneath (viewer/skins_api.py).
// One card under the steps: pick a skin to view, or make one. The person approves the new
// front picture, then the back, before anything is painted; asking again takes a correction.

import { plainError, readableLog } from './jobs.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const POLL_MS = 2000;

/** Only finished Pixal3D models keep the shape a skin repaints. */
export function canSkin(asset) {
  return !!(asset?.finished && asset?.model && /__pixal3d__/.test(asset.model));
}

/** A model's skins that are still being made (not finished), newest first. */
export function draftsFor(asset, skins) {
  return (skins || []).filter((s) => s.base?.finished === asset.finished && s.status !== 'done')
    .sort((a, b) => b.created - a.created);
}

/** Is anything running for this skin right now: a picture being drawn, or the build? */
export function skinBusy(skin) {
  const job = skin.picture_job || {};
  return skin.status === 'building' || job.status === 'queued' || job.status === 'running';
}

/** The newest try for a side, or null. */
export function latestTry(skin, side) {
  const tries = skin.tries?.[side] || [];
  return tries.length ? tries[tries.length - 1] : null;
}

const SIDE_WORDS = { front: 'front', back: 'back' };

/** How far the picture being drawn is (0-100), or null before Qwen reports a step. */
export function drawPercent(skin) {
  const value = Number(skin.picture_job?.last_event?.percent);
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

/** The working area for one skin in the making. `url` turns an output/ path into a served URL. */
export function draftHtml(skin, url) {
  const name = `<p class="skin-name"><b>${esc(skin.name)}</b>${skin.status === 'building' ? '' : ' <button type="button" class="linkish" data-skin-discard>Discard</button>'}</p>`;
  if (skin.status === 'building') {
    const event = skin.build_job?.last_event || {};
    const pct = Number(event.overall_pct);
    return `${name}<div class="job"><div class="bigbar"><span ${Number.isFinite(pct) ? `style="width:${pct}%"` : 'class="busy"'}></span></div>
      <p class="job-msg">${esc(event.message || 'Repainting the model, then finishing it…')}</p>
      <pre class="log">${esc(readableLog(skin.build_job?.log_tail || '', 4))}</pre>
      <div class="row end"><button class="ghost small" data-skin-cancel>Cancel</button></div></div>`;
  }
  if (skin.status === 'error') {
    return `${name}<div class="job failed"><p class="job-msg">${esc(plainError(skin.error || ''))}</p>
      <div class="row end"><button class="primary" data-skin-build>Try building again</button></div></div>`;
  }
  const side = skin.next_side;
  if (!side && skin.ready) {
    const thumbs = ['front', 'back'].filter((s) => skin.approved?.[s])
      .map((s) => `<figure><img src="${esc(url(skin.approved[s]))}" alt="Approved ${s} picture"><figcaption>${s === 'front' ? 'Front' : 'Back'} ✓</figcaption></figure>`).join('');
    return `${name}<div class="skin-pair">${thumbs}</div>
      <p class="hint">Both pictures approved. Painting the model takes a few minutes; its rig and moves will still fit.</p>
      <button class="primary" data-skin-build>Paint the model</button>`;
  }
  const attempt = latestTry(skin, side);
  const drawing = skinBusy(skin);
  const failed = skin.picture_job?.status === 'error' && !drawing;
  const original = skin.base?.[side];
  // `ready` comes from the server; one from before it said so is trusted (its Approve still checks)
  const fresh = attempt && attempt.ready !== false && !drawing && !failed;
  const shown = fresh ? `<img src="${esc(url(attempt.picture))}?v=${attempt.number}" alt="New ${side} picture, try ${attempt.number}">`
    : `<div class="skin-wait">${drawing ? 'Drawing…' : failed || attempt ? 'Could not draw. Ask again.' : ''}</div>`;
  const steps = skin.needs_back ? `<p class="hint">Step ${side === 'front' ? 1 : 2} of 2: the ${SIDE_WORDS[side]} picture.</p>` : '';
  return `${name}${steps}
    <div class="skin-pair">
      <figure><img src="${esc(url(original))}" alt="Original ${side} picture"><figcaption>Original</figcaption></figure>
      <figure>${shown}<figcaption>New${attempt ? ` · try ${attempt.number}` : ''}</figcaption></figure>
    </div>
    ${failed ? `<p class="hint">${esc(plainError(skin.picture_job?.error || ''))}</p>` : ''}
    ${drawing ? `<div class="bigbar"><span ${drawPercent(skin) != null ? `style="width:${drawPercent(skin)}%"` : 'class="busy"'}></span></div><p class="hint">Qwen is redrawing the ${SIDE_WORDS[side]} picture, about a minute or two.</p>` : `
    ${fresh ? `<button class="ghost" data-skin-large="${side}">View large</button>
      <p class="hint">Check that only what you asked for changed: same pose, same parts. Anything extra gets painted on too.</p>
      <button class="primary" data-skin-approve="${side}">Looks right: use this ${SIDE_WORDS[side]}</button>` : ''}
    <label class="field">Ask again, with a correction (optional)<textarea data-k="redo" rows="2" placeholder="Gold skirt, but keep the belt blue"></textarea></label>
    <button class="ghost" data-skin-redo="${side}">Ask again</button>`}`;
}

/** What the big viewer should show for a skin: the original and the newest drawn picture of the
 * side waiting for approval, or null when there is nothing new to look at. */
export function largePair(skin) {
  const side = skin.next_side;
  const attempt = side ? latestTry(skin, side) : null;
  if (!attempt || attempt.ready === false || skinBusy(skin)) return null;
  return { original: skin.base?.[side], fresh: attempt.picture, version: attempt.number,
           caption: `${skin.name}: the ${SIDE_WORDS[side]} picture, try ${attempt.number}. Approve it in Steps, or ask again.` };
}

/** The whole card: viewing skins, the ones in the making, and starting a new one. */
export function skinCardHtml(asset, drafts, viewing, url) {
  const skins = asset.skins || [];
  const picker = skins.length ? `<label class="field">Show<select data-skin-pick>
      <option value="">Original paint</option>
      ${skins.map((s) => `<option value="${esc(s.id)}" ${viewing === s.id ? 'selected' : ''}>${esc(s.name)}</option>`).join('')}
    </select></label>${viewing ? '<button type="button" class="linkish" data-skin-delete>Delete this skin</button>' : ''}
    <button type="button" class="ghost" data-skin-all data-tip="One GLB with the original paint and every skin, switchable in Blender, three.js and other viewers.">Download with every skin</button>` : '';
  const working = drafts.map((d) => `<div class="skin-draft" data-draft="${esc(d.id)}">${draftHtml(d, url)}</div>`).join('');
  const count = skins.length ? `${skins.length} skin${skins.length > 1 ? 's' : ''} · same model, new paint` : 'Same model, new paint';
  return `<li class="step skins" data-step="skins"><header><span class="dot">◐</span><span style="min-width:0"><h3>Skins</h3><div class="sum">${count}</div></span></header>
    <div class="body">${picker}${working}
      <details class="adv"${!skins.length && !drafts.length ? ' open' : ''}><summary>New skin</summary>
        <p class="hint">Say what to change. You approve the new front picture, then the back, before anything is painted. The model's rig and moves still fit.</p>
        <label class="field">What to change<textarea data-k="change" rows="2" placeholder="Make the skirt gold"></textarea></label>
        <button class="primary" data-skin-new>Draw the new picture</button>
      </details>
    </div></li>`;
}

async function call(path, body) {
  const response = await fetch(path, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

/**
 * Put the Skins card at the end of `list` for `asset` and keep it live.
 * ctx: { url(rel), viewing(): skin id or null, view(skin|null), onDone(skin), onBusy(bool) }.
 */
export async function mountSkins(list, asset, ctx) {
  if (!canSkin(asset)) return;
  let all = [];
  try { all = (await call('/api/skins')).skins || []; } catch { /* the card still offers a new skin */ }
  if (ctx.asset() !== asset) return;   // the person moved on while this loaded
  list.querySelector('[data-step="skins"]')?.remove();
  list.insertAdjacentHTML('beforeend', skinCardHtml(asset, draftsFor(asset, all), ctx.viewing(), ctx.url));
  const card = list.querySelector('[data-step="skins"]');
  const note = (message) => {
    card.querySelector('.skin-note')?.remove();
    card.querySelector('.body').insertAdjacentHTML('afterbegin', `<p class="hint skin-note">${esc(plainError(message))}</p>`);
  };

  const pick = card.querySelector('[data-skin-pick]');
  if (pick) pick.onchange = () => ctx.view((asset.skins || []).find((s) => s.id === pick.value) || null);
  card.querySelector('[data-skin-all]')?.addEventListener('click', async (e) => {
    e.target.disabled = true;
    const file = asset.rigged || asset.finished;
    try {
      const made = await call(`/api/skins/all?asset=${encodeURIComponent(asset.id)}&file=${encodeURIComponent(file)}`);
      const link = Object.assign(document.createElement('a'), { href: ctx.url(made.file), download: made.file.split('/').pop() });
      document.body.appendChild(link); link.click(); link.remove();
    } catch (error) { note(error.message); }
    e.target.disabled = false;
  });
  card.querySelector('[data-skin-delete]')?.addEventListener('click', async (e) => {
    if (!confirm('Delete this skin from your disk? This cannot be undone.')) return;
    e.target.disabled = true;
    try { await call(`/api/skins/${ctx.viewing()}/discard`, {}); ctx.onDone?.(null); } catch (error) { note(error.message); e.target.disabled = false; }
  });

  card.querySelector('[data-skin-new]').onclick = async (event) => {
    const change = card.querySelector('[data-k="change"]').value.trim();
    if (!change) { note('Say what to change, for example "make the skirt gold".'); return; }
    event.target.disabled = true;
    try {
      const made = await call('/api/skins', { asset: asset.id, instruction: change });
      follow(made.skin.id);
    } catch (error) { note(error.message); event.target.disabled = false; }
  };

  card.querySelectorAll('[data-draft]').forEach((box) => wireDraft(box, box.dataset.draft));

  function wireDraft(box, id) {
    const act = async (button, path, body) => {
      button.disabled = true;
      try { await call(path, body); follow(id); } catch (error) { note(error.message); button.disabled = false; }
    };
    box.querySelector('[data-skin-approve]')?.addEventListener('click', (e) =>
      act(e.target, `/api/skins/${id}/approve`, { side: e.target.dataset.skinApprove }));
    box.querySelector('[data-skin-redo]')?.addEventListener('click', (e) =>
      act(e.target, `/api/skins/${id}/try`, { side: e.target.dataset.skinRedo, instruction: box.querySelector('[data-k="redo"]')?.value.trim() || undefined }));
    box.querySelector('[data-skin-build]')?.addEventListener('click', (e) => act(e.target, `/api/skins/${id}/build`, {}));
    box.querySelector('[data-skin-discard]')?.addEventListener('click', async (e) => {
      if (!confirm('Discard this skin and its pictures? This cannot be undone.')) return;
      e.target.disabled = true;
      try { await call(`/api/skins/${id}/discard`, {}); box.remove(); } catch (error) { note(error.message); e.target.disabled = false; }
    });
    box.querySelector('[data-skin-large]')?.addEventListener('click', async () => {
      const skin = await call(`/api/skins/${id}`).catch(() => null);
      if (skin) showLarge(skin);
    });
    box.querySelectorAll('.skin-pair img').forEach((img) => { img.style.cursor = 'zoom-in'; img.onclick = () => box.querySelector('[data-skin-large]')?.click(); });
    box.querySelector('[data-skin-cancel]')?.addEventListener('click', async (e) => {
      e.target.disabled = true;
      const skin = await call(`/api/skins/${id}`).catch(() => null);
      if (skin?.build_job) await call(`/api/skins/build/${skin.build_job.job_id}/cancel`, {}).catch(() => {});
      follow(id);
    });
    if (box.dataset.following !== '1') {
      // a reopened page picks up a skin that was drawing or building
      call(`/api/skins/${id}`).then((skin) => { if (skinBusy(skin)) follow(id); }).catch(() => {});
    }
  }

  // Redraw one skin's area until nothing runs for it; then hand a finished skin to the page.
  async function follow(id) {
    let box = card.querySelector(`[data-draft="${id}"]`);
    if (!box) {
      card.querySelector('details.adv')?.removeAttribute('open');
      card.querySelector('.body details.adv').insertAdjacentHTML('beforebegin', `<div class="skin-draft" data-draft="${esc(id)}"></div>`);
      box = card.querySelector(`[data-draft="${id}"]`);
    }
    box.dataset.following = '1';
    ctx.onBusy?.(true);
    let skin;
    for (;;) {
      if (!box.isConnected) { ctx.onBusy?.(false); return; }
      try { skin = await call(`/api/skins/${id}`); } catch { await new Promise((r) => setTimeout(r, POLL_MS)); continue; }
      box.innerHTML = draftHtml(skin, ctx.url);
      wireDraft(box, id);
      if (!skinBusy(skin)) break;
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
    box.dataset.following = '';
    ctx.onBusy?.(false);
    if (skin.status === 'done') ctx.onDone?.(skin);
    else showLarge(skin);   // a new picture just landed: put it in front of the person, full size
  }

  function showLarge(skin) {
    const pair = largePair(skin);
    if (pair) ctx.compare?.(pair.original, pair.fresh, pair.caption, pair.version);
  }
}
