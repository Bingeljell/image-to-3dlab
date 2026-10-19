// Activity and About: quiet sheets over the viewer. Activity is the full history of every
// step that ended (finished, stopped or failed), newest first; failures live here, not in
// the way. About says what this is, who built what, and where to find us.

import { STEP_LABELS } from './plan.js';

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** "4 min 12 s", "38 s", "1 h 3 min": how long a step took, in words. */
export function duration(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
  return `${Math.floor(s / 3600)} h${Math.floor((s % 3600) / 60) ? ` ${Math.floor((s % 3600) / 60)} min` : ''}`;
}

/** History records grouped by day ("Today", "Yesterday", else the date), keeping their order. */
export function byDay(records, today = new Date()) {
  const key = (d) => d.toISOString().slice(0, 10);
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const groups = [];
  for (const record of records) {
    const day = String(record.time || '').slice(0, 10);
    const label = day === key(today) ? 'Today' : day === key(yesterday) ? 'Yesterday' : day || 'Earlier';
    if (!groups.length || groups[groups.length - 1].label !== label) groups.push({ label, items: [] });
    groups[groups.length - 1].items.push(record);
  }
  return groups;
}

/** Put the sheet over the viewer away (Create, Activity, How to use, About), if one is there.
 *  They are opaque and cover the whole stage, so anything that shows a model has to call this
 *  first: the model would otherwise load behind one and the click would look like nothing. */
export function closeSheet(stage) {
  const el = stage.querySelector('.sheet');
  if (!el) return;
  // a sheet that set something up outside itself (Create's paste listener) undoes it here,
  // however it is closed: its own button, or a Library click that clears the stage
  el.onSheetClose?.();
  el.remove();
}

function sheet(stage, inner) {
  closeSheet(stage);
  const el = document.createElement('div');
  el.className = 'sheet';
  el.innerHTML = inner;
  stage.appendChild(el);
  const close = () => el.remove();
  el.querySelector('[data-close]')?.addEventListener('click', close);
  el.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  el.querySelector('[data-close]')?.focus();
  return el;
}

export async function openActivity(stage) {
  const el = sheet(stage, `<div class="sheet-card">
    <div class="sheet-top"><h2>Activity</h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">Every step that has run here, newest first: finished, stopped or failed. Failures keep their log, so nothing is lost.</p>
    <div class="activity" aria-live="polite"><p class="meta">Loading…</p></div></div>`);
  const box = el.querySelector('.activity');
  let records = [];
  try {
    records = (await (await fetch('/api/activity')).json()).history || [];
  } catch {
    box.innerHTML = '<p class="meta">The history could not be read. Restart AssetFurnace and try again.</p>';
    return;
  }
  if (!records.length) {
    box.innerHTML = '<p class="meta">Nothing yet. Everything you make from now on is listed here.</p>';
    return;
  }
  // one line per step; click it for the prompt, what it made and the full log
  box.innerHTML = byDay(records).map((group) => `<h3>${escape(group.label)}</h3><ul class="act-list">${group.items.map((r) => `
    <li class="act ${escape(r.result)}"><details><summary title="${escape(activityLine(r))}">${activityParts(r).map((part, i) => `<span class="${i ? 'meta' : 'act-title'}">${escape(part)}</span>`).join('')}</summary>
      <div class="act-more">
        ${r.result === 'error' && r.note ? `<p class="act-why">${escape(r.note)}</p>` : ''}
        ${r.result === 'error' && r.chain ? `<p><button type="button" class="ghost small" data-report-chain="${escape(r.chain)}" data-report-step="${escape(r.step)}">Report this</button></p>` : ''}
        ${r.prompt ? `<p><span class="meta">Prompt</span><br>${escape(r.prompt)}</p>` : ''}
        ${r.made ? `<p><span class="meta">Made</span><br>${escape(r.made)}</p>` : ''}
        ${r.log ? `<pre class="log">${escape(r.log)}</pre>` : '<p class="meta">No log kept for this step.</p>'}
      </div></details></li>`).join('')}</ul>`).join('');
  box.querySelectorAll('[data-report-chain]').forEach((button) => button.addEventListener('click', () =>
    openReport(stage, { chain: button.dataset.reportChain, step: button.dataset.reportStep })));
}

// How a finished thing is marked. Shared with the Setup page's progress panel, which has the
// same three endings and should not invent its own glyphs.
export const MARK = { done: '✓', error: '!', cancelled: '■' };

/** One Activity line: "✓ Rig · blocky toy knight · 17:19 · 42 s" (and "· stopped" or
 *  "· failed" when it did not finish). */
export function activityLine(record) {
  return activityParts(record).join(' · ');
}

/** The line in two parts: what ran (shortened to fit) and when and how long (always shown). */
export function activityParts(record) {
  const ending = { cancelled: ' · stopped', error: ' · failed' }[record.result] || '';
  return [`${MARK[record.result] || '·'} ${STEP_LABELS[record.step] || record.step} · ${record.title}`,
    `${String(record.time || '').slice(11, 16)} · ${duration(record.seconds)}${ending}`];
}

const CREDITS = [
  ['Pixal3D', 'raven38/pixal3d.cpp', 'https://github.com/raven38/pixal3d.cpp', 'MIT; DINOv3 encoder under its own licence'],
  ['TRELLIS.2', 'Microsoft, on trellis2-apple', 'https://huggingface.co/microsoft/TRELLIS.2-4B', 'MIT; DINOv3 encoder under its own licence'],
  ['Hunyuan3D', 'Tencent, via ZimengXiong and dgrauet’s MLX ports', 'https://github.com/ZimengXiong/Hunyuan3D-MLX', 'Tencent Hunyuan Community License'],
  ['Stable Fast 3D', 'Stability AI', 'https://github.com/Stability-AI/stable-fast-3d', 'Stability AI Community License'],
  ['Qwen-Image', 'Qwen, run by stable-diffusion.cpp', 'https://huggingface.co/Qwen/Qwen-Image-2.1', 'Qwen licence'],
  ['SkinTokens', 'VAST-AI (auto-rig)', 'https://github.com/VAST-AI-Research/SkinTokens', 'MIT'],
  ['Kimodo', 'NVIDIA (the preset moves)', 'https://github.com/nv-tlabs/kimodo', 'Apache-2.0 code, NVIDIA Open Model License'],
  ['MLX, rembg, TinyCLIP, three.js', 'and the people behind them', 'https://github.com/ml-explore/mlx', 'MIT'],
];

export async function openAbout(stage) {
  const el = sheet(stage, `<div class="sheet-card about">
    <div class="sheet-top"><h2>About <span>AssetFurnace</span></h2><button type="button" class="ghost" data-close>Close</button></div>
    <p>Hi, I’m Bingeljell. I love building stuff, specially in and around the vicinity of video games. AssetFurnace is my
      attempt at trying to empower game devs to be able to do more and better.</p>
    <p>This is a young new project and will only survive when people like you use it, critique it, share feedback and most
      of all spread the word. So if you use this and like it, share it with a friend. If you have feedback, write up an issue
      on <a href="https://github.com/Bingeljell/image-to-3dlab/issues" target="_blank" rel="noopener">GitHub</a>, we’re on
      <a href="https://discord.gg/3D4bcEhGx" target="_blank" rel="noopener">Discord</a>, or you can tag me in a post on
      <a href="https://x.com/bingeljell" target="_blank" rel="noopener">X</a>. Most of all, we’d love contributions.</p>
    <p>The goal is to enable you to make game-ready 3D characters and props on your own machine: from an idea, a picture or a model, to a rigged,
      moving character you can drop into Godot, Unity, Unreal or Blender. Nothing leaves this computer.</p>
    <p class="meta" data-version>AssetFurnace by Bingeljell</p>
    <!-- ABOUT COPY: the "What AssetFurnace adds" list below is the part to edit -->
    <h3>What AssetFurnace adds</h3>
    <ul class="adds">
      <li><b>One flow, start to finish.</b> Idea, picture, 3D model, clean-up, rig and moves, each step handing its result to the next.</li>
      <li><b class="pm">Pixel Match</b> puts your picture's real pixels back on the model, so text, logos and faces stay sharp instead of coming back as garbled lookalikes.</li>
      <li><b>Prop sheets.</b> Draw nine props on one sheet and get nine separate, game-ready models back.</li>
      <li><b>Finish.</b> Lighter, cleaner meshes with proper textures, ready for a game engine.</li>
      <li><b>Rigging and moves that fit.</b> Our own retargeting fits preset moves onto generated characters: straightened rest
        poses, arms kept out of bodies and armour, feet on the floor.</li>
      <li><b>Runs on your Mac.</b> Ports and fixes that make these models work on Apple Silicon, on NVIDIA and on AMD.</li>
      <li><b>Licences you can trust.</b> Every asset carries a record of the models that made it and the licences that apply.</li>
    </ul>
    <h3>Built on</h3>
    <p class="meta">Standing on the shoulders of these open models and the people who made them:</p>
    <ul class="credits">${CREDITS.map(([name, who, href, licence]) => `
      <li><a href="${href}" target="_blank" rel="noopener"><b>${escape(name)}</b></a> <span class="meta">${escape(who)} · ${escape(licence)}</span></li>`).join('')}
    </ul>
    <p class="meta">The full list with every licence: <a href="https://github.com/Bingeljell/image-to-3dlab/blob/main/docs/info_and_credits.md" target="_blank" rel="noopener">credits on GitHub</a>.
      AssetFurnace's own code is Apache-2.0.</p>
    <h3>Find us</h3>
    <div class="row">
      <a class="ghost" href="https://assetfurnace.com" target="_blank" rel="noopener">assetfurnace.com</a>
      <a class="ghost" href="https://github.com/Bingeljell/image-to-3dlab" target="_blank" rel="noopener">GitHub</a>
      <a class="ghost" href="https://discord.gg/3D4bcEhGx" target="_blank" rel="noopener">Discord</a>
      <a class="ghost" href="https://github.com/Bingeljell/image-to-3dlab/issues" target="_blank" rel="noopener">Report a problem</a>
    </div>
    <p class="meta signoff" style="text-align:right;opacity:.55;margin-top:1.5em">✝︎ b'tzelem Elohim</p></div>`);
  try {
    const update = await (await fetch('/api/update-check')).json();
    const line = el.querySelector('[data-version]');
    line.textContent = `AssetFurnace by Bingeljell · version ${update.current}`;
    if (update.newer && update.url) {
      line.insertAdjacentHTML('beforeend', ` · <a href="${escape(update.url)}" target="_blank" rel="noopener">version ${escape(update.latest)} is out</a>`);
    }
  } catch { /* the version line just stays without a number */ }
}

// How to use: one short walk through the studio. Each entry is [title, text]; keep it brief.
export const GUIDE = [
  ['1. Make something', 'Press + Create. Start from an idea (type a sentence), a picture, or a 3D model, then pick what you want: a picture, One prop, Nine props (a 3×3 sheet), or a moving character. Press Make it.'],
  ['2. Stop early if you like', 'In Create, click any step in the row to stop there, say after the 3D model. You can carry on later from the Steps panel.'],
  ['3. Watch it run', 'Progress shows in the middle. Each step takes a few minutes. Cancel stops it and keeps what is done. You can close the tab: the run carries on, and the studio picks it up when you come back.'],
  ['4. Your Library', 'Everything you make is listed on the left, newest first. Search, filter by Characters or Props, and click to open. Click a title to rename it. Hide tucks an asset away without deleting it.'],
  ['5. Next steps', 'The Steps panel on the right shows what is done and what comes next: 3D model, Finish (a lighter, cleaner mesh, with Pixel Match keeping your picture sharp), Rig (adds a skeleton) and Animate (add moves). Rig and Animate are for humanoids: two arms, two legs, standing up.'],
  ['6. Look around', 'Drag to orbit, scroll to zoom, right-drag to pan. The buttons top right reset the view and show wireframe, clay (plain grey, to judge the shape), normals (to spot dents and flipped faces), floor grid or skeleton. The face count sits bottom left. Click an axis on the gizmo to look from that side.'],
  ['7. Take it into your game', 'Download GLB saves the model, or the move that is playing. It opens in Godot, Unity, Unreal, Blender and three.js.'],
  ['Open GLB', 'Look at any GLB from your computer, or drop one anywhere on the page. View only: nothing is copied into your Library.'],
  ['Activity', 'Everything that has run, finished, stopped or failed, with logs when something breaks.'],
  ['Report a bug', 'Tell us what broke. The studio adds the version, your machine and the error; you read it all, then send it through our site, GitHub or Discord.'],
  ['Setup', 'Install engines and see disk use. Nothing downloads until you say yes, and each download says how big it is first.'],
  ['About', 'What AssetFurnace is, the open models it builds on, and their licences.'],
];

export function openGuide(stage) {
  sheet(stage, `<div class="sheet-card guide">
    <div class="sheet-top"><h2>How to use <span>AssetFurnace</span></h2><button type="button" class="ghost" data-close>Close</button></div>
    <dl>${GUIDE.map(([title, text]) => `<dt>${escape(title)}</dt><dd>${escape(text)}</dd>`).join('')}</dl></div>`);
}

/** Report a bug: what went wrong in the person's words, plus what the studio fills in (version,
 *  machine, the failed step's log, home folder shown as ~). Nothing is sent from here: each button
 *  hands the report to a place the person sends it from themselves. `chain`/`step` name a failure
 *  from Activity; without them the newest failure is used. */
export async function openReport(stage, { chain = '', step = '' } = {}) {
  const el = sheet(stage, `<div class="sheet-card report">
    <div class="sheet-top"><h2>Report a <span>bug</span></h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">Tell us what went wrong. You see everything that is sent, and nothing leaves this computer until you send it.</p>
    <label class="field">What went wrong?<textarea data-message rows="4" placeholder="What you did, what you expected, and what happened instead."></textarea></label>
    <details class="report-details"><summary>Included automatically <span class="meta" data-summary>Loading…</span></summary>
      <textarea data-details rows="9" spellcheck="false"></textarea>
      <p class="meta">Edit or delete anything. Your home folder is shown as ~.</p></details>
    <label class="check" data-prompt-row hidden><input type="checkbox" data-prompt><span>Include my prompt</span></label>
    <div class="report-send">
      <button type="button" class="primary" data-send="site">Send via assetfurnace.com</button>
      <p class="meta">Opens our site. You sign in with your email first, so we can reply.</p>
      <div class="row"><button type="button" class="ghost" data-send="github">Open on GitHub</button><button type="button" class="ghost" data-send="copy">Copy report</button></div>
      <p class="meta" data-note>GitHub needs an account. Or copy the report and paste it in <a href="https://discord.gg/3D4bcEhGx" target="_blank" rel="noopener">#bugs on our Discord</a>.</p>
    </div></div>`);
  const $ = (sel) => el.querySelector(sel);
  const load = async () => {
    const query = new URLSearchParams({ chain, step, prompt: $('[data-prompt]').checked ? '1' : '' });
    const info = await (await fetch(`/api/bug-report?${query}`)).json().catch(() => ({}));
    $('[data-details]').value = info.details || '';
    $('[data-summary]').textContent = info.failure ? '· version, machine and the last error' : '· version and machine';
    $('[data-prompt-row]').hidden = !info.has_prompt;
  };
  $('[data-prompt]').addEventListener('change', load);
  el.querySelectorAll('[data-send]').forEach((button) => button.addEventListener('click', async () => {
    const message = $('[data-message]').value.trim();
    if (!message) { $('[data-message]').focus(); $('[data-message]').placeholder = 'Say what went wrong first, in a sentence or two.'; return; }
    const response = await fetch('/api/bug-report', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, details: $('[data-details]').value }) });
    const links = await response.json().catch(() => ({}));
    if (!response.ok) { $('[data-note]').textContent = links.error || 'The report could not be made.'; return; }
    if (button.dataset.send === 'copy') {
      try {
        await navigator.clipboard.writeText(links.text);
        $('[data-note]').innerHTML = `Copied. Now paste it in <a href="${escape(links.discord)}" target="_blank" rel="noopener">#bugs on our Discord</a>.`;
      } catch {
        $('[data-note]').textContent = 'Could not copy. Select the text above and copy it by hand.';
      }
      return;
    }
    window.open(button.dataset.send === 'site' ? links.site : links.github, '_blank', 'noopener');
  }));
  $('[data-message]').focus();
  await load();
}
