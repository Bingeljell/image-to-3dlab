import assert from 'node:assert/strict';
import test from 'node:test';
import { canSkin, draftHtml, draftsFor, largePair, mountSkins, skinCardHtml } from '../../viewer/studio/skins.js';
import { installViewerBrowser, flushEvents } from './viewer-browser.js';

const base = { finished: 'finish.glb', front: 'front.png', back: 'back.png' };
const asset = { id: 'id & 1', model: 'file__pixal3d__model.glb', finished: base.finished, rigged: 'rig.glb',
  skins: [{ id: 'gold', name: 'Gold <paint>' }, { id: 'silver', name: 'Silver' }],
};
const pair = { id: 'draft', base, created: 2, name: 'Blue <gold> & "silver"\'', next_side: 'back', needs_back: true,
  tries: { back: [{ picture: 'older.png', number: 1 }, { picture: 'new.png', number: 2 }] } };
const ready = { ...pair, next_side: null, ready: true, approved: { front: 'gold.png', back: 'silver.png' } };
const building = { ...pair, status: 'building', build_job: { job_id: 'build-job' } };
const url = (file) => `/output/${file}`;

test('skin drafts distinguish review, failed pictures, building, and finished skins', () => {
  assert.equal(canSkin(asset), true);
  assert.equal(canSkin(null), false);
  assert.equal(canSkin({ ...asset, model: 'ordinary' }), false);
  assert.deepEqual(draftsFor(asset), []);
  assert.deepEqual(draftsFor(asset, [
    { ...pair, id: 'older', created: 1 }, pair, { ...pair, status: 'done' }, { id: 'unrelated' },
  ]).map(({ id }) => id), ['draft', 'older']);
  assert.match(draftHtml(pair, url), /Step 2 of 2/);
  assert.match(draftHtml(pair, url), /Blue &lt;gold&gt; &amp; &quot;silver&quot;&#39;/);
  assert.match(draftHtml(pair, url), /new.png\?v=2/);
  assert.match(draftHtml({ ...pair, next_side: 'front', tries: undefined }, url), /Step 1 of 2/);
  assert.match(draftHtml({ ...pair, picture_job: { status: 'queued' } }, url), /class="busy"/);
  assert.match(draftHtml({ ...pair, picture_job: { status: 'running', last_event: { percent: 200 } } }, url), /width:100%/);
  assert.match(draftHtml({ ...pair, picture_job: { status: 'error', error: 'memory' } }, url), /Could not draw/);
  assert.match(draftHtml({ ...pair, picture_job: { status: 'error' } }, url), /stopped without saying why/);
  assert.match(draftHtml({ ...pair, tries: { back: [{ ready: false }] } }, url), /Could not draw/);
  assert.match(draftHtml(ready, url), /Both pictures approved/);
  assert.match(draftHtml({ ...ready, approved: { front: 'first.png' } }, url), /Approved front/);
  assert.match(draftHtml({ ...pair, status: 'error', error: 'broken' }, url), /Try building again/);
  assert.match(draftHtml({ ...pair, status: 'error' }, url), /stopped without saying why/);
  assert.match(draftHtml(building, url), /Repainting the model/);
  assert.match(draftHtml({ ...building, build_job: { last_event: { overall_pct: 43, message: 'Painting' }, log_tail: 'detail' } }, url), /width:43%/);
  assert.deepEqual(largePair(pair), {
    original: base.back, fresh: 'new.png', version: 2,
    caption: `${pair.name}: the back picture, try 2. Approve it in Steps, or ask again.`,
  });
  assert.equal(largePair(ready), null);
  assert.equal(largePair({ ...pair, tries: { back: [{ ready: false }] } }), null);
  assert.equal(largePair(building), null);
  assert.match(skinCardHtml(asset, [], 'silver', url), /2 skins/);
  assert.match(skinCardHtml({ skins: [asset.skins[0]] }, [], null, url), /1 skin ·/);
  assert.match(skinCardHtml({}, [], null, url), /class="adv" open/);
});

test('skin card creates a draft, retries polling, reviews a picture and builds it', async (t) => {
  const { document } = installViewerBrowser(t, '<ul></ul>');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const list = document.querySelector('ul');
  const busy = [], viewed = [], done = [], compared = [], requests = [];
  let draft = pair;
  let failed = false;
  let poll = 0;
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    const body = options?.body ? JSON.parse(options.body) : undefined;
    requests.push({ path, body });
    if (path === '/api/skins') return Response.json(body ? { skin: pair } : {});
    if (path.endsWith('/try')) {
      draft = { ...pair, tries: { back: [{ picture: 'corrected.png', number: 3 }] } };
      return Response.json({});
    }
    if (path.endsWith('/approve')) {
      if (failed) return Response.json({ error: 'Picture not ready' }, { status: 409 });
      draft = ready;
      return Response.json({});
    }
    if (path.endsWith('/build')) {
      draft = building;
      return Response.json({});
    }
    if (path.endsWith('/draft')) {
      poll++;
      if (poll === 1) return new Response('temporary', { status: 503 });
      return Response.json(draft);
    }
    assert.fail(path);
  });
  const ctx = { asset: () => asset, viewing: () => 'gold', url,
    onDone: (skin) => done.push(skin), onBusy: (value) => busy.push(value),
    view: (skin) => viewed.push(skin), compare: (...args) => compared.push(args),
  };
  await mountSkins(list, asset, ctx);
  const find = (sel) => list.querySelector(sel);
  const click = async (sel) => { find(sel).click(); await flushEvents(); };
  const picker = find('[data-skin-pick]');
  picker.value = 'silver';
  picker.onchange();
  picker.value = '';
  picker.onchange();
  assert.deepEqual(viewed, [asset.skins[1], null]);

  await click('[data-skin-new]');
  assert.match(find('.skin-note').textContent, /Say what to change/);
  find('[data-k="change"]').value = ' brighten the belt ';
  find('[data-k="name"]').value = ' New belt ';
  await click('[data-skin-new]');
  assert.equal(find('details.adv').open, false);
  assert.deepEqual(requests.find(({ body }) => body).body, {
    asset: asset.id, instruction: 'brighten the belt', name: 'New belt',
  });
  t.mock.timers.tick(2000);
  await flushEvents();
  assert.deepEqual(busy, [true, false]);
  assert.equal(compared.at(-1)[1], 'new.png');

  await click('.skin-pair img');
  assert.equal(compared.length, 2);
  find('[data-k="redo"]').value = ' navy ';
  await click('[data-skin-redo]');
  assert.equal(compared.at(-1)[1], 'corrected.png');
  assert.deepEqual(requests.find(({ path }) => path.endsWith('/try')).body, { side: 'back', instruction: 'navy' });
  failed = true;
  await click('[data-skin-approve]');
  assert.match(find('.skin-note').textContent, /Picture not ready/);
  assert.equal(find('[data-skin-approve]').disabled, false);
  failed = false;
  await click('[data-skin-approve]');
  assert.match(find('[data-draft]').textContent, /Both pictures approved/);
  await click('[data-skin-build]');
  assert.match(find('[data-draft]').textContent, /Repainting the model/);
  assert.equal(busy.at(-1), true);
  draft = { ...pair, status: 'done', file: 'done.glb' };
  t.mock.timers.tick(2000);
  await flushEvents();
  assert.deepEqual(done, [draft]);
  assert.equal(busy.at(-1), false);
});

test('skin card handles stale data, download, delete, discard and cancel', async (t) => {
  const { document, window } = installViewerBrowser(t, '<ul></ul>');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const list = document.querySelector('ul');
  const done = [], busy = [], requests = [], downloads = [];
  let current = null;
  let allow = false;
  let fail = false;
  let draft = pair;
  let fetchAllFails = false;
  const confirm = () => allow;
  Object.defineProperty(globalThis, 'confirm', { configurable: true, value: confirm });
  t.after(() => delete globalThis.confirm);
  t.mock.method(window.HTMLAnchorElement.prototype, 'click', function () { downloads.push(this.download); });
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    requests.push(path);
    if (path === '/api/skins') {
      if (fetchAllFails) return new Response('not JSON', { status: 500 });
      return Response.json({ skins: [draft] });
    }
    if (path.startsWith('/api/skins/all?')) {
      if (fail) return Response.json({ error: 'Bundle unavailable' }, { status: 503 });
      return Response.json({ file: 'bundles/all.glb' });
    }
    if (path.endsWith('/discard')) {
      return fail ? Response.json({ error: 'On disk only' }, { status: 500 }) : Response.json({});
    }
    if (path.endsWith('/cancel')) { draft = ready; return new Response('', { status: 500 }); }
    if (path.endsWith('/draft')) return Response.json(draft);
    assert.fail(path + String(options));
  });
  const ctx = { asset: () => current, viewing: () => 'gold', url, onDone: (skin) => done.push(skin), onBusy: (v) => busy.push(v) };
  const find = (sel) => list.querySelector(sel);
  const click = async (sel) => { find(sel).click(); await flushEvents(); };
  await mountSkins(list, {}, ctx);
  assert.equal(requests.length, 0);
  await mountSkins(list, asset, ctx);
  assert.equal(list.children.length, 0);

  current = asset;
  fetchAllFails = true;
  await mountSkins(list, asset, ctx);
  assert.ok(find('[data-skin-new]'));
  fetchAllFails = false;
  await mountSkins(list, asset, ctx);
  await flushEvents();
  assert.equal(list.children.length, 1);
  await click('[data-skin-delete]');
  await click('[data-skin-discard]');
  assert.equal(requests.some((path) => path.endsWith('/discard')), false);
  allow = fail = true;
  await click('[data-skin-all]');
  assert.match(find('.skin-note').textContent, /Bundle unavailable/);
  await click('[data-skin-delete]');
  assert.equal(find('[data-skin-delete]').disabled, false);
  await click('[data-skin-discard]');
  assert.equal(find('[data-skin-discard]').disabled, false);
  fail = false;
  await click('[data-skin-all]');
  assert.deepEqual(downloads, ['all.glb']);
  assert.ok(requests.includes('/api/skins/all?asset=id%20%26%201&file=rig.glb'));
  await click('[data-skin-delete]');
  assert.deepEqual(done, [null]);
  await click('[data-skin-discard]');
  assert.equal(find('[data-draft]'), null);

  draft = building;
  await mountSkins(list, asset, ctx);
  await flushEvents();
  assert.equal(busy.at(-1), true);
  await click('[data-skin-cancel]');
  assert.ok(requests.includes('/api/skins/build/build-job/cancel'));
  assert.match(find('[data-draft]').textContent, /Both pictures approved/);
  list.remove();
  t.mock.timers.tick(2000);
  await flushEvents();
  assert.equal(busy.at(-1), false);
});

test('a skin without a name can be created and redrawn without correction text', async (t) => {
  const { document } = installViewerBrowser(t, '<ul></ul>');
  const simple = { ...asset, rigged: null, skins: undefined };
  const list = document.querySelector('ul');
  const requests = [];
  let paintAvailable = false;
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    const body = options?.body ? JSON.parse(options.body) : undefined;
    requests.push({ path, body });
    if (path === '/api/skins' && !body) return Response.json({ skins: [] });
    if (path === '/api/skins') {
      return paintAvailable ? Response.json({ skin: pair })
        : Response.json({ error: 'Set up the picture engine' }, { status: 409 });
    }
    if (path.endsWith('/try')) return Response.json({});
    if (path.endsWith('/draft') && paintAvailable) return Response.json({ ...pair, needs_back: false });
    if (path.endsWith('/draft')) return Response.json({}, { status: 500 });
    assert.fail(path);
  });
  await mountSkins(list, simple, { asset: () => simple, viewing: () => null, url });
  const find = (sel) => list.querySelector(sel);
  const click = async (sel) => { find(sel).click(); await flushEvents(); };
  assert.equal(find('[data-skin-pick]'), null);
  find('[data-k="change"]').value = 'Blue belt';
  await click('[data-skin-new]');
  assert.match(find('.skin-note').textContent, /Set up the picture engine/);
  assert.equal(find('[data-skin-new]').disabled, false);
  paintAvailable = true;
  await click('[data-skin-new]');
  assert.deepEqual(requests.at(-2).body, { asset: simple.id, instruction: 'Blue belt' });
  assert.doesNotMatch(find('[data-draft]').textContent, /Step 2 of 2/);
  await click('[data-skin-redo]');
  assert.deepEqual(requests.at(-2).body, { side: 'back' });
  paintAvailable = false;
  await click('[data-skin-large]');
  assert.ok(find('[data-skin-approve]'));
});
