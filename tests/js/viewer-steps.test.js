import assert from 'node:assert/strict';
import test from 'node:test';
import { installViewerBrowser, flushEvents, pointer } from './viewer-browser.js';
import { chainPlan, stepBody, wireStep, wireBrush, wireVersions } from '../../viewer/studio/steps.js';
import { brushForm, brushStage, closeBrushStage, fitSize, mountBrush } from '../../viewer/studio/brush.js';

const engines = [{ id: 'pixal', label: 'Pixal3D', license: 'One' }, { id: 'sf', label: 'SF3D', license: 'Two' }];
const pictureEngines = [{ ...engines[0], installed: true }, { ...engines[1], installed: true }, { id: 'missing' }];
const presets = [{ id: 'walk', label: 'Walk', in_place: true }, { id: 'hook', label: 'Hook', speed: 1.3 }, { take: true }];
const asset = {
  name: 'Knight', kind: 'character', picture: 'front.png', back: 'back.png',
  prompt: 'Knight', model: 'x__pixal3d__test.glb', finished: 'finish.glb', rigged: 'rig.glb', clips: [{ name: 'walk' }],
};
const basic = { engines, pictureEngines, presets, url: (file) => `/files/${file}` };

function makeCard(t, step, source, props = {}) {
  const { document, window } = installViewerBrowser(t, '<main></main><li><div class="body"></div></li>');
  const stage = document.querySelector('main');
  const card = document.querySelector('li');
  card.dataset.step = step;
  const body = card.querySelector('.body');
  const ctx = { ...basic, asset: () => source, ...props };
  body.innerHTML = stepBody(step, source, ctx);
  wireStep(card, step, source, ctx);
  const find = (selector) => card.querySelector(selector);
  const change = (selector, value, event = 'change') => {
    const input = find(selector);
    input.value = value;
    input.dispatchEvent(new window.Event(event, { bubbles: true }));
  };
  return { document, window, stage, card, body, ctx, find, change };
}

test('step cards gate unavailable inputs and preserve older versions', () => {
  const scenarios = [
    ['model', {}, basic, /no picture/],
    ['model', asset, { engines: [] }, /No 3D engine/],
    ['model', asset, basic, /Make another 3D model/],
    ['finished', { kind: 'prop set' }, { blenderReady: false }, /Splitting into props needs Blender/],
    ['finished', asset, { blenderReady: false }, /Finish needs Blender/],
    ['finished', { kind: 'prop set' }, basic, /Split into props/],
    ['finished', { kind: 'prop', rigged: 'rig.glb' }, basic, /Stand upright/],
    ['finished', { ...asset, model: 'plain.glb', finished: null }, basic, /Skipped so far/],
    ['finished', { ...asset, picture: null }, basic, /Hero face needs the picture/],
    ['finished', { ...asset, finished: 'older_hero.glb' }, basic, /Redo Finish/],
    ['rigged', asset, basic, /Re-rig/],
    ['animated', { ...asset, fits_moves: false }, basic, /Preset moves are for humanoids/],
    ['animated', { clips: [] }, { presets: [] }, /No preset moves/],
    ['animated', asset, basic, /Walk ✓/],
    ['picture', { picture: 'single.png' }, { ...basic, frontBack: { ready: true } }, /Draw a back picture/],
    ['picture', { picture: 'single.png' }, { ...basic, pictureEngines: [] }, /^$/],
    ['picture', { kind: 'prop set', picture: 'single.png' }, { frontBack: { ready: true } }, /^$/],
    ['unknown', {}, basic, /^$/],
  ];
  for (const [step, source, ctx, expected] of scenarios) assert.match(stepBody(step, source, ctx).replace(/\s+/g, ' '), expected);
  assert.equal(chainPlan('back', {}, {}), null);
  assert.equal(chainPlan('model', {}, {}), null);
  assert.equal(chainPlan('finished', { kind: 'prop set' }, {}), null);
  assert.equal(chainPlan('finished', { picture: 'a.png' }, {}), null);
  assert.equal(chainPlan('rigged', {}, {}), null);
  assert.equal(chainPlan('animated', asset, {}), null);
  assert.equal(chainPlan('animated', {}, { preset: 'walk' }), null);
  assert.equal(chainPlan('other', asset, {}), null);
});

test('Finish controls reveal only the relevant options and produce a one-step chain', async (t) => {
  const plans = [];
  const { change, find } = makeCard(t, 'finished', asset, { startChain: (plan) => plans.push(plan) });
  for (const [value, active] of [['custom', 'custom'], ['full', 'full'], ['hero', 'hero-hint']]) {
    change('[data-k="faces"]', value);
    for (const name of ['custom', 'full', 'hero-hint']) assert.equal(find(`[data-${name}]`).hidden, name !== active);
  }
  change('[data-k="faces"]', 'custom');
  change('[data-k="customFaces"]', '23000');
  change('[data-k="texture"]', '1024');
  change('[data-k="surface"]', 'drop');
  change('[data-k="cleanup"]', 'rebuild');
  change('[data-k="height"]', '2.2');
  find('[data-k="photo"]').checked = false;
  find('[data-k="edgefade"]').checked = false;
  find('[data-k="nudge"]').checked = false;
  await find('[data-run]:not([data-hero])').onclick();
  assert.deepEqual(plans[0].made, { picture: asset.picture, model: asset.model });
  assert.deepEqual(plans[0].finishSettings, {
    faces: 23000, skip_photo: true, skip_edge_fade: true, skip_nudge: true,
    texture_size: 1024, drop_surface_maps: true, cleanup: 'rebuild', height_m: 2.2,
  });
  await find('[data-hero]').onclick();
  assert.equal(plans[1].finishSettings.hero, true);
});

test('model creation sends the selected engine, then rewires controls after startup failure', async (t) => {
  const requests = [], busy = [], done = [];
  let fail = true;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options });
    if (url.startsWith('/files/')) return new Response('picture');
    if (url === '/api/generate') return fail ? new Response('server busy', { status: 503 }) : Response.json({ job_id: 'm1' });
    return Response.json({ status: 'done', last_event: { overall_pct: 100, message: 'Built' }, log_tail: 'visual hull' });
  });
  const { find, change, body } = makeCard(t, 'model', asset, {
    onBusy: (value) => busy.push(value), onDone: (...args) => done.push(args),
  });
  change('[data-k="engine"]', 'sf');
  assert.equal(find('[data-k="licence"]').textContent, 'Two');
  change('[data-k="engine"]', 'invalid');
  assert.equal(find('[data-k="licence"]').textContent, '');
  change('[data-k="engine"]', 'sf');
  await find('[data-run]').onclick();
  assert.match(body.textContent, /503/);
  find('[data-back]').click();
  fail = false;
  await find('[data-run]').onclick();
  assert.equal(requests[1].options.body.get('image').name, 'front.png');
  assert.equal(JSON.parse(requests[1].options.body.get('settings')).backend, 'sf');
  assert.equal(requests[4].url, '/api/generate/m1/status');
  assert.equal(find('.bigbar span').style.width, '100%');
  assert.deepEqual(done, [['model', { preset: null }]]);
  assert.deepEqual(busy, [true, false, true, false]);
});

test('model, back, rig and prop plan controls preserve inputs appropriate to each route', async () => {
  assert.equal(chainPlan('model', asset, { backend: 'pixal' }).made.back, asset.back);
  assert.deepEqual(chainPlan('model', { picture: 'only.png' }, {}).made, { picture: 'only.png' });
  assert.equal(chainPlan('back', asset, { change: 'Green', backEngine: 'sf' }).backSettings.engine, 'sf');
  assert.equal(chainPlan('back', asset, {}).backSettings.instruction, undefined);
  assert.deepEqual(chainPlan('rigged', asset, { rigSource: 'model', boneNames: 'unity' }).rigSettings, { bone_names: 'unity' });
  assert.equal(chainPlan('rigged', { model: 'raw.glb' }, {}).made.model, 'raw.glb');
  assert.equal(chainPlan('rigged', asset, { boneNames: 'numbered' }).made.finished, asset.finished);
  assert.deepEqual(chainPlan('finished', { ...asset, kind: 'prop set' }, {}).steps, ['split']);
  assert.equal(chainPlan('finished', { ...asset, kind: 'prop' }, { upright: false }).finishSettings.upright, false);
});

test('animation card selects per-move defaults and sends user changes as JSON', async (t) => {
  let payload;
  const done = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url === '/api/animate/play') {
      payload = JSON.parse(options.body);
      assert.equal(options.headers['Content-Type'], 'application/json');
      return Response.json({ status_url: '/animate-status' });
    }
    assert.equal(url, '/animate-status');
    return Response.json({ status: 'done' });
  });
  const { find, change } = makeCard(t, 'animated', asset, { onDone: (...args) => done.push(args) });
  assert.equal(find('[data-k="inplace"]').checked, true);
  find('[data-preset="hook"]').click();
  assert.equal(find('[data-k="inplace"]').checked, false);
  assert.equal(find('[data-o="speed"]').textContent, '1.30×');
  change('[data-k="speed"]', '1.1', 'input');
  change('[data-k="spread"]', '8', 'input');
  assert.equal(find('[data-o="speed"]').textContent, '1.10×');
  assert.equal(find('[data-o="spread"]').textContent, '8°');
  await find('[data-run]').onclick();
  assert.deepEqual(payload, { model: asset.rigged, preset: 'hook', speed: 1.1, arm_spread: 8, in_place: false });
  assert.deepEqual(done, [['animated', { preset: 'hook' }]]);
});

test('standalone Finish, split and rig send the correct files', async (t) => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, body: options?.body });
    if (url.startsWith('/files/')) return new Response('some file');
    if (url.endsWith('/status')) return Response.json({ status: 'done' });
    return Response.json({ job_id: 'created' });
  });
  for (const [step, source] of [['finished', { ...asset, picture: null }], ['finished', { ...asset, kind: 'prop set' }], ['finished', asset], ['rigged', asset]]) {
    const { find, change } = makeCard(t, step, source);
    if (step === 'rigged') {
      change('[data-k="rigsource"]', 'model');
      change('[data-k="bonenames"]', 'unity');
      await find('[data-humanoid="no"]').onclick();
      assert.match(find('.job').textContent, /Rigging anyway/);
    } else await find('[data-run]:not([data-hero])').onclick();
  }
  const finishes = requests.filter(({ url }) => url === '/api/finish');
  assert.equal(finishes[0].body.get('image'), null);
  assert.equal(finishes[0].body.get('asset').name, asset.model);
  assert.equal(finishes[1].body.get('image').name, asset.picture);
  assert.equal(requests.find(({ url }) => url === '/api/props').body.get('generated'), asset.model);
  const rig = requests.find(({ url }) => url === '/api/animate/rig').body;
  assert.equal(rig.get('model'), asset.model);
  assert.equal(rig.get('bone_names'), 'unity');
});

test('progress can recover a stalled poll, report a failure, and cancel a retry', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
  let poll = 0;
  let hold;
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options });
    if (url === '/api/animate/rig') return Response.json({ job_id: 'rig-id' });
    if (url.endsWith('/cancel')) throw new TypeError('offline moment');
    if (url === '/api/bug-report') return Response.json({ text: 'scrubbed' });
    if (url.endsWith('/status')) {
      poll++;
      if (poll <= 2) return Response.json({ status: 'running', last_event: { phase: 'Weighting' }, log_tail: 'detail' });
      if (poll === 3) return Response.json({ status: 'error', last_event: { message: 'Out of memory' }, log_tail: 'full detail' });
      return new Promise((resolve) => { hold = resolve; });
    }
    assert.fail(url);
  });
  const { find } = makeCard(t, 'rigged', asset);
  const run = find('[data-run]').onclick();
  await flushEvents();
  assert.match(find('.job-msg').textContent, /Weighting/);
  assert.equal(find('.bigbar span').classList.contains('busy'), true);
  t.mock.timers.setTime(181000);
  t.mock.timers.tick(1200);
  await flushEvents();
  assert.match(find('.stallnote').textContent, /may be stuck/);
  t.mock.timers.tick(1200);
  await run;
  assert.match(find('.job-msg').textContent, /ran out of memory/);
  find('[data-copy]').click();
  await flushEvents();
  assert.equal(JSON.parse(requests.find(({ url }) => url === '/api/bug-report').options.body).details, 'full detail');
  find('[data-back]').click();
  const retry = find('[data-run]').onclick();
  await flushEvents();
  const cancel = find('[data-cancel]');
  await cancel.onclick({ target: cancel });
  assert.equal(cancel.disabled, true);
  hold(Response.json({ status: 'cancelled' }));
  await retry;
  assert.match(find('.job-msg').textContent, /Cancelled/);
});

test('back versions show, keep, delete and restore pictures and surface request errors', async (t) => {
  const shown = [], requests = [], done = [];
  const source = { picture: asset.picture, back: 'back & front.png' };
  const { window, card, ctx, find } = makeCard(t, 'picture', source, {
    showBack: (file) => shown.push(file), onDone: (step) => done.push(step),
  });
  let fail = true;
  let versions = [{ file: 'old.png', note: '<older>' }];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options });
    if (fail) return new Response('bad response', { status: 500 });
    if (url.endsWith('/delete')) versions = [];
    return Response.json({ current: { file: source.back }, versions });
  });
  wireVersions(card, source, ctx);
  const details = find('[data-versions]');
  const toggle = () => details.dispatchEvent(new window.Event('toggle'));
  details.open = true;
  toggle();
  await flushEvents();
  assert.match(find('[data-versions-error]').textContent, /Could not list/);
  fail = false;
  toggle();
  await flushEvents();
  assert.match(find('[data-file]').textContent, /<older>/);
  find('[data-show="old.png"]').click();
  find('[data-use]').click();
  await flushEvents();
  assert.deepEqual(done, ['picture']);
  assert.deepEqual(shown, ['old.png', source.back]);
  fail = true;
  find('[data-use]').click();
  await flushEvents();
  assert.match(find('[data-versions-error]').textContent, /Could not use/);
  find('[data-delete]').click();
  assert.match(find('[data-file]').textContent, /Delete for good/);
  fail = false;
  find('[data-keep]').click();
  await flushEvents();
  find('[data-delete]').click();
  find('[data-sure]').click();
  await flushEvents();
  assert.match(details.textContent, /No other versions/);
  assert.equal(find('[data-versions-error]').hidden, true);
  assert.equal(requests[0].url, '/api/image/versions?back=back%20%26%20front.png');
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), { back: source.back, version: 'old.png' });
});

function installBrushCanvas(t, window) {
  const contexts = new Map();
  const images = [];
  Object.defineProperty(globalThis, 'Image', { configurable: true, value: class {
    constructor() { images.push(this); }
  } });
  const proto = window.HTMLCanvasElement.prototype;
  t.mock.method(proto, 'getContext', function () {
    if (!contexts.has(this)) {
      const entries = [];
      const methods = ['clearRect', 'drawImage', 'fillRect', 'beginPath', 'moveTo', 'lineTo', 'stroke'];
      const context = Object.fromEntries(methods.map((name) => [name, (...args) => entries.push([name, ...args])]));
      contexts.set(this, Object.assign(context, { entries }));
    }
    return contexts.get(this);
  });
  t.mock.method(proto, 'toBlob', function (callback, type) { callback(new Blob(['mask'], { type })); });
  proto.setPointerCapture = () => {};
  return {
    contexts, images,
    load(w = 800, h = 600) {
      Object.assign(images.at(-1), { naturalWidth: w, naturalHeight: h });
      images.at(-1).onload();
    },
  };
}

test('image brush records strokes in source pixels and resets the mask', async (t) => {
  const { document, window, observers } = installViewerBrowser(t, '<main></main>');
  const canvas = installBrushCanvas(t, window);
  const stage = document.querySelector('main');
  const layer = brushStage(stage);
  Object.defineProperties(layer, { clientWidth: { value: 432 }, clientHeight: { value: 372 } });
  const brush = mountBrush(layer, '/paint.png', { fit: true });
  canvas.load();
  const shown = layer.querySelector('canvas');
  assert.deepEqual([shown.width, shown.height], [400, 300]);
  assert.deepEqual(fitSize({ w: 200, h: 100 }, { w: -5, h: 10 }), { w: 1, h: 1 });
  shown.getBoundingClientRect = () => ({ left: 10, top: 20, width: 400, height: 300 });
  const draw = (type, x, y) => pointer(window, shown, type, { clientX: x, clientY: y });
  draw('pointermove', 30, 60);
  assert.equal(brush.painted(), false);
  brush.setSize('64');
  draw('pointerdown', 110, 120);
  draw('pointermove', 210, 220);
  draw('pointercancel', 210, 220);
  brush.setSize('');
  assert.equal(brush.painted(), true);
  let maskContext = [...canvas.contexts.values()].find((context) => context.strokeStyle === '#fff');
  assert.ok(maskContext.entries.some(([method, x, y]) => method === 'lineTo' && x === 400 && y === 400));
  assert.equal(maskContext.lineWidth, 50);
  observers[0].callback();
  const blob = await brush.maskBlob();
  assert.equal(blob.type, 'image/png');
  assert.equal(brushForm('a.png', 'gold', blob, 42).get('seed'), '42');
  assert.equal(brushForm('a.png', 'gold', blob, 42, 'sf').get('mask').name, 'mask.png');
  brush.clear();
  assert.equal(brush.painted(), false);
  assert.equal(maskContext.fillStyle, '#000');

  const broad = mountBrush(layer, '/broad.png');
  canvas.load(600, 5000);
  assert.match(broad.blocked(), /600x5000/);
  assert.match(layer.querySelector('.error').textContent, /4096/);
  closeBrushStage(null);
  closeBrushStage(stage);
  assert.equal(stage.children.length, 0);
});

test('picture brush validates paint and words before posting the mask', async (t) => {
  const requests = [], done = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options });
    return Response.json(url.endsWith('/brush') ? { job_id: 'brush' } : { status: 'done' });
  });
  const { window, stage, card, find, ctx } = makeCard(t, 'picture', { ...asset, model: null }, { onDone: (...args) => done.push(args) });
  const canvas = installBrushCanvas(t, window);
  ctx.stage = () => stage;
  wireBrush(card, { ...asset, model: null }, ctx);
  const box = find('[data-brush="front"]');
  box.open = true;
  box.dispatchEvent(new window.Event('toggle'));
  canvas.load(5000, 600);
  const button = box.querySelector('[data-brush-run]');
  await button.onclick();
  assert.match(box.querySelector('[data-brush-error]').textContent, /4096/);
  box.open = false;
  box.dispatchEvent(new window.Event('toggle'));
  box.open = true;
  box.dispatchEvent(new window.Event('toggle'));
  canvas.load();
  await button.onclick();
  assert.match(box.querySelector('[data-brush-error]').textContent, /Paint over/);
  const shown = stage.querySelector('canvas');
  shown.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 100 });
  pointer(window, shown, 'pointerdown', { clientX: 100, clientY: 40 });
  pointer(window, shown, 'pointerup');
  await button.onclick();
  assert.match(box.querySelector('[data-brush-error]').textContent, /Say what goes/);
  box.querySelector('[data-brush-clear]').click();
  pointer(window, shown, 'pointerdown', { clientX: 100, clientY: 40 });
  box.querySelector('[data-brush-text]').value = ' brass pin ';
  box.querySelector('[data-brushengine]').value = 'sf';
  const size = box.querySelector('[data-brush-size]');
  size.value = '64';
  size.dispatchEvent(new window.Event('input'));
  await button.onclick();
  assert.equal(stage.querySelector('.brushstage'), null);
  assert.equal(requests[0].url, '/api/image/brush');
  assert.equal(requests[0].options.body.get('instruction'), 'brass pin');
  assert.equal(requests[0].options.body.get('engine'), 'sf');
  assert.equal(requests[0].options.body.get('mask').type, 'image/png');
  assert.deepEqual(done, [['picture', { preset: null }]]);
});

test('steps missing the file or selected move stop before posting a job', async (t) => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url) => { requests.push(url); return new Response('missing', { status: 404 }); });
  const raw = { picture: 'lost.png' };
  const model = makeCard(t, 'model', raw);
  assert.match(model.find('[data-run]').textContent, /Make 3D model/);
  await model.find('[data-run]').onclick();
  assert.match(model.find('.job-msg').textContent, /could not read lost.png/);
  const animate = makeCard(t, 'animated', { ...asset, clips: [] });
  animate.find('[data-preset="walk"]').setAttribute('aria-pressed', 'false');
  await animate.find('[data-run]').onclick();
  assert.match(animate.find('.job-msg').textContent, /Pick a move first/);
  assert.deepEqual(requests, ['/files/lost.png']);
});

test('back redraw starts a chain with the chosen picture settings', async (t) => {
  const plans = [];
  const { change, find } = makeCard(t, 'picture', asset, { startChain: (plan) => plans.push(plan) });
  change('[data-k="change"]', ' cloth texture ');
  change('[data-backengine]', 'sf');
  await find('[data-run="back"]').onclick();
  assert.equal(plans[0].backSettings.instruction, 'cloth texture');
  assert.equal(plans[0].backSettings.engine, 'sf');
  assert.deepEqual(plans[0].steps, ['back']);
  assert.deepEqual(plans[0].made, { picture: asset.picture });
});

test('without a main view, open brush boxes show a smaller canvas inside the current card', async (t) => {
  const source = { ...asset, model: null };
  const { card, window, find, ctx } = makeCard(t, 'picture', source);
  const canvas = installBrushCanvas(t, window);
  const front = find('[data-brush="front"]'), back = find('[data-brush="back"]');
  front.open = true;
  const noResize = globalThis.ResizeObserver;
  delete globalThis.ResizeObserver;
  try {
    wireBrush(card, source, ctx);
    canvas.load();
    const toggle = (box) => box.dispatchEvent(new window.Event('toggle'));
    toggle(front);
    assert.equal(canvas.images.length, 1);
    const shown = front.querySelector('canvas');
    assert.deepEqual([shown.width, shown.height], [320, 240]);
    back.open = true;
    toggle(back);
    canvas.load();
    assert.equal(front.open, false);
    assert.ok(back.querySelector('canvas'));
    assert.equal(canvas.images.length, 2);
    mountBrush(back, 'fit-without-observer.png', { fit: true });
    canvas.load();
    assert.equal(canvas.images.length, 3);
  } finally {
    globalThis.ResizeObserver = noResize;
  }
});
