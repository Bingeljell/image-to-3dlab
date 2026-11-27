import assert from 'node:assert/strict';
import test from 'node:test';
import { browser, change, click, replaceGlobal, requests, waitFor } from './app-helpers.js';
import { closeSheet } from '../../viewer/studio/pages.js';
import {
  chainEvents, fileSizeText, openCreate, pastedPicture, pictureDefaults, pictureSettings,
  planForm, preloadPicture, retryPictureSettings, runChain,
} from '../../viewer/studio/create.js';

const engines = [{ id: 'pixal3d', label: 'Pixal3D' }, { id: 'fast', label: 'Fast' }];
const pictureEngines = [
  { id: 'fast', label: 'Fast picture', installed: true, suggested: true, license: { name: 'MIT' }, defaults: { width: 768, steps: 4 }, picture_only_steps: 13 },
  { id: 'quality', label: 'Detailed picture', installed: true },
  { id: 'missing', label: 'Unavailable' },
];
const presets = [{ id: 'run', label: 'Run' }, { id: 'walk', label: 'Walk' }, { id: 'take', take: 1 }];
const recipes = { goals: { character: { label: 'Make a character', wording: 'even lighting',
  tips: ['<a href="/docs/prompt-guide.md">Picture guide</a>'] } },
trouble: [{ goals: ['character'], words: ['cloak'], message: 'Use close-fitting clothes.' }] };

function input(selector, value) {
  const el = document.querySelector(selector);
  assert.ok(el);
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

function submit() {
  document.querySelector('.sheet form').dispatchEvent(new Event('submit', { cancelable: true }));
}

function drop(window, target, file) {
  const event = new window.Event('drop', { cancelable: true, bubbles: true });
  Object.defineProperty(event, 'dataTransfer', { value: { files: file ? [file] : [] } });
  target.dispatchEvent(event);
  assert.ok(event.defaultPrevented);
}

function objectUrls(t) {
  const revoked = [];
  let id = 0;
  t.mock.method(URL, 'createObjectURL', () => `blob:test/${++id}`);
  t.mock.method(URL, 'revokeObjectURL', (url) => revoked.push(url));
  return revoked;
}

test('Create turns a prompt and advanced settings into a run, keeping open sections on rerender', async (t) => {
  const { window } = browser(t);
  const stage = document.querySelector('main');
  let started;
  const observers = [];
  replaceGlobal(t, 'ResizeObserver', class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(node) { this.node = node; }
    disconnect() { this.disconnected = true; }
  });
  requests(t, { 'GET /docs/prompt-guide.md': () => new Response('# Picture guide') });
  openCreate({ stage, recipes, engines, presets, pictureEngines, frontBack: { ready: true, shape_ready: true }, onStart: (plan) => { started = plan; } });
  const text = () => document.querySelector('[data-k="text"]');
  assert.equal(document.activeElement, text());
  submit();
  assert.match(document.querySelector('[data-error]').textContent, /Describe what/);
  const observer = observers.at(-1);
  observer.callback();
  Object.defineProperties(text(), { clientWidth: { value: 100 }, offsetHeight: { value: 20 }, clientHeight: { value: 16 }, scrollHeight: { value: 56 } });
  observer.callback();
  assert.equal(text().style.height, '60px');
  input('[data-k="text"]', '   clockwork\n   knight with a cloak  ');
  assert.match(stage.querySelector('.warnings').textContent, /Use close-fitting clothes/);
  click('[data-tips]');
  click('.tipsbox b');
  click('.tipsbox a[href$="prompt-guide.md"]');
  await waitFor(() => document.querySelector('.guide-body h2'));
  click('.modal [data-close]');
  assert.match(text().value, /clockwork/);
  change('[data-k="picengine"]', 'quality');
  assert.ok(stage.querySelector('[data-detail="picture"]').open);
  assert.equal(stage.querySelector('[data-pic="steps"]').value, '8');
  change('[data-k="picengine"]', 'fast');
  assert.equal(stage.querySelector('[data-pic="size"]').value, '768');
  const sheet = stage.querySelector('.sheet');
  sheet.scrollTop = 123;
  change('[data-pic="newEachTime"]', false);
  assert.equal(sheet.scrollTop, 123);
  assert.equal(document.activeElement.dataset.pic, 'newEachTime');
  change('[data-pic="size"]', '512');
  change('[data-pic="steps"]', '12');
  change('[data-pic="seed"]', '55');
  change('[data-k="engine"]', 'fast');
  assert.ok(stage.querySelector('[data-face="hero"]').disabled);
  assert.ok(stage.querySelector('[data-detail="model"]').open);
  change('[data-k="engine"]', 'pixal3d');
  assert.ok(stage.querySelector('[data-stop="back"]'));
  click('[data-stop="back"]');
  click('[data-stop="back"]');
  change('[data-k="frontback"]', false);
  assert.equal(stage.querySelector('[data-stop="back"]'), null);
  click('[data-stop="model"]');
  assert.ok(stage.querySelector('[data-stop="finished"]').classList.contains('skip'));
  click('[data-stop="model"]');
  click('[data-face="hero"]');
  click('[data-face="quick"]');
  click('[data-face="hero"]');
  change('[data-k="move"]', 'walk');
  change('[data-k="keep"]', true);
  change('[data-k="recipe"]', false);
  input('[data-k="height"]', '2.1');
  assert.ok(observer.disconnected);
  assert.equal(text().value, '   clockwork\n   knight with a cloak  ');
  submit();
  assert.deepEqual(started.steps, ['picture', 'model', 'finished', 'rigged', 'animated']);
  assert.equal(started.prompt, 'clockwork knight with a cloak');
  assert.equal(started.firstMove, 'walk');
  assert.deepEqual(started.pictureSettings, { width: 512, height: 512, steps: 12, seed: 55 });
  assert.deepEqual(started.finishSettings, { faces: 32000, hero: true, height_m: 2.1 });
  assert.equal(started.keepEverything, true);
  assert.equal(stage.querySelector('.sheet'), null);
  assert.equal(window.document, document);
});

test('Create accepts picture paste, back-picture drop and model input, and releases previews', (t) => {
  const { window } = browser(t);
  const revoked = objectUrls(t);
  const stage = document.querySelector('main');
  const changes = [];
  let plan;
  const context = { stage, recipes, engines, presets, frontBack: { hint: 'Download the view weights first', shape_ready: true },
    onStart: (value) => { plan = value; }, onSourceChange: (file) => changes.push(file) };
  const front = new File(['picture'], '', { type: 'image/png' });
  const original = new File(['other'], 'front.jpg', { type: 'image/jpeg' });
  const back = new File(['back'], 'back.webp', { type: 'image/webp' });
  const model = new File(['model'], 'boat.glb');
  const paste = (file) => {
    const event = new Event('paste', { cancelable: true });
    Object.defineProperty(event, 'clipboardData', { value: { files: file ? [file] : [] } });
    document.dispatchEvent(event);
    return event.defaultPrevented;
  };
  const handle = openCreate(context);
  assert.equal(handle.canUseLibraryPicture(), false);
  assert.equal(handle.pictureName(), null);
  assert.match(stage.textContent, /Front \+ back is off/);
  assert.equal(paste(), false);
  assert.equal(paste(front), true);
  assert.equal(handle.canUseLibraryPicture(), true);
  assert.match(handle.pictureName(), /^pasted-\d+.png$/);
  assert.equal(pastedPicture({ files: [original] }), original);
  const picture = stage.querySelector('[data-drop]');
  const picker = stage.querySelector('[data-file]');
  let pickerOpens = 0;
  picker.addEventListener('click', () => { pickerOpens++; });
  picture.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'a' }));
  picture.dispatchEvent(new window.KeyboardEvent('keydown', { key: ' ' }));
  click('[data-drop]');
  assert.equal(pickerOpens, 2);
  Object.defineProperty(picker, 'files', { value: [original] });
  picker.dispatchEvent(new Event('change'));
  assert.equal(handle.pictureName(), 'front.jpg');
  assert.deepEqual(revoked, ['blob:test/1']);

  const backDrop = () => stage.querySelector('[data-backdrop]');
  const backPicker = stage.querySelector('[data-backfile]');
  let backOpens = 0;
  backPicker.addEventListener('click', () => { backOpens++; });
  backDrop().dispatchEvent(new window.KeyboardEvent('keydown', { key: 'a' }));
  backDrop().dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter' }));
  click('[data-backdrop]');
  assert.equal(backOpens, 2);
  drop(window, backDrop(), model);
  assert.match(backDrop().textContent, /Have a back view/);
  Object.defineProperty(backPicker, 'files', { value: [back] });
  backPicker.dispatchEvent(new Event('change'));
  assert.match(backDrop().textContent, /Back: back.webp/);
  click('[data-backclear]');
  assert.match(backDrop().textContent, /Have a back view/);
  drop(window, backDrop(), back);
  click('[data-want="set"]');
  assert.equal(backDrop(), null);
  click('[data-want="prop"]');
  assert.match(backDrop().textContent, /Have a back view/);
  drop(window, backDrop(), back);
  change('[data-k="upright"]', false);
  submit();
  assert.equal(plan.file, original);
  assert.equal(plan.backFile, back);
  assert.equal(plan.pictureSettings, null);
  assert.equal(plan.prompt, null);
  assert.deepEqual(plan.finishSettings, { upright: false });
  assert.equal(changes.at(-1), null);
  assert.deepEqual(new Set(revoked), new Set(['blob:test/1', 'blob:test/2', 'blob:test/3', 'blob:test/4', 'blob:test/5']));
  assert.equal(paste(front), false);

  openCreate(context);
  click('[data-h="model"]');
  submit();
  assert.match(stage.querySelector('[data-error]').textContent, /Choose a GLB/);
  const sheet = stage.querySelector('.sheet');
  const dragover = new Event('dragover', { cancelable: true });
  sheet.dispatchEvent(dragover);
  assert.ok(dragover.defaultPrevented);
  drop(window, sheet);
  drop(window, sheet, model);
  assert.equal(stage.querySelector('[data-source-name]').textContent, 'boat.glb');
  assert.equal(stage.querySelector('[data-source-preview]'), null);
  submit();
  assert.equal(plan.file, model);
  assert.deepEqual(plan.steps, ['rigged', 'animated']);
  assert.equal(plan.modelSeed, null);

  openCreate(context);
  click('[data-have="picture"]');
  submit();
  assert.match(stage.querySelector('[data-error]').textContent, /Choose a picture/);
  stage.querySelector('.sheet').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
  assert.equal(stage.querySelector('.sheet'), null);
});

test('Create keeps a remembered seed and guards against stale library previews', async (t) => {
  browser(t);
  const revoked = objectUrls(t);
  const stage = document.querySelector('main');
  const images = [];
  replaceGlobal(t, 'Image', class { constructor() { images.push(this); } });
  const context = { stage, engines, presets, pictureEngines, onStart: () => assert.fail('Unexpected submit') };
  const restored = openCreate({ ...context, prefill: { text: 'An earlier prompt', want: 'picture',
    pictureSettings: { width: 768, height: 768, steps: 24, seed: 0 } } });
  assert.equal(stage.querySelector('textarea').value, 'An earlier prompt');
  assert.equal(stage.querySelector('[data-pic="size"]').value, '768');
  assert.equal(stage.querySelector('[data-pic="steps"]').value, '24');
  assert.equal(restored.isSourceCurrent(restored.sourceRevision()), true);
  closeSheet(stage);
  assert.equal(restored.isSourceCurrent(restored.sourceRevision()), false);
  assert.equal(await restored.usePicture(new File([], 'closed.png')), false);
  const handle = openCreate(context);
  handle.showError('Load failed');
  assert.equal(stage.querySelector('[data-error]').textContent, 'Load failed');
  assert.equal(await handle.usePicture(null), false);
  click('[data-have="picture"]');
  const file = (name) => new File(['picture'], name, { type: 'image/png' });
  const first = handle.usePicture(file('first.png'));
  assert.equal(stage.querySelector('[data-source-preview]'), null);
  images.at(-1).onload();
  assert.equal(await first, true);
  const preview = stage.querySelector('[data-source-preview]');
  const revision = handle.sourceRevision();
  const second = handle.usePicture(file('second.png'));
  images.at(-1).onload();
  assert.equal(await second, true);
  assert.equal(stage.querySelector('[data-source-preview]'), preview);
  assert.equal(stage.querySelector('[data-source-name]').textContent, 'second.png');
  assert.equal(handle.isSourceCurrent(revision), false);
  assert.deepEqual(revoked, ['blob:test/1']);
  const failing = handle.usePicture(file('bad.png'));
  images.at(-1).onerror();
  assert.equal(await failing, false);
  assert.equal(stage.querySelector('[data-source-name]').textContent, 'second.png');
  assert.ok(revoked.includes('blob:test/3'));
  const slow = handle.usePicture(file('slow.png'));
  const pending = images.at(-1);
  click('[data-have="idea"]');
  pending.onload();
  assert.equal(await slow, false);
  assert.ok(revoked.includes('blob:test/4'));
  const beforeClose = handle.usePicture(file('late.png'));
  click('[data-close]');
  images.at(-1).onload();
  assert.equal(await beforeClose, false);
  assert.equal(handle.canUseLibraryPicture(), false);
  handle.showError('After close');
});

test('picture settings clamp numbers, retain supported saved sizes and preload via browser decoding', async () => {
  assert.equal(fileSizeText(1), '1 B');
  assert.equal(fileSizeText(2047), '1 KB');
  assert.equal(fileSizeText(1572864), '1.5 MB');
  assert.deepEqual(pictureDefaults({ defaults: { width: 768, steps: 6 } }, true), { size: 768, steps: 6 });
  assert.deepEqual(pictureDefaults(null), { size: 1024, steps: 8 });
  assert.deepEqual(pictureDefaults(pictureEngines[0], true), { size: 768, steps: 13 });
  assert.deepEqual(pictureSettings(undefined, () => 0), { width: 1024, height: 1024, steps: 8, seed: 0 });
  assert.deepEqual(pictureSettings({ size: 50, steps: 0, seed: Infinity, newEachTime: false }), { width: 1024, height: 1024, steps: 10, seed: 42 });
  assert.deepEqual(pictureSettings({ size: '512', steps: 55, seed: 3.9, newEachTime: false }), { width: 512, height: 512, steps: 50, seed: 3 });
  assert.equal(pictureSettings({ steps: -3 }, () => 0).steps, 1);
  assert.equal(retryPictureSettings(null), null);
  assert.equal(retryPictureSettings('old'), null);
  assert.equal(retryPictureSettings({ steps: Infinity, seed: 1 }), null);
  assert.equal(retryPictureSettings({ steps: 8, seed: 0.5 }), null);
  assert.deepEqual(retryPictureSettings({ steps: 1, seed: 0 }), { size: 1024, steps: 1, seed: 0, touched: true });
  assert.deepEqual(retryPictureSettings({ steps: 59, seed: 5, width: 768, height: 1024 }), { steps: 50, seed: 5, touched: true });
  assert.deepEqual(retryPictureSettings({ steps: -2, seed: 5, width: 100, height: 100 }), { steps: 1, seed: 5, touched: true });
  assert.equal(pastedPicture(null), null);
  await preloadPicture('unused', null);
  let decoded;
  class Decoder {
    constructor() { decoded = this; }
    decode() { assert.equal(this.src, 'asset.png'); return Promise.resolve(); }
  }
  await preloadPicture('asset.png', Decoder);
  decoded.onerror(); // a later event cannot change the settled preview
  await preloadPicture('ok.png', class {
    set src(_) { queueMicrotask(() => this.onload()); }
    decode() { return Promise.reject(new Error('Unsupported')); }
  });
});

test('chain hands off the real form, refreshes completed files, and handles stops and polling errors', async (t) => {
  browser(t);
  const file = new File(['mesh'], 'asset.glb');
  const backFile = new File(['back'], 'back.png');
  const plan = { description: 'Asset', steps: ['model', 'finished'], want: 'prop', file, backFile,
    prompt: 'boat', engine: 'fast', firstMove: 'walk', pictureSettings: { seed: 0 }, pictureEngine: 'fast',
    modelSeed: 0, keepEverything: true, finishSettings: { enabled: true }, moveSettings: {},
    rigSettings: {}, backSettings: {}, made: { picture: 'input.png' } };
  const entries = Object.fromEntries(planForm(plan, plan.made));
  assert.deepEqual(entries, {
    title: 'Asset', steps: '["model","finished"]', want: 'prop', prompt: 'boat', engine: 'fast', first_move: 'walk',
    picture_settings: '{"seed":0}', picture_engine: 'fast', model_seed: '0', keep_everything: '1',
    finish_settings: '{"enabled":true}', move_settings: '{}', rig_settings: '{}', back_settings: '{}',
    made: '{"picture":"input.png"}', file, back_file: backFile,
  });
  assert.equal(planForm({ steps: [], file }).get('title'), 'asset.glb');
  assert.equal(planForm({ steps: [] }).get('title'), 'New asset');

  const events = [];
  const ui = Object.fromEntries(['begin', 'step', 'progress', 'done', 'finish', 'fail', 'cancelled']
    .map((name) => [name, (...args) => events.push([name, ...args])]));
  let cancel;
  ui.cancelWith = (callback) => { cancel = callback; };
  const matches = [];
  const ctx = { reload: async (match) => {
    matches.push(['unrelated', 'same.png', 'same.glb', 'same-finished.glb', 'same-rig.glb'].map((name) => !!match({
      picture: name, model: name, finished: name, rigged: name,
    })));
  } };
  const steps = plan.steps;
  const made = { picture: 'same.png', model: 'same.glb', finished: 'same-finished.glb', rigged: 'same-rig.glb' };
  const statuses = [
    new Error('Temporary connection loss'),
    { steps, done: [], status: 'running', current: 'model', percent: 10, message: 'Starting' },
    { steps, done: ['model'], current: 'finished', status: 'running', made, step_seconds: { model: 10 } },
    { steps, done: ['model', 'finished'], status: 'done', made },
  ];
  let create = { id: 'run' };
  const calls = requests(t, {
    'POST /api/chains': () => create,
    'GET /api/chains/run': () => {
      const status = statuses.shift();
      if (status instanceof Error) throw status;
      return Response.json(status);
    },
    'POST /api/chains/run/cancel': () => { throw new Error('Already finished'); },
  });
  assert.equal(await runChain(plan, { ui, ctx, interval: 0 }), true);
  assert.deepEqual(events.map(([name]) => name), ['begin', 'step', 'progress', 'done', 'step', 'progress', 'done', 'finish']);
  assert.equal(calls[0].body.get('made'), '{"picture":"input.png"}');
  assert.deepEqual(matches, Array.from({ length: 3 }, () => [false, true, true, true, true]));
  await cancel();
  assert.equal(calls.at(-1).path, '/api/chains/run/cancel');
  statuses.push({ steps, done: [], status: 'cancelled', current: 'model' });
  assert.equal(await runChain({ steps }, { ui, ctx, attach: 'run' }), false);
  assert.deepEqual(events.at(-1), ['cancelled', 'model']);
  statuses.push({ steps, done: [], status: 'error', current: 'model', error: 'Bad mesh' });
  assert.equal(await runChain({ steps }, { ui, ctx, attach: 'run' }), false);
  assert.deepEqual(events.at(-1), ['fail', 'model', 'Bad mesh', '']);
  statuses.push(null);
  assert.equal(await runChain({ steps }, { ui, ctx, attach: 'run' }), false);
  assert.match(events.at(-1)[2], /no longer known/);
  create = Response.json({ error: 'Disk low' }, { status: 400 });
  assert.equal(await runChain({ steps }, { ui, ctx }), false);
  assert.match(events.at(-1)[2], /Disk low/);
  create = Response.json({}, { status: 503 });
  assert.equal(await runChain({ steps }, { ui, ctx }), false);
  assert.match(events.at(-1)[2], /503/);
  assert.deepEqual(chainEvents({ current: 'model', done: [] }, { status: 'running', current: 'model', done: [] }).map(([name]) => name), ['progress']);
});

test('Create offers recipe defaults for prop sheets and handles picture files without a MIME type', (t) => {
  const { window } = browser(t);
  objectUrls(t);
  const stage = document.querySelector('main');
  const plans = [];
  const context = { stage, engines: [], presets: [], pictureEngines: [pictureEngines[0]],
    onStart: (plan) => plans.push(plan) };
  openCreate({ ...context, prefill: { text: 'An earlier robot' } });
  assert.equal(stage.querySelector('[data-want="character"]').getAttribute('aria-pressed'), 'true');
  click('[data-want="prop"]');
  assert.match(stage.querySelector('textarea').placeholder, /wooden tavern barrel/);
  click('[data-want="set"]');
  submit();
  assert.deepEqual(plans[0].finishSettings, {});
  assert.deepEqual(plans[0].steps, ['picture', 'model', 'split']);
  openCreate(context);
  click('[data-want="picture"]');
  assert.equal(stage.querySelector('[data-pic="steps"]').value, '13');
  const food = new File(['picture'], 'banana.png');
  drop(window, stage.querySelector('.sheet'), food);
  assert.equal(stage.querySelector('[data-source-name]').textContent, 'banana.png');
  assert.equal(stage.querySelector('[data-source-preview]').src, 'blob:test/1');
  const unfamiliar = new File(['contents'], 'banana.txt');
  drop(window, stage.querySelector('.sheet'), unfamiliar);
  assert.equal(stage.querySelector('[data-source-preview]'), null);
  assert.equal(stage.querySelector('[data-have="picture"]').getAttribute('aria-pressed'), 'true');
  click('[data-close]');
});
