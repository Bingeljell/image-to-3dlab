import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import test from 'node:test';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { browser, change, click, replaceGlobal, requests, waitFor } from './app-helpers.js';

const studioUrl = new URL('../../viewer/studio/app.js', import.meta.url);
const rendererUrl = new URL('./app-renderer.js', import.meta.url);
const shape = (id, values = {}) => ({
  id, family: id, version: 1, name: id, kind: 'model', updated: 1730000000,
  model: null, finished: null, rigged: null, picture: null, clips: [], props: [], ...values,
});
const robot = shape('acb000000001', { name: 'Robot', family: 'robot-family', versions: 2, kind: 'character',
  model: 'robot_pixal3d.glb', finished: 'robot_10k.glb', rigged: 'robot_rigged.glb',
  clips: [{ name: 'jump_up', file: 'robot_jump.glb' }, { name: 'walk', file: 'robot_walk.glb' }],
  picture: 'front.png', back: 'back.png', prompt: 'A robot', pixel_match: 'good',
  picture_settings: { width: 512, height: 512, steps: 8, seed: 5 },
  facing: 'back', timings: { model: 8 }, made_with: 'Fast',
});
const before = shape('robot-old', { name: 'Old Robot', family: 'robot-family', version: 2, versions: 2, kind: 'character',
  model: 'old.glb', finished: 'old_hero.glb' });
const propSet = shape('set', { kind: 'prop set', name: 'Props', model: 'set_raw.glb', picture: 'props.png',
  props: [{ name: 'blue-box', file: 'blue.glb' }], hidden: true });
const prop = shape('prop', { kind: 'prop', name: 'Vase', model: 'vase_raw.glb', finished: 'vase_full.glb', picture: 'vase.png' });
const picture = shape('picture', { kind: 'picture', name: 'Park', picture: 'park.png' });
const empty = shape('empty', { kind: 'other', name: 'Empty' });
const bad = (message, status = 409) => Response.json(message ? { error: message } : {}, { status });
const stageTitle = () => document.querySelector('#vTitle').textContent;

async function openStudio(t, { url, handlers, thumbnail = async () => null }) {
  const { window } = browser(t, { file: 'studio.html', url });
  const trace = { calls: [], thumbnail };
  window.renderTrace = trace;
  const calls = requests(t, {
    'GET /api/notices': { unread: 2, notices: [] },
    'GET /api/usage': { noticed: true },
    'GET /api/catalog': { backends: [] },
    'GET /api/animate/models': { installed: false },
    'GET /api/finish/capabilities': { ready: false },
    'GET /api/recipes': {},
    'GET /api/chains': {},
    'GET /api/model-info': (_body, path) => {
      const file = new URL(path, location.href).searchParams.get('file');
      return { file, path: `/local/${file}`, stats: { triangles: 77 } };
    },
    ...handlers,
  });
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (context.parentURL?.split('?')[0] === studioUrl.href &&
          ['./viewer.js', './model-thumb-renderer.js'].includes(specifier)) {
        return nextResolve(rendererUrl.href, context);
      }
      return nextResolve(specifier, context);
    },
  });
  try {
    await import(`${studioUrl.href}?scenario=${encodeURIComponent(t.name)}`);
  } finally {
    hook.deregister();
  }
  return { window, trace, calls };
}

function asset(id) {
  return document.querySelector(`[data-id="${id}"]:not(.sub)`) ?? document.querySelector(`[data-id="${id}"]`);
}

function key(window, target, name) {
  target.dispatchEvent(new window.KeyboardEvent('keydown', { key: name, bubbles: true }));
}

test('studio opens a linked asset and switches models, pictures, library filters and page overlays', async (t) => {
  let rows = [robot, before, propSet, prop, picture, empty, ...Array.from({ length: 35 }, (_, index) =>
    shape(`asset-${index}`, { name: `Extra ${index}`, updated: 1700000000 - index, picture: `extra-${index}.png` }))];
  let clips;
  let nextInfo;
  let thumbnailRequest = 0;
  const { window, trace, calls } = await openStudio(t, {
    url: 'http://localhost/studio.html#asset=acb000000001',
    thumbnail: async () => { thumbnailRequest++; return 'data:image/png;base64,AA=='; },
    handlers: {
      'GET /api/assets': () => ({ base: '/output/', assets: rows }),
      'GET /api/model-info': (_body, path) => {
        const file = new URL(path, location.href).searchParams.get('file');
        if (nextInfo) { const result = nextInfo; nextInfo = null; return result; }
        return { file, path: `/local/${file}`, stats: file === 'robot_10k.glb' ? null
          : { triangles: 77, textures: [{ width: 1024, height: 512 }] } };
      },
      'GET /api/activity': {},
      'GET /api/addons': { addons: [] },
      'GET /api/update-check': { current: '1' },
      'GET /api/bug-report': {},
    },
  });
  await waitFor(() => stageTitle() === 'Robot');
  assert.equal(document.querySelector('.sheet'), null);
  assert.equal(document.querySelector('[data-step="model"] .time').textContent, '8s');
  clips = trace.calls.find(([name, items]) => name === 'setClips' && items.length === 2);
  assert.deepEqual(clips[1].map(({ name }) => name), ['jump up', 'walk']);
  assert.equal(clips[2].index, 1);
  assert.equal(clips[2].restUrl, '/output/robot_rigged.glb');
  assert.equal(document.querySelector('#download').getAttribute('download'), 'robot_rigged.glb');
  assert.match(document.querySelector('[data-step="finished"]').textContent, /10,000 triangles · Pixel Match: good/);
  assert.ok(!!document.querySelector('#showMore'));
  click('#showMore');
  assert.ok(asset('asset-34'));
  click('[data-family="robot-family"]');
  assert.ok(document.querySelector('[data-id="robot-old"].sub'));
  click('[data-family="robot-family"]');
  assert.ok(document.querySelector('[data-family="robot-family"]'));

  click('[data-model-step="model"]');
  assert.equal(document.querySelector('#download').getAttribute('download'), 'robot_pixal3d.glb');
  assert.equal(document.querySelector('#vMeta').textContent, '3D model');
  click('[data-viewstep="model"]');
  assert.equal(document.querySelector('#download').getAttribute('download'), 'robot_rigged.glb');
  click('[data-viewstep="finished"]');
  await waitFor(() => !document.querySelector('#fileCard').hidden && document.querySelector('#modelStats').hidden);
  click('[data-pictures]');
  assert.equal(trace.calls.at(-1)[0], 'showImage');
  assert.match(trace.calls.at(-1)[1], /^\/output\/front.png\?v=/);
  assert.match(trace.calls.at(-1)[3], /^\/output\/back.png\?v=/);
  click('[data-pictures]');
  assert.equal(document.querySelector('#download').getAttribute('download'), 'robot_rigged.glb');
  trace.options.onClipChange({ file: 'robot_jump.glb' });
  assert.equal(document.querySelector('#download').getAttribute('download'), 'robot_jump.glb');
  click('[data-newpicture]');
  assert.equal(document.querySelector('.sheet textarea').value, robot.prompt);
  click('.sheet [data-close]');
  assert.equal(stageTitle(), 'Robot');

  change('#showHidden', true);
  asset('set').click();
  assert.equal(stageTitle(), 'Props');
  assert.equal(document.querySelector('[data-step="finished"]').textContent.includes('1 props cut out'), true);
  assert.equal(trace.calls.at(-1)[2].still, true);
  assert.equal(trace.calls.at(-1)[1][0].name, 'blue box');
  asset('prop').click();
  assert.match(document.querySelector('[data-step="finished"]').textContent, /Full detail/);
  asset('picture').click();
  assert.equal(document.querySelector('#download').hidden, true);
  assert.match(trace.calls.at(-1)[2], /picture so far/);
  asset('empty').click();
  assert.match(trace.calls.at(-1)[1], /Nothing to show for this asset/);
  asset('empty').click();

  const search = document.querySelector('#search');
  search.value = 'unknown title';
  search.dispatchEvent(new Event('input', { bubbles: true }));
  assert.match(document.querySelector('#assetList').textContent, /Nothing matches/);
  search.value = '';
  search.dispatchEvent(new Event('input', { bubbles: true }));
  const setFilter = document.querySelector('.seg [data-filter]:not([data-filter="all"])');
  setFilter.click();
  assert.equal(setFilter.getAttribute('aria-pressed'), 'true');
  click('.seg [data-filter="all"]');
  for (const id of ['stageFilter', 'engineFilter', 'sortBy']) {
    const select = document.getElementById(id);
    change(`#${id}`, select.options[1].value);
    change(`#${id}`, select.options[0].value);
  }
  change('#starredOnly', true);
  change('#starredOnly', false);

  asset(robot.id).click();
  for (const [button, heading] of [['guideBtn', 'How to use'], ['activityBtn', 'Activity'],
    ['addonsBtn', 'Add-ons'], ['newsBtn', 'News'], ['aboutBtn', 'About'], ['reportBtn', 'Report a']]) {
    click(`#${button}`);
    await waitFor(() => document.querySelector('.sheet h2')?.textContent.includes(heading));
    click('.sheet [data-close]');
  }
  assert.match(document.querySelector('#newsBtn').textContent, /2/);
  assert.equal(calls.filter(({ path }) => path === '/api/bug-report?chain=&step=&prompt=').length, 1);
  // The real thumbnail queue runs on an active tab and updates the visible model row.
  Object.defineProperty(document, 'hidden', { value: false, configurable: true });
  document.dispatchEvent(new Event('visibilitychange'));
  await waitFor(() => thumbnailRequest > 0);
  await waitFor(() => document.querySelector('[data-thumb-id="robot-old"].mark') === null);
  change('#engineFilter', 'made outside the app');
  const extras = rows;
  rows = [shape('acb000000002', { name: 'Added externally', picture: 'new.png' })];
  window.history.replaceState(null, '', '#asset=acb000000002');
  window.dispatchEvent(new window.HashChangeEvent('hashchange'));
  await waitFor(() => stageTitle() === 'Added externally');
  assert.equal(document.querySelector('#engineFilter').value, 'all');
  rows = [...rows, ...extras];
  window.history.replaceState(null, '', '#asset=ffffffffffff');
  window.dispatchEvent(new window.HashChangeEvent('hashchange'));
  await waitFor(() => asset(robot.id));
  const pending = Promise.withResolvers();
  const bodyRead = Promise.withResolvers();
  nextInfo = pending.promise;
  asset(robot.id).click();
  assert.equal(nextInfo, null, 'the Robot info request is waiting for its response');
  asset('prop').click();
  const actions = document.querySelector('#fileActions');
  await waitFor(() => actions.dataset.file === 'vase_full.glb');
  const oldResponse = Response.json({ file: 'robot_rigged.glb', path: '/stale' });
  const read = oldResponse.json.bind(oldResponse);
  oldResponse.json = async () => {
    const body = await read();
    bodyRead.resolve();
    return body;
  };
  pending.resolve(oldResponse);
  await bodyRead.promise;
  await nextTurn(); // Let Studio apply or discard the parsed response.
  assert.equal(actions.dataset.file, 'vase_full.glb');
  assert.equal(actions.dataset.path, '/local/vase_full.glb');
});

test('studio actions honor failures and confirmation before changing files', async (t) => {
  let rows = [robot, before, prop];
  let patchResponse = { preview: 'preview.glb' }, hiddenResponse = bad(), mainResponse = bad();
  let keepMode = 'new-version';
  let deletePlan = { refused: 'Only version' }, deletion = bad('Busy');
  let exportResponse = { file: 'roblox.glb', warnings: [], triangles: 300 };
  let renameResponse = bad('Name in use');
  let exportWait;
  const { window, trace, calls } = await openStudio(t, {
    url: 'http://127.0.0.1/studio.html#asset=acb000000001',
    handlers: {
      'GET /api/assets': () => ({ base: '/output/', assets: rows }),
      'POST /api/fix-spot/preview': () => patchResponse,
      'POST /api/fix-spot/keep': () => {
        if (keepMode === 'found-file') return { file: prop.finished };
        if (keepMode === 'missing-file') return { file: 'missing.glb' };
        rows = [{ ...robot, id: 'robot-edit', version: 3, name: 'Robot patched', finished: 'patched.glb' }, ...rows];
        return { file: 'patched.glb' };
      },
      'POST /api/fix-spot/discard': () => { throw new Error('Already gone'); },
      'POST /api/assets/robot-edit/hidden': () => hiddenResponse,
      'POST /api/assets/robot-edit/main': () => mainResponse,
      'GET /api/assets/robot-edit/delete-plan': () => deletePlan,
      'POST /api/assets/robot-edit/delete': () => {
        if (deletion instanceof Response && !deletion.ok) return deletion;
        rows = rows.filter(({ id }) => id !== 'robot-edit');
        return {};
      },
      'POST /api/assets/robot-edit/name': ({ name }) => {
        if (renameResponse instanceof Response && !renameResponse.ok) return renameResponse;
        rows = rows.map((row) => row.id === 'robot-edit' ? { ...row, name } : row);
        return {};
      },
      'POST /api/reveal': {},
      'POST /api/roblox-export': () => {
        if (exportWait) return exportWait;
        if (exportResponse instanceof Error) throw exportResponse;
        return exportResponse;
      },
    },
  });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  await waitFor(() => stageTitle() === 'Robot');
  const fix = trace.options.fixSession;
  assert.equal(fix.canFix(), false);
  assert.match(fix.blockedNote(), /Finish/);
  assert.deepEqual(await fix.patch([{ x: 1 }], 'base.glb'), { preview: 'preview.glb', url: '/output/preview.glb' });
  patchResponse = bad('Patch failed');
  await assert.rejects(fix.patch([], null), /Patch failed/);
  patchResponse = new Response('', { status: 400 });
  await assert.rejects(fix.patch([], null), /That did not work/);
  await fix.discard('preview.glb');
  await fix.keep('preview.glb');
  assert.equal(stageTitle(), 'Robot patched');
  keepMode = 'found-file';
  await fix.keep('preview.glb');
  assert.equal(stageTitle(), 'Vase');
  keepMode = 'missing-file';
  await fix.keep('preview.glb');
  assert.equal(stageTitle(), '');
  assert.match(trace.calls.at(-1)[1], /Nothing to show yet/);
  click('[data-family="robot-family"]');
  asset('robot-edit').click();

  click('#hideBtn');
  await waitFor(() => calls.at(-1).path.endsWith('/hidden'));
  assert.equal(document.querySelector('#hideBtn').textContent, 'Hide');
  hiddenResponse = {};
  click('#hideBtn');
  await waitFor(() => document.querySelector('#hideBtn').textContent === 'Unhide');
  change('#showHidden', true);
  click('#hideBtn');
  await waitFor(() => document.querySelector('#hideBtn').textContent === 'Hide');
  click('#mainBtn');
  await waitFor(() => calls.at(-1).path.endsWith('/main'));
  assert.equal(document.querySelector('#mainBtn').textContent, '☆ Main');
  rows = rows.map((row) => row.id === 'robot-edit' ? { ...row, main: true } : row);
  mainResponse = {};
  click('#mainBtn');
  await waitFor(() => document.querySelector('#mainBtn').textContent === '★ Main');

  const rename = (value, event = 'Enter') => {
    click('#vTitle');
    click('#vTitle');
    const input = document.querySelector('#vTitle input');
    if (value !== undefined) input.value = value;
    if (event === 'blur') input.dispatchEvent(new Event('blur'));
    else key(window, input, event);
    input.dispatchEvent(new Event('blur'));
  };
  rename('Cancelled', 'Escape');
  assert.equal(stageTitle(), 'Robot patched');
  rename();
  assert.equal(stageTitle(), 'Robot patched');
  rename('Taken');
  await waitFor(() => calls.at(-1).path.endsWith('/name'));
  assert.equal(stageTitle(), 'Robot patched');
  renameResponse = {};
  rename('  New Robot  ', 'blur');
  await waitFor(() => stageTitle() === 'New Robot');
  assert.deepEqual(calls.findLast(({ path }) => path.endsWith('/name')).body, { name: 'New Robot' });

  let clipboardWorks = true;
  const copied = [];
  Object.defineProperty(navigator, 'clipboard', { value: { writeText: async (path) => {
    if (!clipboardWorks) throw new Error('Denied');
    copied.push(path);
  } } });
  await waitFor(() => !document.querySelector('#fileCard').hidden);
  assert.equal(document.querySelector('#revealBtn').hidden, false);
  click('#revealBtn');
  assert.deepEqual(calls.at(-1).body, { file: 'robot_rigged.glb' });
  click('#copyPathBtn');
  await waitFor(() => document.querySelector('#copyPathBtn').textContent === 'Copied');
  assert.deepEqual(copied, ['/local/robot_rigged.glb']);
  t.mock.timers.tick(1500);
  assert.equal(document.querySelector('#copyPathBtn').textContent, 'Copy path');
  clipboardWorks = false;
  click('#copyPathBtn');
  await waitFor(() => document.querySelector('#copyPathBtn').textContent === 'Copy failed');
  click('#robloxBtn');
  await waitFor(() => calls.at(-1).body?.file === 'roblox.glb');
  assert.ok(!document.querySelector('#robloxResult').hidden);
  exportResponse = bad('Too large');
  click('#robloxBtn');
  await waitFor(() => document.querySelector('#robloxResult').textContent.includes('Too large'));
  exportResponse = new Error('No connection');
  click('#robloxBtn');
  await waitFor(() => document.querySelector('#robloxResult').textContent.includes('export did not run'));
  const pending = Promise.withResolvers();
  exportWait = pending.promise;
  click('#robloxBtn');
  asset('prop').click();
  await waitFor(() => document.querySelector('#fileActions').dataset.file === 'vase_full.glb');
  pending.resolve({ file: 'stale.glb' });
  await waitFor(() => !document.querySelector('#robloxBtn').disabled);
  assert.equal(calls.some(({ body }) => body?.file === 'stale.glb'), false);
  asset('robot-edit').click();

  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar').textContent.includes('Only version'));
  click('#deleteBar [data-close]');
  deletePlan = bad('Disk unavailable');
  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar').textContent.includes('Disk unavailable'));
  click('#deleteBar [data-close]');
  deletePlan = new Response('invalid', { status: 409 });
  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar').textContent.includes('could not be checked'));
  click('#deleteBar [data-close]');
  deletePlan = { folders: [], bytes: 22 };
  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar [data-delete]'));
  click('#deleteBar [data-delete]');
  await waitFor(() => document.querySelector('#deleteBar').textContent.includes('Busy'));
  click('#deleteBar [data-close]');
  deletion = new Response('invalid', { status: 409 });
  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar [data-delete]'));
  click('#deleteBar [data-delete]');
  await waitFor(() => document.querySelector('#deleteBar').textContent.includes('Could not delete'));
  click('#deleteBar [data-close]');
  deletion = {};
  click('#deleteBtn');
  await waitFor(() => document.querySelector('#deleteBar [data-delete]'));
  click('#deleteBar [data-delete]');
  await waitFor(() => stageTitle() === 'Robot');
  assert.equal(asset('robot-edit'), null);
  assert.deepEqual(calls.at(-3).body, { confirm: true });
});

test('studio reconnects to a run, updates and moves the progress card, then selects the finished asset', async (t) => {
  const steps = ['picture', 'back', 'model', 'finished'];
  const title = 'A particularly long new character name for a model that was started in another tab';
  const made = { picture: 'new.png', back: 'new-back.png', model: 'new.glb', finished: 'final.glb' };
  const rows = [picture];
  let active = { steps, done: [], status: 'running', current: 'picture', message: 'Sketching', full_log: 'Started', stalled: true };
  const { window, trace, calls } = await openStudio(t, {
    handlers: {
      'GET /api/assets': () => ({ base: '/output/', assets: rows }),
      'GET /api/chains': { running: { id: 'existing', steps, title, want: 'prop' } },
      'GET /api/chains/existing': () => active,
      'POST /api/chains/existing/cancel': {},
    },
  });
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  await waitFor(() => /Sketching/.test(document.querySelector('.job-msg')?.textContent));
  assert.equal(stageTitle(), title);
  assert.ok(document.querySelector('#vTitle').classList.contains('long'));
  assert.match(document.querySelector('.stallnote').textContent, /No news/);
  assert.equal(document.querySelector('.pcard .log').textContent, 'Started');

  const card = document.querySelector('.pcard');
  const stage = document.querySelector('#progress');
  Object.defineProperties(stage, { clientWidth: { value: 800 }, clientHeight: { value: 600 } });
  card.getBoundingClientRect = () => ({ left: 20, top: 20, width: 350, height: 250 });
  const grip = card.querySelector('.grip');
  card.setPointerCapture = grip.setPointerCapture = () => {};
  const pointer = (target, name, x, y, id = 1, button = 0) => {
    const event = new window.MouseEvent(name, { clientX: x, clientY: y, button, bubbles: true, cancelable: true });
    Object.defineProperty(event, 'pointerId', { value: id });
    target.dispatchEvent(event);
  };
  pointer(card, 'pointerdown', 25, 25, 1, 2);
  pointer(card.querySelector('[data-x]'), 'pointerdown', 25, 25);
  assert.ok(!card.classList.contains('placed'));
  pointer(card, 'pointerdown', 25, 25);
  pointer(card, 'pointermove', 100, 100, 2);
  assert.equal(card.style.left, '20px');
  pointer(card, 'pointermove', 55, 65);
  assert.equal(card.style.left, '50px');
  assert.equal(card.style.top, '60px');
  pointer(card, 'pointerup', 55, 65, 2);
  pointer(card, 'pointerup', 55, 65);
  pointer(card, 'pointermove', 200, 200);
  assert.equal(card.style.left, '50px');
  pointer(grip, 'pointerdown', 25, 25, 1, 2);
  pointer(grip, 'pointerdown', 25, 25);
  pointer(grip, 'pointermove', 125, 75);
  assert.equal(card.style.width, '450px');
  assert.equal(card.style.height, '300px');
  pointer(grip, 'pointercancel', 125, 75);

  active = { ...active, message: 'Almost there', stalled: false, percent: 60, full_log: 'Second log' };
  t.mock.timers.tick(1200);
  await waitFor(() => document.querySelector('.job-msg').textContent.includes('Almost there'));
  assert.equal(document.querySelector('.stallnote'), null);
  assert.equal(document.querySelector('.pcard'), card);
  assert.equal(card.querySelector('.log').textContent, 'Second log');
  assert.equal(card.querySelector('.bigbar span').style.width, '60%');
  t.mock.timers.tick(1200);
  assert.equal(card.querySelector('.log').textContent, 'Second log');
  active = { ...active, message: null, stalled: true, full_log: '' };
  t.mock.timers.tick(1200);
  await waitFor(() => document.querySelector('.job-msg').textContent.includes('Working'));
  assert.equal(card.querySelector('.log'), null);
  assert.ok(document.querySelector('.stallnote'));
  active = { steps, done: ['picture', 'back'], current: 'model', status: 'running',
    made: { picture: made.picture, back: made.back }, step_seconds: { picture: 12 }, message: 'Building' };
  rows.push(shape('new', { ...active.made, name: 'New picture', kind: 'picture' }));
  t.mock.timers.tick(1200);
  await waitFor(() => document.querySelector('#progress').classList.contains('compact'));
  assert.match(document.querySelector('#steps').textContent, /12s/);
  assert.deepEqual(trace.calls.findLast(([method]) => method === 'showImage').slice(1),
    ['/output/new.png', '', '/output/new-back.png']);
  click('[data-making]');
  click('#createBtn');
  click('#vTitle');
  assert.equal(document.querySelector('#vTitle input'), null);
  click('#progress [data-x]');
  assert.equal(calls.at(-1).path, '/api/chains/existing/cancel');
  asset('new').click();
  assert.match(trace.calls.at(-1)[2], /Being made into 3D/);
  t.mock.timers.tick(1000);
  asset('picture').click();
  rows[1] = shape('new', { ...made, name: 'Finished prop', kind: 'prop' });
  active = { steps, done: steps, status: 'done', made };
  t.mock.timers.tick(1200);
  await waitFor(() => !document.body.classList.contains('busy'));
  assert.match(document.querySelector('[data-making]').textContent, /Done/);
  click('[data-making]');
  assert.match(document.querySelector('.job-msg').textContent, /All done/);
  assert.deepEqual(trace.calls.findLast(([method]) => method === 'load').slice(1), ['/output/final.glb', { label: title }]);
  click('#progress [data-close]');
  assert.equal(stageTitle(), 'Finished prop');
  assert.equal(document.querySelector('[data-making]'), null);

  const event = new window.Event('dragover', { cancelable: true });
  window.dispatchEvent(event);
  assert.ok(event.defaultPrevented);
  const modelPicker = document.querySelector('#openFile');
  let pick = 0;
  modelPicker.addEventListener('click', () => pick++);
  click('#openBtn');
  assert.equal(pick, 1);
  let fileId = 0;
  t.mock.method(URL, 'createObjectURL', () => `blob:studio/${++fileId}`);
  const revoke = t.mock.method(URL, 'revokeObjectURL', () => {});
  const drop = (file, prevented = false) => {
    const event = new window.Event('drop', { cancelable: true });
    if (prevented) event.preventDefault();
    if (file) Object.defineProperty(event, 'dataTransfer', { value: { files: [file] } });
    window.dispatchEvent(event);
  };
  drop(new File([], 'ignored.glb'), true);
  drop();
  assert.equal(stageTitle(), 'Finished prop');
  drop(new File([], 'image.png'));
  assert.match(trace.calls.at(-1)[1], /Only GLB/);
  Object.defineProperty(modelPicker, 'files', { value: [new File([], 'first.glb')] });
  modelPicker.dispatchEvent(new Event('change'));
  assert.equal(stageTitle(), 'first.glb');
  for (const control of ['#hideBtn', '#mainBtn', '#deleteBtn', '#vTitle']) click(control);
  assert.equal(document.querySelector('#vTitle input'), null);
  assert.match(document.querySelector('#vMeta').textContent, /Opened from your computer/);
  assert.deepEqual(trace.calls.at(-1), ['load', 'blob:studio/1', { label: 'first.glb', autoplay: true }]);
  drop(new File([], 'second.GLB'));
  assert.equal(stageTitle(), 'second.GLB');
  assert.deepEqual(revoke.mock.calls[0].arguments, ['blob:studio/1']);
});

test('studio recovers an unavailable library, uses a saved picture, and shows stopped runs', async (t) => {
  const pictureAsset = shape('photo', { kind: 'picture', name: 'Restored picture', picture: 'restored.png' });
  const initial = Promise.withResolvers();
  let library = initial.promise;
  let latest = { status: 'error', done: [], steps: ['model'], current: 'model', error: 'Broken model', full_log: 'Geometry log' };
  let create = { id: 'new-run' };
  const { window, calls, trace } = await openStudio(t, {
    handlers: {
      'GET /api/chains': () => { throw Error('Older server'); },
      'GET /api/assets': () => library,
      'GET /output/restored.png': () => new Response('image', { headers: { 'Content-Type': 'image/png' } }),
      'POST /api/chains': () => create,
      'GET /api/chains/new-run': () => latest,
      'GET /api/bug-report': { details: 'machine' },
      'POST /api/bug-report': ({ details }) => ({ text: details }),
    },
  });
  t.mock.method(URL, 'createObjectURL', () => 'blob:source');
  t.mock.method(URL, 'revokeObjectURL', () => {});
  replaceGlobal(t, 'Image', class {
    set src(_) { queueMicrotask(() => this.onload()); }
  });
  const copies = [];
  Object.defineProperty(navigator, 'clipboard', { value: { writeText: async (text) => copies.push(text) } });
  await waitFor(() => calls.some(({ path }) => path === '/api/assets'));
  assert.match(document.querySelector('#libLoad').textContent, /Loading your assets/);
  document.querySelector('#search').dispatchEvent(new Event('input'));
  initial.resolve(bad('Library unavailable'));
  await waitFor(() => document.querySelector('.sheet textarea'));
  assert.match(document.querySelector('#libLoad').textContent, /Library unavailable/);
  assert.match(trace.calls.at(-1)[1], /Library did not load/);
  library = { base: '/output/', assets: [pictureAsset] };
  const retry = [...document.querySelectorAll('#assetList button, .lib button')].find((button) => /Try again/.test(button.textContent));
  assert.ok(retry);
  retry.click();
  await waitFor(() => asset('photo'));
  await waitFor(() => !document.querySelector('.sheet'));
  click('#createBtn');
  click('[data-have="picture"]');
  asset('photo').click();
  await waitFor(() => document.querySelector('[data-source-name]')?.textContent === 'restored.png');
  assert.equal(asset('photo').getAttribute('aria-current'), 'true');
  click('[data-stop="model"]');
  document.querySelector('.sheet form').dispatchEvent(new Event('submit', { cancelable: true }));
  await waitFor(() => document.querySelector('#progress').dataset.kind === 'failed');
  assert.match(document.querySelector('[data-making]').textContent, /Stopped at 3D model/);
  assert.equal(document.querySelector('.job-msg').textContent, 'Broken model');
  assert.equal(document.querySelector('.pcard .log').textContent, 'Geometry log');
  assert.equal(calls.find(({ method, path }) => method === 'POST' && path === '/api/chains').body.get('file').name, 'restored.png');
  click('#progress [data-copy]');
  await waitFor(() => copies[0]?.includes('Geometry log'));
  click('#progress [data-report]');
  await waitFor(() => document.querySelector('.report [data-details]')?.value === 'machine');
  click('.report [data-close]');
  click('#progress [data-close]');
  assert.equal(stageTitle(), 'Restored picture');

  const begin = () => {
    click('#createBtn');
    click('[data-w="picture"][data-h="idea"]');
    const text = document.querySelector('.sheet textarea');
    text.value = 'A pine tree';
    text.dispatchEvent(new Event('input'));
    document.querySelector('.sheet form').dispatchEvent(new Event('submit', { cancelable: true }));
  };
  latest = { status: 'cancelled', done: [], steps: ['picture', 'model'], current: 'picture' };
  begin();
  await waitFor(() => document.querySelector('#progress').dataset.kind === 'cancelled');
  assert.match(document.querySelector('.job-msg').textContent, /Everything finished before this step is kept/);
  click('#progress [data-close]');
  create = bad('Queue full');
  begin();
  await waitFor(() => document.querySelector('#progress').dataset.kind === 'failed');
  assert.match(document.querySelector('.job-msg').textContent, /Queue full/);
  assert.equal(document.querySelector('.pcard .log'), null);
  assert.equal(window.document, document);
});

test('studio previews kept backs and skins while ignoring a completed dress request for a different asset', async (t) => {
  const skin = { id: 'gold', name: 'Gold paint', file: 'gold.glb' };
  const draft = { id: 'blue', name: 'Blue <paint>', status: 'review', next_side: 'front',
    base: { finished: robot.finished, front: robot.picture },
    tries: { front: [{ picture: 'blue.png', number: 2 }] } };
  const dressedRobot = { ...robot, model: 'robot__pixal3d__raw.glb', skins: [skin], clips: [robot.clips[0]] };
  const skinProp = { ...prop, model: 'prop__pixal3d__raw.glb', skins: [skin] };
  let rows = [dressedRobot, skinProp];
  let libraryFails = false;
  let dressFails = false;
  let holdDress;
  let offlineRefresh = false;
  const { window, trace } = await openStudio(t, {
    url: 'http://remote.test/studio.html#asset=acb000000001',
    handlers: {
      'GET /api/catalog': { backends: [{ id: 'pixal3d', kind: '3d', state: 'ready', label: 'Pixal3D' }],
        picture_engines: [], front_back: { ready: true } },
      'GET /api/animate/models': { installed: true, presets: [{ id: 'idle', label: 'Idle' }] },
      'GET /api/finish/capabilities': () => { throw new Error('Older server'); },
      'GET /api/assets': () => libraryFails || offlineRefresh ? bad('Offline') : { base: '/output/', assets: rows },
      'GET /api/skins': () => ({ skins: [draft] }),
      'GET /api/skins/blue': () => draft,
      'GET /api/skins/gold/dressed': async (_body, path) => {
        if (holdDress) await holdDress;
        if (dressFails) throw new Error('Missing model');
        return { file: `gold-${new URL(path, location.href).searchParams.get('file')}` };
      },
      'POST /api/skins/gold/discard': () => {
        rows = rows.map((row) => row.id === skinProp.id ? { ...row, skins: [] } : row);
        return {};
      },
      'POST /api/skins': () => ({ skin: { id: 'gold' } }),
      'GET /api/skins/gold': () => ({ ...skin, status: 'done' }),
      'POST /api/roblox-export': { file: 'remote.glb', size_studs: [1, 2.5, 1] },
      'GET /api/image/versions': () => ({ current: { file: 'back.png' }, versions: [{ file: 'old-back.png' }] }),
      'POST /api/image/versions/use': () => {
        rows = [{ ...rows[0], id: 'new-back', updated: dressedRobot.updated + 1, name: 'New back' }, ...rows];
        return {};
      },
    },
  });
  replaceGlobal(t, 'confirm', () => true);
  await waitFor(() => document.querySelector('[data-skin-pick]'));
  assert.equal(document.querySelector('#revealBtn').hidden, true);
  click('#robloxBtn');
  await waitFor(() => document.querySelector('#robloxResult').textContent.includes('2.5 studs'));

  change('[data-skin-pick]', 'gold');
  await waitFor(() => trace.calls.at(-1)[0] === 'setClips' && trace.calls.at(-1)[2]?.restUrl === '/output/gold-robot_rigged.glb');
  assert.equal(trace.calls.at(-1)[1][0].file, 'gold-robot_jump.glb');
  await waitFor(() => document.querySelector('[data-skin-delete]'));
  change('[data-skin-pick]', '');
  await waitFor(() => document.querySelector('[data-skin-pick]')?.value === '');
  dressFails = true;
  change('[data-skin-pick]', 'gold');
  await waitFor(() => trace.calls.at(-1)[0] === 'showEmpty');
  assert.match(trace.calls.at(-1)[1], /Could not put this skin/);
  await waitFor(() => document.querySelector('[data-skin-delete]'));
  change('[data-skin-pick]', '');
  await waitFor(() => document.querySelector('[data-skin-pick]'));
  dressFails = false;
  const waiting = Promise.withResolvers();
  holdDress = waiting.promise;
  change('[data-skin-pick]', 'gold');
  asset('prop').click();
  await waitFor(() => stageTitle() === 'Vase' && document.querySelector('[data-skin-pick]'));
  waiting.resolve();
  await waitFor(() => stageTitle() === 'Vase');
  assert.equal(trace.calls.at(-1)[1], '/output/vase_full.glb');

  change('[data-skin-pick]', 'gold');
  await waitFor(() => document.querySelector('[data-skin-delete]'));
  assert.equal(trace.calls.at(-1)[1], '/output/gold.glb');
  assert.equal(document.querySelector('#download').getAttribute('download'), 'gold.glb');
  click('[data-skin-delete]');
  await waitFor(() => document.querySelector('.skins') && !document.querySelector('[data-skin-pick]'));

  asset(robot.id).click();
  await waitFor(() => document.querySelector('[data-skin-large]'));
  click('[data-skin-large]');
  await waitFor(() => trace.calls.at(-1)[0] === 'showImage');
  assert.equal(trace.calls.at(-1)[1], '/output/front.png');
  assert.match(trace.calls.at(-1)[2], /Blue &lt;paint&gt;/);
  assert.equal(trace.calls.at(-1)[3], '/output/blue.png?v=2');
  assert.deepEqual(trace.calls.at(-1)[4], ['Original', 'New']);

  const backBox = document.querySelector('[data-versions]');
  backBox.open = true;
  backBox.dispatchEvent(new Event('toggle'));
  await waitFor(() => document.querySelector('[data-show="old-back.png"]'));
  click('[data-show="old-back.png"]');
  assert.equal(trace.calls.at(-1)[2], 'A kept version of the back.');
  click('[data-use]');
  await waitFor(() => stageTitle() === 'New back');
  await waitFor(() => trace.calls.at(-1)[0] === 'showImage');
  assert.equal(trace.calls.at(-1)[2], 'The back in use.');

  await waitFor(() => document.querySelector('.skins [data-k="change"]'));
  document.querySelector('.skins [data-k="change"]').value = 'Gold cotton';
  click('[data-skin-new]');
  await waitFor(() => document.querySelector('[data-skin-pick]')?.value === 'gold');

  // A failed refresh reports the problem and keeps the last known asset available.
  libraryFails = true;
  window.history.replaceState(null, '', '#asset=acb000000001');
  window.dispatchEvent(new window.HashChangeEvent('hashchange'));
  await waitFor(() => stageTitle() === 'Robot');
  await waitFor(() => document.querySelector('[data-skin-pick]'));
  change('[data-skin-pick]', 'gold');
  await waitFor(() => document.querySelector('[data-skin-delete]'));
  click('[data-skin-delete]');
  await waitFor(() => trace.calls.some(([method, message]) =>
    method === 'showEmpty' && message.includes('Library did not load')));
  await waitFor(() => stageTitle() === 'Robot' && document.querySelector('[data-skin-pick]'));
  libraryFails = false;
  offlineRefresh = true;
  window.history.replaceState(null, '', '#asset=ffffffffffff');
  window.dispatchEvent(new window.HashChangeEvent('hashchange'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(stageTitle(), 'Robot');
});

test('studio only renders visible thumbnails and can use an in-memory cache when browser storage is blocked', async (t) => {
  const pending = Promise.withResolvers();
  const rendering = [];
  const observed = new Set();
  let observer;
  replaceGlobal(t, 'IntersectionObserver', class {
    constructor(callback, options) {
      this.fire = callback;
      assert.equal(options.root.className, 'scroll');
      observer = this;
    }
    disconnect() { observed.clear(); }
    observe(node) { observed.add(node); }
    unobserve(node) { observed.delete(node); }
  });
  const { window } = await openStudio(t, {
    url: 'file:///studio.html',
    handlers: {
      'GET /api/assets': { base: '/output/', assets: [before, empty, shape('another', { model: 'second.glb' })] },
      'GET /api/model-info': () => { throw new Error('Unavailable'); },
    },
    thumbnail: async (file) => { rendering.push(file); return pending.promise; },
  });
  await waitFor(() => observed.size === 2);
  document.dispatchEvent(new Event('visibilitychange'));
  assert.deepEqual(rendering, []);
  const img = [...observed][0];
  observer.fire([{ target: img, isIntersecting: false }]);
  assert.equal(observed.size, 2);
  Object.defineProperty(document, 'hidden', { value: false });
  observer.fire([{ target: img, isIntersecting: true }]);
  assert.equal(observed.size, 1);
  assert.deepEqual(rendering, ['/output/old_hero.glb']);
  const search = document.querySelector('#search');
  search.value = 'missing';
  search.dispatchEvent(new Event('input'));
  pending.resolve('data:image/png;base64,AA==');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(document.querySelector('[data-thumb-id]'), null);
  search.value = '';
  search.dispatchEvent(new Event('input'));
  assert.equal(document.querySelector('[data-thumb-id="robot-old"]').classList.contains('mark'), false);
  assert.equal(rendering.length, 1);
  asset('robot-old').click();
  await waitFor(() => stageTitle() === 'Old Robot');
  assert.match(document.querySelector('[data-step="finished"]').textContent, /Hero/);
  assert.equal(document.querySelector('#fileCard').hidden, true);
  assert.equal(window.document, document);
});

test('studio shows and dismisses a new run with no resulting asset, then opens a matching asset automatically', async (t) => {
  const steps = ['picture'];
  let rows = [];
  let status = { steps, status: 'running', current: 'picture', done: [], message: 'Waiting' };
  const { trace, calls } = await openStudio(t, {
    handlers: {
      'GET /api/assets': () => ({ base: '/output/', assets: rows }),
      'GET /api/catalog': { backends: [{ id: 'pixal3d', kind: '3d', state: 'ready', label: 'Pixal3D' }] },
      'POST /api/chains': { id: 'run' },
      'GET /api/chains/run': () => status,
    },
  });
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  await waitFor(() => document.querySelector('.sheet'));
  assert.match(document.querySelector('#assetList').textContent, /No assets yet/);
  change('#showHidden', true);
  assert.equal(document.querySelector('.hidden-toggle').hidden, false);
  const start = () => {
    click('[data-w="picture"][data-h="idea"]');
    const text = document.querySelector('.sheet textarea');
    text.value = 'Sketch of a bridge';
    text.dispatchEvent(new Event('input'));
    document.querySelector('.sheet form').dispatchEvent(new Event('submit', { cancelable: true }));
  };
  start();
  click('#progress [data-x]'); // The request is still starting, so no cancellation callback exists yet.
  await waitFor(() => document.querySelector('.job-msg').textContent.includes('Waiting'));
  status = { ...status, message: 'Still waiting', stalled: true, full_log: 'Late log' };
  t.mock.timers.tick(1200);
  await waitFor(() => document.querySelector('.pcard .log')?.textContent === 'Late log');
  status = { steps, status: 'done', current: null, done: ['picture'], made: { picture: 'drawn.png' } };
  t.mock.timers.tick(1200);
  await waitFor(() => !document.body.classList.contains('busy'));
  click('[data-making]');
  assert.match(document.querySelector('.job-msg').textContent, /All done/);
  assert.deepEqual(trace.calls.findLast(([method]) => method === 'showImage').slice(1), ['/output/drawn.png', '', null]);
  click('#progress [data-close]');
  assert.equal(stageTitle(), '');

  click('#createBtn');
  rows = [shape('drawn', { picture: 'drawn.png', name: 'Drawn', kind: 'picture' })];
  start();
  await waitFor(() => stageTitle() === 'Drawn');
  assert.equal(document.querySelector('[data-making]'), null);
  assert.equal(document.querySelector('#progress').hidden, true);
  assert.equal(calls.filter(({ method, path }) => method === 'POST' && path === '/api/chains').length, 2);
  await waitFor(() => !document.body.classList.contains('busy'));

  status = { steps: ['model'], status: 'running', current: 'model', done: [], message: 'Working' };
  click('[data-step="model"] [data-run]');
  await waitFor(() => document.querySelector('.job-msg').textContent.includes('Working'));
  const stepRun = calls.findLast(({ method, path }) => method === 'POST' && path === '/api/chains');
  assert.equal(stepRun.body.get('steps'), '["model"]');
  assert.equal(stepRun.body.get('made'), '{"picture":"drawn.png"}');
  asset('drawn').click();
  assert.ok(document.querySelector('#progress').hidden);
  status = { ...status, status: 'cancelled' };
  t.mock.timers.tick(1200);
  await waitFor(() => !document.body.classList.contains('busy'));
  assert.equal(stageTitle(), 'Drawn');
  assert.ok(document.querySelector('#progress').hidden);
});
