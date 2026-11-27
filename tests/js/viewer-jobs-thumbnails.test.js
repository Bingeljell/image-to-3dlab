import assert from 'node:assert/strict';
import test from 'node:test';
import { flushEvents, installViewerBrowser } from './viewer-browser.js';
import { copyDetails, followJob, jobEnd, logScroll, moveBox, percentOf, plainError, readableLog, readyEngines, resizeBox } from '../../viewer/studio/jobs.js';
import { createThumbCache, createThumbQueue, thumbFile, thumbKey } from '../../viewer/studio/model-thumbs.js';
import { activeChoice, modelChoices } from '../../viewer/studio/model-switch.js';

test('job display respects engine readiness, progress units and error details', async () => {
  assert.deepEqual(readyEngines(null), []);
  assert.deepEqual(readyEngines({ backends: [
    { id: 'second', kind: '3d', state: 'ready', rank: 1, license: { name: 'MIT' }, caveat: 'local' },
    { id: 'third', kind: '3d', state: 'ready' },
    { id: 'absent', kind: '3d' }, { id: 'picture', kind: 'image', state: 'ready' },
  ] }).map(({ id, license, caveat }) => [id, license, caveat]), [
    ['second', 'MIT', 'local'], ['third', '', null],
  ]);
  for (const [event, pct] of [[null, null], [{ pct: -20 }, 0], [{ percent: 200 }, 100],
    [{ progress: 0.12 }, 12], [{ progress: 29 }, 29], [{ pct: 0.5 }, 0.5]]) {
    assert.equal(percentOf(event), pct);
  }
  for (const [raw, expected] of [
    ['Install Blender', /Install Blender/], ['Killed', /ran out of memory/],
    ['needs_setup', /engine is not installed/], ['wait for it to finish', /Another job/],
    ['could not map', /humanoid/], ['Blender crashed', /Blender stopped/], ['Message omitted', /Message omitted/],
  ]) assert.match(plainError(raw), expected);
  assert.match(plainError(), /stopped without saying why/);
  assert.equal(readableLog('1\n2\n[DEBUG] skip\n3\n4\n5'), '2\n3\n4\n5');
  assert.equal(jobEnd({ status: 'error', log_tail: 'last line' }).why, 'last line');
  assert.equal(jobEnd({}).log, '');
  assert.equal(logScroll(null, 50), 50);
  assert.equal(logScroll({ scrollTop: 10, clientHeight: 20, scrollHeight: 50 }, 60), 10);
  const copied = [];
  const source = async (url, options) => {
    assert.equal(url, '/api/bug-report');
    assert.deepEqual(JSON.parse(options.body), { message: 'A step failed', details: 'D'.repeat(30000) });
    return Response.json({ text: 'scrubbed' });
  };
  assert.equal(await copyDetails('', 'A' + 'D'.repeat(30000), {
    send: source, clipboard: { writeText: (text) => copied.push(text) },
  }), true);
  assert.deepEqual(copied, ['scrubbed']);
  assert.equal(await copyDetails(null, null, { send: () => new Response('', { status: 500 }) }), false);
  assert.equal(await copyDetails('', undefined, { send: async (url, opts) => {
    assert.equal(JSON.parse(opts.body).details, '');
    return new Response('', { status: 500 });
  } }), false);
  assert.deepEqual(moveBox({ x: 20, y: 20, w: 300, h: 250 }, -60, 100, { w: 400, h: 300 }),
    { x: 0, y: 50, w: 300, h: 250 });
  assert.deepEqual(resizeBox({ x: 20, y: 20, w: 300, h: 250 }, 50, -80, { w: 325, h: 350 }),
    { x: 0, y: 20, w: 325, h: 220 });
});

test('job follower recovers after a missed request and can stop polling', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const updates = [];
  let polls = 0;
  const job = followJob('/status', {
    interval: 100, now: () => 1000, onUpdate: (value) => updates.push(value),
    fetchImpl: () => {
      polls++;
      if (polls === 1) throw new TypeError('temporary');
      return Response.json(polls === 2 ? { status: 'running', last_event: { progress: 0.2 } } : { status: 'done' });
    },
  });
  await flushEvents();
  assert.deepEqual(updates, []);
  t.mock.timers.tick(100);
  await flushEvents();
  assert.equal(updates[0].percent, 20);
  t.mock.timers.tick(100);
  assert.equal((await job.done).status, 'done');
  const abandoned = followJob('/status', { interval: 10, fetchImpl: () => { polls++; return Response.json({ status: 'running' }); } });
  await flushEvents();
  abandoned.stop();
  t.mock.timers.tick(100);
  await flushEvents();
  assert.equal(polls, 4);
  t.mock.method(globalThis, 'fetch', async () => Response.json({ status: 'done' }));
  assert.equal((await followJob('/status').done).status, 'done');
});

test('model switch shows the available stages and selects an explicit stage over the latest file', () => {
  const asset = { model: 'raw.glb', finished: 'finish.glb', rigged: 'rig.glb' };
  assert.deepEqual(modelChoices(null), []);
  assert.deepEqual(modelChoices({ ...asset, kind: 'prop set' }), []);
  assert.deepEqual(modelChoices({ picture: 'a.png' }), []);
  const choices = modelChoices(asset);
  assert.deepEqual(choices.map(({ step, file }) => [step, file]), [
    ['model', asset.model], ['finished', asset.finished], ['rigged', asset.rigged],
  ]);
  assert.equal(activeChoice(choices, { step: 'finished', latest: asset.rigged }), 'finished');
  assert.equal(activeChoice(choices, { step: 'picture' }), null);
  assert.equal(activeChoice(choices, { latest: asset.rigged }), 'rigged');
  assert.equal(activeChoice(choices, { latest: 'unknown' }), null);
  assert.equal(activeChoice(choices), null);
  assert.equal(thumbFile(asset), asset.finished);
  assert.equal(thumbFile({ ...asset, picture: 'image.png' }), null);
  assert.equal(thumbFile({ model: 'a', rigged: 'b' }), 'a');
  assert.equal(thumbFile({ rigged: 'b' }), 'b');
  assert.equal(thumbFile(null), null);
  assert.equal(thumbFile({}), null);
});

test('thumbnail cache uses browser storage when available and evicts older entries', (t) => {
  const { window } = installViewerBrowser(t);
  const { localStorage } = window;
  localStorage.setItem('af-thumb-index', 'broken');
  const cache = createThumbCache({ storage: localStorage, limit: 2 });
  cache.set(thumbKey('a'), 'a.png');
  cache.set(thumbKey('b'), 'b.png');
  cache.set(thumbKey('a'), 'updated.png');
  cache.set(thumbKey('c'), 'c.png');
  assert.equal(cache.get(thumbKey('a')), 'updated.png');
  assert.equal(cache.get(thumbKey('b')), null);
  assert.equal(localStorage.getItem(thumbKey('b')), null);
  const reloaded = createThumbCache({ storage: localStorage });
  assert.equal(reloaded.get(thumbKey('a')), 'updated.png');
  localStorage.setItem('af-thumb-index', JSON.stringify({ invalid: 1 }));
  assert.equal(createThumbCache({ storage: localStorage }).get(thumbKey('c')), 'c.png');
  localStorage.setItem('af-thumb-index', JSON.stringify([null, thumbKey('a')]));
  const filtered = createThumbCache({ storage: localStorage, limit: 2 });
  filtered.set(thumbKey('c'), 'c.png');
  assert.deepEqual(JSON.parse(localStorage.getItem('af-thumb-index')), [thumbKey('a'), thumbKey('c')]);

  const denied = () => { throw new Error('storage denied'); };
  const memory = createThumbCache({ storage: { getItem: denied, setItem: denied, removeItem: denied }, limit: 1 });
  memory.set('first', 'picture');
  memory.set('second', 'newer');
  assert.equal(memory.get('first'), null);
  assert.equal(memory.get('second'), 'newer');
});

test('thumbnail queue pauses for hidden tabs, deduplicates and caps work after an error', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const cache = createThumbCache();
  let visible = false, resolveFirst;
  const rendered = [], thumbs = [];
  const queue = createThumbQueue({
    cache, limit: 2, gap: 10, isHidden: () => !visible,
    render: (file) => {
      rendered.push(file);
      if (file === 'broken') throw new Error('Unreadable mesh');
      return new Promise((resolve) => { resolveFirst = resolve; });
    },
    onThumb: (item, data) => thumbs.push([item.file, data]),
  });
  const item = { file: 'first' };
  assert.equal(queue.offer(item), 'wanted');
  queue.enqueue(item);
  queue.enqueue({ file: 'broken' });
  queue.enqueue(item);
  assert.deepEqual(rendered, []);
  assert.equal(queue.pending(), 2);
  visible = true;
  queue.resume();
  queue.enqueue(item);
  assert.deepEqual(rendered, ['first']);
  resolveFirst('displayable');
  await flushEvents();
  t.mock.timers.tick(10);
  await flushEvents();
  assert.equal(queue.rendered(), 2);
  queue.enqueue({ file: 'ignored' });
  queue.enqueue({ file: 'broken' });
  queue.enqueue(item);
  assert.equal(queue.pending(), 0);
  assert.equal(queue.offer(item), 'cached');
  assert.equal(queue.offer({ file: 'broken' }), 'skip');
  assert.deepEqual(rendered, ['first', 'broken']);
  assert.deepEqual(thumbs, [['first', 'displayable'], ['first', 'displayable']]);
  const defaults = createThumbQueue({ cache, render: () => 'fresh' });
  defaults.enqueue({ file: 'second' });
  await flushEvents();
  assert.equal(defaults.rendered(), 1);
});
