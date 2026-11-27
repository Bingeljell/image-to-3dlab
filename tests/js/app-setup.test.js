import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { browser, click, replaceGlobal, requests, waitFor } from './app-helpers.js';

const bad = (message, status = 400) => Response.json(message ? { error: message } : {}, { status });
const backend = (id, fields = {}) => ({
  id, label: id, best_for: 'Small game assets', tradeoff: 'Runs on your machine.',
  state: 'missing', action: 'download', bytes_present: 0, bytes_expected: 2 ** 31,
  human_expected: '2 GB', license: { name: 'MIT', url: 'https://example.test/license' },
  weights: [{ label: 'Weights', source: 'demo/weights', path: '/models/demo/weights.bin',
    source_url: 'https://example.test/weights', human_expected: '2 GB' }],
  extra_steps: [], ...fields,
});

test('setup confirms downloads and removals, follows streamed events and reconnects to active runs', async (t) => {
  const { weightRow } = await import('../../viewer/studio/setup.js');
  const { window } = browser(t, { file: 'setup.html' });
  window.HTMLElement.prototype.scrollIntoView = () => {};
  replaceGlobal(t, 'CSS', { escape: (identifier) => identifier });
  const streams = [];
  replaceGlobal(t, 'EventSource', class {
    constructor(url) {
      this.url = url;
      this.closed = false;
      streams.push(this);
    }
    close() { this.closed = true; }
    emit(event) { this.onmessage({ data: JSON.stringify(event) }); }
  });
  const questions = [], prompts = [], alerts = [];
  let accept = false, typed = null;
  window.confirm = (question) => { questions.push(question); return accept; };
  window.prompt = (prompt) => { prompts.push(prompt); return typed; };
  window.alert = (message) => alerts.push(message);

  const routes = [
    backend('metal', { action: 'build', recommended: true, build_hint: 'The build is missing.',
      caveat: 'Some files need approval.', platform_note: 'Runs on Metal.',
      also_fetches: [{ from: 'https://example.test/extra', what: 'Utility', size: '1 MB', licence: 'MIT' }],
      extra_steps: ['Read docs/hunyuan-mlx-recipes.md before installing.'] }),
    backend('cuda', { action: 'build', setup_fetches_weights: true, rebuild_reason: 'New build', rebuild_downloads: '50 MB' }),
    backend('partial', { action: 'resume', state: 'partial', bytes_present: 2 ** 20,
      human_present: '1 MB', weights: [{ label: 'Weights', source: 'Weights', path: '/cache/huggingface/partial.bin',
        present: true, human_present: '1 MB', note: 'Already here' }] }),
    backend('ready', { action: 'none', state: 'ready', bytes_present: 2 ** 30, bytes_removable: 512,
      human_present: '1 GB', rebuild_reason: 'Patch available',
      weights: [{ label: 'Weights', path: '/models/weights.bin', present: true }] }),
    backend('manual', { action: 'manual', install: './lab manual' }),
    backend('unsupported', { supported_here: false, state: 'unexpected', rebuild_reason: 'Unavailable' }),
    backend('download', { bytes_expected: 0 }),
  ];
  const host = { id: 'mac', label: 'Mac', any_backend_runs_here: true, supported: ['Mac', 'Linux'] };
  const platforms = [
    { id: 'mac', label: 'Mac' },
    { id: 'linux', label: 'Linux' },
    { id: 'future', label: 'Future', coming: true },
  ];
  let catalog = { host, platforms, backends: routes, views: { mac: routes, linux: routes.slice(0, 1) } };
  let installError;
  let removeError;
  let hf = { signed_in: false, repos: [
    { repo: 'yes', access: 'yes', for: 'A' },
    { repo: 'no', access: 'no', for: 'B', request_url: 'https://example.test/access' },
    { repo: 'unknown', for: 'C' },
  ] };
  let hfError = bad('Bad token');
  let folder = { default: '/default', saved: '/extra', restart_needed: true };
  let pendingFolder;
  let saveFolder = { saved: '/other', restart_needed: true };
  let blender = { blender_installable: true };
  let gltfpack = { gltfpack_installable: true };
  let blenderInstall = bad('No space');
  let gltfInstall = bad();
  const response = (fixture) => {
    if (fixture instanceof Error) throw fixture;
    return fixture;
  };
  const calls = requests(t, {
    'GET /api/catalog': () => response(catalog),
    'GET /api/finish/capabilities': () => response(blender),
    'GET /api/props/tools': () => response(gltfpack),
    'GET /api/hf/status': () => response(hf),
    'GET /api/models-folder': () => pendingFolder ?? response(folder),
    'POST /api/hf/sign-in': () => hfError ?? {},
    'POST /api/models-folder': () => saveFolder,
    'POST /api/blender/install': () => blenderInstall,
    'POST /api/gltfpack/install': () => gltfInstall,
    'POST /api/setup/metal/download': () => installError ?? { events_url: '/events/metal' },
    'POST /api/setup/metal/cancel': {},
    'POST /api/setup/cuda/rebuild': { events_url: '/events/cuda' },
    'POST /api/setup/partial/remove': () => removeError ?? { freed: '1 MB' },
    'POST /api/setup/ready/remove': () => removeError ?? { freed: '512 B' },
  });
  let loads = 0;
  document.addEventListener('viewer:catalog', () => loads++);
  const { load } = await import('../../viewer/studio/setup.js?page');
  const card = (id) => document.querySelector(`[data-backend="${id}"]`);
  const label = (id) => card(id).querySelector('.setup-card-state').textContent;
  const summary = () => document.querySelector('#setup-summary').textContent;
  const panel = () => document.querySelector('#setup-run');
  const runDetail = () => document.querySelector('#setup-run-detail').textContent;
  const runButton = (id) => click('.setup-card-action button', card(id));
  const tab = (id) => {
    const button = [...document.querySelectorAll('.setup-tab')].find((item) => item.textContent.startsWith(id));
    assert.ok(button);
    button.click();
  };
  await waitFor(() => loads === 1);
  assert.match(summary(), /1 of 7 backends installed/);
  assert.equal(document.querySelector('#setup-backends').firstElementChild.dataset.backend, 'blender');
  assert.match(card('metal').textContent, /start here/);
  assert.match(card('metal').textContent, /The build is missing/);
  assert.equal(card('metal').querySelector('.setup-card-extra a').getAttribute('href'), '/docs/hunyuan-mlx-recipes.md');
  assert.match(card('metal').textContent, /Utility/);
  assert.match(label('partial'), /partly downloaded · 1 MB/);
  assert.match(card('partial').textContent, /Already here/);
  assert.match(card('manual').querySelector('code').textContent, /lab manual/);
  assert.match(card('unsupported').textContent, /not on this OS yet/);
  assert.match(card('ready').textContent, /✓ ready/);
  assert.match(weightRow({ source_url: '/missing' }), /<a href="\/missing"[^>]*><\/a> — $/);

  tab('Linux');
  assert.match(summary(), /Showing what runs on Linux/);
  assert.equal(document.querySelectorAll('#setup-backends [data-backend]').length, 1);
  assert.equal(card('metal').querySelector('button'), null);
  assert.match(label('metal'), /on that machine · 2 GB/);
  assert.doesNotMatch(card('metal').textContent, /Runs on Metal/);
  tab('Future');
  assert.match(document.querySelector('#setup-backends').textContent, /support is coming/);
  const lateFolder = Promise.withResolvers();
  const bodyRead = Promise.withResolvers();
  pendingFolder = lateFolder.promise;
  const folderRequests = () => calls.filter(({ path }) => path === '/api/models-folder').length;
  const requestsBeforeSwitch = folderRequests();
  tab('Mac');
  await waitFor(() => folderRequests() > requestsBeforeSwitch);
  tab('Linux'); // a tab change supersedes the pending local tool queries
  const oldResponse = Response.json(folder);
  const read = oldResponse.json.bind(oldResponse);
  oldResponse.json = async () => {
    const body = await read();
    bodyRead.resolve();
    return body;
  };
  lateFolder.resolve(oldResponse);
  await bodyRead.promise;
  await setImmediate(); // Let Setup apply or discard the parsed response.
  pendingFolder = null;
  assert.match(summary(), /Showing what runs on Linux/);
  assert.equal(document.querySelector('.setup-tab[aria-selected="true"]').textContent, 'Linux');
  for (const tool of ['blender', 'gltfpack', 'hf', 'models-folder']) assert.equal(card(tool), null, tool);
  assert.equal(document.querySelectorAll('#setup-backends .elsewhere').length, 1);
  tab('Mac');
  await waitFor(() => card('models-folder'));

  runButton('metal');
  assert.match(questions.at(-1), /downloads no weights/);
  assert.match(questions.at(-1), /MIT/);
  assert.match(questions.at(-1), /Some files need approval/);
  runButton('partial');
  assert.match(questions.at(-1), /About 2.0 GB will be downloaded/);
  runButton('cuda');
  assert.match(questions.at(-1), /This builds the code/);
  runButton('download');
  assert.match(questions.at(-1), /About 0 B/);
  runButton('ready');
  assert.match(questions.at(-1), /downloads nothing/);
  assert.equal(calls.some(({ method }) => method === 'POST'), false);
  accept = true;
  installError = bad('Disk full');
  runButton('metal');
  await waitFor(() => /Disk full/.test(runDetail()));
  assert.equal(card('metal').querySelector('button').disabled, false);
  installError = bad();
  runButton('metal');
  await waitFor(() => /HTTP 400/.test(runDetail()));
  installError = null;
  runButton('metal');
  await waitFor(() => streams.length === 1);
  assert.equal(card('metal').nextElementSibling, panel());
  streams[0].emit(null);
  streams[0].emit({ phase: 'queued', log: 'queued' });
  streams[0].emit({ phase: 'building', detail: 'Compiling' });
  assert.ok(panel().querySelector('.progress-track').classList.contains('busy'));
  assert.equal(runDetail(), 'Compiling');
  streams[0].emit({ phase: 'download', overall_pct: 50, log: 'downloading' });
  assert.equal(panel().querySelector('#setup-run-bar').style.width, '50%');
  assert.match(panel().querySelector('#setup-log').textContent, /queued\ndownloading/);
  click('#setup-run-cancel');
  assert.equal(calls.at(-1).path, '/api/setup/metal/cancel');
  streams[0].emit({ phase: 'cancelled' });
  await waitFor(() => loads === 2);
  assert.ok(streams[0].closed);
  assert.ok(panel().querySelector('#setup-run-cancel').hidden);
  const rebuild = [...card('cuda').querySelectorAll('button')].find((button) => button.textContent === 'Rebuild');
  rebuild.click();
  await waitFor(() => streams.length === 2);
  assert.match(questions.at(-1), /downloads 50 MB/);
  streams[1].emit({ phase: 'error', detail: 'Interrupted' });
  await waitFor(() => loads === 3);
  catalog.running_setup = { backend: 'unlisted', rebuild: true, events_url: '/events/resume' };
  await load();
  assert.equal(runDetail(), 'reconnecting…');
  assert.equal(streams.at(-1).url, '/events/resume');
  await load(); // reloading with an open stream leaves it alone
  assert.equal(streams.length, 3);
  delete catalog.running_setup;
  streams.at(-1).emit({ phase: 'done' });
  await waitFor(() => loads === 6);
  catalog.running_setup = { backend: 'metal', events_url: '/events/metal-resume' };
  await load();
  delete catalog.running_setup;
  assert.equal(card('metal').nextElementSibling, panel());
  streams.at(-1).emit({ phase: 'done' });
  await waitFor(() => loads === 8);

  const remove = (id) => click('.setup-remove', card(id));
  remove('partial');
  assert.match(prompts.at(-1), /shared Hugging Face cache/);
  typed = 'wrong';
  remove('partial');
  assert.match(alerts.at(-1), /Not deleted/);
  typed = ' PARTIAL ';
  removeError = bad('In use');
  remove('partial');
  await waitFor(() => loads === 9);
  assert.match(alerts.at(-1), /In use/);
  removeError = bad();
  remove('partial');
  await waitFor(() => loads === 10);
  assert.match(alerts.at(-1), /HTTP 400/);
  removeError = null;
  remove('partial');
  await waitFor(() => loads === 11);
  typed = null;
  remove('ready');
  assert.match(prompts.at(-1), /Delete 512 B/);
  assert.match(prompts.at(-1), /cannot be undone/);

  const signIn = () => click('.setup-hf-form button', card('huggingface'));
  assert.match(card('huggingface').textContent, /Request access/);
  assert.match(card('huggingface').textContent, /sign in to check/);
  card('huggingface').querySelector('input').value = 'synthetic-token';
  signIn();
  await waitFor(() => /Bad token/.test(card('huggingface').textContent));
  assert.deepEqual(calls.at(-1).body, { token: 'synthetic-token' });
  assert.equal(card('huggingface').querySelector('input').value, '');
  hfError = bad();
  signIn();
  await waitFor(() => /HTTP 400/.test(card('huggingface').textContent));
  hf = { signed_in: true, user: 'tester', repos: [] };
  hfError = null;
  signIn();
  await waitFor(() => /signed in as tester/.test(card('huggingface').textContent));

  const save = async (fixture, message) => {
    saveFolder = fixture;
    click('button', card('models-folder'));
    await waitFor(() => card('models-folder').querySelector('[data-note]').textContent.includes(message));
  };
  assert.match(card('models-folder').querySelector('[data-where]').textContent, /\/default/);
  assert.match(card('models-folder').querySelector('[data-note]').textContent, /\/extra/);
  await save({ saved: '/other', restart_needed: true }, '/other');
  await save({}, 'Saved.');
  await save(bad('Invalid location'), 'Invalid location');
  await save(bad(), 'HTTP 400');

  const install = async (id, streamName, setPayload, firstMessage) => {
    const button = card(id).querySelector('button');
    accept = false;
    const count = calls.length;
    button.click();
    assert.equal(calls.length, count);
    accept = true;
    button.click();
    await waitFor(() => card(id).textContent.includes(firstMessage));
    setPayload(bad());
    button.click();
    await waitFor(() => card(id).textContent.includes('HTTP 400'));
    setPayload({ events_url: streamName });
    button.click();
    await waitFor(() => streams.at(-1).url === streamName);
    const stream = streams.at(-1);
    stream.emit({ phase: 'queued' });
    stream.emit({ phase: 'download', message: 'Installing files' });
    assert.match(card(id).textContent, /Installing files/);
    const before = loads;
    stream.emit({ phase: 'setup_done' });
    await waitFor(() => loads > before);
    assert.ok(stream.closed);
  };
  await install('blender', '/events/blender', (value) => { blenderInstall = value; }, 'No space');
  gltfInstall = bad('Unavailable');
  await install('gltfpack', '/events/gltfpack', (value) => { gltfInstall = value; }, 'Unavailable');

  blender = { blender: '/bin/blender', blender_version: '4.2' };
  gltfpack = { gltfpack: '/bin/gltfpack' };
  folder = { in_use: '/active' };
  await load();
  assert.match(card('blender').textContent, /4.2 at \/bin\/blender/);
  assert.match(card('gltfpack').textContent, /found at \/bin\/gltfpack/);
  assert.match(card('models-folder').textContent, /\/active/);
  blender = { blender: '/bin/blender' };
  await load();
  assert.match(card('blender').textContent, /at \/bin\/blender/);
  blender = { blender_problem: 'Wrong version' };
  gltfpack = hf = folder = new Error('Older server');
  await load();
  assert.match(card('blender').textContent, /Wrong version/);
  assert.equal(card('models-folder'), null);
  assert.match(card('huggingface').textContent, /not signed in/);
  blender = bad();
  gltfpack = hf = folder = bad();
  await load();
  assert.match(card('blender').textContent, /not found/);
  assert.match(card('gltfpack').textContent, /not found/);

  catalog = { host, backends: [routes[2]] };
  await load();
  assert.match(summary(), /partial is partly downloaded/);
  catalog.backends = [routes[2], { ...routes[2], id: 'other', label: 'other' }];
  await load();
  assert.match(summary(), /partial, other are partly downloaded/);
  catalog.backends = [routes[0]];
  await load();
  assert.match(summary(), /No backend installed/);
  catalog.host = { ...host, any_backend_runs_here: false, label: 'Other' };
  await load();
  assert.match(summary(), /Every backend here needs Mac or Linux/);
  catalog = { host: { ...host, id: 'unknown-host' }, platforms, backends: routes,
    views: { mac: routes, linux: routes.slice(0, 1) } };
  const { load: loadForNewHost } = await import('../../viewer/studio/setup.js?unknown-host');
  await waitFor(() => /Showing what runs on Mac/.test(summary()));
  catalog = { ...catalog, platforms: platforms.slice(1) };
  await loadForNewHost();
  assert.match(summary(), /Showing what runs on mac/); // A stale tab falls back to its identifier.
  catalog = bad(null, 503);
  await load();
  assert.match(summary(), /Could not read machine status: HTTP 503/);
});
