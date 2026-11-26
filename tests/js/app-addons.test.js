import assert from 'node:assert/strict';
import test from 'node:test';
import { browser, change, click, requests, waitFor } from './app-helpers.js';
import { openAddons } from '../../viewer/studio/addons_sheet.js';

const tools = [{
  id: 'shape tool', label: 'Build <shape>', hint: 'A small test shape.',
  inputs: [
    { id: 'kind', type: 'choice', label: 'Kind', choices: ['Plain', 'A&B'] },
    { id: 'smooth', type: 'toggle', label: 'Smooth' },
    { id: 'size', type: 'number', label: 'Size', min: 0, max: 10 },
    { id: 'amount', type: 'number', label: 'Amount' },
    { id: 'picture', type: 'picture', label: 'Picture' },
    { id: 'model', type: 'model', label: 'Model' },
    { id: 'name', type: 'text', label: 'Name' },
  ],
}, { id: 'second', label: 'Second tool', hint: '', inputs: [] }];
const installed = { id: 'shape builder', installed: true, name: 'Shapes', author: { name: 'Ada' }, tools, tier: 'verified' };
const catalog = (addons, online = true) => ({ online, addons });
const error = (message) => Response.json(message ? { error: message } : {}, { status: 400 });

function submit(form) {
  form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
}

test('add-ons page lists installed tools and confirms changes before sending requests', async (t) => {
  const { authorLine, pictureHtml, ratingLine, tierBadge } = await import('../../viewer/studio/addons_page.js');
  const { window } = browser(t, { file: 'addons.html' });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let addons = [installed, { id: 'standalone', name: 'Standalone', installed: true }, {
    id: 'new', name: 'Water & sky', tier: 'listed', runs: 'cloud', rating: 4, ratings_count: 2,
    author: { name: 'N. Doe', url: 'https://example.test/author' },
    links: { repo: 'https://example.test/repo', site: 'https://example.test/site' },
    picture: 'https://example.test/drawing.png', summary: 'Make <clouds>',
  }];
  let online = false;
  let loadError;
  let offlineMessage = 'Offline catalogue';
  let start = { job_id: 'job 1' };
  let installError, removeError, runError, statusError;
  let plan = { id: 'new', author: 'N. Doe', licence: 'MIT', name: 'Water & sky', version: '2', warning: 'Check weights', models: [{ name: 'Sky', licence: 'MIT', size: '2 GB' }] };
  let status = { status: 'queued', log_tail: 'Working' };
  const calls = requests(t, {
    'GET /api/addons': () => loadError ?? { ...catalog(addons, online), error: offlineMessage },
    'POST /api/addons/new/install': (body) => {
      if (installError) return installError;
      if (!body.confirm) return plan;
      addons = addons.map((addon) => addon.id === 'new' ? { ...addon, installed: true } : addon);
      return {};
    },
    'POST /api/addons/new/remove': () => {
      if (removeError) return removeError;
      addons = addons.filter((addon) => addon.id !== 'new');
      return {};
    },
    'POST /api/addons/shape%20builder/run/shape%20tool': () => runError ?? start,
    'GET /api/addons/jobs/job%201/status': () => statusError ?? status,
  });

  await import('../../viewer/studio/addons_page.js?page');
  await waitFor(() => document.querySelectorAll('.addon-card').length === 3);
  assert.equal(document.querySelector('.addon-card').dataset.addon, installed.id);
  assert.match(document.querySelector('#addons-summary').textContent, /Offline catalogue/);
  assert.equal(document.querySelector('.addon-picture').alt, 'What Water & sky makes');
  assert.match(document.querySelector('.addon-rating').textContent, /4 \(2\)/);
  assert.equal(document.querySelector('a[href="https://example.test/repo"]').textContent, 'Source');
  assert.equal(document.querySelector('a[href="https://example.test/site"]').textContent, 'Site page');
  assert.equal(authorLine({ author: { url: 'http://example.test' } }), 'by unknown');
  assert.match(tierBadge({ tier: 'listed', run_checked: false }), /Not run-checked/);
  assert.match(tierBadge({ tier: 'listed', run_skipped: ['<a&"b>'] }), /file: ab/);
  assert.match(tierBadge({ tier: 'listed' }), /Listed/);
  assert.match(tierBadge({ no_longer_listed: true }), /No longer listed/);
  assert.match(pictureHtml({ picture: 'https://example.test/pic' }), /this add-on/);
  assert.equal(pictureHtml({ picture: 'http://example.test/pic' }), '');
  assert.match(ratingLine({ rating: 0 }), /★ 0<\/span>/);

  const newCard = () => document.querySelector('[data-addon="new"].addon-card');
  installError = error();
  click('.addon-install', newCard());
  await waitFor(() => /Could not load install plan: unknown error/.test(newCard().textContent));
  assert.equal(newCard().querySelector('.addon-install').disabled, false);
  installError = null;
  click('.addon-install', newCard());
  await waitFor(() => newCard().querySelector('.addon-confirm'));
  assert.match(newCard().querySelector('.addon-plan').textContent, /Sky.*MIT, 2 GB/);
  assert.equal(calls.filter(({ body }) => body?.confirm).length, 0);
  installError = error();
  click('.addon-confirm', newCard());
  await waitFor(() => /Install failed: unknown error/.test(newCard().textContent));

  // The app reloads its page after a successful install or removal.
  await import('../../viewer/studio/addons_page.js?install-retry');
  await waitFor(() => newCard().querySelector('.addon-install')?.disabled === false);
  installError = null;
  online = true;
  plan = { id: 'new', name: 'Water & sky', author: 'N. Doe', licence: 'MIT', models: [] };
  click('.addon-install', newCard());
  await waitFor(() => newCard().querySelector('.addon-confirm'));
  assert.equal(newCard().querySelector('.addon-plan ul'), null);
  click('.addon-confirm', newCard());
  await waitFor(() => newCard().querySelector('.addon-remove'));
  assert.ok(document.querySelector('#addons-summary').hidden);
  assert.deepEqual(calls.filter(({ body }) => body?.confirm).map(({ body }) => body), [{ confirm: true }, { confirm: true }]);
  click('.addon-remove', newCard());
  click('.addon-cancel-remove', newCard());
  assert.equal(newCard().querySelector('.addon-plan-box'), null);
  click('.addon-remove', newCard());
  removeError = error('In use');
  click('.addon-confirm-remove', newCard());
  await waitFor(() => /Remove failed: In use/.test(newCard().textContent));
  click('.addon-remove', newCard());
  removeError = error();
  click('.addon-confirm-remove', newCard());
  await waitFor(() => /Remove failed: unknown error/.test(newCard().textContent));
  click('.addon-remove', newCard());
  removeError = null;
  click('.addon-confirm-remove', newCard());
  await waitFor(() => newCard() === null);

  const toggle = click('.addon-tools-toggle');
  assert.equal(toggle.textContent, 'Hide tools');
  click('.addon-tools-toggle');
  assert.equal(toggle.textContent, 'Tools (2)');
  click('.addon-tools-toggle');
  const [first, second] = document.querySelectorAll('.addon-tool-form');
  change('[data-input="kind"]', 'A&B', first);
  change('[data-input="smooth"]', true, first);
  change('[data-input="size"]', '3', first);
  change('[data-input="name"]', 'Boat', first);
  assert.equal(first.querySelector('[data-input="model"]').placeholder, 'Library or file path');
  assert.equal(first.querySelector('[data-input="name"]').placeholder, '');
  submit(first);
  const log = first.parentElement.querySelector('.addon-log');
  await waitFor(() => log.textContent === 'Working');
  assert.deepEqual(calls.find(({ path }) => path.includes('/run/')).body, {
    inputs: { kind: 'A&B', smooth: true, size: '3', amount: '', picture: '', model: '', name: 'Boat' },
  });
  submit(second);
  submit(second);
  assert.match(second.parentElement.textContent, /already in progress/);
  status = { status: 'done', result: 'boat.glb' };
  t.mock.timers.tick(1000);
  await waitFor(() => log.textContent.includes('Done: boat.glb'));
  assert.equal(first.querySelector('button').disabled, false);
  status = { status: 'error', error: 'Tool broke' };
  submit(first);
  await waitFor(() => log.textContent.includes('Tool broke'));
  status = { status: 'error' };
  submit(first);
  await waitFor(() => log.textContent.includes('run failed'));
  statusError = error();
  submit(first);
  await waitFor(() => log.textContent.includes('unknown error'));
  statusError = error('Missing job');
  submit(first);
  await waitFor(() => log.textContent.includes('Missing job'));
  runError = error('Invalid input');
  submit(first);
  await waitFor(() => log.textContent.includes('Invalid input'));
  runError = error();
  submit(first);
  await waitFor(() => log.textContent.includes('unknown error'));
  runError = null;
  start = {}; // Malformed job response must leave the tool usable.
  submit(first);
  await waitFor(() => !first.querySelector('button').disabled);
  online = false;
  offlineMessage = undefined;
  const load = () => import(`../../viewer/studio/addons_page.js?refresh=${calls.length}`);
  addons = [];
  await load();
  await waitFor(() => /No add-ons listed/.test(document.querySelector('#addons-list').textContent));
  assert.match(document.querySelector('#addons-summary').textContent, /list is unreachable/);
  loadError = error();
  await load();
  await waitFor(() => /Could not load add-ons: HTTP 400/.test(document.querySelector('#addons-summary').textContent));
  assert.equal(window.document, document);
});

test('studio add-ons sheet filters the catalogue and reports tool results to the library', async (t) => {
  const { window } = browser(t);
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const stage = document.querySelector('main');
  let response = catalog([
    installed,
    { id: 'missing', tools, installed: false },
    { id: 'no-tools', installed: true },
    { ...installed, id: 'unknown-author', author: undefined, tools: [tools[1]] },
  ]);
  let run = { job_id: 'id' };
  let status = { status: 'running', log_tail: 'Running' };
  let refreshed = 0;
  document.addEventListener('viewer:refresh-library', () => { refreshed++; });
  const calls = requests(t, {
    'GET /api/addons': () => response,
    'POST /api/addons/shape%20builder/run/shape%20tool': () => run,
    'GET /api/addons/jobs/id/status': () => status,
  });
  await openAddons(stage);
  assert.equal(stage.querySelectorAll('form').length, 3);
  assert.match(stage.textContent, /by Ada/);
  assert.match(stage.textContent, /by unknown/);
  const [form, second] = stage.querySelectorAll('form');
  change('[data-input="smooth"]', true, form);
  assert.equal(form.querySelector('[data-input="size"]').min, '0');
  assert.equal(form.querySelector('[data-input="picture"]').placeholder, 'Library or file path');
  submit(form);
  const log = form.parentElement.querySelector('.addon-log');
  await waitFor(() => log.textContent === 'Running');
  assert.ok(calls.find(({ path }) => path.includes('/run/')).body.inputs.smooth);
  submit(second);
  submit(second);
  assert.match(second.parentElement.textContent, /already in progress/);
  status = { status: 'done', result: 'built.glb' };
  t.mock.timers.tick(1000);
  await waitFor(() => refreshed === 1);
  assert.match(log.textContent, /Done: built.glb/);
  status = { status: 'error' };
  submit(form);
  await waitFor(() => log.textContent.includes('run failed'));
  status = { status: 'error', error: 'Bad mesh' };
  submit(form);
  await waitFor(() => log.textContent.includes('Bad mesh'));
  status = error('Job lost');
  submit(form);
  await waitFor(() => log.textContent.includes('Job lost'));
  status = error();
  submit(form);
  await waitFor(() => log.textContent.includes('unknown error'));
  run = error('Invalid argument');
  submit(form);
  await waitFor(() => log.textContent.includes('Invalid argument'));
  run = error();
  submit(form);
  await waitFor(() => log.textContent.includes('unknown error'));
  run = {};
  submit(form);
  await waitFor(() => !form.querySelector('button').disabled);
  run = { job_id: 'id' };
  status = { status: 'running', log_tail: 'Wait' };
  submit(form);
  await waitFor(() => log.textContent === 'Wait');
  const sheet = stage.querySelector('.sheet');
  sheet.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter' }));
  assert.ok(sheet.isConnected);
  sheet.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
  const callCount = calls.length;
  t.mock.timers.tick(1000);
  assert.equal(calls.length, callCount);
  assert.equal(stage.children.length, 0);
  // Opening an empty sheet is safe even if the previous poll already ended.
  response = error();
  await openAddons(stage);
  assert.match(stage.textContent, /No add-ons installed/);
  click('[data-close]');
});
