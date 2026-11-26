import assert from 'node:assert/strict';
import test from 'node:test';
import { browser, change, click, requests, waitFor } from './app-helpers.js';
import { GUIDE_URL, isGuideLink, openPromptGuide, renderGuide } from '../../viewer/studio/guide.js';
import {
  byDay, closeSheet, duration, newsCount, openAbout, openActivity, openGuide,
  openNews, openReport, refreshNews, usageNotice,
} from '../../viewer/studio/pages.js';

test('picture guide renders the supported Markdown and resolves relative links', () => {
  const html = renderGuide([
    '# Pictures', 'A line with **bold** and `<text>`.', '- First', '  wrapped line',
    '- [More](other.md)', '', '![Example](../demo.png)', '```text', '<code>', '```',
    '### Another heading', '```', 'unfinished',
  ].join('\r\n'));
  assert.match(html, /<h2>Pictures<\/h2>/);
  assert.match(html, /<b>bold<\/b> and <code>&lt;text&gt;<\/code>/);
  assert.match(html, /<li>First wrapped line<\/li>/);
  assert.match(html, /href="http:\/\/studio\/docs\/other.md"/);
  assert.match(html, /src="\/demo.png" alt="Example"/);
  assert.match(html, /<pre class="log">&lt;code&gt;<\/pre>/);
  assert.match(html, /<h4>Another heading<\/h4>/);
  assert.match(html, /<pre class="log">unfinished<\/pre>$/);
  assert.ok(isGuideLink(GUIDE_URL + '#settings'));
  assert.equal(isGuideLink(), false);
});

test('guide loads over existing work and closes from each supported control', async (t) => {
  const { window } = browser(t, { html: '<textarea>Unsubmitted prompt</textarea><div class="modal">Old popup</div>' });
  let available = true;
  requests(t, {
    [`GET ${GUIDE_URL}`]: () => available ? new Response('# Draw it\n[More](other.md)') : new Response('', { status: 503 }),
  });
  await openPromptGuide();
  assert.equal(document.querySelectorAll('.modal').length, 1);
  assert.equal(document.querySelector('.guide-body a').href, 'http://studio.test/docs/other.md');
  document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter' }));
  click('.guide-body h2');
  assert.ok(document.querySelector('.modal'));
  document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
  assert.equal(document.querySelector('.modal'), null);
  available = false;
  await openPromptGuide();
  assert.match(document.querySelector('.guide-body').textContent, /did not load/);
  click('[data-close]');
  await openPromptGuide();
  click('.modal');
  assert.equal(document.querySelector('.modal'), null);
  assert.equal(document.querySelector('textarea').value, 'Unsubmitted prompt');
});

test('news updates the header, marks only unread notices, and handles empty and failed loads', async (t) => {
  browser(t, { html: '<main id="stage"></main><button><span class="news-count"></span></button>' });
  const stage = document.querySelector('main');
  const button = document.querySelector('button');
  let answer = {
    unread: 2,
    notices: [{ id: 'first', title: '<Beta>', text: '& fixes', url: 'https://example.test' },
      { id: 'second', title: 'Read', read: true },
      { id: 'third', title: 'Version', url: '/version', link_text: 'Changes' }],
  };
  const calls = requests(t, {
    'GET /api/notices': () => {
      if (answer instanceof Error) throw answer;
      return Response.json(answer);
    },
    'POST /api/notices/read': () => { answer.unread = 0; return {}; },
  });
  await refreshNews(button);
  assert.equal(button.textContent, '2');
  assert.ok(button.classList.contains('has-news'));
  await openNews(stage, button);
  assert.equal(stage.querySelectorAll('.notice.unread').length, 2);
  assert.match(stage.textContent, /<Beta>.*NEW/s);
  assert.match(stage.textContent, /Read more/);
  assert.match(stage.textContent, /Changes/);
  assert.deepEqual(calls.find(({ method }) => method === 'POST').body, { ids: ['first', 'third'] });
  await waitFor(() => button.querySelector('.news-count').hidden);
  answer = { notices: [{ id: 'without-button', title: 'Opened directly' }] };
  await openNews(stage);
  assert.deepEqual(calls.at(-1).body, { ids: ['without-button'] });
  answer = { enabled: false, unread: -1 };
  await refreshNews(button);
  assert.ok(button.hidden);
  assert.equal(newsCount(null), 0);
  await openNews(stage);
  assert.match(stage.textContent, /caught up/);
  answer = new Error('Offline');
  assert.equal(await refreshNews(button), null);
  await openNews(stage);
  assert.match(stage.textContent, /Could not load the news/);
});

test('activity groups records and opens the report for the selected failed step', async (t) => {
  const { window } = browser(t);
  const stage = document.querySelector('main');
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  let history = [
    { result: 'error', time: now.toISOString(), step: 'rig', title: 'Knight', seconds: 61,
      note: 'An error', log: '<failed>', chain: 'run-1', prompt: 'Shiny', made: 'body.glb' },
    { result: 'done', time: now.toISOString(), step: 'rig', title: 'Mage', seconds: 60 },
    { result: 'cancelled', time: yesterday.toISOString(), step: 'extra', title: 'Boss', seconds: 3600 },
    { result: 'other', title: 'Old', step: 'extra', seconds: 3670 },
  ];
  const calls = requests(t, {
    'GET /api/activity': () => {
      if (history === null) throw new Error('Offline');
      return history ? { history } : {};
    },
    'GET /api/bug-report': { details: 'machine' },
  });
  assert.deepEqual(byDay(history, now).map(({ label }) => label), ['Today', 'Yesterday', 'Earlier']);
  assert.equal(duration(-1), '0 s');
  assert.equal(duration('not a number'), '0 s');
  await openActivity(stage);
  assert.match(stage.textContent, /Knight.*1 min 1 s.*failed/s);
  assert.match(stage.textContent, /Mage.*1 min/s);
  assert.match(stage.textContent, /Boss.*1 h.*stopped/s);
  assert.match(stage.textContent, /Old.*1 h 1 min/s);
  assert.equal(stage.querySelector('.act-why').textContent, 'An error');
  assert.equal(stage.querySelector('.log').textContent, '<failed>');
  click('[data-report-chain]');
  await waitFor(() => stage.querySelector('[data-details]').value === 'machine');
  assert.equal(new URL(calls.at(-1).path, location.href).searchParams.get('chain'), 'run-1');
  stage.querySelector('.sheet').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
  assert.equal(stage.querySelector('.sheet'), null);
  history = undefined;
  await openActivity(stage);
  assert.match(stage.textContent, /Nothing yet/);
  history = null;
  await openActivity(stage);
  assert.match(stage.textContent, /history could not be read/);
});

test('About displays current version, usage payload and the result of saving a preference', async (t) => {
  const { window } = browser(t);
  const stage = document.querySelector('main');
  let update = { current: '1.0', newer: true, latest: '2.0', url: '/release' };
  let state = { enabled: true, sends: { count: 3 } };
  let save = true;
  let scrolled = false;
  window.HTMLElement.prototype.scrollIntoView = () => { scrolled = true; };
  const calls = requests(t, {
    'GET /api/update-check': () => {
      if (!update) throw Error('Offline');
      return update;
    },
    'GET /api/usage': () => {
      if (!state) throw Error('Offline');
      return state;
    },
    'POST /api/usage': ({ share }) => {
      if (!save) throw Error('Offline');
      return { enabled: share };
    },
  });
  openGuide(stage);
  let closed = 0;
  stage.querySelector('.sheet').onSheetClose = () => { closed++; };
  await openAbout(stage);
  assert.equal(closed, 1);
  const box = stage.querySelector('[data-usage-box]');
  await waitFor(() => box.checked);
  assert.equal(stage.querySelector('[data-version] a').getAttribute('href'), '/release');
  assert.match(stage.querySelector('[data-usage-sends] pre').textContent, /"count": 3/);
  click('[data-usage-see]');
  assert.equal(scrolled, true);
  change('[data-usage-box]', false);
  await waitFor(() => stage.querySelector('[data-usage-sends]').hidden);
  click('[data-usage-see]');
  save = false;
  change('[data-usage-box]', true);
  await waitFor(() => !box.checked);
  assert.deepEqual(calls.filter(({ method }) => method === 'POST').map(({ body }) => body), [{ share: false }, { share: true }]);
  state = { allowed: false };
  update = { current: '2.0', newer: true };
  await openAbout(stage);
  await waitFor(() => stage.querySelector('[data-usage-box]').disabled);
  assert.equal(stage.querySelector('[data-version] a'), null);
  click('[data-close]');
  update = state = null;
  await openAbout(stage);
  await waitFor(() => stage.querySelector('[data-usage]').hidden);
  assert.equal(stage.querySelector('[data-version]').textContent, 'AssetFurnace by Bingeljell');
  closeSheet(stage);
  assert.equal(stage.children.length, 0);
});

test('first-start usage notice waits for an opt-out to save and never saves on preview', async (t) => {
  browser(t);
  let state = { policy: 'https://example.test/privacy' };
  let save = false;
  let about = 0;
  const calls = requests(t, {
    'GET /api/usage': () => {
      if (!state) throw Error('Offline');
      return state;
    },
    'POST /api/usage': () => new Response(null, { status: save ? 204 : 503 }),
  });
  const open = () => usageNotice(() => { about++; });
  await open();
  const off = click('[data-about]');
  await waitFor(() => /Try again/.test(off.textContent));
  assert.ok(document.querySelector('.usage-notice'));
  save = true;
  click('[data-about]');
  await waitFor(() => about === 1);
  assert.equal(document.querySelector('.usage-notice'), null);
  await open();
  click('[data-see]');
  assert.equal(about, 2);
  await open();
  click('[data-ok]');
  assert.equal(document.querySelector('.usage-notice'), null);
  assert.deepEqual(calls.filter(({ method }) => method === 'POST').map(({ body }) => body),
    [{ share: false }, { share: false }, { noticed: true }]);
  state = { noticed: true };
  await open();
  state = null;
  await open();
  assert.equal(document.querySelector('.usage-notice'), null);
});

test('report previews private data, validates input and uses the chosen handoff', async (t) => {
  const { window } = browser(t);
  const stage = document.querySelector('main');
  const opened = [], copied = [];
  window.open = (...args) => opened.push(args);
  let clipboardWorks = true;
  Object.defineProperty(window.navigator, 'clipboard', { value: { writeText: async (value) => {
    if (!clipboardWorks) throw Error('Denied');
    copied.push(value);
  } } });
  let getFails = true;
  let failure = null;
  const links = { text: 'Report body', discord: '/discord', site: '/site', github: '/github' };
  const calls = requests(t, {
    'GET /api/bug-report': (_body, path) => getFails ? new Response('invalid JSON') : {
      failure: true, has_prompt: true, details: path.endsWith('prompt=1') ? 'machine, prompt' : 'machine',
    },
    'POST /api/bug-report': () => failure ?? links,
  });
  await openReport(stage);
  assert.match(stage.querySelector('[data-summary]').textContent, /version and machine/);
  assert.ok(stage.querySelector('[data-prompt-row]').hidden);
  getFails = false;
  await openReport(stage, { chain: 'test chain', step: 'finish' });
  assert.equal(stage.querySelector('[data-details]').value, 'machine');
  assert.equal(stage.querySelector('[data-prompt-row]').hidden, false);
  change('[data-prompt]', true);
  await waitFor(() => stage.querySelector('[data-details]').value === 'machine, prompt');
  const query = new URL(calls.at(-1).path, location.href).searchParams;
  assert.deepEqual(Object.fromEntries(query), { chain: 'test chain', step: 'finish', prompt: '1' });
  const message = stage.querySelector('[data-message]');
  click('[data-send="site"]');
  assert.match(message.placeholder, /Say what went wrong/);
  assert.equal(document.activeElement, message);
  message.value = '  It broke  ';
  const note = stage.querySelector('[data-note]');
  failure = new Response('not JSON', { status: 400 });
  click('[data-send="site"]');
  await waitFor(() => /could not be made/.test(note.textContent));
  failure = Response.json({ error: 'Try again' }, { status: 409 });
  click('[data-send="site"]');
  await waitFor(() => note.textContent === 'Try again');
  failure = null;
  click('[data-send="site"]');
  click('[data-send="github"]');
  await waitFor(() => opened.length === 2);
  assert.deepEqual(opened.map(([url]) => url), ['/site', '/github']);
  click('[data-send="copy"]');
  await waitFor(() => /Copied/.test(note.textContent));
  assert.deepEqual(copied, ['Report body']);
  assert.deepEqual(calls.at(-1).body, { message: 'It broke', details: 'machine, prompt' });
  clipboardWorks = false;
  click('[data-send="copy"]');
  await waitFor(() => /Could not copy/.test(note.textContent));
});
