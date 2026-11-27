import assert from 'node:assert/strict';
import test from 'node:test';
import { installViewerBrowser, pointer } from './viewer-browser.js';
import { createLibraryPictureLoader } from '../../viewer/studio/library-picture.js';
import { libraryFamilyOpen, libraryRowSelection } from '../../viewer/studio/library-selection.js';
import { runLibraryLoad, setLibraryLoading, showLibraryError, showLibraryLoadError } from '../../viewer/studio/library-loading.js';
import { attachZoom, MAX_ZOOM, zoomAt, RESET } from '../../viewer/studio/zoom.js';
import { installTooltips, tooltipPlacement } from '../../viewer/studio/tooltips.js';
import { alsoFetchesLines } from '../../viewer/components/also-fetches.js';
import { showsCaveat } from '../../viewer/components/gated-caveat.js';
import { panelState, runTitle, setupVerb } from '../../viewer/components/setup-progress.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

test('Library hides controls while loading and reports plain text with a retry button', (t) => {
  const { document } = installViewerBrowser(t, '<section><div id="libControls"></div><div id="libLoad"></div><footer id="libFoot"></footer></section>');
  const root = document.querySelector('section');
  setLibraryLoading(root, true);
  const controls = root.querySelector('#libControls'), indicator = root.querySelector('#libLoad');
  assert.equal(controls.hidden, true);
  assert.equal(controls.inert, true);
  assert.equal(indicator.hidden, false);
  assert.equal(indicator.querySelector('.loader').getAttribute('aria-hidden'), 'true');
  assert.match(indicator.textContent, /Loading your assets/);
  setLibraryLoading(root, false);
  assert.equal(controls.hidden, false);
  assert.equal(controls.inert, false);
  assert.equal(indicator.hidden, true);
  let retries = 0;
  const message = '<b>offline</b>';
  showLibraryLoadError(root, message, () => retries++);
  showLibraryError(root, message, () => retries++);
  for (const box of [indicator, root.querySelector('footer')]) {
    assert.match(box.textContent, /<b>offline<\/b>/);
    assert.equal(box.querySelector('b'), null);
    const retry = box.querySelector('button');
    retry.click();
    assert.equal(retry.disabled, true);
  }
  assert.equal(indicator.hidden, false);
  assert.equal(retries, 2);
  const empty = document.createElement('div');
  setLibraryLoading(empty, true);
  showLibraryLoadError(empty, message, () => assert.fail());
  showLibraryError(empty, message, () => assert.fail());
  assert.equal(empty.childElementCount, 0);
  // A whole document is a valid container and has no ownerDocument.
  setLibraryLoading(document, true);
  assert.ok(indicator.querySelector('.loader'));
  showLibraryLoadError(document, 'failed', () => {});
  assert.match(indicator.textContent, /failed/);
  showLibraryError(document, 'try again later', () => {});
  assert.match(root.querySelector('footer').textContent, /try again later/);
});

test('Library commits results and errors only for the most recent request', async () => {
  let current = 0;
  const committed = [], errors = [];
  const run = (fetchAssets) => {
    const requestId = ++current;
    return runLibraryLoad({ requestId, isCurrent: () => requestId === current, fetchAssets,
      onSuccess: (value) => committed.push(value), onError: (err) => errors.push(err.message),
    });
  };
  const pending = deferred();
  const stale = run(() => pending.promise);
  assert.equal(await run(async () => ['new']), true);
  pending.resolve(['old']);
  assert.equal(await stale, false);
  const pendingFailure = deferred();
  const staleFailure = run(() => pendingFailure.promise);
  assert.equal(await run(async () => { throw new Error('current error'); }), true);
  pendingFailure.reject(new Error('old error'));
  assert.equal(await staleFailure, false);
  assert.deepEqual(committed, [['new']]);
  assert.deepEqual(errors, ['current error']);
});

test('Library picture loader discards old requests and reports errors only to the current draft', async () => {
  const old = deferred(), staleDraft = deferred();
  const responses = [old.promise, new Response('picture bytes', { headers: { 'Content-Type': 'image/webp' } }),
    new Response(new Blob(['PNG bytes'])), staleDraft.promise, new Response('', { status: 404 }), new Response('ignored')];
  const used = [], files = [], errors = [], urls = [];
  let revision = 0;
  let allow = true;
  let accepted = true;
  let target = {
    sourceRevision: () => revision, isSourceCurrent: (old) => old === revision,
    canUseLibraryPicture: () => allow, usePicture: (file) => { files.push(file); return accepted; },
    showError: (error) => errors.push(error),
  };
  const loader = createLibraryPictureLoader({
    fetchPicture: (path) => { urls.push(path); return responses.shift(); }, freshUrl: (path) => `/files/${path}?new`,
    getTarget: () => target, makeFile: (parts, name, opts) => new File(parts, name, opts), onUsed: (asset) => used.push(asset),
  });
  const waiting = loader.load({ picture: 'stale.png' });
  const ok = { picture: 'result/blue%20coat.webp' };
  assert.equal(await loader.load(ok), true);
  old.resolve(new Response('old image'));
  assert.equal(await waiting, false);
  assert.deepEqual(used, [ok]);
  assert.equal(files[0].name, 'blue coat.webp');
  assert.equal(files[0].type, 'image/webp');
  assert.equal(await files[0].text(), 'picture bytes');
  assert.equal(urls[1], '/files/result/blue%20coat.webp?new');
  accepted = false;
  assert.equal(await loader.load({ picture: 'result/' }), false);
  assert.equal(files[1].name, 'library-picture.png');
  assert.equal(files[1].type, 'image/png');
  const changing = loader.load({ picture: 'changes.png' });
  revision++;
  staleDraft.resolve(new Response('stale draft'));
  assert.equal(await changing, false);
  assert.equal(await loader.load({ picture: 'missing.png' }), false);
  assert.deepEqual(errors, ['The picture could not be read (404).']);
  allow = false;
  assert.equal(await loader.load({ picture: 'disallowed.png' }), false);
  allow = true;
  const fallback = createLibraryPictureLoader({
    fetchPicture: async () => { throw { type: 'offline' }; },
    freshUrl: (path) => path, getTarget: () => target, makeFile: () => assert.fail(),
  });
  assert.equal(await fallback.load({ picture: 'offline.png' }), false);
  assert.equal(errors.at(-1), 'The picture could not be read.');
  target = null;
  loader.invalidate();
});

test('Library stays open for selected input rows, current models and manually opened families', () => {
  const original = { id: 'original' }, version = { id: 'version' }, unrelated = { id: 'outside' };
  const openFamilies = new Set(['pinned']);
  const ctx = { current: original, inputId: version.id, openFamilies, key: 'shut' };
  assert.equal(libraryRowSelection(original, ctx), false);
  assert.equal(libraryRowSelection(version, ctx), true);
  assert.equal(libraryRowSelection(original, { current: original }), true);
  assert.equal(libraryRowSelection(version, { ...ctx, viewingMaking: true }), false);
  assert.equal(libraryFamilyOpen([unrelated], ctx), false);
  assert.equal(libraryFamilyOpen([unrelated], { ...ctx, key: 'pinned' }), true);
  assert.equal(libraryFamilyOpen([original], ctx), true);
  assert.equal(libraryFamilyOpen([version], ctx), true);
});

test('zoom uses a shared transform for both pictures and suppresses the click after a drag', (t) => {
  const { document, window } = installViewerBrowser(t, `<section>
    <div class="zoombox"><img></div><div class="zoombox"><img></div><div class="zoombar"></div></section><aside></aside>`);
  const root = document.querySelector('section');
  attachZoom(document.querySelector('aside'));
  const [first, second] = root.querySelectorAll('.zoombox');
  for (const box of [first, second]) {
    box.getBoundingClientRect = () => ({ left: 10, top: 0, width: 200, height: 100 });
    box.setPointerCapture = () => {};
  }
  attachZoom(root);
  const zoom = (value) => root.querySelector(`[data-zoom="${value}"]`);
  const output = () => root.querySelector('output').textContent;
  const click = (type, clientX, clientY) => pointer(window, first, type, { clientX, clientY });
  assert.equal(output(), '100%');
  click('pointermove', 55, 50);
  click('pointerdown', 110, 50);
  click('pointerup', 110, 50);
  click('click', 110, 50);
  assert.equal(output(), '200%');
  assert.equal(first.firstChild.draggable, false);
  assert.equal(zoom(2).getAttribute('aria-pressed'), 'true');
  assert.equal(first.firstChild.style.cursor, 'grab');
  click('pointerdown', 110, 50);
  assert.equal(first.firstChild.style.cursor, 'grabbing');
  click('pointermove', 110, 50);
  click('pointermove', 110, 75);
  click('pointermove', 90, 75);
  click('pointercancel', 90, 75);
  assert.equal(first.firstChild.style.transform, 'translate(-120px, -25px) scale(2)');
  assert.equal(second.firstChild.style.transform, first.firstChild.style.transform);
  zoom(1).click();
  click('click', 110, 50);
  assert.equal(output(), '100%');
  click('click', 110, 50);
  click('click', 110, 50);
  assert.equal(output(), '200%');
  const wheel = new window.WheelEvent('wheel', { clientX: 70, clientY: 35, deltaY: -200, cancelable: true });
  first.dispatchEvent(wheel);
  assert.equal(wheel.defaultPrevented, true);
  assert.equal(output(), '212%');
  click('dblclick', 110, 50);
  assert.equal(output(), '100%');
  assert.equal(first.firstChild.style.cursor, 'zoom-in');
  assert.equal(zoomAt(RESET, 100, 1, 1, 100, 100).scale, MAX_ZOOM);
  const standalone = document.querySelector('aside');
  standalone.innerHTML = '<div class="zoombox"><img></div>';
  attachZoom(standalone);
  assert.equal(standalone.querySelector('img').style.cursor, 'zoom-in');
});

test('tooltip position follows browser size and changes between hover and keyboard focus', (t) => {
  const { window, document } = installViewerBrowser(t, '<div><button data-tip="Help"><span>?</span></button><button>Plain</button></div>');
  window.innerWidth = 800;
  window.innerHeight = 600;
  installTooltips();
  assert.deepEqual(tooltipPlacement({ left: 300, right: 320, top: 300, bottom: 320 }, { width: 800, height: 600 }),
    { vertical: 'down', horizontal: '' });
  const target = document.querySelector('[data-tip]');
  let pos = { left: 10, right: 30, top: 100, bottom: 120 };
  target.getBoundingClientRect = () => pos;
  pointer(window, document.querySelector('span'), 'pointerover');
  assert.deepEqual({ ...target.dataset }, { tip: 'Help', tipV: 'down', tipH: 'left' });
  pos = { left: 710, right: 780, top: 510, bottom: 530 };
  target.dispatchEvent(new window.FocusEvent('focusin', { bubbles: true }));
  assert.equal(target.dataset.tipV, 'up');
  assert.equal(target.dataset.tipH, 'right');
  pos = { left: 450, right: 470, top: 100, bottom: 120 };
  target.classList.add('tip-up');
  pointer(window, target, 'pointerover');
  assert.equal(target.dataset.tipV, 'down');
  assert.equal(target.dataset.tipH, undefined);
  pointer(window, document.querySelector('button:last-child'), 'pointerover');
  assert.equal(document.querySelector('button:last-child').dataset.tipV, undefined);
});

test('setup shows download source and permissions only when they apply', () => {
  assert.deepEqual(alsoFetchesLines({}), []);
  assert.deepEqual(alsoFetchesLines({ also_fetches: [] }), []);
  assert.deepEqual(alsoFetchesLines({ also_fetches: [{ what: 'Paint', size: '4 GB', licence: 'MIT', from: 'project' }] }),
    ['', 'Also downloads:', '  Paint: 4 GB, MIT\n    from project']);
  assert.equal(showsCaveat({}, {}), false);
  assert.equal(showsCaveat({ caveat: 'Region' }, {}), true);
  assert.equal(showsCaveat({ caveat: 'Sign in', gated_repo: 'project' }, {}), true);
  assert.equal(showsCaveat({ caveat: 'Sign in', gated_repo: 'project' }, { project: 'yes' }), false);
  const engine = { label: 'Engine', build_present: true, setup_fetches_weights: false };
  assert.equal(panelState(null), 'running');
  assert.equal(runTitle(engine, 'rebuild'), 'Rebuilding Engine');
  assert.equal(runTitle(engine, 'download', { phase: 'done' }), '✓ Installed · Engine');
  assert.equal(runTitle(null, null, { phase: 'error' }), '! Setup failed · this backend');
  assert.equal(runTitle({}, null, { phase: 'cancelled' }), '■ Setup cancelled · this backend');
  assert.equal(setupVerb({ build_present: false }, 'setup'), 'Setting up');
  assert.equal(setupVerb(engine, 'download'), 'Downloading');
  assert.equal(setupVerb(engine, 'resume'), 'Downloading');
  assert.equal(setupVerb({ ...engine, setup_fetches_weights: true }, 'setup'), 'Downloading');
  assert.equal(setupVerb(engine, 'setup'), 'Building');
  assert.equal(setupVerb(null), 'Building');
});

test('studio request marker affects only same-origin writes and preserves explicit headers', async (t) => {
  const { window } = installViewerBrowser(t);
  const calls = [];
  window.fetch = function (url, init) { calls.push({ url, init, receiver: this }); };
  Object.defineProperty(globalThis, 'location', { configurable: true, value: window.location });
  t.after(() => delete globalThis.location);
  await import('../../viewer/request-header.js');
  window.fetch('/api/library');
  window.fetch('/api/library', { method: 'head' });
  window.fetch('https://elsewhere.test/api', { method: 'POST' });
  const write = new Request('https://studio.test/api', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
  window.fetch(write);
  window.fetch('/api/library', { method: 'DELETE', headers: { 'X-AssetFurnace-Request': 'explicit' } });
  window.fetch(new Request('https://studio.test/api', { method: 'GET' }), { method: 'PUT' });
  window.fetch('/api/library', { method: 'POST' });
  assert.ok(calls.every(({ receiver }) => receiver === window));
  assert.ok(calls.slice(0, 3).every(({ init }) => !init.headers));
  assert.equal(calls[3].init.headers.get('Content-Type'), 'application/json');
  assert.equal(calls[3].init.headers.get('X-AssetFurnace-Request'), 'studio');
  assert.equal(calls[4].init.headers.get('X-AssetFurnace-Request'), 'explicit');
  assert.equal(calls[5].init.headers.get('X-AssetFurnace-Request'), 'studio');
  assert.equal(calls[6].init.headers.get('X-AssetFurnace-Request'), 'studio');
});

test('asset metadata handles imported, incomplete and unrecognised stages', async () => {
  const {
    assetEngine, displayModel, engineOf, madeVersion, robloxLines, servedUrl,
    statsRows, statusText, stepReady, visibleAssets, visibleFamilies,
  } = await import('../../viewer/studio/library.js');
  assert.equal(stepReady({}, 'unknown'), false);
  assert.equal(statusText({ kind: 'prop set' }), '3D model, not split yet');
  assert.equal(statusText({ kind: 'prop set', props: [{}, {}] }), '2 props, game-ready');
  assert.equal(statusText({ stage: 'animated' }), 'Animated · 0 moves');
  assert.equal(statusText({ stage: 'animated', clips: [{}] }), 'Animated · 1 move');
  assert.equal(statusText({ stage: 'imported' }), 'imported');
  assert.equal(displayModel({ kind: 'prop set', model: 'raw.glb' }), 'raw.glb');
  assert.equal(displayModel({ kind: 'prop set' }), null);
  assert.equal(servedUrl('/output/', ''), null);
  assert.equal(engineOf(), 'made outside the app');
  assert.equal(engineOf('output/run__hunyuan-mlx__/model.glb'), 'Hunyuan3D');
  assert.equal(assetEngine({ finished: 'old.glb' }), 'made outside the app');
  const original = { id: 'root', name: 'Tool', updated: 10 };
  const old = { id: 'v2', family: original.id, name: 'Tool', updated: 11 };
  const fresh = { id: 'v3', family: original.id, name: 'Tool', updated: 13 };
  const unrelated = { id: 'other', name: 'Hat', updated: 12 };
  assert.deepEqual(visibleAssets([original]), { rows: [original], more: 0, hiddenCount: 0 });
  const { rows: families } = visibleFamilies([original, old, unrelated]);
  assert.deepEqual(families.map(({ key, shown }) => [key, shown.id]), [['root', 'v2'], ['other', 'other']]);
  assert.equal(madeVersion([], [fresh, old, unrelated], original.id), fresh);
  assert.equal(madeVersion([], [unrelated], unrelated.id), unrelated);
  const stats = { vertices: 15, triangles: 5, bytes: 1500 };
  assert.equal(statsRows(stats).length, 3);
  assert.deepEqual(statsRows({ ...stats, textures: [{ width: 64, height: 128 }] }).find(([label]) => label === 'Textures'),
    ['Textures', '1 × 64×128']);
  assert.deepEqual(robloxLines({}), []);
});

test('plan keeps empty goals empty and checks warnings only for the intended goal', async () => {
  const { normalizePromptText, planFor, troubleFor } = await import('../../viewer/studio/plan.js');
  assert.deepEqual(planFor('model', 'unknown'), []);
  assert.equal(normalizePromptText(), '');
  assert.deepEqual(troubleFor(undefined, 'picture'), []);
  const recipes = { trouble: [{ goals: ['prop'], words: ['eyes'], message: 'One object works best' }] };
  assert.deepEqual(troubleFor('A box with eyes', 'picture', recipes), []);
});
