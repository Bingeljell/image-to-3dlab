import assert from 'node:assert/strict';
import test from 'node:test';
import { installViewerBrowser, flushEvents, pointer } from './viewer-browser.js';
import { mapViewerThree } from './viewer-three.js';
import * as THREE from './viewer-renderer.js';

const hooks = mapViewerThree({ headless: true });
const { createStudioViewer, pictureMaxHeight } = await import('../../viewer/studio/viewer.js');
const { createModelViewport, disposeModelViewport } = await import('../../viewer/components/model-viewport.js');
const { createThumbnailRenderer } = await import('../../viewer/studio/model-thumb-renderer.js');
hooks.deregister();

function gltf({ animated = false, indexed = false, rigged = false } = {}) {
  const buffers = [
    new Float32Array([-2, 0, 0, 2, 0, 0, 0, 4, 0]),
    new Float32Array([0, 2]),
    new Float32Array([0, 0, 0, 1, 0, 0]),
    new Uint16Array([0, 1, 2]),
    new Uint16Array(12),
    new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
  ];
  return {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: rigged ? [0, 1] : [0] }],
    nodes: [{ mesh: 0, name: 'figure', ...(rigged ? { skin: 0 } : {}) },
      ...(rigged ? [{ name: 'hip', children: [2] }, { name: 'hand', translation: [0, 2, 0] }] : [])],
    ...(rigged ? { skins: [{ joints: [1, 2], skeleton: 1 }] } : {}),
    meshes: [{ primitives: [{
      attributes: { POSITION: 0, ...(rigged ? { JOINTS_0: 4, WEIGHTS_0: 5 } : {}) },
      ...(indexed ? { indices: 3 } : {}), material: 0,
    }] }],
    materials: [{ doubleSided: true }],
    buffers: buffers.map(({ buffer }) => ({
      byteLength: buffer.byteLength, uri: `data:application/octet-stream;base64,${Buffer.from(buffer).toString('base64')}`,
    })),
    bufferViews: buffers.map(({ byteLength }, buffer) => ({ buffer, byteLength })),
    accessors: [
      { bufferView: 0, type: 'VEC3', componentType: 5126, count: 3, min: [-2, 0, 0], max: [2, 4, 0] },
      { bufferView: 1, type: 'SCALAR', componentType: 5126, count: 2, min: [0], max: [2] },
      { bufferView: 2, type: 'VEC3', componentType: 5126, count: 2 },
      { bufferView: 3, type: 'SCALAR', componentType: 5123, count: 3 },
      { bufferView: 4, type: 'VEC4', componentType: 5123, count: 3 },
      { bufferView: 5, type: 'VEC4', componentType: 5126, count: 3 },
    ],
    ...(animated ? {
      animations: [{
        name: 'shift',
        samplers: [{ input: 1, output: 2, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
    } : {}),
  };
}

const local = (name) => `https://studio.test/${name}`;

function serveModels(t, optionsByPath = {}) {
  const realFetch = globalThis.fetch;
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (request, ...options) => {
    const url = String(request.url || request);
    if (url.startsWith('data:')) return realFetch(request, ...options);
    const name = new URL(url).pathname.slice(1);
    requests.push(name);
    const data = optionsByPath[name];
    if (typeof data === 'function') return data();
    if (!data) return new Response('no model', { status: 404 });
    const text = typeof data === 'string' ? data : JSON.stringify(gltf(data));
    return new Response(text, { headers: { 'content-length': String(text.length) } });
  });
  return requests;
}

async function waitFor(check) {
  for (let attempt = 0; attempt < 50; attempt++) {
    if (check()) return;
    await flushEvents();
  }
  assert.ok(check(), 'model load did not settle');
}

test('model viewport loads both supported formats, normalizes bounds and releases the renderer', async (t) => {
  const { document } = installViewerBrowser(t, '<div><span class="stats"></span></div>');
  const requests = serveModels(t, { obj: 'v 0 0 0\nv 0 2 0\nv 2 0 0\nf 1 2 3', glb: { indexed: true } });
  const pane = document.querySelector('div');
  for (const ext of ['obj', 'glb']) {
    let view;
    let changes = 0;
    const spec = { url: local(ext), ext };
    if (ext === 'glb') spec.manager = new THREE.LoadingManager();
    const loaded = new Promise((resolve, reject) => {
      view = createModelViewport({ pane, spec, slotIndex: 2, onChange: () => changes++, onLoaded: resolve, onError: reject });
    });
    assert.equal(await loaded, view);
    assert.equal(view.slotIndex, 2);
    assert.equal(view.stats.textContent, '1 faces\n3 verts (as stored)');
    assert.equal(view.renderer.pixelRatio, 2);
    assert.ok(view.materials[0].mesh.geometry.attributes.normal);
    assert.equal(view.modelRoot.rotation.x, ext === 'obj' ? -Math.PI / 2 : 0);
    assert.ok(new THREE.Box3().setFromObject(view.root).getCenter(new THREE.Vector3()).length() < 1e-8);
    assert.equal(view.root.scale.x, ext === 'obj' ? 0.5 : 0.25);
    view.controls.dispatchEvent({ type: 'change' });
    assert.equal(changes, 1);
    view.resetPose();
    disposeModelViewport(view);
    assert.equal(view.renderer.disposed, true);
    assert.equal(view.renderer.lost, true);
  }
  const failure = await new Promise((resolve) => createModelViewport({
    pane, spec: { url: local('unknown'), ext: 'obj' }, onError: resolve,
  }));
  assert.match(failure.message, /404/);
  assert.equal(pane.querySelector('.stats').textContent, 'FAILED');
  assert.deepEqual(requests, ['obj', 'glb', 'unknown']);
});

test('studio switches between loading, error, pictures and animated model controls', async (t) => {
  const { document, window, observers, frame } = installViewerBrowser(t, '<main><div class="hud tl"></div></main>');
  const requests = serveModels(t, { first: { animated: true }, second: { animated: true }, rest: {} });
  const stage = document.querySelector('main');
  const changes = [];
  const controller = createStudioViewer({ stage, logoUrl: '/logo.svg', onClipChange: (clip) => changes.push(clip.name) });
  const find = (selector) => stage.querySelector(selector);
  const pane = find('.sv-pane');
  Object.defineProperties(pane, { clientWidth: { value: 400 }, clientHeight: { value: 300 } });
  stage.getBoundingClientRect = () => ({ top: 30, height: 500 });
  find('.hud').getBoundingClientRect = () => ({ bottom: 90 });

  controller.showEmpty('Pick a model');
  assert.match(find('.sv-overlay').textContent, /Pick a model/);
  assert.equal(find('.sv-tools').inert, true);
  controller.showImage('/front.png', 'Study', '/back.png', ['Reference', 'New']);
  assert.equal(find('.sv-picture').classList.contains('two'), true);
  assert.equal(find('img[alt="New"]').getAttribute('src'), '/back.png');
  assert.equal(find('.sv-overlay').style.getPropertyValue('--sv-picture-top'), '72px');
  assert.equal(pictureMaxHeight({ top: 30, height: 20 }, { bottom: 90 }, 24, 12), 0);
  observers.find((observer) => observer.targets.includes(stage)).callback();

  controller.load(local('missing'));
  assert.match(find('.sv-overlay').textContent, /Loading model/);
  await waitFor(() => /didn't load/.test(find('.sv-overlay').textContent));
  assert.ok(find('.sv-overlay').textContent.includes('404'));

  controller.setFacing([0, 0, 1]);
  controller.setClips([{ name: 'Walk', url: local('first') }, { name: 'Run', url: local('second') }], { restUrl: local('rest') });
  await waitFor(() => find('.sv-overlay').hidden);
  frame(0);
  frame(500);
  assert.deepEqual(changes, ['Walk']);
  assert.equal(find('.sv-tools').hidden, false);
  assert.equal(find('.stats').textContent, '1 faces');
  assert.equal(find('[data-frame]').textContent, 'frame 15 / 60');
  const renderer = THREE.WebGLRenderer.instances.at(-1);
  const material = renderer.scene.getObjectByName('figure').material;
  const act = (name) => find(`[data-act="${name}"]`).click();
  act('wire');
  assert.equal(material.wireframe, true);
  act('grid');
  act('clay');
  assert.notEqual(renderer.scene.getObjectByName('figure').material, material);
  act('normal');
  assert.equal(renderer.scene.getObjectByName('figure').material.isMeshNormalMaterial, true);
  assert.equal(find('[data-act="clay"]').getAttribute('aria-pressed'), 'false');
  act('normal');
  assert.equal(renderer.scene.getObjectByName('figure').material, material);

  find('[aria-label="Look from Top (+Y)"]').click();
  assert.ok(renderer.camera.position.y > 2);
  window.dispatchEvent(new window.KeyboardEvent('keydown', { key: '3' }));
  assert.ok(renderer.camera.position.x > 2);
  act('reset');
  assert.ok(renderer.camera.position.x > 1 && renderer.camera.position.z > 2);
  const scrub = find('[data-scrub]');
  scrub.value = '750';
  scrub.dispatchEvent(new window.Event('input', { bubbles: true }));
  assert.equal(find('[data-frame]').textContent, 'frame 45 / 60');
  assert.equal(find('[data-clip-act="play"]').dataset.tip, 'Play');
  find('[data-clip-act="play"]').click();
  assert.equal(find('[data-clip-act="play"]').dataset.tip, 'Pause');
  find('[data-clip-act="rest"]').click();
  assert.equal(find('[data-clip-act="play"]').dataset.tip, 'Play');

  find('[data-clip="1"]').click();
  await waitFor(() => find('.sv-overlay').hidden);
  assert.deepEqual(changes, ['Walk', 'Run']);
  assert.equal(renderer.lost, true);
  assert.equal(find('[data-clip="1"]').getAttribute('aria-pressed'), 'true');
  controller.setClips([], { restUrl: local('rest') });
  await waitFor(() => find('.sv-overlay').hidden);
  controller.setClips([]);
  assert.equal(find('.sv-clips').hidden, true);
  controller.showImage('/still.png', 'Single');
  frame(750);
  assert.equal(find('canvas'), null);
  assert.equal(find('.stats').hidden, true);
  assert.deepEqual(requests, ['missing', 'first', 'second', 'rest']);
});

test('thumbnail renderer loads real geometry, renders and reuses its context', async (t) => {
  installViewerBrowser(t);
  serveModels(t, { first: {}, second: { indexed: true } });
  const before = THREE.WebGLRenderer.instances.length;
  const thumbs = createThumbnailRenderer({ size: 80 });
  assert.equal(await thumbs.render(local('first')), 'data:image/png;base64,test-render');
  const gl = THREE.WebGLRenderer.instances.at(-1);
  assert.deepEqual(gl.size, [80, 80]);
  assert.ok(gl.camera.position.z > 0);
  assert.ok(gl.scene.getObjectByName('figure').geometry.attributes.position);
  assert.equal(await thumbs.render(local('second')), 'data:image/png;base64,test-render');
  assert.equal(THREE.WebGLRenderer.instances.length, before + 1);
  await assert.rejects(thumbs.render(local('missing')), /404/);
  thumbs.dispose();
  assert.ok(gl.disposed && gl.lost);
  thumbs.dispose();
});

test('spot repair submits local coordinates and review preserves the model camera', async (t) => {
  const { document, window, frame, observers } = installViewerBrowser(t, '<main></main>');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let mayFix = false;
  let failPatch = true;
  const patches = [], kept = [], discarded = [];
  const confirm = t.mock.fn(() => false);
  Object.defineProperty(globalThis, 'confirm', { configurable: true, value: confirm });
  t.after(() => delete globalThis.confirm);
  let sendOld;
  const requests = serveModels(t, {
    base: {}, patched: {},
    old: () => new Promise((resolve) => { sendOld = resolve; }),
  });
  const stage = document.querySelector('main');
  const controller = createStudioViewer({
    stage, logoUrl: '/logo.svg',
    fixSession: {
      canFix: () => mayFix, blockedNote: () => 'Finish this first',
      async patch(spots, preview) {
        patches.push({ spots, preview });
        if (failPatch) throw 'Patch failed';
        return { preview: 'saved-temp', url: local('patched') };
      },
      keep: (id) => kept.push(id),
      discard: (id) => discarded.push(id),
    },
  });
  const find = (selector) => stage.querySelector(selector);
  const tool = (name) => find(`[data-act="${name}"]`).click();
  const fix = (name) => find(`[data-fix="${name}"]`);
  const pane = find('.sv-pane');
  const count = () => find('[data-count]').textContent;
  const move = (x, y, buttons = 0) => pointer(window, pane, 'pointermove', { clientX: x, clientY: y, buttons });
  const mark = (x, y, movement = 0) => {
    pointer(window, pane, 'pointerdown', { clientX: x, clientY: y });
    pointer(window, pane, 'pointerup', { clientX: x + movement, clientY: y });
  };
  observers[0].callback();
  controller.snap('reset');
  tool('wire');
  controller.load(local('old'));
  await waitFor(() => sendOld);
  controller.load(local('base'));
  await waitFor(() => find('.sv-overlay').hidden);
  sendOld(new Response('disappeared', { status: 404 }));
  await flushEvents();
  assert.equal(find('.sv-overlay').hidden, true);

  controller.snap('unknown');
  controller.snap('front');
  frame();
  const renderer = THREE.WebGLRenderer.instances.at(-1);
  const position = renderer.camera.position.toArray();
  tool('bones');
  assert.match(stage.textContent, /No skeleton yet/);
  tool('fix');
  mark(200, 150);
  assert.match(stage.textContent, /Finish this first/);
  mayFix = true;
  mark(0, 0);
  assert.match(stage.textContent, /Missed the model/);
  mark(200, 150, 12);
  assert.equal(count(), 'Click to mark spots');
  move(200, 150, 1);
  assert.equal(pane.classList.contains('ringing'), false);
  move(200, 150);
  assert.equal(pane.classList.contains('ringing'), true);
  const size = find('[aria-label="Brush size"]');
  const strength = find('[aria-label="Smooth strength"]');
  window.dispatchEvent(new window.KeyboardEvent('keydown', { key: ']' }));
  size.dispatchEvent(new window.KeyboardEvent('keydown', { key: ']', bubbles: true }));
  assert.equal(Number(size.value), 0.025);
  window.dispatchEvent(new window.KeyboardEvent('keydown', { key: '[' }));
  assert.equal(Number(size.value), 0.02);
  move(0, 0);
  assert.equal(pane.classList.contains('ringing'), false);
  size.value = '0.04';
  size.dispatchEvent(new window.Event('input'));
  find('[data-mode="smooth"]').click();
  assert.equal(find('[data-strength]').hidden, false);
  strength.value = '0.8';
  strength.dispatchEvent(new window.Event('input'));
  mark(200, 150);
  mark(210, 152);
  assert.equal(count(), '2 spots marked');
  fix('undo').click();
  assert.equal(count(), '1 spot marked');
  fix('patch').click();
  await flushEvents();
  assert.match(stage.textContent, /Patch failed/);
  assert.equal(count(), '1 spot marked');
  assert.deepEqual(patches[0].spots.map(({ mode, radius, strength }) => ({ mode, radius, strength })), [
    { mode: 'smooth', radius: 0.04, strength: 0.8 },
  ]);
  assert.ok(Math.abs(patches[0].spots[0].point[0]) < 0.00001);
  assert.ok(Math.abs(patches[0].spots[0].point[1] - 2) < 0.00001);

  failPatch = false;
  fix('patch').click();
  await waitFor(() => find('.sv-overlay').hidden && !find('[data-review]').hidden);
  frame();
  assert.deepEqual(THREE.WebGLRenderer.instances.at(-1).camera.position.toArray(), position);
  assert.equal(fix('compare').textContent, 'Showing: after');
  fix('compare').click();
  await waitFor(() => find('.sv-overlay').hidden);
  frame();
  mark(200, 150);
  assert.match(stage.textContent, /Switch to "after"/);
  fix('compare').click();
  await waitFor(() => find('.sv-overlay').hidden);
  frame();
  mark(200, 150);
  fix('patch').click();
  await waitFor(() => find('.sv-overlay').hidden && !find('[data-review]').hidden);
  assert.equal(patches.at(-1).preview, 'saved-temp');
  tool('fix');
  assert.equal(find('.sv-brush').hidden, false);
  assert.equal(confirm.mock.callCount(), 1);
  fix('discard').click();
  await waitFor(() => find('.sv-overlay').hidden);
  assert.deepEqual(discarded, ['saved-temp']);

  frame();
  mark(200, 150);
  fix('patch').click();
  await waitFor(() => find('.sv-overlay').hidden && !find('[data-review]').hidden);
  fix('keep').click();
  assert.deepEqual(kept, ['saved-temp']);
  assert.equal(find('[data-review]').hidden, true);
  tool('fix');
  assert.equal(find('.sv-brush').hidden, true);
  assert.deepEqual(requests, ['old', 'base', 'patched', 'base', 'patched', 'patched', 'base', 'patched']);
  t.mock.timers.tick(2400);
});

test('studio keeps skeleton and look settings across model versions', async (t) => {
  const { document, window, frame, observers } = installViewerBrowser(t, '<main></main>');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let sendOlder;
  serveModels(t, {
    rigged: { rigged: true }, animated: { rigged: true, animated: true }, unrigged: {},
    older: () => new Promise((resolve) => { sendOlder = resolve; }),
  });
  const stage = document.querySelector('main');
  const controller = createStudioViewer({ stage, logoUrl: '/logo.svg' });
  const find = (sel) => stage.querySelector(sel);
  const click = (name) => find(`[data-act="${name}"]`).click();
  const key = (key) => window.dispatchEvent(new window.KeyboardEvent('keydown', { key }));
  const settle = async () => {
    await waitFor(() => find('.sv-overlay').hidden);
    frame();
    return THREE.WebGLRenderer.instances.at(-1);
  };
  const pane = find('.sv-pane');
  pointer(window, pane, 'pointermove');
  pointer(window, pane, 'pointerup');
  click('clay');
  click('fix');
  click('fix');
  controller.load(local('older'));
  await waitFor(() => sendOlder);
  controller.load(local('rigged'));
  let renderer = await settle();
  sendOlder(Response.json(gltf()));
  await flushEvents();
  assert.equal(find('.sv-overlay').hidden, true);

  const mesh = renderer.scene.getObjectByName('figure');
  assert.equal(mesh.isSkinnedMesh, true);
  assert.ok(mesh.material.color.equals(new THREE.Color(0x8a8580)));
  click('bones');
  frame();
  let display = renderer.scene.children.find(({ renderOrder }) => renderOrder === 999);
  assert.equal(display.children.length, 3);
  click('bones');
  assert.equal(display.parent, null);
  click('bones');
  click('normal');

  controller.setClips([{ name: 'Painted knight', url: local('animated') }], { still: true });
  renderer = await settle();
  assert.equal(find('[data-clip-act]'), null);
  assert.equal(renderer.scene.getObjectByName('figure').material.isMeshNormalMaterial, true);
  display = renderer.scene.children.find(({ renderOrder }) => renderOrder === 999);
  assert.equal(display.children.length, 3);
  key('Home');
  observers[0].callback();
  controller.load(local('animated'));
  renderer = await settle();
  assert.ok(renderer.scene.children.some(({ renderOrder }) => renderOrder === 999));
  controller.load(local('unrigged'));
  await settle();
  assert.equal(find('[data-act="bones"]').getAttribute('aria-pressed'), 'false');
  controller.dispose();
});

test('viewport shows failure with no callback and ignores unknown load size', async (t) => {
  const { document } = installViewerBrowser(t, '<main><span class="stats"></span></main>');
  const pane = document.querySelector('main');
  serveModels(t, {
    stream: () => new Response('v 0 0 0\nv 0 2 0\nv 2 0 0\nf 1 2 3'),
  });
  const view = createModelViewport({ pane, spec: { url: local('stream'), ext: 'obj' } });
  await waitFor(() => view.root);
  assert.match(view.stats.textContent, /1 faces/);
  disposeModelViewport(view);
  const failed = createModelViewport({ pane, spec: { url: local('missing'), ext: 'obj' } });
  await waitFor(() => failed.stats.textContent === 'FAILED');
  assert.equal(failed.root, null);
  disposeModelViewport(failed);
});
