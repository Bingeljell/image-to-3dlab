import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { mapViewerThree } from './viewer-three.js';
import { flushEvents, installViewerBrowser } from './viewer-browser.js';
import { Vector3, WebGLRenderer } from './viewer-renderer.js';

test('winding controls reverse the triangle, update the display and restore the original mesh', async (t) => {
  const html = await readFile(new URL('../../understanding_3d/mesh-winding-demo.html', import.meta.url), 'utf8');
  const { window, document } = installViewerBrowser(t, html);
  const frames = [];
  window.requestAnimationFrame = (callback) => frames.push(callback);
  const entry = document.querySelector('script[src]');
  // A local file may load a classic script beside it. A module script needs an HTTP origin.
  assert.equal(entry.hasAttribute('type'), false);
  const hooks = mapViewerThree({ headless: true });
  t.after(() => hooks.deregister());
  await import(new URL(entry.getAttribute('src'), new URL('../../understanding_3d/mesh-winding-demo.html', import.meta.url)));
  for (let attempt = 0; !frames.length && attempt < 20; attempt++) await flushEvents();
  assert.equal(frames.length, 1);
  const [renderer, labelRenderer] = WebGLRenderer.instances;
  const byId = (id) => document.getElementById(id);
  const click = (id) => byId(id).click();
  const mesh = renderer.scene.children.find((object) => object.isMesh);
  const [vertices, labels, arrows] = renderer.scene.children.filter((object) => object.isGroup);
  const wire = renderer.scene.children.find((object) => object.isLineSegments && object.geometry.type === 'WireframeGeometry');
  assert.equal(vertices.children.length, 8);
  assert.deepEqual(labels.children.map(({ element }) => element.textContent), ['0', '1', '2', '3', '4', '5', '6', '7']);
  assert.equal(arrows.children.length, 12);
  assert.deepEqual(mesh.geometry.index.array.slice(0, 3), new Uint16Array([0, 1, 2]));

  for (const [id, object] of [['btn-normals', arrows], ['btn-labels', labels], ['btn-wire', wire]]) {
    assert.equal(byId(id).classList.contains('active'), true);
    click(id);
    assert.equal(object.visible, false);
    assert.equal(byId(id).classList.contains('active'), false);
    click(id);
    assert.equal(object.visible, true);
  }

  const firstWire = wire.geometry;
  const originalMaterial = mesh.material;
  click('btn-flip');
  assert.match(byId('status').textContent, /Clockwise \(back/);
  assert.equal(byId('status').classList.contains('flipped'), true);
  assert.equal(byId('index-display').textContent, '0, 2, 1');
  assert.notEqual(wire.geometry, firstWire);
  assert.deepEqual(mesh.geometry.index.array.slice(0, 3), new Uint16Array([0, 2, 1]));
  assert.ok(new Vector3(0, 1, 0).applyQuaternion(arrows.children[0].quaternion).z < -0.99);

  click('btn-orientation');
  assert.equal(mesh.material.vertexColors, true);
  const color = mesh.geometry.getAttribute('color');
  assert.ok(color.getX(1) > color.getZ(1));
  click('btn-flip');
  assert.equal(byId('index-display').textContent, '0, 1, 2');
  assert.equal(byId('status').classList.contains('flipped'), false);
  assert.ok(mesh.geometry.getAttribute('color').getZ(1) > mesh.geometry.getAttribute('color').getX(1));
  click('btn-orientation');
  assert.equal(mesh.material, originalMaterial);

  window.innerWidth = 640;
  window.innerHeight = 360;
  window.dispatchEvent(new window.Event('resize'));
  frames.shift()();
  assert.deepEqual(renderer.size, [640, 360]);
  assert.deepEqual(labelRenderer.size, [640, 360]);
  assert.equal(renderer.camera.aspect, 640 / 360);
  assert.equal(frames.length, 1);
});
