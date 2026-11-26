import assert from 'node:assert/strict';
import { test } from 'node:test';

test('browser imports use the same bundled Three.js as the studio', async () => {
  const browser = await import('three');
  const shipped = await import('../../viewer/vendor/three.module.js');
  assert.equal(browser.Mesh, shipped.Mesh);
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  assert.equal(typeof GLTFLoader, 'function');
});
