import assert from 'node:assert/strict';
import test from 'node:test';
import { mapViewerThree } from './viewer-three.js';

const hooks = mapViewerThree();
const { IndexedOBJLoader } = await import('../../viewer/IndexedOBJLoader.js');
hooks.deregister();

function streamText(parts) {
  const encoder = new TextEncoder();
  const strings = parts[Symbol.iterator]();
  return new ReadableStream({
    pull(controller) {
      const { value, done } = strings.next();
      if (done) controller.close();
      else controller.enqueue(encoder.encode(value));
    },
  });
}

test('OBJ streaming joins split lines, resolves negative indices and triangulates polygons', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(streamText([
    '# a quad\nv 0 0 ', '0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2\nf -4/7 ', '-3 -2 -1\n',
  ])));
  let mesh;
  const progress = [];
  await new IndexedOBJLoader().load('/quad.obj', (model) => { mesh = model; }, (event) => progress.push(event));
  assert.deepEqual([...mesh.geometry.index.array], [0, 1, 2, 0, 2, 3]);
  assert.deepEqual(mesh.geometry.boundingBox.max.toArray(), [1, 1, 0]);
  assert.equal(mesh.geometry.attributes.normal.getZ(0), 1);
  assert.equal(progress.length, 3);
  assert.ok(progress.every(({ lengthComputable, total }) => !lengthComputable && total === 0));
  assert.ok(progress[1].loaded > progress[0].loaded);
});

test('OBJ loading rejects an empty response body without a failure callback', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(null));
  await assert.rejects(new IndexedOBJLoader().load('/empty.obj', () => assert.fail()), /streaming response body unavailable/);
});

test('indexed OBJ loader expands both buffers for meshes over a million vertices and faces', async (t) => {
  const count = 1024 * 1024;
  function* parts() {
    const vertices = 'v 0 1 0\n'.repeat(64 * 1024);
    const faces = 'f 1 2 3\n'.repeat(64 * 1024);
    yield 'v 0 0 0\nv 1 0 0\n';
    for (let i = 0; i < 16; i++) yield vertices;
    for (let i = 0; i < 16; i++) yield faces;
    yield 'f 1 2 -1';
  }
  t.mock.method(globalThis, 'fetch', async () => new Response(streamText(parts())));
  let geometry;
  await new IndexedOBJLoader().load('/big.obj', (model) => { geometry = model.geometry; });
  assert.equal(geometry.attributes.position.count, count + 2);
  assert.equal(geometry.index.count / 3, count + 1);
  assert.equal(geometry.index.array.at(-1), count + 1);
  assert.equal(geometry.index.array[0], 0);
  assert.equal(geometry.attributes.normal.getZ(0), 1);
});
