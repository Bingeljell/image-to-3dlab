import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { setImmediate } from 'node:timers/promises';
import { JSDOM } from 'jsdom';

export function browser(t, { file, html = '<main id="stage"></main>', url = 'http://studio.test/' } = {}) {
  const source = file ? readFileSync(new URL(`../../viewer/${file}`, import.meta.url), 'utf8') : html;
  const dom = new JSDOM(source, { url });
  const { window } = dom;
  for (const name of ['window', 'document', 'location', 'navigator', 'CustomEvent', 'Event', 'Image', 'HTMLElement', 'File', 'FileReader', 'FormData']) {
    replaceGlobal(t, name, name === 'window' ? window : window[name]);
  }
  t.after(() => window.close());
  return dom;
}

export function replaceGlobal(t, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  });
}

export async function waitFor(check) {
  const deadline = performance.now() + 1000;
  while (performance.now() < deadline) {
    if (check()) return;
    await setImmediate();
  }
  assert.ok(check(), 'Expected the browser to reach the asserted state');
}

export function requests(t, handlers) {
  const calls = [];
  const unexpected = [];
  replaceGlobal(t, 'fetch', async (url, options = {}) => {
    const method = options.method ?? 'GET';
    const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
    const path = String(url);
    calls.push({ method, path, body });
    const handler = handlers[`${method} ${path}`] ?? handlers[`${method} ${path.split('?')[0]}`];
    if (!handler) {
      unexpected.push({ method, path });
      throw new Error(`Unexpected request: ${method} ${path}`);
    }
    const payload = typeof handler === 'function' ? await handler(body, path) : handler;
    return payload instanceof Response ? payload : Response.json(payload);
  });
  t.after(() => assert.deepEqual(unexpected, []));
  return calls;
}

export function click(selector, root = document) {
  const element = root.querySelector(selector);
  assert.ok(element, `Missing control ${selector}`);
  element.click();
  return element;
}

export function change(selector, value, root = document) {
  const element = root.querySelector(selector);
  assert.ok(element, `Missing field ${selector}`);
  if (element.type === 'checkbox') element.checked = value;
  else element.value = value;
  element.dispatchEvent(new Event('change', { bubbles: true }));
  return element;
}
