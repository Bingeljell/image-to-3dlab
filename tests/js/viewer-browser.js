import { JSDOM } from 'jsdom';

export function installViewerBrowser(t, body = '') {
  const { window } = new JSDOM(body, { url: 'https://studio.test/' });
  const observers = [];
  const frames = new Map();
  let frameId = 0;
  const globals = {
    window, document: window.document,
    Image: window.Image,
    devicePixelRatio: 3,
    ProgressEvent: window.ProgressEvent,
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; this.targets = []; }
      observe(node) { this.targets.push(node); observers.push(this); }
    },
    requestAnimationFrame(callback) { frames.set(++frameId, callback); return frameId; },
    cancelAnimationFrame(id) { frames.delete(id); },
  };
  for (const [name, value] of Object.entries(globals)) {
    const old = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    t.after(() => old ? Object.defineProperty(globalThis, name, old) : delete globalThis[name]);
  }
  t.after(() => window.close());
  return {
    document: window.document, window, observers,
    frame(time = 0) {
      const starting = [...frames];
      for (const [id, callback] of starting) if (frames.delete(id)) callback(time);
    },
  };
}

export function pointer(window, node, type, options = {}) {
  return node.dispatchEvent(new window.MouseEvent(type, { bubbles: true, cancelable: true, ...options }));
}

export function flushEvents() {
  return new Promise((resolve) => setImmediate(resolve));
}
