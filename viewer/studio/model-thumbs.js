// Library thumbnails rendered from a model, for assets that have no picture to show.
//
// Rendering happens in the browser, because three.js is already loaded for the viewer and
// the server has no renderer. It is deliberately frugal: only rows the person can see, one
// model at a time, a short gap between them, a cap per page load, and the result cached so a
// second visit costs nothing. No timer runs while the tab is hidden.

export const THUMB_PREFIX = 'af-thumb-v1:';
const INDEX_KEY = 'af-thumb-index';

/** The file a row's thumbnail should come from, or null when the row already has a picture. */
export function thumbFile(asset) {
  if (!asset || asset.picture) return null;
  // Finished first: it is the one that looks like the asset, without a rig's rest pose.
  return asset.finished || asset.model || asset.rigged || null;
}

export function thumbKey(file) {
  return THUMB_PREFIX + file;
}

/** A small LRU cache that survives a reload when the browser allows storage. */
export function createThumbCache({ storage = null, limit = 24 } = {}) {
  const memory = new Map();
  let index = [];
  try {
    const raw = storage?.getItem(INDEX_KEY);
    const list = raw ? JSON.parse(raw) : [];
    if (Array.isArray(list)) index = list.filter((key) => typeof key === 'string');
  } catch {
    index = [];   // a blocked or unreadable store still gets the in-memory copy
  }
  const persist = () => {
    try { storage?.setItem(INDEX_KEY, JSON.stringify(index)); } catch { /* full or blocked */ }
  };
  return {
    get(key) {
      if (memory.has(key)) return memory.get(key);
      try {
        const value = storage?.getItem(key);
        if (value) { memory.set(key, value); return value; }
      } catch { /* blocked */ }
      return null;
    },
    set(key, value) {
      memory.set(key, value);
      index = index.filter((entry) => entry !== key);
      index.push(key);
      while (index.length > limit) {
        const old = index.shift();
        memory.delete(old);
        try { storage?.removeItem(old); } catch { /* blocked */ }
      }
      persist();
      try { storage?.setItem(key, value); } catch { /* full or blocked */ }
    },
  };
}

/**
 * One render at a time, only for rows that are on screen, and never the same file twice.
 * `render(file)` returns a data URL; `onThumb(item, data)` puts it in the row.
 */
export function createThumbQueue({ render, cache, onThumb = () => {}, limit = 24, gap = 150,
                                   wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
                                   isHidden = () => false }) {
  const pending = [];
  const queued = new Set();
  const failed = new Set();
  let running = false;
  let rendered = 0;

  const pump = async () => {
    if (running) return;
    running = true;
    while (pending.length && rendered < limit) {
      if (isHidden()) break;   // resume() picks it up when the tab comes back
      const item = pending.shift();
      try {
        const data = await render(item.file);
        cache.set(item.key, data);
        onThumb(item, data);
      } catch {
        failed.add(item.key);   // one unreadable model must not stall the rest
      }
      queued.delete(item.key);  // late on purpose: a duplicate enqueue mid-render must be ignored
      rendered += 1;
      if (pending.length) await wait(gap);
    }
    running = false;
  };

  return {
    /** Show a cached thumbnail immediately; otherwise say whether the row wants one. */
    offer(item) {
      const key = thumbKey(item.file);
      const cached = cache.get(key);
      if (cached) { onThumb({ ...item, key }, cached); return 'cached'; }
      return failed.has(key) ? 'skip' : 'wanted';
    },
    /** Render this row's thumbnail when the queue reaches it. */
    enqueue(item) {
      const key = thumbKey(item.file);
      if (cache.get(key) || failed.has(key) || queued.has(key) || rendered >= limit) return;
      queued.add(key);
      pending.push({ ...item, key });
      void pump();
    },
    resume() { void pump(); },
    pending: () => pending.length,
    rendered: () => rendered,
  };
}
