// Zoom and pan for pictures in the viewer: wheel to zoom at the pointer, drag to pan,
// double-click to reset. Pictures shown side by side move together, so the same spot of the
// original and the new picture is always in view.

export const MAX_ZOOM = 8;

/** The transform after zooming by `factor` around (px, py), a point inside a w x h box.
 * State is {scale, x, y} for `translate(x, y) scale(scale)` with the origin at the top left. */
export function zoomAt(state, factor, px, py, w, h) {
  const scale = Math.min(MAX_ZOOM, Math.max(1, state.scale * factor));
  const ratio = scale / state.scale;
  return clampPan({ scale, x: px - (px - state.x) * ratio, y: py - (py - state.y) * ratio }, w, h);
}

/** Moved by (dx, dy) pixels, kept so the picture still covers its box. */
export function panBy(state, dx, dy, w, h) {
  return clampPan({ ...state, x: state.x + dx, y: state.y + dy }, w, h);
}

export function clampPan(state, w, h) {
  const clamp = (value, size) => Math.min(0, Math.max(size - size * state.scale, value));
  return { scale: state.scale, x: clamp(state.x, w), y: clamp(state.y, h) };
}

export const RESET = Object.freeze({ scale: 1, x: 0, y: 0 });
export const STEPS = [1, 1.25, 1.5, 2, 3, 4];

/** How much one wheel event zooms. Proportional to how far it scrolled, and never more than
 * 6% per event: a trackpad sends dozens of tiny events per swipe, a mouse a few big ones. */
export function wheelFactor(deltaY) {
  const factor = Math.exp(-Number(deltaY || 0) * 0.002);
  return Math.min(1.06, Math.max(1 / 1.06, factor));
}

/** Zoom to `scale`, keeping the middle of the picture in the middle. */
export function zoomTo(state, scale, w, h) {
  return zoomAt(state, scale / state.scale, w / 2, h / 2, w, h);
}

/** Wire zoom and pan onto every `.zoombox img` inside `root`, all sharing one state. Each picture
 * sits in its own `.zoombox` (overflow hidden, never transformed), which is what gets measured. */
export function attachZoom(root) {
  const images = [...root.querySelectorAll('.zoombox img')];
  if (!images.length) return;
  let state = RESET, drag = null;
  const bar = root.querySelector('.zoombar');
  if (bar) {
    bar.innerHTML = STEPS.map((s) => `<button type="button" class="chip" data-zoom="${s}">${Math.round(s * 100)}%</button>`).join('')
      + '<output aria-live="polite"></output>';
    bar.querySelectorAll('[data-zoom]').forEach((button) => button.onclick = () => {
      const r = images[0].parentElement.getBoundingClientRect();
      state = zoomTo(state, Number(button.dataset.zoom), r.width, r.height);
      apply();
    });
  }
  const apply = () => {
    images.forEach((img) => {
      img.style.transformOrigin = '0 0';
      img.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
      img.style.cursor = state.scale > 1 ? (drag ? 'grabbing' : 'grab') : 'zoom-in';
    });
    if (!bar) return;
    bar.querySelector('output').textContent = `${Math.round(state.scale * 100)}%`;
    bar.querySelectorAll('[data-zoom]').forEach((b) => b.setAttribute('aria-pressed', String(Math.abs(Number(b.dataset.zoom) - state.scale) < 0.01)));
  };
  images.forEach((img) => {
    const box = img.parentElement;
    img.draggable = false;
    box.addEventListener('wheel', (event) => {
      event.preventDefault();
      const r = box.getBoundingClientRect();
      state = zoomAt(state, wheelFactor(event.deltaY), event.clientX - r.left, event.clientY - r.top, r.width, r.height);
      apply();
    }, { passive: false });
    box.addEventListener('pointerdown', (event) => {
      if (state.scale === 1) return;
      drag = { x: event.clientX, y: event.clientY };
      box.setPointerCapture(event.pointerId);
      apply();
    });
    box.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const r = box.getBoundingClientRect();
      state = panBy(state, event.clientX - drag.x, event.clientY - drag.y, r.width, r.height);
      drag = { x: event.clientX, y: event.clientY };
      apply();
    });
    const stop = () => { drag = null; apply(); };
    box.addEventListener('pointerup', stop);
    box.addEventListener('pointercancel', stop);
    box.addEventListener('dblclick', () => { state = RESET; apply(); });
  });
  apply();
}
