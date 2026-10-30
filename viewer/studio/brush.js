// The back picture's brush: paint over the part to change, say what goes there, and only that
// part is drawn again. The painted mask is kept at the picture's own size, so a stroke lands on
// the same pixels whatever size the picture is shown at.

/** A point on the shown canvas, in the picture's own pixels. */
export function toPicture(x, y, shown, natural) {
  return [x * natural.w / shown.w, y * natural.h / shown.h];
}

/** The request the server's /api/image/brush reads. */
export function brushForm(picture, instruction, mask, seed, engine = '') {
  const form = new FormData();
  if (engine) form.append('engine', engine);   // empty: the engine that drew the back
  form.append('picture', picture);
  form.append('instruction', instruction);
  form.append('seed', String(seed));
  form.append('mask', mask, 'mask.png');
  return form;
}

/**
 * A paintable copy of the picture at `imageUrl`, put inside `box`. Returns `painted()` (any
 * stroke yet), `clear()`, and `maskBlob()` (a PNG, white where painted, at the picture's size).
 */
export function mountBrush(box, imageUrl, { size = 48, tint = 'rgba(255, 60, 40, 0.5)' } = {}) {
  const shown = document.createElement('canvas');
  shown.className = 'brushcanvas';
  shown.style.touchAction = 'none';
  box.appendChild(shown);
  const mask = document.createElement('canvas'), strokes = document.createElement('canvas');
  const image = new Image();
  let width = 0, painted = false, last = null;
  const state = { size };

  const redraw = () => {
    const g = shown.getContext('2d');
    g.clearRect(0, 0, shown.width, shown.height);
    g.drawImage(image, 0, 0, shown.width, shown.height);
    g.drawImage(strokes, 0, 0, shown.width, shown.height);
  };
  image.onload = () => {
    width = Math.min(box.clientWidth || 320, 420);
    shown.width = width;
    shown.height = Math.round(width * image.naturalHeight / image.naturalWidth);
    for (const c of [mask, strokes]) { c.width = image.naturalWidth; c.height = image.naturalHeight; }
    const m = mask.getContext('2d');
    m.fillStyle = '#000';
    m.fillRect(0, 0, mask.width, mask.height);
    redraw();
  };
  image.src = imageUrl;

  const at = (event) => {
    const rect = shown.getBoundingClientRect();
    return toPicture(event.clientX - rect.left, event.clientY - rect.top, { w: rect.width, h: rect.height },
      { w: mask.width, h: mask.height });
  };
  const line = (from, to) => {
    for (const [c, colour] of [[mask, '#fff'], [strokes, tint]]) {
      const g = c.getContext('2d');
      g.strokeStyle = colour;
      g.lineWidth = state.size * mask.width / 1024;
      g.lineCap = g.lineJoin = 'round';
      g.beginPath();
      g.moveTo(...from);
      g.lineTo(...to);
      g.stroke();
    }
    painted = true;
    redraw();
  };
  shown.addEventListener('pointerdown', (event) => {
    shown.setPointerCapture(event.pointerId);
    last = at(event);
    line(last, last);
  });
  shown.addEventListener('pointermove', (event) => {
    if (!last) return;
    const point = at(event);
    line(last, point);
    last = point;
  });
  const stop = () => { last = null; };
  shown.addEventListener('pointerup', stop);
  shown.addEventListener('pointercancel', stop);

  return {
    painted: () => painted,
    setSize: (value) => { state.size = Number(value) || size; },
    clear: () => {
      painted = false;
      const m = mask.getContext('2d');
      m.fillStyle = '#000';
      m.fillRect(0, 0, mask.width, mask.height);
      strokes.getContext('2d').clearRect(0, 0, strokes.width, strokes.height);
      redraw();
    },
    maskBlob: () => new Promise((resolve) => mask.toBlob(resolve, 'image/png')),
  };
}
