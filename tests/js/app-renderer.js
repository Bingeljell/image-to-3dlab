// Rendering calls are observed without starting a graphics context in Node.
export function createStudioViewer(options) {
  const trace = options.stage.ownerDocument.defaultView.renderTrace;
  trace.options = options;
  return Object.fromEntries(['load', 'showEmpty', 'showImage', 'setFacing', 'setClips'].map((name) => [
    name, (...args) => { trace.calls.push([name, ...args]); },
  ]));
}

export function createThumbnailRenderer() {
  return { render: (file) => window.renderTrace.thumbnail(file) };
}
