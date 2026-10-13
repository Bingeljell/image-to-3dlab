// Library logic for the studio: what an asset has reached, how to say it, what to show.
// No DOM here, so Node tests import this exact file.

export const STEPS = [
  { id: 'picture', label: 'Picture' },
  { id: 'model', label: '3D model' },
  { id: 'finished', label: 'Finish' },
  { id: 'rigged', label: 'Rig' },
  { id: 'animated', label: 'Animate' },
];

/** How many steps this asset can go through: props stop at Finish. */
export function stepCount(asset) {
  return asset.kind === 'prop set' ? 3 : STEPS.length;
}

/** Whether one step is done, judged by the file it makes, not by order: a model can be
 *  rigged without being finished, and an uploaded model with no picture counts past it. */
export function stepDone(asset, stepId) {
  const set = asset.kind === 'prop set';
  const has = {
    model: set ? !!asset.model : !!(asset.model || asset.finished || asset.rigged),
    finished: set ? !!asset.props?.length : !!asset.finished,
    rigged: !!asset.rigged,
    animated: !!asset.clips?.length,
  };
  if (stepId === 'picture') return !!asset.picture || has.model;
  return !!has[stepId];
}

/** Whether a step that is not done can run now: the file it starts from is there. */
export function stepReady(asset, stepId) {
  const model = !!(asset.model || asset.finished);
  return { model: !!asset.picture, finished: model, rigged: model, animated: !!asset.rigged }[stepId] ?? false;
}

/** 'done', 'now' (can run) or 'later' (waits on an earlier step), per step. */
export function stepState(asset, stepId) {
  if (stepDone(asset, stepId)) return 'done';
  return stepReady(asset, stepId) ? 'now' : 'later';
}

/** How many of those steps are done. */
export function doneCount(asset) {
  return STEPS.slice(0, stepCount(asset)).filter((step) => stepDone(asset, step.id)).length;
}

/** The first step not done yet, or null when the asset has gone as far as it can. */
export function nextStep(asset) {
  return STEPS.slice(0, stepCount(asset)).find((step) => !stepDone(asset, step.id)) ?? null;
}

/** The triangle count Finish's Detail control asks for: a preset, or the custom box. */
export function finishFaces(choice, custom) {
  return Math.round(Number(choice === 'custom' ? custom : choice));
}

/** One short line for the Library card. */
export function statusText(asset) {
  if (asset.kind === 'prop set') {
    const n = asset.props?.length || 0;
    return n ? `${n} prop${n === 1 ? '' : 's'}, game-ready` : '3D model, not split yet';
  }
  if (asset.stage === 'animated') {
    const n = asset.clips?.length || 0;
    return `Animated · ${n} move${n === 1 ? '' : 's'}`;
  }
  return { picture: 'Picture', model: '3D model', finished: 'Finished', rigged: 'Rigged, ready to animate' }[asset.stage] || asset.stage;
}

const PROP_KINDS = new Set(['prop set', 'prop']);

/** Library filter: all, characters (anything that could be one), or props (single props
 *  made as props, and prop sheets). */
export function filterAssets(assets, filter) {
  if (filter === 'props') return assets.filter((a) => PROP_KINDS.has(a.kind));
  if (filter === 'characters') return assets.filter((a) => !PROP_KINDS.has(a.kind));
  return assets;
}

/** The furthest-along 3D file to show in the viewer, or null for a picture-only asset. */
export function displayModel(asset) {
  if (asset.kind === 'prop set') return asset.props?.[0]?.file || asset.model || null;
  return asset.rigged || asset.finished || asset.model || null;
}

/** The GLB one finished step made, so each step can be downloaded on its own: the raw
 *  model from before Finish and Pixel Match, not only the latest one. */
export function stepFile(asset, stepId) {
  return { model: asset.model, finished: asset.finished, rigged: asset.rigged }[stepId] || null;
}

/** Whether the Picture step offers "View pictures": there is a picture, and a model is
 *  covering it in the viewer (a picture-only asset already shows its picture). */
export function canViewPictures(asset) {
  return !!asset.picture && !!displayModel(asset);
}

/** A served URL for a path the assets API gave, relative to its base. */
export function servedUrl(base, relative) {
  if (!relative) return null;
  return base + relative.split('/').map(encodeURIComponent).join('/');
}

/**
 * What the Library list shows: filtered, searched, hidden ones out unless asked for, and
 * only the first `limit`. Returns the rows plus how many more there are and how many are hidden.
 */
export function visibleAssets(assets, { filter = 'all', query = '', showHidden = false, limit = 30 } = {}) {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matching = filterAssets(assets, filter)
    .filter((a) => words.every((w) => a.name.toLowerCase().includes(w)));
  const hiddenCount = matching.filter((a) => a.hidden).length;
  const pool = showHidden ? matching : matching.filter((a) => !a.hidden);
  return { rows: pool.slice(0, limit), more: Math.max(0, pool.length - limit), hiddenCount };
}

/**
 * What the studio shows when it opens: a run still going on the server, else Create. Never the
 * last asset: most visits are to make something, and the Library beside it is one click away.
 */
export function openingView(running) {
  return running ? 'running' : 'create';
}

/** Whether a file can be opened in the viewer. GLB only: a .gltf points at files beside it that a dropped file doesn't bring. */
export function isModelFile(name) {
  return /.\.glb$/i.test(name || '');
}

/** How long a step took, in plain words: "42s", "4m 12s", "1h 3m"; "" when unknown. */
export function duration(seconds) {
  const s = Number(seconds);
  if (seconds == null || !Number.isFinite(s) || s < 0) return '';
  const whole = Math.floor(s);
  if (whole < 60) return `${whole}s`;
  if (whole < 3600) return `${Math.floor(whole / 60)}m ${whole % 60}s`;
  return `${Math.floor(whole / 3600)}h ${Math.floor((whole % 3600) / 60)}m`;
}
