// Library logic for the studio: what an asset has reached, how to say it, what to show.
// No DOM here, so Node tests import this exact file.

export const STEPS = [
  { id: 'picture', label: 'Picture' },
  { id: 'model', label: '3D model' },
  { id: 'finished', label: 'Finish' },
  { id: 'rigged', label: 'Rig' },
  { id: 'animated', label: 'Animate' },
];

const STAGE_INDEX = Object.fromEntries(STEPS.map((step, i) => [step.id, i]));

/** How many steps this asset can go through: props stop at Finish. */
export function stepCount(asset) {
  return asset.kind === 'prop set' ? 3 : STEPS.length;
}

/** How many of those steps are done. */
export function doneCount(asset) {
  return Math.min(stepCount(asset), (STAGE_INDEX[asset.stage] ?? -1) + 1);
}

/** The step to offer next, or null when the asset has gone as far as it can. */
export function nextStep(asset) {
  const done = doneCount(asset);
  return done < stepCount(asset) ? STEPS[done] : null;
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

/** Library filter: all, characters (anything that could be one), or prop sets. */
export function filterAssets(assets, filter) {
  if (filter === 'props') return assets.filter((a) => a.kind === 'prop set');
  if (filter === 'characters') return assets.filter((a) => a.kind !== 'prop set');
  return assets;
}

/** The furthest-along 3D file to show in the viewer, or null for a picture-only asset. */
export function displayModel(asset) {
  if (asset.kind === 'prop set') return asset.props?.[0]?.file || asset.model || null;
  return asset.rigged || asset.finished || asset.model || null;
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
