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

/** What Finish's Detail control sends: a triangle count, or full detail (every triangle kept). */
export function finishDetailSettings(choice, custom) {
  return choice === 'full' ? { keep_mesh: true } : { faces: finishFaces(choice, custom) };
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

/** Whether a done step offers "View": it made its own GLB and that is not the one the viewer
 *  already shows. This is how the raw 3D model (Pixal3D, TRELLIS, ...) stays one click away
 *  after Finish, Rig and Animate have moved past it. */
export function canViewStep(asset, stepId) {
  if (asset.kind === 'prop set' || stepId === 'rigged') return false;
  const file = stepFile(asset, stepId);
  return !!file && file !== displayModel(asset);
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

/** Which 3D engine made a model, read from its run folder's name. */
export function engineOf(path) {
  const run = path?.split('/').slice(-2, -1)[0] || '';
  const match = run.match(/__(pixal3d|trellis|hunyuan[a-z-]*|sf3d)__/);
  return match ? { pixal3d: 'Pixal3D', trellis: 'TRELLIS.2', sf3d: 'Stable Fast 3D' }[match[1]] || 'Hunyuan3D' : 'made outside the app';
}

/** The engine for the Library's filter: nothing for a picture with no model yet. */
export function assetEngine(asset) {
  return asset.model ? engineOf(asset.model) : asset.finished ? 'made outside the app' : '';
}

/** The engines present in the Library, for the filter's choices. */
export function engineChoices(assets) {
  return [...new Set(assets.map(assetEngine).filter(Boolean))].sort();
}

/**
 * The Library as families: one row per picture's family, showing the version marked main (the star),
 * else the newest. `members` holds every shown version, the original first. Paging counts families.
 */
export function visibleFamilies(assets, { limit = 30, stage = 'all', engine = 'all', starred = false, sort = 'newest', ...options } = {}) {
  const found = visibleAssets(assets, { ...options, limit: Infinity });
  const { hiddenCount } = found;
  const rows = found.rows.filter((a) => (stage === 'all' || a.stage === stage)
    && (engine === 'all' || assetEngine(a) === engine) && (!starred || a.main));
  const families = new Map();
  for (const asset of rows) {
    const key = asset.family || asset.id;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(asset);
  }
  const all = [...families.entries()].map(([key, members]) => {
    members.sort((a, b) => (a.version || 1) - (b.version || 1));
    const shown = members.find((m) => m.main) ?? members.reduce((best, m) => (m.updated > best.updated ? m : best));
    return { key, shown, members };
  });
  // newest first is the order the assets already arrive in; by name goes by the original's name
  if (sort === 'name') all.sort((a, b) => a.members[0].name.localeCompare(b.members[0].name, undefined, { sensitivity: 'base' }));
  return { rows: all.slice(0, limit), more: Math.max(0, all.length - limit), hiddenCount };
}

/** After a step finishes: the version it just made (an id the Library did not have, in the
 *  same family), or null when it only added to the version already open. */
export function madeVersion(before, after, familyKey) {
  const known = new Set(before.map((a) => a.id));
  const fresh = after.filter((a) => !known.has(a.id) && (a.family || a.id) === familyKey);
  return fresh.reduce((best, a) => (!best || a.updated > best.updated ? a : best), null);
}

/**
 * What the studio shows when it opens: a run still going on the server, else Create. Never the
 * last asset: most visits are to make something, and the Library beside it is one click away.
 */
export function openingView(running) {
  return running ? 'running' : 'create';
}

/** The asset a link asks for (`studio.html#asset=<id>`, what an agent opens), or null. */
export function linkedAssetId(hash) {
  const id = new URLSearchParams(String(hash || '').replace(/^#/, '')).get('asset');
  return id && /^[0-9a-f]{12}$/.test(id) ? id : null;
}

/** Whether a file can be opened in the viewer. GLB only: a .gltf points at files beside it that a dropped file doesn't bring. */
export function isModelFile(name) {
  return /.\.glb$/i.test(name || '');
}

/** A byte count in plain words: "480 KB", "12 MB", "1.4 GB". */
export function sizeText(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return '';
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  if (n < 1024 ** 3) return `${Math.round(n / 1024 ** 2)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

/** A model's stats (GET /api/model-info) as [label, value] rows. Bones and moves only when
 *  there are some. Textures are grouped by size: "2 × 2048², 1 × 1024²". */
export function statsRows(stats) {
  if (!stats) return [];
  const count = (n) => Number(n).toLocaleString('en-US');
  const sizes = new Map();
  for (const t of stats.textures || []) {
    const key = t.width && t.height ? (t.width === t.height ? `${t.width}²` : `${t.width}×${t.height}`) : '';
    sizes.set(key, (sizes.get(key) || 0) + 1);
  }
  const textures = [...sizes].every(([key]) => key)
    ? [...sizes].map(([key, n]) => `${n} × ${key}`).join(', ')
    : String((stats.textures || []).length);
  const rows = [['Triangles', count(stats.triangles)], ['Vertices', count(stats.vertices)]];
  if (stats.textures?.length) rows.push(['Textures', textures]);
  if (stats.bones) rows.push(['Bones', count(stats.bones)]);
  if (stats.moves?.length) rows.push(['Moves', count(stats.moves.length)]);
  rows.push(['File size', sizeText(stats.bytes)]);
  return rows;
}

/** Show in folder opens a window on the server's screen, so only offer it there. */
export function canReveal(hostname) {
  return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(hostname);
}

/** What deleting a version removes, in words, from the server's plan: "its rig and moves, its finished model". */
export function deletionWords(plan) {
  const part = (folder) => {
    const top = folder.path.split('/')[0];
    return { animate: 'its rig and moves', finish: 'its finished model', images: 'its picture', props: 'its split props' }[top] || 'its 3D model';
  };
  const words = [...new Set(plan.folders.map(part))];
  return words.length > 1 ? `${words.slice(0, -1).join(', ')} and ${words.at(-1)}` : words[0] || 'nothing';
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
