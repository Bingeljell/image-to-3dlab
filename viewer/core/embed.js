// The Generate tab previews its result by loading this same page in an iframe with
// ?restricted=1. That copy must be a bare 3D view: no tab bar, no About, no Setup landing,
// no update banner -- otherwise the whole app appears nested inside its own panel.
export function isEmbedded(search) {
  return new URLSearchParams(search).get('restricted') === '1';
}

// A link to a model (?a=..., which `serve.py --open` writes) asks to look at that model,
// so it opens on Compare, with the rest of the app around it. Compare loads from any of
// a to d, and an empty value loads nothing, so only a filled one counts.
export function linksAModel(search) {
  const params = new URLSearchParams(search);
  return ['a', 'b', 'c', 'd'].some((key) => (params.get(key) || '').trim() !== '');
}

export function landingMode({ embedded, linked = false, skipSetup }) {
  if (embedded || linked) return 'compare';
  return skipSetup ? 'generate' : 'setup';
}
