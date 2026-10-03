// --- Welcome card content ---------------------------------------------------------------
// What the card says, decided without touching the page, so Node can test it. The page
// half lives in modes/welcome.js. The payload comes from GET /api/welcome
// (viewer/welcome_api.py).

const SECTION_ORDER = ['Added', 'Changed', 'Fixed', 'Removed', 'Security'];

/** First visit, or news since the version this browser last saw. Not when the page was
 *  opened on a model (`linked`): that visit is for the model, and the card waits. */
export function shouldShow(lastSeen, payload, linked = false) {
  return !linked && (!lastSeen || (payload.news || []).length > 0);
}

/** The kicker and title. `reopened` is a click on the brand, not an arrival. */
export function greeting(lastSeen, payload, reopened = false) {
  const name = payload.brand.name;
  if (reopened || !lastSeen) {
    return { kicker: reopened ? `Version ${payload.version}` : 'Hello',
             title: `Welcome to ${name}` };
  }
  return { kicker: `Updated to version ${payload.version}`, title: `Welcome back to ${name}` };
}

/** A release's headlines, Added first, cut to `limit`, with a count of the rest. */
export function newsItems(release, limit = 6) {
  const all = [];
  for (const section of SECTION_ORDER) all.push(...(release.sections[section] || []));
  return { items: all.slice(0, limit), more: Math.max(0, all.length - limit) };
}

/** HTML-escape, then turn `backticks` into <code>. Changelog text is ours, but still. */
export function inlineCode(text) {
  const escaped = String(text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  return escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
}

/** Where "Get started" goes: setup until something runs, then the first tool that does. */
export function primaryAction(payload) {
  const routes = payload.routes || [];
  if (!routes.length) return null;
  const ready = routes.filter((route) => route.state === 'ready');
  if (!ready.length) return { label: 'Set up a route', mode: 'setup' };
  const imageReady = ready.some((route) => route.id === 'qwen-image');
  return { label: 'Make something', mode: imageReady ? 'generate-image' : 'generate' };
}
