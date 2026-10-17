// The progress panel on Setup & Status: what this setup is doing right now, what it became
// of, and the state the panel is painted with.
//
// The title used to be written once, when the download was started, and never touched again
// — so a 16 GB fetch that finished perfectly said "Downloading" forever, beside a bar at 100%
// and a detail line reading "done · 16.8 GB on disk". Two copies of a ternary wrote that
// sentence, which is how it drifted out of sync with its own log.
//
// Pure, so it can be exercised under Node (tests/test_viewer_js_modules.py) against the
// shipped file rather than a copy of these rules.

import { MARK } from '../studio/pages.js';

// The endings, and the mark each one wears. Taken from the studio's Activity list, which
// already marks a finished thing this way: one convention, not two.
const ENDINGS = {
  done: { mark: MARK.done, verb: 'Installed' },
  error: { mark: MARK.error, verb: 'Setup failed' },
  cancelled: { mark: MARK.cancelled, verb: 'Setup cancelled' },
};

/** Where a run has got to: one of the three endings, or still going. */
export function panelState(event) {
  const phase = event && event.phase;
  return phase in ENDINGS ? phase : 'running';
}

/** The verb for a run that has not finished yet, from what we know when it is started.

 *  `build_present` decides "Setting up" over "Downloading" and stays right for the whole run:
 *  a route with no build yet compiles first, and a route that is built only fetches weights. */
export function setupVerb(backend, action) {
  if (action === 'rebuild') return 'Rebuilding';
  if (backend && !backend.build_present) return 'Setting up';
  return backend && backend.setup_fetches_weights ? 'Downloading' : 'Building';
}

/** The panel's title: what it became, marked, once the run is over; what it is doing until then.

 *  `done` is the phase the server emits after the exit code is checked and the bytes are
 *  counted, so it is the only honest moment to call the backend installed. */
export function runTitle(backend, action, event) {
  const label = (backend && backend.label) || 'this backend';
  const ending = ENDINGS[panelState(event)];
  if (ending) return `${ending.mark} ${ending.verb} · ${label}`;
  return `${setupVerb(backend, action)} ${label}`;
}