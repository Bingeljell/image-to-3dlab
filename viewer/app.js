import './modes/compare.js';
import { skipRequested } from './modes/setup.js';
import { welcomeOnArrival } from './modes/welcome.js';
import { checkForUpdates } from './modes/update.js';

import './modes/generate-image.js';
import './modes/generate.js';
import './modes/finish.js';
import './modes/props.js';
import './modes/rig-review.js';
import './modes/animate.js';
import './modes/animate-flow.js';
import { subscribeRigEditState } from './core/rig-edit-state.js';
import { isEmbedded, landingMode, linksAModel } from './core/embed.js';
import { sidewaysScroll } from './core/sideways-wheel.js';

const byId = (id) => document.getElementById(id);
const modes = {
  setup: byId('setup-view'),
  compare: byId('compare-view'),
  'generate-image': byId('generate-image-view'),
  generate: byId('generate-view'),
  finish: byId('finish-view'),
  props: byId('props-view'),
  rig: byId('rig-view'),
  animate: byId('animate-view'),
  about: byId('about-view'),
};

function setMode(activeMode) {
  modes.setup.hidden = activeMode !== 'setup';
  modes.compare.classList.toggle('hidden', activeMode !== 'compare');
  modes['generate-image'].hidden = activeMode !== 'generate-image';
  modes.generate.hidden = activeMode !== 'generate';
  modes.finish.hidden = activeMode !== 'finish';
  modes.props.hidden = activeMode !== 'props';
  modes.rig.hidden = activeMode !== 'rig';
  modes.animate.hidden = activeMode !== 'animate';
  modes.about.hidden = activeMode !== 'about';
  for (const mode of Object.keys(modes)) {
    byId(`mode-${mode}`).classList.toggle('on', mode === activeMode);
  }
  document.dispatchEvent(new CustomEvent('viewer:modechange', { detail: { mode: activeMode } }));
}

for (const mode of Object.keys(modes)) {
  byId(`mode-${mode}`).onclick = () => setMode(mode);
}

// The About page's "Get started" button, and its first-visit landing, ask for a screen
// by name.
document.addEventListener('viewer:navigate', (event) => {
  if (modes[event.detail?.mode]) setMode(event.detail.mode);
});

subscribeRigEditState(({ pendingCount }) => {
  const button = byId('mode-rig');
  button.classList.toggle('has-pending', pendingCount > 0);
  button.title = pendingCount
    ? `${pendingCount} rig edit${pendingCount === 1 ? '' : 's'} awaiting Blender rebind`
    : 'Correct the rest skeleton';
});

// First run lands on Setup & Status, because a fresh clone can generate nothing until
// weights exist and the page is where that is explained. Once the user ticks "skip this
// next time" it is never the landing page again -- it stays one click away in the menu.
// Inside the Generate tab's preview iframe the page is just a 3D view: hide the app chrome
// and skip the About/update checks, which would otherwise navigate the iframe elsewhere.
const embedded = isEmbedded(location.search);
const linked = linksAModel(location.search);
document.documentElement.classList.toggle('embedded', embedded);
setMode(landingMode({ embedded, linked, skipSetup: skipRequested() }));
// The studio links here by tab name (index.html#setup); a known name wins over the landing tab.
if (modes[location.hash.slice(1)]) setMode(location.hash.slice(1));
if (!embedded) {
  welcomeOnArrival({ linked });
  checkForUpdates();
}

// In a narrow window the menu scrolls sideways with no scrollbar; let a mouse wheel do it.
const menu = byId('mode-switch');
menu?.addEventListener('wheel', (event) => {
  const next = sidewaysScroll(event, menu);
  if (next === null) return;
  event.preventDefault();
  menu.scrollLeft = next;
}, { passive: false });
