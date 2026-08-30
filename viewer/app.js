import './modes/compare.js';

import './modes/generate.js';
import './modes/rig-review.js';
import './modes/animate.js';

const byId = (id) => document.getElementById(id);
const modes = {
  compare: byId('compare-view'),
  generate: byId('generate-view'),
  rig: byId('rig-view'),
  animate: byId('animate-view'),
  credits: byId('credits-view'),
};

function setMode(activeMode) {
  modes.compare.classList.toggle('hidden', activeMode !== 'compare');
  modes.generate.hidden = activeMode !== 'generate';
  modes.rig.hidden = activeMode !== 'rig';
  modes.animate.hidden = activeMode !== 'animate';
  modes.credits.hidden = activeMode !== 'credits';
  for (const mode of Object.keys(modes)) {
    byId(`mode-${mode}`).classList.toggle('on', mode === activeMode);
  }
  document.dispatchEvent(new CustomEvent('viewer:modechange', { detail: { mode: activeMode } }));
}

for (const mode of Object.keys(modes)) {
  byId(`mode-${mode}`).onclick = () => setMode(mode);
}

setMode('compare');
