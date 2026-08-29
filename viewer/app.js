import './modes/compare.js';

import './modes/generate.js';

const byId = (id) => document.getElementById(id);
const modes = {
  compare: byId('compare-view'),
  generate: byId('generate-view'),
  credits: byId('credits-view'),
};

function setMode(activeMode) {
  modes.compare.classList.toggle('hidden', activeMode !== 'compare');
  modes.generate.hidden = activeMode !== 'generate';
  modes.credits.hidden = activeMode !== 'credits';
  for (const mode of Object.keys(modes)) {
    byId(`mode-${mode}`).classList.toggle('on', mode === activeMode);
  }
}

for (const mode of Object.keys(modes)) {
  byId(`mode-${mode}`).onclick = () => setMode(mode);
}

setMode('compare');
