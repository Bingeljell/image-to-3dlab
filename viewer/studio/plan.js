// Create: "what do you have?" x "what do you want?" -> which steps run, and the prompt sent.
// No DOM here, so Node tests import this exact file.

export const HAVES = { idea: 'An idea', picture: 'A picture', model: 'A 3D model' };
export const WANTS = {
  picture: { label: 'A picture', hint: 'Concept art, or the start of a 3D model later.' },
  prop: { label: 'One prop', hint: 'One object, game-ready, with detail levels.' },
  set: { label: 'Nine props', hint: 'A 3×3 sheet of props, cut into nine game-ready models.' },
  character: { label: 'A moving character', hint: 'Rigged and animated. Humanoids only for now.' },
};

// What each starting point can lead to; a string is the reason it can't.
export const ALLOWED = {
  idea: { picture: true, prop: true, set: true, character: true },
  picture: { picture: 'You already have the picture.', prop: true, set: true, character: true },
  model: { picture: 'Start from an idea to make a picture.', prop: true, set: 'Upload the 3×3 sheet as a picture instead.', character: true },
};

// Steps by id; `set` finishes by splitting the sheet into props.
// The step-by-step guide on our site; How to use, Create's tips and About all link it.
export const FULL_GUIDE = 'https://assetfurnace.com/blog/how-to-use-assetfurnace';
export const STEP_LABELS = { picture: 'Picture', back: 'Back picture', model: '3D model', finished: 'Finish', split: 'Split into props', rigged: 'Rig', animated: 'Animate' };

/** The steps to run, in order, before any "stop after".
 *  `frontBack`: draw a back picture and build the 3D model from front and back (one prop or
 *  a character; a sheet of nine has no back). */
export function planFor(have, want, { finishUpload = false, frontBack = false } = {}) {
  const full = {
    picture: ['picture'],
    prop: ['picture', 'model', 'finished'],
    set: ['picture', 'model', 'split'],
    character: ['picture', 'model', 'finished', 'rigged', 'animated'],
  }[want] || [];
  const skip = have === 'picture' ? ['picture'] : have === 'model' ? ['picture', 'model'] : [];
  let steps = full.filter((s) => !skip.includes(s));
  // an uploaded model may already be game-ready; finishing it is the user's choice
  if (have === 'model' && !finishUpload) steps = steps.filter((s) => s !== 'finished');
  if (frontBack && want !== 'set' && steps.includes('model')) {
    steps.splice(steps.indexOf('model'), 0, 'back');
  }
  return steps;
}

/** Cut a plan after a step the user clicked, or keep it whole. */
export function stopAfter(steps, last) {
  const i = steps.indexOf(last);
  return i < 0 ? steps : steps.slice(0, i + 1);
}

/** The prompt actually sent: the description plus the goal's proven wording. */
export function composePrompt(text, goal, recipes, useRecipe = true) {
  const base = String(text || '').trim().replace(/[\s,.]+$/, '');
  const wording = useRecipe ? recipes?.goals?.[goal]?.wording : '';
  return wording ? `${base}, ${wording}` : base;
}

/** Gentle warnings for words we know cause trouble for this goal. */
export function troubleFor(text, goal, recipes) {
  const lower = ` ${String(text || '').toLowerCase()} `;
  const found = [];
  for (const rule of recipes?.trouble || []) {
    if (!rule.goals.includes(goal)) continue;
    const hit = rule.words.find((w) => new RegExp(`(^|[^a-z])${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^a-z]|$)`).test(lower));
    if (hit && !found.some((f) => f.message === rule.message)) found.push({ word: hit, message: rule.message });
  }
  return found;
}
