// Which of an asset's models the viewer can show, for the switcher above it.
//
// The Steps panel's "View" links do the same job, but they sit in a crowded column and only
// appear when a step's file differs from the latest one. So after opening the raw 3D model
// there is no obvious way back to Finished: the Finished row offers nothing, because it is
// what the viewer would show by default. This lists every model the asset actually has.

const MODELS = [
  { step: 'model', label: '3D model' },
  { step: 'finished', label: 'Finished' },
  { step: 'rigged', label: 'Rigged' },
];

/** The models this asset has, in the order they were made. Empty for a picture or a prop set. */
export function modelChoices(asset) {
  if (!asset || asset.kind === 'prop set') return [];
  const files = { model: asset.model, finished: asset.finished, rigged: asset.rigged };
  return MODELS
    .map(({ step, label }) => ({ step, label, file: files[step] }))
    .filter((choice) => Boolean(choice.file));
}

/** Which chip is showing: the step picked by hand, else the step whose file the viewer shows
 *  by default (`latest`). Null when a picture is on screen instead of a model. */
export function activeChoice(choices, { step = null, latest = null } = {}) {
  if (step) return choices.find((choice) => choice.step === step)?.step ?? null;
  if (!latest) return null;
  return choices.find((choice) => choice.file === latest)?.step ?? null;
}
