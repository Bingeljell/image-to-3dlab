// The Animate tab's model list, grouped for its dropdown. Rigged models come first: they
// skip straight to a preset. Pure, so it runs under Node in the tests.
export const KIND_LABELS = { rigged: 'Rigged', finished: 'Finished', generated: 'Generated' };

export function groupModels(models) {
  const groups = [];
  for (const kind of Object.keys(KIND_LABELS)) {
    const items = models.filter((m) => m.kind === kind);
    if (items.length) groups.push({ kind, label: KIND_LABELS[kind], items });
  }
  return groups;
}
