// What a Setup card's download brings beyond its listed weights (a picture engine, a
// vision projector), named in the confirmation before anyone says yes (AGENTS.md).
/** The confirmation's lines for what setup fetches beyond the weights (an engine, a projector). */
export function alsoFetchesLines(backend) {
  const extra = backend.also_fetches || [];
  return extra.length ? ['', 'Also downloads:',
    ...extra.map((x) => `  ${x.what}: ${x.size}, ${x.licence}\n    from ${x.from}`)] : [];
}
