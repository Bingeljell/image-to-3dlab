/** Selection state used by the real Library row renderer. */
export function libraryRowSelection(asset, { current, inputId, viewingMaking = false }) {
  return !viewingMaking && (inputId ? asset.id === inputId : asset === current);
}

export function libraryFamilyOpen(members, { key, openFamilies, current, inputId }) {
  return openFamilies.has(key) || members.includes(current) || members.some((asset) => asset.id === inputId);
}
