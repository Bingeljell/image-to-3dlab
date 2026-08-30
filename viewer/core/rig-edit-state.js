const listeners = new Set();
let state = Object.freeze({ pendingCount: 0, assetLabel: null });

/** Publish only cross-room status; correction data remains owned by Rig Edit. */
export function setRigEditState({ pendingCount = 0, assetLabel = null } = {}) {
  state = Object.freeze({
    pendingCount: Math.max(0, Math.trunc(Number(pendingCount) || 0)),
    assetLabel: typeof assetLabel === 'string' && assetLabel ? assetLabel : null,
  });
  listeners.forEach((listener) => listener(state));
  return state;
}

export function getRigEditState() {
  return state;
}

export function subscribeRigEditState(listener) {
  listeners.add(listener);
  listener(state);
  return () => listeners.delete(listener);
}
