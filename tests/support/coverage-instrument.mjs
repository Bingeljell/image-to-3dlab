import { createInstrumenter } from 'istanbul-lib-instrument';

export function instrument() {
  return createInstrumenter({
    coverageGlobalScope: 'globalThis',
    coverageGlobalScopeFunc: false,
    compact: false,
  });
}
