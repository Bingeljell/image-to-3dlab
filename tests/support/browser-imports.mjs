import { registerHooks } from 'node:module';

const vendor = new URL('../../viewer/vendor/', import.meta.url);

// Mirror the studio page's import map with the shipped, offline modules.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'three') {
      return nextResolve(new URL('three.module.js', vendor).href, context);
    }
    if (specifier.startsWith('three/addons/')) {
      const path = specifier.slice('three/addons/'.length);
      return nextResolve(new URL(path, vendor).href, context);
    }
    return nextResolve(specifier, context);
  },
});
