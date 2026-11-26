// Mirror the browser's import map for the Three.js modules shipped with the viewer.
import { registerHooks } from 'node:module';

const vendor = new URL('../../viewer/vendor/', import.meta.url);

export function mapViewerThree({ headless = false } = {}) {
  return registerHooks({
    resolve(specifier, context, nextResolve) {
      if (specifier === 'three') {
        const target = headless ? new URL('./viewer-renderer.js', import.meta.url) : new URL('three.module.js', vendor);
        return nextResolve(target.href, context);
      }
      if (headless && specifier === 'three/addons/renderers/CSS2DRenderer.js') {
        return nextResolve(new URL('./viewer-renderer.js', import.meta.url).href, context);
      }
      if (specifier.startsWith('three/addons/')) {
        return nextResolve(new URL(specifier.slice('three/addons/'.length), vendor).href, context);
      }
      return nextResolve(specifier, context);
    },
  });
}
