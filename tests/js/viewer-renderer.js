// Keep geometry, loaders, cameras, controls, raycasts and animation real.
export * from '../../viewer/vendor/three.module.js';

export class WebGLRenderer {
  static instances = [];

  constructor(options) {
    this.options = options;
    this.domElement = document.createElement('canvas');
    WebGLRenderer.instances.push(this);
    this.domElement.toDataURL = (format) => `data:${format};base64,test-render`;
    this.domElement.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 300 });
  }
  setPixelRatio(value) { this.pixelRatio = value; }
  setSize(width, height) { this.size = [width, height]; }
  render(scene, camera) {
    scene.updateMatrixWorld();
    camera.updateMatrixWorld();
    Object.assign(this, { scene, camera });
  }
  dispose() { this.disposed = true; }
  forceContextLoss() { this.lost = true; }
}
export class PMREMGenerator {
  fromScene() { return { texture: {} }; }
  dispose() {}
}

import { Object3D } from '../../viewer/vendor/three.module.js';

export class CSS2DRenderer extends WebGLRenderer {}
export class CSS2DObject extends Object3D {
  constructor(element) {
    super();
    this.element = element;
  }
}
