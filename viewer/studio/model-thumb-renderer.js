// The three.js half of the Library thumbnails: one small renderer, reused for every row.
//
// Kept apart from model-thumbs.js so the planning, caching and queueing can be tested under
// Node without WebGL. Everything a render allocates (geometry, materials, textures) is
// disposed straight after the frame; the 160 px context is reused and can be dropped with
// dispose().

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const SIZE = 160;
const FOV = 32;

export function createThumbnailRenderer({ size = SIZE } = {}) {
  let renderer = null;
  let loader = null;

  const context = () => {
    if (renderer) return renderer;
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    renderer.setSize(size, size, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    loader = new GLTFLoader();
    return renderer;
  };

  const disposeObject = (root) => {
    root.traverse((object) => {
      object.geometry?.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (!material) continue;
        for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
        material.dispose();
      }
    });
  };

  return {
    async render(url) {
      const target = context();
      const gltf = await loader.loadAsync(url);
      const scene = new THREE.Scene();
      const root = gltf.scene;
      scene.add(root);
      scene.add(new THREE.HemisphereLight(0xffffff, 0x2a211c, 2.4));
      const key = new THREE.DirectionalLight(0xffffff, 2.2);
      key.position.set(2, 4, 3);
      scene.add(key);

      // Frame the model the way the viewer does: its bounding box, a little breathing room.
      const box = new THREE.Box3().setFromObject(root);
      const extent = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const radius = Math.max(extent.x, extent.y, extent.z) || 1;
      const camera = new THREE.PerspectiveCamera(FOV, 1, radius / 100, radius * 30);
      const distance = (radius / 2) / Math.tan((FOV * Math.PI / 180) / 2) * 1.2;
      camera.position.copy(center)
        .add(new THREE.Vector3(0.55, 0.25, 1).normalize().multiplyScalar(distance));
      camera.lookAt(center);

      target.render(scene, camera);
      const data = target.domElement.toDataURL('image/png');
      disposeObject(root);
      return data;
    },
    dispose() {
      renderer?.dispose();
      renderer?.forceContextLoss?.();
      renderer = null;
      loader = null;
    },
  };
}
