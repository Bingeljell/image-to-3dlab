import * as THREE from 'three';

export const MODEL_EXTS = new Set(['glb', 'gltf', 'obj']);
export const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'avif']);

export const extOf = (name) => name.split('.').pop().toLowerCase();
export const isModelFile = (file) => MODEL_EXTS.has(extOf(file.name));
export const isImageFile = (file) => IMAGE_EXTS.has(extOf(file.name));

// A bundle is one root file plus sibling resources referenced by a .gltf. Local files do
// not have server paths, so the LoadingManager resolves those references to blob URLs by
// basename. GLB and OBJ inputs use the same asset-spec shape for every workshop room.
function makeManager(blobByName) {
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => {
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    const base = decodeURIComponent(url.split('/').pop().split('?')[0]);
    return blobByName.get(base) || url;
  });
  return manager;
}

/** Convert a dropped file collection into model or reference-image asset specs. */
export function specsFromFiles(fileList) {
  const files = [...fileList];
  const models = files.filter(isModelFile);
  if (models.length) {
    const blobByName = new Map();
    const urls = [];
    for (const file of files) {
      const url = URL.createObjectURL(file);
      urls.push(url);
      blobByName.set(file.name, url);
    }
    const manager = makeManager(blobByName);
    return models.map((file) => ({
      kind: 'model',
      url: blobByName.get(file.name),
      label: file.name,
      ext: extOf(file.name),
      manager,
      revoke: urls,
    }));
  }

  const images = files.filter(isImageFile);
  if (images.length) {
    return images.map((file) => {
      const url = URL.createObjectURL(file);
      return { kind: 'image', url, label: file.name, revoke: [url] };
    });
  }

  throw new Error('nothing loadable — expected a .glb/.gltf/.obj model or a PNG/JPG image');
}

