// A floor to stand on: a faint grid at height 0, Blender-style axis lines through the
// origin (red X, blue Z), and a dot at 0,0,0. Models are lifted so their feet touch it.

import * as THREE from 'three';

export const HOME_OFFSET = [1.45, 0.35, 2.45];   // camera, relative to the model's middle

/** Where the starting camera sits and looks, for a model `height` tall standing on the floor. */
export function homeView(height) {
  const target = [0, height / 2, 0];
  return { target, position: target.map((v, i) => v + HOME_OFFSET[i]) };
}

export function createFloor({ size = 4, divisions = 20 } = {}) {
  const group = new THREE.Group();
  group.name = 'studio-floor';
  const grid = new THREE.GridHelper(size, divisions, 0x3a2d26, 0x2a201b);
  grid.material.transparent = true;
  grid.material.opacity = 0.9;
  group.add(grid);
  const line = (from, to, colour) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...from), new THREE.Vector3(...to)]);
    const mesh = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.85 }));
    mesh.position.y = 0.001;  // just above the grid, so it wins
    return mesh;
  };
  const half = size / 2;
  group.add(line([-half, 0, 0], [half, 0, 0], 0xe2574c));   // X: red, as in Blender
  group.add(line([0, 0, -half], [0, 0, half], 0x5b9bf0));   // Z (three.js forward): blue
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: 0xf4e8de }));
  group.add(dot);
  return group;
}

/** Lift a loaded model so its lowest point sits on the floor; returns its height. */
export function standOnFloor(root) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return 1;
  root.position.y -= box.min.y;
  root.updateMatrixWorld(true);
  return box.max.y - box.min.y;
}
