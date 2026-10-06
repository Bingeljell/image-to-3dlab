// A floor to stand on: a wide grid at height 0 that fades out with distance (no visible
// edge to walk off), Blender-style axis lines through the origin (red X, blue Z), and a
// dot at 0,0,0. Models are lifted so their feet touch it.

import * as THREE from 'three';

export const HOME_OFFSET = [1.45, 0.35, 2.45];   // camera, relative to the model's middle

/** Where the starting camera sits and looks, for a model `height` tall standing on the floor. */
export function homeView(height) {
  const target = [0, height / 2, 0];
  return { target, position: target.map((v, i) => v + HOME_OFFSET[i]) };
}

const GRID_VERTEX = `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }`;

// Lines drawn per pixel from world position, so they stay crisp at any distance; a minor
// grid every `cell`, a major one every 10 cells, the X and Z axes coloured, all fading out.
const GRID_FRAGMENT = `
  uniform float cell;
  uniform float fadeFrom;
  uniform float fadeTo;
  varying vec3 vWorld;
  float lines(vec2 p, float size) {
    vec2 g = abs(fract(p / size - 0.5) - 0.5) / fwidth(p / size);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }
  void main() {
    vec2 p = vWorld.xz;
    float minor = lines(p, cell) * 0.35;
    float major = lines(p, cell * 10.0) * 0.6;
    float fade = 1.0 - smoothstep(fadeFrom, fadeTo, length(p));
    vec3 colour = vec3(0.24, 0.19, 0.16);
    float alpha = max(minor, major);
    vec2 axis = abs(p) / fwidth(p);
    if (axis.y < 1.0) { colour = vec3(0.886, 0.341, 0.298); alpha = 0.9; }      // X axis (z = 0): red
    else if (axis.x < 1.0) { colour = vec3(0.357, 0.608, 0.941); alpha = 0.9; } // Z axis (x = 0): blue
    alpha *= fade;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(colour, alpha);
  }`;

export function createFloor({ cell = 0.1, radius = 12 } = {}) {
  const group = new THREE.Group();
  group.name = 'studio-floor';
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(radius * 2, radius * 2).rotateX(-Math.PI / 2),
    new THREE.ShaderMaterial({
      vertexShader: GRID_VERTEX,
      fragmentShader: GRID_FRAGMENT,
      uniforms: { cell: { value: cell }, fadeFrom: { value: radius * 0.15 }, fadeTo: { value: radius * 0.6 } },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      extensions: { derivatives: true },
    }),
  );
  plane.renderOrder = -1;
  group.add(plane);
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

/** The turn about the up axis that makes a character whose feet point along `forward`
 * ([x, y, z], glTF space) face +Z, towards the starting camera. 0 when unknown. */
export function facingAngle(forward) {
  if (!forward || forward.length < 3) return 0;
  const [x, , z] = forward;
  return Math.hypot(x, z) < 1e-6 ? 0 : -Math.atan2(x, z);
}
